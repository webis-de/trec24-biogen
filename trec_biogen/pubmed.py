from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from random import shuffle
from re import compile as re_compile
from typing import Iterator, Iterable
from warnings import warn

from elasticsearch7_dsl import Document, Date, Text, Keyword, InnerDoc, Nested
from joblib import Memory
from pubmed_parser import parse_medline_xml
from tqdm.auto import tqdm

from trec_biogen import PROJECT_DIR


_memory = Memory(
    location=PROJECT_DIR / "data" / "cache",
    verbose=0,
)


class Author(InnerDoc):
    lastname: str | None = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore
    forename: str | None = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore
    initials: str | None = Keyword()  # type: ignore
    orcid: str | None = Keyword()  # type: ignore
    affiliation: str | None = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore

    @property
    def pmc_id_url(self) -> str | None:
        if self.pmc_id is None:
            return None
        return f"https://ncbi.nlm.nih.gov/pmc/articles/{self.pmc_id}"


class MeshTerm(InnerDoc):
    mesh_id: str = Keyword()  # type: ignore
    """MeSH ID"""
    term: str = Keyword()  # type: ignore
    qualifiers: list[str] = Keyword()  # type: ignore


def _parse_required(value: str) -> str:
    if len(value) == 0:
        raise RuntimeError("String must not be empty.")
    return value


def _parse_optional(value: str) -> str | None:
    if len(value) == 0:
        return None
    return value


def _parse_list(value: str) -> list[str]:
    if len(value) == 0:
        return []
    return [item.strip() for item in value.split("; ")]


_PATTERN_ORCID = re_compile(
    r"(?:"
    r"https?:\/\/:?(?:www\.)?orcid\.org\/\/?"
    r"|"
    r"%20|s|\""
    r")?"
    r"(?:"
    r"(?P<part1>[0-9]{4})[- ]"
    r"(?P<part2>[0-9]{4})[- ]"
    r"(?P<part3>[0-9]{4})[- ]"
    r"(?P<part4>[0-9]{3}[0-9Xx]?)"
    r"|"
    r"(?P<part1_alt>[0-9]{4})"
    r"(?P<part2_alt>[0-9]{4})"
    r"(?P<part3_alt>[0-9]{4})"
    r"(?P<part4_alt>[0-9]{3}[0-9Xx]?)"
    r")"
    r"(?:"  # Suffix
    r"\/"
    r")?"
)


def _parse_orcid(value: str) -> str | None:
    value = value.replace("​", "").strip()
    if len(value) == 0:
        return None
    match = _PATTERN_ORCID.fullmatch(value)
    if match is None:
        warn(RuntimeWarning(
            f"Could not parse author identifier: {value}"))
        return None
    groups = match.groupdict()
    part1: str
    part2: str
    part3: str
    part4: str
    if groups["part1_alt"] is not None:
        part1 = groups["part1_alt"]
        part2 = groups["part2_alt"]
        part3 = groups["part3_alt"]
        part4 = groups["part4_alt"]
        part4 = part4.ljust(4, "X")
    else:
        part1 = groups["part1"]
        part2 = groups["part2"]
        part3 = groups["part3"]
        part4 = groups["part4"]
        if part1 is None:
            part1 = "0000"
        part1 = part1.rjust(4, "0")
        part2 = part2.rjust(4, "0")
        part3 = part3.rjust(4, "0")
        part4 = part4.ljust(4, "X")
    for part in (part1, part2, part3, part4):
        if part is None or len(part) != 4:
            warn(RuntimeWarning(
                f"Could not parse author identifier: {value}"))
            return None
    return f"{part1}-{part2}-{part3}-{part4}".upper()


def _parse_authors(values: list[dict[str, str]]) -> list[Author]:
    return [
        Author(
            lastname=_parse_optional(author["lastname"]),
            forename=_parse_optional(author["forename"]),
            initials=_parse_optional(author["initials"]),
            orcid=_parse_orcid(author["identifier"]),
            affiliation=_parse_optional(author["affiliation"]),
        )
        for author in values
    ]


def _parse_mesh_terms(value: str) -> list[MeshTerm]:
    if len(value) == 0:
        return []
    mesh_terms_split: Iterable[list[str]] = (
        mesh_term.strip().split(":", maxsplit=1)
        for mesh_term in value.split("; ")
    )
    mesh_terms: list[MeshTerm] = []
    for mesh_id_term in mesh_terms_split:
        if len(mesh_id_term) == 1 and len(mesh_terms) > 0:
            mesh_terms[-1].qualifiers.append(mesh_id_term[0])
        else:
            mesh_id, term = mesh_id_term
            mesh_terms.append(MeshTerm(
                mesh_id=mesh_id.strip(),
                term=term.strip(),
                qualifiers=[],
            ))
    return mesh_terms


def _parse_date(value: str) -> datetime:
    if value.count("-") == 0:
        return datetime.strptime(value, "%Y")
    elif value.count("-") == 1:
        return datetime.strptime(value, "%Y-%m")
    elif value.count("-") == 2:
        return datetime.strptime(value, "%Y-%m-%d")
    else:
        raise RuntimeError(f"Unsupported date format: {value}")


class Article(Document):
    class Index:
        settings = {
            "number_of_shards": 3,
            "number_of_replicas": 2,
        }

    pubmed_id: str = Keyword(required=True)  # type: ignore
    """PubMed ID"""
    pmc_id: str | None = Keyword()  # type: ignore
    """PubMed Central ID"""
    doi: str | None = Keyword()  # type: ignore
    """DOI"""
    other_ids: list[str] = Keyword(multi=True)  # type: ignore
    """Other IDs of the same article."""
    title: str | None = Text()  # type: ignore
    """Title of the article."""
    abstract: str | None = Text()  # type: ignore
    """Abstract of the article."""
    authors: list[Author] = Nested(Author)  # type: ignore
    mesh_terms: list[MeshTerm] = Nested(MeshTerm)  # type: ignore
    """List of MeSH terms."""
    publication_types: list[MeshTerm] = Nested(MeshTerm)  # type: ignore
    """List of publication types."""
    keywords: list[str] = Keyword(multi=True)  # type: ignore
    """List of keywords."""
    chemicals: list[MeshTerm] = Nested(MeshTerm)  # type: ignore
    """List of chemical terms."""
    publication_date: datetime = Date(
        default_timezone="UTC", required=True)  # type: ignore
    """Publication date."""
    journal: str | None = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore
    """Journal of the article."""
    journal_abbreviation: str | None = Keyword()  # type: ignore
    """Journal abbreviation."""
    nlm_id: str = Keyword(required=True)  # type: ignore
    """NLM unique identification."""
    issn: str | None = Keyword()  # type: ignore
    """ISSN of the journal."""
    country: str | None = Keyword()  # type: ignore
    """Country of the journal."""
    references_pubmed_ids: list[str] = Keyword(multi=True)  # type: ignore
    """PubMed IDs of references made to the article."""
    languages: list[str] = Keyword(multi=True)  # type: ignore
    """List of languages."""
    source_file: str = Keyword(required=True)  # type: ignore
    """Basename of the XML file that contains this article."""

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pubmed_id}"

    @property
    def pmc_url(self) -> str | None:
        if self.pmc_id is None:
            return None
        return f"https://ncbi.nlm.nih.gov/pmc/articles/{self.pmc_id}"

    @property
    def doi_url(self) -> str | None:
        if self.doi is None:
            return None
        return f"https://doi.org/{self.doi}"

    @classmethod
    def parse(cls, article: dict, path: Path) -> "Article | None":
        if article["delete"]:
            return None
        pubmed_id = _parse_required(article["pmid"])
        pmc_id = _parse_optional(article["pmc"])
        doi = _parse_optional(article["doi"])
        other_ids = _parse_list(article["other_id"])
        title = _parse_optional(article["title"])
        abstract = _parse_optional(article["abstract"])
        authors = _parse_authors(article["authors"])
        mesh_terms: list[MeshTerm] = _parse_mesh_terms(article["mesh_terms"])
        publication_types: list[MeshTerm] = _parse_mesh_terms(
            article["publication_types"])
        keywords = _parse_list(article["keywords"])
        chemicals: list[MeshTerm] = _parse_mesh_terms(article["chemical_list"])
        publication_date = _parse_date(article["pubdate"])
        journal = _parse_optional(article["journal"])
        journal_abbreviation = _parse_optional(article["medline_ta"])
        nlm_id = _parse_required(article["nlm_unique_id"])
        issn = _parse_optional(article["issn_linking"])
        country = _parse_optional(article["country"])
        references_pubmed_ids = _parse_list(article.get("reference", ""))
        languages = _parse_list(article.get("languages", ""))
        return Article(
            meta={"id": pubmed_id},
            pubmed_id=pubmed_id,
            pmc_id=pmc_id,
            doi=doi,
            other_ids=other_ids,
            title=title,
            abstract=abstract,
            authors=authors,
            mesh_terms=mesh_terms,
            publication_types=publication_types,
            keywords=keywords,
            chemicals=chemicals,
            publication_date=publication_date,
            journal=journal,
            journal_abbreviation=journal_abbreviation,
            nlm_id=nlm_id,
            issn=issn,
            country=country,
            references_pubmed_ids=references_pubmed_ids,
            languages=languages,
            source_file=path.name,
        )


@dataclass(frozen=True)
class PubMedBaseline(Iterable[Article]):
    directory: Path

    def __post_init__(self):
        if not self.directory.is_dir():
            raise RuntimeError(
                f"Cannot read PubMed baseline from: {self.directory}")

    @staticmethod
    def _parse_articles(path: Path) -> Iterator[Article]:
        articles = parse_medline_xml(
            path=str(path),
            year_info_only=False,
            nlm_category=False,
            author_list=True,
            reference_list=False,
        )
        for article in tqdm(
            articles,
            desc=f"Parse {path}",
            unit="article",
        ):
            parsed = Article.parse(
                article=article,
                path=path,
            )
            if parsed is None:
                continue
            yield parsed

    @cached_property
    def _paths(self) -> list[Path]:
        paths = list(self.directory.glob("pubmed*n*.xml.gz"))
        shuffle(paths)
        return paths

    def __iter__(self) -> Iterator[Article]:
        for path in tqdm(
            self._paths,
            desc="Parse articles",
            unit="path",
        ):
            yield from self._parse_articles(path)
