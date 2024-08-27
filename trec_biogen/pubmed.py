from datetime import datetime

from elasticsearch7_dsl import Document, Date, Text, Keyword, InnerDoc, Nested


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


class Article(Document):
    class Index:
        name = "corpus_pubmed_2024"
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
    full_text: str | None = Text()  # type: ignore
    """Extracted full text of the article."""
    last_fetched_full_text: datetime | None = Date(
        default_timezone="UTC")  # type: ignore
    """Last date at which the full text has been extracted."""
    is_included_trec_biogen_2024: bool | None = Keyword()  # type: ignore
    """Whether the article is in the subset used for TREC BioGen 2024."""

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
