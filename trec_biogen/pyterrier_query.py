from dataclasses import dataclass
from functools import cached_property
from typing import Hashable, Iterable, Literal
from warnings import catch_warnings, simplefilter

from elasticsearch7_dsl.query import Nested, Query, Bool, Exists, Match, Terms
from pandas import DataFrame, Series
from pyterrier.transformer import Transformer
from spacy import Language, load as spacy_load
from tqdm.auto import tqdm

from trec_biogen.model import PartialAnswer


@dataclass(frozen=True)
class ContextQueryTransformer(Transformer):
    include_question: bool
    include_query: bool
    include_narrative: bool
    include_summary: bool
    include_exact: bool
    progress: bool = False

    def _build_query(self, row: Series) -> str:
        context: PartialAnswer = row["context"]

        query_parts = []
        if self.include_question:
            query_parts.append(context.text)
        if self.include_query and context.query is not None:
            query_parts.append(context.query)
        if self.include_narrative and context.narrative is not None:
            query_parts.append(context.narrative)
        if self.include_summary and context.summary is not None:
            query_parts.extend(
                sentence.sentence
                for sentence in context.summary
            )
        if self.include_exact and context.exact is not None:
            if isinstance(context.exact, str):
                query_parts.append(context.exact)
            else:
                query_parts.extend(context.exact)

        return " ".join(query_parts)

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res = topics_or_res.copy()
        rows: Iterable[tuple[Hashable, Series]] = topics_or_res.iterrows()
        if self.progress:
            rows = tqdm(
                rows,
                total=len(topics_or_res),
                desc="Build queries",
                unit="query"
            )
        topics_or_res["query"] = [
            self._build_query(row)
            for _, row in rows
        ]
        return topics_or_res


_DISALLOWED_PUBLICATION_TYPES = [
    "Letter",
    "Comment",
    "Editorial",
    "News",
    "Biography",
    "Congress",
    "Video-Audio Media",
    "Interview",
    "Overall",
    "Retraction of Publication",
    "Retracted Publication",
    "Newspaper Article",
    "Bibliography",
    "Legal Case",
    "Directory",
    "Personal Narrative",
    "Address",
    "Randomized Controlled Trial, Veterinary",
    "Autobiography",
    "Dataset",
    "Clinical Trial, Veterinary",
    "Festschrift",
    "Webcast",
    "Observational Study, Veterinary",
    "Dictionary",
    "Periodical Index",
    "Interactive Tutorial",
]
@dataclass(frozen=True)
class ContextElasticsearchQueryTransformer(Transformer):
    require_title: bool
    require_abstract: bool
    filter_publication_types: bool
    remove_stopwords: bool
    match_title: Literal["must", "should"] | None
    match_abstract: Literal["must", "should"] | None
    match_mesh_terms: Literal["must", "should"] | None
    progress: bool = False

    @cached_property
    def _nlp(self) -> Language:
        with catch_warnings():
            simplefilter(action="ignore", category=FutureWarning)
            return spacy_load("en_core_sci_sm")

    def _build_query(self, row: Series) -> Query:
        query: str = row["query"]
        # context: PartialAnswer = row["context"]
 
        doc = self._nlp(query)

        if self.remove_stopwords:
            query = " ".join(
                token.text
                for token in doc
                if not token.is_stop
            )

        filters = []
        if self.require_title:
            filters.append(Exists(field="title"))
        if self.require_abstract:
            filters.append(Exists(field="abstract"))
        if self.filter_publication_types:
            filters.append(~Nested(
                path="publication_types",
                query=Terms(term=_DISALLOWED_PUBLICATION_TYPES),
            ))

        musts = []
        shoulds = []

        match_title = Match(title=query)
        if self.match_title == "must":
            musts.append(match_title)
        elif self.match_title == "should":
            shoulds.append(match_title)
        
        match_abstract = Match(abstract=query)
        if self.match_abstract == "must":
            musts.append(match_abstract)
        elif self.match_abstract == "should":
            shoulds.append(match_abstract)

        match_mesh_terms = Nested(
            path="mesh_terms",
            query=Bool(
                should=[
                    Match(term=entity.text)
                    for entity in doc.ents
                ]
            ),
        )
        if self.match_mesh_terms == "must":
            musts.append(match_mesh_terms)
        elif self.match_mesh_terms == "should":
            shoulds.append(match_mesh_terms)

        elasticsearch_query =  Bool(
            filter=filters,
            must=musts,
            should=shoulds,
        )
        return elasticsearch_query


    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res = topics_or_res.copy()
        rows: Iterable[tuple[Hashable, Series]] = topics_or_res.iterrows()
        if self.progress:
            rows = tqdm(
                rows,
                total=len(topics_or_res),
                desc="Build Elasticsearch queries",
                unit="query"
            )
        topics_or_res["elasticsearch_query"] = [
            self._build_query(row)
            for _, row in rows
        ]
        return topics_or_res
