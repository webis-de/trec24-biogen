from dataclasses import dataclass
from functools import cached_property
from typing import Any, Hashable

from elasticsearch7_dsl.query import Query, Match, Exists, Bool
from pandas import DataFrame
from pyterrier.transformer import Transformer
from pyterrier.text import MaxPassage

from trec_biogen.pubmed import Article
from trec_biogen.elasticsearch import elasticsearch_connection
from trec_biogen.elasticsearch_pyterrier import ElasticsearchTransformer


def _build_query(row: dict[Hashable, Any]) -> Query:
    query = str(row["query"])

    es_query = Bool(
        filter=[
            # Only consider articles with an abstract.
            Exists(field="abstract"),
        ],
        must=[
            Bool(
                must=[
                    # Require to at least match the title or abstract.
                    Bool(
                        should=[
                            Match(title=query),
                            Match(abstract=query),
                        ]
                    )
                ],
            ),
        ],
    )
    return es_query


def _build_result(article: Article) -> dict[Hashable, Any]:
    return {
        "title": article.title,
        "abstract": article.abstract,
        "text": f"{article.title}\n\n{article.abstract}",
        "url": article.pubmed_url,
        "doi": article.doi,
        "doi_url": article.doi_url,
        "pmc_id": article.pmc_id,
        "pmc_url": article.pmc_url,
        "nlm_id": article.nlm_id,
        "author_full_names": [
            f"{author.forename} {author.lastname}"
            for author in article.authors
            if author.forename is not None and author.lastname is not None
        ],
        "author_orcids": [
            author.orcid
            for author in article.authors
            if author.orcid is not None
        ],
        "mesh_terms": [
            term.term
            for term in article.mesh_terms
        ],
        "publication_types": [
            term.term
            for term in article.publication_types
        ],
        "chemicals": [
            term.term
            for term in article.chemicals
        ],
        "keywords": list(article.keywords),
        "all_terms": [
            term.term
            for terms in (
                article.mesh_terms,
                article.publication_types,
                article.chemicals,
            )
            for term in terms
        ] + list(article.keywords),
        "publication_date": article.publication_date,
        "journal": article.journal,
        "journal_abbreviation": article.journal_abbreviation,
        "issn": article.issn,
        "country": article.country,
        "languages": article.languages,
    }


@dataclass(frozen=True)
class Pipeline(Transformer):
    elasticsearch_url: str
    elasticsearch_username: str | None
    elasticsearch_password: str | None
    elasticsearch_index: str | None

    @cached_property
    def _pipeline(self) -> Transformer:
        pipeline = Transformer.identity()

        # Retrieve or re-rank documents with Elasticsearch (BM25).
        pipeline = pipeline >> ElasticsearchTransformer(
            document_type=Article,
            client=elasticsearch_connection(
                elasticsearch_url=self.elasticsearch_url,
                elasticsearch_username=self.elasticsearch_username,
                elasticsearch_password=self.elasticsearch_password,
            ),
            query_builder=_build_query,
            result_builder=_build_result,
            num_results=10,
            index=self.elasticsearch_index,
            verbose=True,
        )

        # TODO: Re-rank documents?

        # TODO: Split passages.

        # TODO: Re-rank passages.

        # De-passage and aggregate documents.
        de_passager = MaxPassage()
        pipeline = pipeline >> de_passager

        return pipeline

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._pipeline.transform(topics_or_res)
