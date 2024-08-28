from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from os import environ
from typing import Annotated, Any, Hashable, Iterable, Iterator

from annotated_types import Ge
from elasticsearch7 import Elasticsearch
from elasticsearch7_dsl.query import Query
from more_itertools import sliding_window
from pandas import DataFrame, Series
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer
from spacy import load as spacy_load, Language
from spacy.tokens.span import Span

from trec_biogen.elasticsearch_pyterrier import ElasticsearchRetrieve
from trec_biogen.model import Snippet
from trec_biogen.pubmed import Article


@dataclass(frozen=True)
class PubMedElasticsearchRetrieve(Transformer):
    include_title_text: bool
    include_abstract_text: bool

    @staticmethod
    def _build_query(row: dict[Hashable, Any]) -> Query:
        return row["elasticsearch_query"]

    def _build_result(self, article: Article) -> dict[Hashable, Any]:
        text_parts: list[str] = []
        if self.include_title_text and article.title is not None:
            text_parts.append(article.title)
        if self.include_abstract_text and article.abstract is not None:
            text_parts.append(article.abstract)
        return {
            "text": "\n".join(text_parts),
            "start_section": None,
            "start_offset": None,
            "end_section": None,
            "end_offset": None,
            "article": article,
        }

    @cached_property
    def _elasticsearch_retrieve(self) -> Transformer:
        elasticsearch = Elasticsearch(
            hosts=environ["ELASTICSEARCH_URL"],
            http_auth=(
                environ["ELASTICSEARCH_USERNAME"],
                environ["ELASTICSEARCH_PASSWORD"],
            ),
            timeout=60,
            request_timeout=60,
            read_timeout=60,
            max_retries=10,
        )
        elasticsearch_retrieve = ElasticsearchRetrieve(
            document_type=Article,
            client=elasticsearch,
            query_builder=self._build_query,
            result_builder=self._build_result,
            num_results=10,
            index=environ.get("ELASTICSEARCH_INDEX_PUBMED"),
            verbose=True,
        )
        return elasticsearch_retrieve

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._elasticsearch_retrieve.transform(topics_or_res)


@dataclass(frozen=True)
class PubMedSentencePassager(Transformer):
    """
    Split a PubMed article into snippets consisting of either:
    - the full title or
    - one or more sentences from the abstract.
    The sentences are split using the NLTK and the maximum number of sentences can be configured.
    """

    include_title_snippets: bool
    include_abstract_snippets: bool
    max_sentences: Annotated[int, Ge(1)]

    def _iter_title_snippets(self, row: Series) -> Iterator[Snippet]:
        article: Article = row["article"]
        # TODO: Remove after debugging.
        # print(article.to_dict())
        title = article.title
        if title is None:
            return
        yield Snippet(
            text=title,
            start_section="title",
            start_offset=0,
            end_section="title",
            end_offset=len(title)
        )


    @cached_property
    def _nlp(self) -> Language:
        return spacy_load("en_core_web_sm")

    def _iter_abstract_snippets(self, row: Series) -> Iterator[Snippet]:
        article: Article = row["article"]
        abstract = article.abstract
        if abstract is None:
            return
        
        doc = self._nlp(abstract)

        sentences: Iterable[Span] = doc.sents
        sentence_tuples = chain.from_iterable(
            sliding_window(sentences, n)
            for n in range(1, self.max_sentences + 1)
        )
        for sentence_tuple in sentence_tuples:
            yield Snippet(
                text=" ".join(
                    sentence.text
                    for sentence in sentence_tuple
                ),
                start_section="abstract",
                start_offset=sentence_tuple[0].start,
                end_section="abstract",
                end_offset=sentence_tuple[-1].end,
            )

    def _iter_snippets(self, row: Series) -> Iterator[Snippet]:
        if self.include_title_snippets:
            yield from self._iter_title_snippets(row)
        if self.include_abstract_snippets:
            yield from self._iter_abstract_snippets(row)

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        columns = list(set(topics_or_res.columns) | {
            "start_section",
            "start_offset",
            "end_section",
            "end_offset",
        })
        topics_or_res = DataFrame([
            {
                "docno": f"{row['docno']}%p({snippet.start_section},{snippet.start_offset:d},{snippet.end_section},{snippet.end_offset:d})",
                **row[list(set(row.index) - {"docno", "text"})],
                "text": snippet.text,
                "start_section": snippet.start_section,
                "start_offset": snippet.start_offset,
                "end_section": snippet.end_section,
                "end_offset": snippet.end_offset,
            }
            for _, row in topics_or_res.iterrows()
            for snippet in self._iter_snippets(row)
        ], columns=columns)

        if "score" in topics_or_res.columns:
            topics_or_res.sort_values(
                by=["qid", "score"],
                ascending=[True, False],
                inplace=True,
            )
            topics_or_res = add_ranks(topics_or_res)

        return topics_or_res

