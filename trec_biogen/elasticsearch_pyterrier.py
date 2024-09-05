from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from json import dumps
from typing import Any, Callable, Generic, Hashable, Iterable, Type, TypeVar
from warnings import warn

from elasticsearch7 import Elasticsearch
from elasticsearch7_dsl import Document, Search
from elasticsearch7_dsl.query import Query, Terms
from elasticsearch7_dsl.response import Hit
from pandas import DataFrame, Series
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer
from tqdm.auto import tqdm


T = TypeVar("T", bound=Document)


@dataclass(frozen=True)
class ElasticsearchRetrieve(Generic[T], Transformer):
    """
    Retrieve documents from an Elasticsearch index.

    :param document_type: Elasticsearch document type. Must extend `Document`.
    :param client: Elasticsearch client to execute searches.
    :param query_builder: A function that builds an Elasticsearch query from the data frame row.
    :param result_builder: A function that extracts a dict from the document returned by Elasticsearch.
    :param num_results: Number of results to be retrieved. Defaults to 10 results.
    :param index: The Elasticsearch index name to retrieve documents from. Defaults to the index specified in the document type.
    :param verbose: Whether to show a progress bar when retrieving results. Defaults to `False`.
    """

    document_type: Type[T]
    client: Elasticsearch
    query_builder: Callable[[dict[Hashable, Any]], Query] = field(repr=False)
    result_builder: Callable[[T], dict[Hashable, Any]] = field(repr=False)
    num_results: int = 10
    index: str | None = None
    verbose: bool = False

    def _merge_result(
            self,
            row: dict[Hashable, Any],
            hit: Hit,
    ) -> dict[Hashable, Any]:
        result: T = self.document_type()
        result._from_dict(hit._source.to_dict())
        return {
            **row,
            "docno": hit._id,
            "score": hit._score,
            **self.result_builder(result),
        }

    def _transform_query(self, topic: DataFrame) -> DataFrame:
        row: Series = topic.iloc[0]

        search: Search = self.document_type.search(
            using=self.client, index=self.index)
        
        query = self.query_builder(row.to_dict())
        search = search.query(query)
        search = search.extra(size=self.num_results)

        response = search.execute()
        hits: Iterable[Hit] = response.hits.hits  # type: ignore
        hits = islice(hits, self.num_results)
        res = DataFrame([
            self._merge_result(row.to_dict(), hit)
            for hit in hits
        ])
        if len(res) == 0:
            warn(UserWarning(f"Did not find any results for query: {dumps({"query": query.to_dict()})}"))
            
            # Fix columns when no results could be retrieved.
            res = DataFrame(columns=["qid", "query", "docno", "score"])
        return res

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if not {"qid", "query"}.issubset(topics_or_res.columns):
            raise RuntimeError("Needs qid and query columns.")
        if len(topics_or_res) == 0:
            return topics_or_res

        topics_by_query = topics_or_res.groupby(
            by=["qid", "query"],
            as_index=False,
            sort=False,
        )
        if self.verbose:
            tqdm.pandas(
                desc="Retrieve with Elasticsearch",
                unit="query",
            )
            topics_or_res = topics_by_query.progress_apply(
                self._transform_query
            )  # type: ignore
        else:
            topics_or_res = topics_by_query.apply(self._transform_query)

        topics_or_res.reset_index(drop=True, inplace=True)
        topics_or_res.sort_values(by=["qid", "score"], ascending=[
                                  True, False], inplace=True)
        topics_or_res = add_ranks(topics_or_res)

        return topics_or_res


@dataclass(frozen=True)
class ElasticsearchRerank(Generic[T], Transformer):
    """
    Re-rank documents based on an Elasticsearch index.
    The `docno` column is expected to contain the same IDs as used as Elasticsearch IDs.

    :param document_type: Elasticsearch document type. Must extend `Document`.
    :param client: Elasticsearch client to execute searches.
    :param query_builder: A function that builds an Elasticsearch query from the data frame row.
    :param index: The Elasticsearch index name to retrieve documents from. Defaults to the index specified in the document type.
    :param verbose: Whether to show a progress bar when re-ranking results. Defaults to `False`.
    """

    document_type: Type[T]
    client: Elasticsearch
    query_builder: Callable[[dict[Hashable, Any]], Query] = field(repr=False)
    index: str | None = None
    verbose: bool = False

    def _transform_query(self, res: DataFrame) -> DataFrame:
        search: Search = self.document_type.search(
            using=self.client, index=self.index)
        search = search.query(self.query_builder(res.iloc[0].to_dict()))
        search = search.filter(Terms(_id=[
            row["docno"]
            for _, row in res.iterrows()
        ]))
        search = search.extra(size=len(res))
        search = search.extra(_source=False)

        response = search.execute()
        hits: Iterable[Hit] = response.hits.hits  # type: ignore
        scores = DataFrame([
            {
                "docno": hit._id,
                "score": hit._score,
            }
            for hit in hits
        ])

        res = res.merge(scores, how="left", on="docno")
        if res["score"].isna().sum() > 0:
            not_reranked = res[res["score"].isna()]["docno"]
            warn(RuntimeWarning(
                f"Could not re-rank documents: {', '.join(not_reranked)}"))
        res = res[res["score"].notna()]
        return res

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if not {"qid", "query"}.issubset(topics_or_res.columns):
            raise RuntimeError("Needs qid and query columns.")
        if len(topics_or_res) == 0:
            return topics_or_res

        topics_by_query = topics_or_res.groupby(
            by=["qid", "query"],
            as_index=False,
            sort=False,
        )
        if self.verbose:
            tqdm.pandas(
                desc="Re-rank with Elasticsearch",
                unit="query",
            )
            topics_or_res = topics_by_query.progress_apply(
                self._transform_query
            )  # type: ignore
        else:
            topics_or_res = topics_by_query.apply(self._transform_query)

        topics_or_res.reset_index(drop=True, inplace=True)
        topics_or_res.sort_values(by=["qid", "score"], ascending=[
                                  True, False], inplace=True)
        topics_or_res = add_ranks(topics_or_res)
        return topics_or_res


@dataclass(frozen=True)
class ElasticsearchGet(Generic[T], Transformer):
    """
    Get document fields from an Elasticsearch index.
    The `docno` column is expected to contain the same IDs as used as Elasticsearch IDs.

    :param document_type: Elasticsearch document type. Must extend `Document`.
    :param client: Elasticsearch client to execute searches.
    :param result_builder: A function that extracts a dict from the document returned by Elasticsearch.
    :param index: The Elasticsearch index name to get documents from. Defaults to the index specified in the document type.
    :param verbose: Whether to show a progress bar when getting results. Defaults to `False`.
    """

    document_type: Type[T]
    client: Elasticsearch
    result_builder: Callable[[T], dict[Hashable, Any]] = field(repr=False)
    index: str | None = None
    verbose: bool = False

    def _merge_result(
            self,
            row: dict[Hashable, Any],
            document: T
    ) -> dict[Hashable, Any]:
        return {
            **row,
            **self.result_builder(document),
        }

    def _transform_query(self, res: DataFrame) -> DataFrame:
        if not isinstance(res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if "docno" not in res.columns:
            raise RuntimeError("Needs docno column.")
        if len(res) == 0:
            return res

        ids = {str(id) for id in res["docno"].to_list()}
        sorted_ids = sorted(ids)
        sorted_documents: list[T] = self.document_type.mget(
            docs=sorted_ids,
            using=self.client,
            index=self.index,
        )

        documents: dict[str, T] = dict(zip(sorted_ids, sorted_documents))
        return DataFrame([
            self._merge_result(row.to_dict(), documents[row["docno"]])
            for _, row in res.iterrows()
        ])

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if "docno" not in topics_or_res.columns:
            raise RuntimeError("Needs docno column.")
        if len(topics_or_res) == 0:
            return topics_or_res
        if not {"qid", "query"}.issubset(topics_or_res.columns):
            return self._transform_query(topics_or_res)

        topics_by_query = topics_or_res.groupby(
            by=["qid", "query"],
            as_index=False,
            sort=False,
        )
        if self.verbose:
            tqdm.pandas(
                desc="Get with Elasticsearch",
                unit="query",
            )
            topics_or_res = topics_by_query.progress_apply(
                self._transform_query
            )  # type: ignore
        else:
            topics_or_res = topics_by_query.apply(self._transform_query)

        topics_or_res.reset_index(drop=True, inplace=True)
        return topics_or_res


@dataclass(frozen=True)
class ElasticsearchTransformer(Generic[T], Transformer):
    """
    Generic `Transformer` that either retrieves or re-ranks documents from an Elasticsearch index, depending on whether the `docno` field is present.

    :param document_type: Elasticsearch document type. Must extend `Document`.
    :param client: Elasticsearch client to execute searches.
    :param query_builder: A function that builds an Elasticsearch query from the data frame row.
    :param result_builder: A function that extracts a dict from the document returned by Elasticsearch.
    :param num_results: Number of results to be retrieved. Defaults to 10 results.
    :param index: The Elasticsearch index name to retrieve documents from. Defaults to the index specified in the document type.
    :param verbose: Whether to show a progress bar when retrieving/re-ranking results. Defaults to `False`.
    """
    document_type: Type[T]
    client: Elasticsearch
    query_builder: Callable[[dict[Hashable, Any]], Query] = field(repr=False)
    result_builder: Callable[[T], dict[Hashable, Any]] = field(repr=False)
    num_results: int = 10
    index: str | None = None
    verbose: bool = False

    @cached_property
    def _retrieve(self) -> ElasticsearchRetrieve:
        return ElasticsearchRetrieve(
            document_type=self.document_type,
            client=self.client,
            query_builder=self.query_builder,
            result_builder=self.result_builder,
            num_results=self.num_results,
            index=self.index,
            verbose=self.verbose,
        )

    @cached_property
    def _rerank(self) -> ElasticsearchRerank:
        return ElasticsearchRerank(
            document_type=self.document_type,
            client=self.client,
            query_builder=self.query_builder,
            index=self.index,
            verbose=self.verbose,
        )

    @cached_property
    def _get(self) -> ElasticsearchGet:
        return ElasticsearchGet(
            document_type=self.document_type,
            client=self.client,
            result_builder=self.result_builder,
            index=self.index,
            verbose=self.verbose,
        )

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if "docno" in topics_or_res.columns:
            return ((self._rerank >> self._get) ^ self._retrieve).transform(topics_or_res)
        if {"qid", "query"}.issubset(topics_or_res.columns):
            return self._retrieve.transform(topics_or_res)
        raise RuntimeError("Needs qid and query columns or docno column.")
