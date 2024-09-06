from asyncio import run
from datetime import datetime, timezone
from os import environ

from dotenv import find_dotenv, load_dotenv
from elasticsearch7 import Elasticsearch
from elasticsearch7_dsl.query import Exists, Range
from pandas import DataFrame
from pyarrow import Table, field, schema, string, struct
from ray import init as ray_init
from ray.data import read_datasource
from ray.data.block import DataBatch
from ray_elasticsearch import ElasticsearchDslDatasink, ElasticsearchDslDatasource

from trec_biogen.pubmed import Article
from trec_biogen.pubmed_full_text import get_full_text_dict


def index_pubmed_full_texts(
    dry_run: bool = False,
    refetch: bool = False,
    sample: float | None = None,
) -> None:
    ray_init()  # Connect to Ray cluster.

    if find_dotenv():
        load_dotenv()  # Load .env file.

    es_kwargs: dict = dict(
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

    elasticsearch = Elasticsearch(**es_kwargs)
    Article.init(using=elasticsearch)

    def get_full_text_batch(input_batch: DataBatch) -> DataBatch:
        input_table: Table
        if isinstance(input_batch, dict):
            input_table = Table.from_pydict(input_batch)
        elif isinstance(input_batch, DataFrame):
            input_table = Table.from_pandas(input_batch)
        else:
            input_table = input_batch
        input_data = input_table.to_pylist()
        pubmed_ids = {row["_id"] for row in input_data}
        full_texts = run(get_full_text_dict(pubmed_ids))
        output_table = Table.from_pylist(
            mapping=[
                {
                    "_id": row["_id"],
                    "doc": {
                        "full_text": full_texts[row["_id"]],
                        "last_fetched_full_text": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                }
                for row in input_data
            ],
            schema=schema(
                [
                    field(name="_id", type=string(), nullable=False),
                    field(
                        name="doc",
                        type=struct(
                            [
                                field(
                                    name="full_text",
                                    type=string(),
                                    nullable=True,
                                ),
                                field(
                                    name="last_fetched_full_text",
                                    type=string(),
                                    nullable=False,
                                ),
                            ]
                        ),
                        nullable=False,
                    ),
                ]
            ),
        )
        return output_table

    # Not fetched yet.
    query_not_fetched = ~Exists(field="last_fetched_full_text")
    query_outdated = Range(
        last_fetched_full_text={
            "lt": "now-1M",  # Last fetched over 1 month ago.
        }
    )
    source = ElasticsearchDslDatasource(
        index=Article,
        query=(query_not_fetched | query_outdated if refetch else query_not_fetched),
        keep_alive="15m",
        client_kwargs=es_kwargs,
        schema=schema(
            [
                field(name="_id", type=string(), nullable=False),
                field(
                    name="_source",
                    type=struct(
                        [
                            field(
                                name="pubmed_id",
                                type=string(),
                                nullable=False,
                            ),
                        ]
                    ),
                    nullable=False,
                ),
            ]
        ),
    )
    sink = ElasticsearchDslDatasink(
        index=Article,
        op_type="update",
        chunk_size=50,
        initial_backoff=4,
        max_backoff=600,
        client_kwargs=es_kwargs,
    )

    data = read_datasource(source, concurrency=6)  # 3 shards x 2 threads
    if sample is not None:
        data = data.random_sample(fraction=sample, seed=0)
    data = data.map_batches(
        get_full_text_batch,
        batch_size=10,
        num_cpus=0.25,
        concurrency=6,
    )
    if dry_run:
        data = data.filter(lambda x: x["doc"]["full_text"] is not None)
        full_texts = data.take(10)
        for full_text in full_texts:
            print(
                full_text["_id"],
                full_text["doc"]["last_fetched_full_text"],
                full_text["doc"]["full_text"][:100].replace("\n", " "),
            )
    else:
        data.write_datasink(sink, concurrency=3)  # 3 shards x 1 thread
