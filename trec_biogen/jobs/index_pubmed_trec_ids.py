from gzip import open as gzip_open
from os import environ
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from elasticsearch7 import Elasticsearch
from pandas import DataFrame
from pyarrow import Table, field, schema, string, struct, bool_
from ray import init as ray_init
from ray.data import from_items
from ray.data.block import DataBatch
from ray_elasticsearch import ElasticsearchDslDatasink
from tqdm import tqdm

from trec_biogen.pubmed import Article


def index_pubmed_trec_ids(
    trec_ids_path: Path,
    dry_run: bool = False,
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

    def _map_ids_batch(input_batch: DataBatch) -> DataBatch:
        input_table: Table
        if isinstance(input_batch, dict):
            input_table = Table.from_pydict(input_batch)
        elif isinstance(input_batch, DataFrame):
            input_table = Table.from_pandas(input_batch)
        else:
            input_table = input_batch
        input_data = input_table.to_pylist()
        output_table = Table.from_pylist(
            mapping=[
                {
                    "_id": row["item"],
                    "doc": {
                        "is_included_trec_biogen_2024": True,
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
                                    name="is_included_trec_biogen_2024",
                                    type=bool_(),
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

    sink = ElasticsearchDslDatasink(
        index=Article,
        op_type="update",
        chunk_size=50,
        initial_backoff=4,
        max_backoff=600,
        client_kwargs=es_kwargs,
    )

    with gzip_open(trec_ids_path, "rt", encoding="utf8") as file:
        trec_ids = [
            line.strip()
            for line in tqdm(file)
        ]

    data = from_items(trec_ids)
    data = data.map_batches(
        _map_ids_batch,
        batch_size=100,
        num_cpus=0.25,
        concurrency=6,
    )
    if dry_run:
        data = data.filter(lambda x: x["doc"]["is_included_trec_biogen_2024"])
        full_texts = data.take(10)
        for full_text in full_texts:
            print(
                full_text["_id"],
                full_text["doc"]["is_included_trec_biogen_2024"],
            )
    else:
        data.write_datasink(sink, concurrency=3)  # 3 shards x 1 thread
