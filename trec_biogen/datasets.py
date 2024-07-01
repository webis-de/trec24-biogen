from typing import Any, Collection, Iterable, Literal, TypeAlias

from ir_measures import Qrel, ScoredDoc
from pydantic import BaseModel


Split: TypeAlias = Literal["train", "dev", "test"]


RetrievalDatasetName: TypeAlias = Literal[
    "bioasq-task-b",
    # "trec-biogen",
]


class RetrievalDataset(BaseModel):
    run: Iterable[ScoredDoc]
    qrels: Iterable[Qrel]


def load_retrieval_dataset(
    dataset_name: RetrievalDatasetName,
    split: Split,
) -> RetrievalDataset:
    """
    Load retrieval examples for training, development, or testing.
    """
    # TODO: Load dataset.
    return NotImplemented


GenerationDatasetName: TypeAlias = Literal[
    "bioasq-task-b",
    # "trec-biogen",
]


class GenerationExample(BaseModel):
    # TODO: What should correspond to the inputs and outputs of the generation module.
    todo: Any


class GenerationDataset(BaseModel):
    examples: Collection[GenerationExample]


def load_generation_dataset(
    dataset_name: GenerationDatasetName,
    split: Split,
) -> GenerationDataset:
    """
    Load retrieval examples for training, development, or testing.
    """
    # TODO: Load dataset.
    return NotImplemented
