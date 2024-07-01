from typing import Any, Iterable, Literal, TypeAlias
from pydantic import BaseModel


Split: TypeAlias = Literal["train", "dev", "test"]


class RetrievalExample(BaseModel):
    # TODO: What should correspond to the inputs and outputs of the generation module.
    todo: Any


RetrievalDataset: TypeAlias = Literal[
    "bioasq-task-b",
    # "trec-biogen",
]


def load_retrieval_examples(
    dataset: RetrievalDataset,
    split: Split,
) -> Iterable[RetrievalExample]:
    """
    Load retrieval examples for training, development, or testing.
    """
    return NotImplemented


class GenerationExample(BaseModel):
    # TODO: What should correspond to the inputs and outputs of the generation module.
    todo: Any


GenerationDataset: TypeAlias = Literal[
    "bioasq-task-b",
    # "trec-biogen",
]


def load_generation_examples(
    dataset: GenerationDataset,
    split: Split,
) -> Iterable[GenerationExample]:
    """
    Load retrieval examples for training, development, or testing.
    """
    return NotImplemented
