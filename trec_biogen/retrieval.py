from dataclasses import dataclass
from typing import Any

from pandas import DataFrame
from pyterrier.transformer import Transformer

from trec_biogen.modules import GenerationModule, RetrievalModule


@dataclass(frozen=True)
class PyterrierRetrievalModule(RetrievalModule, Transformer):
    """
    Retrieve relevant context using the PyTerrier framework.
    """

    todo: Any  # TODO: Define hyper-parameters.

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return NotImplemented

    # TODO: Define parameters (per question) and output(s).
    def retrieve(self, todo: Any) -> Any:
        # TODO: Use PyTerrier's `search()` function to retrieve results for one query.
        res: DataFrame = self.search(NotImplemented)
        return NotImplemented


@dataclass(frozen=True)
class GenerationAugmentedRetrievalModule(RetrievalModule):
    """
    Retrieve relevant context based on the generated answer from some generation module, known as Generation-augmented Retrieval.
    """

    generation_module: GenerationModule
    retrieval_module: RetrievalModule

    todo: Any  # TODO: Define hyper-parameters (e.g., how to use answer for retrieval).

    # TODO: Define parameters (per question) and output(s).
    def retrieve(self, todo: Any) -> Any:
        foo = self.generation_module.generate(NotImplemented)
        bar = self.retrieval_module.retrieve(foo)
        return bar
