from abc import ABCMeta
from dataclasses import dataclass
from typing import Any

from dspy import Module, ProgramMeta

from trec_biogen.modules import GenerationModule, RetrievalModule


class _ABCProgramMeta(ABCMeta, ProgramMeta):
    """
    Combine the metaclasses of ABC and DSPy.
    This is necessary, because otherwise, the metaclasses
    of DSPy's `Module` and `ABC` conflict.
    """


@dataclass(frozen=True)
class DspyGenerationModule(GenerationModule, Module, metaclass=_ABCProgramMeta):
    """
    Generate an answer using the DSPy LLM programming framework.
    """

    todo: Any  # TODO: Define hyper-parameters.

    # TODO: Define parameters (per question) and output(s).
    def generate(self, todo: Any) -> Any:
        return NotImplemented


@dataclass(frozen=True)
class RetrievalAugmentedGenerationModule(GenerationModule):
    """
    Generate an answer based on the retrieved context from some retrieval module, known as Retrieval-augmented Generation.
    """

    retrieval_module: RetrievalModule
    generation_module: GenerationModule

    todo: Any  # TODO: Define hyper-parameters (e.g., how to use retrieved context for generation).

    # TODO: Define parameters (per question) and output(s).
    def generate(self, todo: Any) -> Any:
        foo = self.retrieval_module.retrieve(NotImplemented)
        bar = self.generation_module.generate(foo)
        return bar
