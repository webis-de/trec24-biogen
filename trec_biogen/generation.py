from abc import ABCMeta
from dataclasses import dataclass
from typing import Any, Sequence

from dspy import Module, ProgramMeta

from trec_biogen.model import GenerationAnswer, PartialAnswer
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

    def generate(self, context: PartialAnswer) -> GenerationAnswer:
        return NotImplemented


@dataclass(frozen=True)
class RetrievalThenGenerationModule(GenerationModule):
    """
    Generate an answer based on the retrieved context from some retrieval module, known as Retrieval-augmented Generation.
    """

    retrieval_module: RetrievalModule
    generation_module: GenerationModule

    def generate(self, context: PartialAnswer) -> GenerationAnswer:
        context = self.retrieval_module.retrieve(context)
        return self.generation_module.generate(context)

    def generate_many(
        self, contexts: Sequence[PartialAnswer]
    ) -> Sequence[GenerationAnswer]:
        contexts = self.retrieval_module.retrieve_many(contexts)
        return self.generation_module.generate_many(contexts)
