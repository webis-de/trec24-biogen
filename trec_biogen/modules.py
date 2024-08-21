from abc import ABC, abstractmethod
from typing import Sequence

from trec_biogen.model import Answer, GenerationAnswer, PartialAnswer, RetrievalAnswer


class RetrievalModule(ABC):
    """
    Module to retrieve relevant context for a medical question.
    """

    @abstractmethod
    def retrieve(self, context: PartialAnswer) -> RetrievalAnswer:
        return NotImplemented

    def retrieve_many(
        self, contexts: Sequence[PartialAnswer]
    ) -> Sequence[RetrievalAnswer]:
        return [self.retrieve(context) for context in contexts]


class GenerationModule(ABC):
    """
    Module to generate an answer to a medical question.
    """

    @abstractmethod
    def generate(self, context: PartialAnswer) -> GenerationAnswer:
        return NotImplemented

    def generate_many(
        self, contexts: Sequence[PartialAnswer]
    ) -> Sequence[GenerationAnswer]:
        return [self.generate(context) for context in contexts]


class AnsweringModule(ABC):
    """
    Module to answer a medical question.
    """

    @abstractmethod
    def answer(self, context: PartialAnswer) -> Answer:
        return NotImplemented

    def answer_many(self, contexts: Sequence[PartialAnswer]) -> Sequence[Answer]:
        return [self.answer(context) for context in contexts]
