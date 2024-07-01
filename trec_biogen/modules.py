from abc import ABC, abstractmethod
from typing import Any


class RetrievalModule(ABC):
    """
    Module to retrieve relevant context for a medical question.
    """

    # TODO: Define parameters and output(s): question, additional context from generation?
    @abstractmethod
    def retrieve(self, todo: Any) -> Any:
        return NotImplemented

    # TODO (later): Add function to retrieve for multiple queries.
    #  That would be helpful for bulk-optimizations.


class GenerationModule(ABC):
    """
    Module to generate an answer to a medical question.
    """

    # TODO: Define parameters and output(s): question, additional context from retrieval?
    @abstractmethod
    def generate(self, todo: Any) -> Any:
        return NotImplemented

    # TODO (later): Add function to generate for multiple questions.
    #  That would be helpful for bulk-optimizations.
