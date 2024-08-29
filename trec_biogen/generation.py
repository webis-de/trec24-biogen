from dataclasses import dataclass
from typing import Iterable, Sequence

from dspy import LM, settings as dspy_settings
from tqdm.auto import tqdm

from trec_biogen.dspy_generation import GenerationAnswerPredict
from trec_biogen.model import GenerationAnswer, PartialAnswer
from trec_biogen.modules import GenerationModule, RetrievalModule


@dataclass(frozen=True)
class DspyGenerationModule(GenerationModule):
    """
    Generate an answer using the DSPy LLM programming framework.
    """

    predict: GenerationAnswerPredict
    language_model: LM
    progress: bool = False

    def generate(self, context: PartialAnswer) -> GenerationAnswer:
        with dspy_settings.context(
            lm=self.language_model,
        ):
            prediction = self.predict.forward(
                context=context,
            )
        answer: GenerationAnswer = prediction.answer
        return answer

    def generate_many(
        self, contexts: Sequence[PartialAnswer]
    ) -> Sequence[GenerationAnswer]:
        contexts_iterable: Iterable[PartialAnswer] = contexts
        if self.progress:
            contexts_iterable = tqdm(
                contexts_iterable,
                desc="Generate answers with DSPy",
                unit="question",
            )
        return [self.generate(context) for context in contexts_iterable]


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
