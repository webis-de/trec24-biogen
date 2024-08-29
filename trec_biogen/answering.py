from dataclasses import dataclass
from typing import Annotated, Sequence

from annotated_types import Ge


from trec_biogen.model import (
    Answer,
    GenerationAnswer,
    PartialAnswer,
    PubMedReferenceSentence,
    RankedPubMedReference,
    RetrievalAnswer,
)
from trec_biogen.modules import GenerationModule, RetrievalModule, AnsweringModule


def _remove_dangling_references(
    sentence: PubMedReferenceSentence,
    references: Sequence[RankedPubMedReference],
) -> PubMedReferenceSentence:
    sentence_references = [
        sentence_reference
        for sentence_reference in sentence.references
        if any(sentence_reference in reference for reference in references)
    ]
    return PubMedReferenceSentence(
        sentence=sentence.sentence,
        references=sentence_references,
    )


def _merge_answers(
    context: PartialAnswer,
    retrieval_answer: RetrievalAnswer,
    generation_answer: GenerationAnswer,
) -> Answer:
    summary = [
        _remove_dangling_references(sentence, retrieval_answer.references)
        for sentence in generation_answer.summary
    ]
    return Answer(
        id=context.id,
        text=context.text,
        type=context.type,
        query=context.query,
        narrative=context.narrative,
        summary=summary,
        exact=generation_answer.exact,
        references=retrieval_answer.references,
    )


@dataclass(frozen=True)
class IndependentAnsweringModule(AnsweringModule):
    """
    Answer by independently retrieving references and generating an answer.
    """

    retrieval_module: RetrievalModule
    generation_module: GenerationModule

    def answer(self, context: PartialAnswer) -> Answer:
        retrieval_answer = self.retrieval_module.retrieve(context)
        generation_answer = self.generation_module.generate(context)
        return _merge_answers(
            context=context,
            retrieval_answer=retrieval_answer,
            generation_answer=generation_answer,
        )

    def answer_many(self, contexts: Sequence[PartialAnswer]) -> Sequence[Answer]:
        retrieval_answers = self.retrieval_module.retrieve_many(contexts)
        generation_answers = self.generation_module.generate_many(contexts)
        return [
            _merge_answers(
                context=context,
                retrieval_answer=retrieval_answer,
                generation_answer=generation_answer,
            )
            for context, retrieval_answer, generation_answer in zip(
                contexts, retrieval_answers, generation_answers
            )
        ]


@dataclass(frozen=True)
class RecurrentAnsweringModule(AnsweringModule):
    """
    Recurrently find and then refine an answer.
    """

    answering_module: AnsweringModule
    steps: Annotated[int, Ge(1)]

    def answer(self, context: PartialAnswer) -> Answer:
        answer = self.answering_module.answer(context)
        steps = self.steps - 1
        for _ in range(steps):
            answer = self.answering_module.answer(answer)
        return answer

    def answer_many(self, contexts: Sequence[PartialAnswer]) -> Sequence[Answer]:
        answers = self.answering_module.answer_many(contexts)
        steps = self.steps - 1
        for _ in range(steps):
            answers = self.answering_module.answer_many(answers)
        return answers
