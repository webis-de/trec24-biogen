from functools import cached_property
from re import compile as re_compile
from typing import Annotated, Literal, TypeAlias
from warnings import warn

from dspy import (
    Signature,
    InputField,
    OutputField,
    Module,
    Predict,
    ChainOfThought,
    Prediction,
)
from spacy import load as spacy_load, Language

from trec_biogen.model import (
    ExactAnswer,
    GenerationAnswer,
    PartialAnswer,
    PubMedReferenceSentence,
    PubMedReferencesSummary,
    RankedPubMedReference,
)

PredictType: TypeAlias = Literal["predict", "chain-of-thought"]


class _SummarySignature(Signature):
    """Answer the medical question based on the given reference snippets (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given references if the references are factually correct."""

    question: Annotated[
        str,
        InputField(
            description="The question that should be answered.",
        ),
    ]
    references: Annotated[
        str,
        InputField(
            description='Snippets from medical abstracts that should be used as references to the answer. Snippets are given in the form "[1]: This is a snippet.", where "[1]" denotes the number of the snippet.',
        ),
    ]
    answer: Annotated[
        str,
        OutputField(
            description='The summary answer to the question consisting of 1 to 3 sentences that also explain the answer. The answer should be grammatically correct, concise, and precise. The answer should cite the snippets from the references by including the numbers of relevant snippets in brackets, e.g., "This is an answer [1]." or "This is another answer [2,3]."',
        ),
    ]


_PATTERN_REFERENCE = re_compile(r".*\[(\d+(?:,\s*\d+)*)\]\s*[.!?]?")


class _SummaryPredict(Module):
    _predict: Predict | ChainOfThought

    @cached_property
    def _nlp(self) -> Language:
        return spacy_load("en_core_web_sm")

    def __init__(
        self,
        predict_type: PredictType,
    ) -> None:
        if predict_type == "predict":
            self._predict = Predict(
                signature=_SummarySignature,
            )
        elif predict_type == "chain-of-thought":
            self._predict = ChainOfThought(
                signature=_SummarySignature,
            )

    def _parse_sentence(
        self,
        sentence: str,
        all_references: dict[int, RankedPubMedReference],
    ) -> PubMedReferenceSentence:
        match = _PATTERN_REFERENCE.match(sentence)
        if match is None:
            return PubMedReferenceSentence(
                sentence=sentence,
                references=[],
            )
        references_group: str = match.group(1)
        reference_ids = {int(id.strip()) for id in references_group.split(",")}
        return PubMedReferenceSentence(
            sentence=sentence,
            references=[
                all_references[id]
                for id in reference_ids
                if id in all_references.keys()
            ],
        )

    def forward(self, context: PartialAnswer, **kwargs) -> Prediction:
        input_question = context.text
        references: dict[int, RankedPubMedReference] = (
            {
                i: reference
                for i, reference in enumerate(context.references)
                if reference.snippet is not None
            }
            if context.references is not None
            else {}
        )
        input_references = "\n" + "\n".join(
            f"\t[{i}] {reference.snippet.text}"
            for i, reference in references.items()
            if reference.snippet is not None
        )
        prediction: Prediction = self._predict.forward(
            question=input_question,
            references=input_references,
            **kwargs,
        )
        output_answer: str = prediction["answer"]

        doc = self._nlp(output_answer)
        summary: PubMedReferencesSummary = [
            self._parse_sentence(
                sentence=sentence.text,
                all_references=references,
            )
            for sentence in doc.sents
        ]
        return Prediction(
            summary=summary,
        )


class _ExactYesNoSignature(Signature):
    """Answer the medical yes-no question based on the given reference snippets (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given references if the references are factually correct."""

    question: Annotated[
        str,
        InputField(
            description="The question that should be answered.",
        ),
    ]
    references: Annotated[
        str,
        InputField(
            description='Snippets from medical abstracts that should be used as references to the answer. Snippets are given in the form "[1]: This is a snippet.", where "[1]" denotes the number of the snippet.',
        ),
    ]
    answer: Annotated[
        str,
        OutputField(
            description='The yes-no answer to the question, i.e., either "yes" or "no". Do not include references. Do not include an explanation.',
        ),
    ]


class _ExactFactualSignature(Signature):
    """Answer the medical factual question based on the given reference snippets (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given references if the references are factually correct."""

    question: Annotated[
        str,
        InputField(
            description="The question that should be answered.",
        ),
    ]
    references: Annotated[
        str,
        InputField(
            description='Snippets from medical abstracts that should be used as references to the answer. Snippets are given in the form "[1]: This is a snippet.", where "[1]" denotes the number of the snippet.',
        ),
    ]
    answer: Annotated[
        str,
        OutputField(
            description="The factual answer to the question, i.e., a single entity or number. Do not include references. Do not include an explanation. Do not use more than 10 words or 50 characters.",
        ),
    ]


class _ExactListSignature(Signature):
    """Answer the medical list question based on the given reference snippets (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given references if the references are factually correct."""

    question: Annotated[
        str,
        InputField(
            description="The question that should be answered.",
        ),
    ]
    references: Annotated[
        str,
        InputField(
            description='Snippets from medical abstracts that should be used as references to the answer. Snippets are given in the form "[1]: This is a snippet.", where "[1]" denotes the number of the snippet.',
        ),
    ]
    answer: Annotated[
        str,
        OutputField(
            description="The list answer to the question, i.e., a newline-separated list of entities, one entity per line. Do not include references. Do not include an explanation.",
        ),
    ]


class _ExactPredict(Module):
    _predict_yes_no: Predict | ChainOfThought
    _predict_factual: Predict | ChainOfThought
    _predict_list: Predict | ChainOfThought

    @cached_property
    def _nlp(self) -> Language:
        return spacy_load("en_core_web_sm")

    def __init__(
        self,
        predict_type: PredictType,
        assertions_max_backtracks: int = 0,
    ) -> None:
        if predict_type == "predict":
            self._predict_yes_no = Predict(
                signature=_ExactYesNoSignature,
            )
            self._predict_factual = Predict(
                signature=_ExactFactualSignature,
            )
            self._predict_list = Predict(
                signature=_ExactListSignature,
            )
        elif predict_type == "chain-of-thought":
            self._predict_yes_no = ChainOfThought(
                signature=_ExactYesNoSignature,
            )
            self._predict_factual = ChainOfThought(
                signature=_ExactFactualSignature,
            )
            self._predict_list = ChainOfThought(
                signature=_ExactListSignature,
            )

    def forward(self, context: PartialAnswer, **kwargs) -> Prediction:
        if "past_outputs" not in kwargs.keys():
            kwargs["past_outputs"] = {}

        input_question = context.text
        references: dict[int, RankedPubMedReference] = (
            {
                i: reference
                for i, reference in enumerate(context.references)
                if reference.snippet is not None
            }
            if context.references is not None
            else {}
        )
        input_references = "\n" + "\n".join(
            f"\t[{i}] {reference.snippet.text}"
            for i, reference in references.items()
            if reference.snippet is not None
        )
        prediction: Prediction
        output_answer: str
        exact: ExactAnswer | None
        if context.type == "yes-no":
            prediction = self._predict_yes_no.forward(
                question=input_question,
                references=input_references,
                **kwargs,
            )
            output_answer = prediction["answer"]
            # Consider only first line.
            output_answer = output_answer.split("\n")[0]
            output_answer = output_answer.strip()
            # Ignore period at the end and lower-case.
            output_answer = output_answer.removesuffix(".").lower()
            if output_answer == "yes":
                exact = "yes"
            elif output_answer == "no":
                exact = "no"
            else:
                warn(
                    UserWarning(
                        f"Invalid yes-no answer: {prediction['answer']}\nReturning \"no\" answer."
                    )
                )
                exact = "no"
        elif context.type == "factual":
            prediction = self._predict_factual.forward(
                question=input_question,
                references=input_references,
                **kwargs,
            )
            output_answer = prediction["answer"]
            # Consider only first line.
            output_answer = output_answer.split("\n")[0]
            output_answer = output_answer.strip()
            if len(output_answer) == 0:
                warn(
                    UserWarning(
                        f"Empty factual answer: {prediction['answer']}\nReturning empty answer."
                    )
                )
                exact = None
            elif len(output_answer) > 50:
                warn(
                    UserWarning(
                        f"Too long factual answer (>50 characters): {prediction['answer']}\nReturning empty answer."
                    )
                )
                exact = None
            elif sum(1 for _ in self._nlp(output_answer)) > 10:
                warn(
                    UserWarning(
                        f"Too long factual answer (>10 words): {prediction['answer']}\nReturning empty answer."
                    )
                )
                exact = None
            else:
                exact = output_answer
        elif context.type == "list":
            prediction = self._predict_list.forward(
                question=input_question,
                references=input_references,
                **kwargs,
            )
            output_answer = prediction["answer"]
            items = [item.strip() for item in output_answer.split("\n")]
            items = [item for item in items if len(item) > 0]
            if len(items) == 0:
                warn(
                    UserWarning(
                        f"Empty list answer: {prediction['answer']}\nReturning empty answer."
                    )
                )
                exact = None
            elif len(items) <= 1:
                warn(
                    UserWarning(
                        f"Single item list answer: {prediction['answer']}\nReturning single-item list."
                    )
                )
                exact = items
            else:
                exact = items
        elif context.type == "definition":
            exact = None
        elif context.type == "reasoning":
            exact = None
        else:
            raise ValueError(f"Unknown question type: {context.type}")
        return Prediction(
            exact=exact,
        )


class GenerationAnswerPredict(Module):
    _summary_predict: _SummaryPredict
    _exact_predict: _ExactPredict

    def __init__(
        self,
        summary_predict_type: PredictType,
        exact_predict_type: PredictType,
        assertions_max_backtracks: int = 0,
    ) -> None:
        self._summary_predict = _SummaryPredict(
            predict_type=summary_predict_type,
        )
        self._exact_predict = _ExactPredict(
            predict_type=exact_predict_type,
            assertions_max_backtracks=assertions_max_backtracks,
        )

    def forward(self, context: PartialAnswer, **kwargs) -> Prediction:
        summary_prediction = self._summary_predict.forward(
            context=context,
            **kwargs,
        )
        summary: PubMedReferencesSummary = summary_prediction.summary
        exact_prediction = self._exact_predict.forward(
            context=context,
            **kwargs,
        )
        exact: ExactAnswer = exact_prediction.exact

        answer = GenerationAnswer(
            id=context.id,
            text=context.text,
            type=context.type,
            query=context.query,
            narrative=context.narrative,
            summary=summary,
            exact=exact,
            references=context.references,
        )
        return Prediction(
            answer=answer,
        )
