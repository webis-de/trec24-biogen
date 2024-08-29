from functools import cached_property
from re import compile as re_compile
from typing import Annotated, Literal, Self, Sequence, TypeAlias, TypeVar
from warnings import warn

from dspy import (
    Signature,
    InputField,
    OutputField,
    Module,
    Predict,
    ChainOfThought,
    Prediction,
    Example,
)
from dspy.teleprompt import LabeledFewShot, BootstrapFewShot
from optuna.trial._trial import Trial
from spacy import load as spacy_load, Language

from trec_biogen.model import (
    Answer,
    ExactAnswer,
    GenerationAnswer,
    PartialAnswer,
    PubMedReferenceSentence,
    PubMedReferencesSummary,
    RankedPubMedReference,
)

_ModuleType = TypeVar("_ModuleType", bound=Module)


OptimizerType: TypeAlias = Literal[
    "labeled-few-shot",
    "bootstrap-few-shot",
    # TODO (maybe later): Add other DSPy optimizers.
]


def _dspy_optimize(
    module: _ModuleType,
    trial: Trial,
    examples: Sequence[Example],
) -> _ModuleType:

    optimizer_type: OptimizerType | None = trial.suggest_categorical(
        name="dspy_optimizer",
        choices=[
            "labeled-few-shot",
            "bootstrap-few-shot",
            None,
        ],
    )  # type: ignore
    if optimizer_type is None:
        return module
    elif len(examples) == 0:
        print(f"Not optimizing DSPy module {module!r} due to no examples.")
        return module
    elif optimizer_type == "labeled-few-shot":
        # Tune the generation module with DSPy (to select few-shot examples).
        few_shot_k = trial.suggest_int(
            name="dspy_labeled_few_shot_k",
            low=1,
            high=3,
        )
        optimizer = LabeledFewShot(k=few_shot_k)
        print(f"Optimizing DSPy module {module!r} with labeled few-shot optimizer...")
        return optimizer.compile(
            student=module,
            trainset=examples,
            sample=True,
        )
    elif optimizer_type == "bootstrap-few-shot":
        # Tune the generation module with DSPy (to select few-shot examples).
        max_bootstrapped_demos = trial.suggest_int(
            name="dspy_bootstrap_few_shot_max_bootstrapped_demos",
            low=2,
            high=4,
        )
        max_labeled_demos = trial.suggest_int(
            name="dspy_bootstrap_few_shot_max_labeled_demos",
            low=8,
            high=16,
        )
        max_rounds = trial.suggest_int(
            name="dspy_bootstrap_few_shot_max_rounds",
            low=1,
            high=3,
        )
        optimizer = BootstrapFewShot(
            metric=None,
            metric_threshold=None,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
            max_errors=5,
        )
        print(f"Optimizing DSPy module {module} with bootstrap few-shot optimizer...")
        return optimizer.compile(
            student=module,
            trainset=examples,
        )
    else:
        raise ValueError(f"Unkown optimizer: {optimizer_type}")


PredictType: TypeAlias = Literal[
    "predict",
    "chain-of-thought",
]


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


def _question(context: PartialAnswer) -> str:
    return context.text


def _references(
    context: PartialAnswer,
) -> tuple[str, dict[int, RankedPubMedReference]]:
    references: dict[int, RankedPubMedReference] = (
        {i: reference for i, reference in enumerate(context.references)}
        if context.references is not None
        else {}
    )
    references_text = "\n" + "\n".join(
        f"\t[{i}] {reference.snippet.text}"
        for i, reference in references.items()
        if reference.snippet is not None
    )
    return references_text, references


def _summary_answer_sentence(
    sentence: PubMedReferenceSentence, references: dict[str, int]
) -> str:
    reference_string: str
    if len(sentence.references) > 0:
        reference_ids = [
            f"{references[reference.pubmed_id]}" for reference in sentence.references
        ]
        reference_ids_string = ",".join(reference_ids)
        reference_string = f" [{reference_ids_string}]"
    else:
        reference_string = ""
    sentence_text = sentence.sentence.strip()
    if len(sentence_text) > 1 and sentence_text[-1] in (".", ":", "!", "?"):
        return f"{sentence_text[:-1]}{reference_string}{sentence_text[-1]}"
    else:
        return f"{sentence_text}{reference_string}."


def _summary_answer(answer: Answer) -> str:
    _, references = _references(answer)
    references_inverse = {
        f"{reference.pubmed_id}": i for i, reference in references.items()
    }
    return " ".join(
        _summary_answer_sentence(sentence, references_inverse)
        for sentence in answer.summary
    )


def _summary_examples(
    answers: Sequence[Answer],
) -> list[Example]:
    return [
        Example(
            question=_question(answer),
            references=_references(answer)[0],
            answer=_summary_answer(answer),
        ).with_inputs("question", "references")
        for answer in answers
    ]


class SummaryPredict(Module):
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

    def optimize(
        self,
        trial: Trial,
        answers: Sequence[Answer],
    ) -> Self:
        self._predict = _dspy_optimize(
            module=self._predict,
            trial=trial,
            examples=_summary_examples(answers),
        )
        return self

    def forward(self, context: PartialAnswer) -> Prediction:
        question = _question(context)
        references_text, references = _references(context)
        prediction: Prediction = self._predict.forward(
            question=question,
            references=references_text,
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


def _exact_yes_no_examples(
    answers: Sequence[Answer],
) -> list[Example]:
    return [
        Example(
            question=_question(answer),
            references=_references(answer)[0],
            answer=answer.exact,
        ).with_inputs("question", "references")
        for answer in answers
        if answer.type == "yes-no" and answer.exact is not None and isinstance(answer.exact, str)
    ]


def _exact_factual_examples(
    answers: Sequence[Answer],
) -> list[Example]:
    return [
        Example(
            question=_question(answer),
            references=_references(answer)[0],
            answer=answer.exact,
        ).with_inputs("question", "references")
        for answer in answers
        if answer.type == "factual" and answer.exact is not None and isinstance(answer.exact, str)
    ]


def _exact_list_examples(
    answers: Sequence[Answer],
) -> list[Example]:
    return [
        Example(
            question=_question(answer),
            references=_references(answer)[0],
            answer="\n" + "\n".join(answer.exact),
        ).with_inputs("question", "references")
        for answer in answers
        if answer.type == "list" and answer.exact is not None and not isinstance(answer.exact, str)
    ]


class ExactPredict(Module):
    _predict_yes_no: Predict | ChainOfThought
    _predict_factual: Predict | ChainOfThought
    _predict_list: Predict | ChainOfThought

    @cached_property
    def _nlp(self) -> Language:
        return spacy_load("en_core_web_sm")

    def __init__(
        self,
        predict_type: PredictType,
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

    def optimize(
        self,
        trial: Trial,
        answers: Sequence[Answer],
    ) -> Self:
        self._predict_yes_no = _dspy_optimize(
            module=self._predict_yes_no,
            trial=trial,
            examples=_exact_yes_no_examples(answers),
        )
        self._predict_factual = _dspy_optimize(
            module=self._predict_factual,
            trial=trial,
            examples=_exact_factual_examples(answers),
        )
        self._predict_list = _dspy_optimize(
            module=self._predict_list,
            trial=trial,
            examples=_exact_list_examples(answers),
        )
        return self

    def forward(self, context: PartialAnswer) -> Prediction:
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
            )
            output_answer = prediction["answer"]
            # Consider only first line.
            output_answer = output_answer.split("\n")[0]
            # Consider only text before the first divider (consecutive dashes).
            output_answer = output_answer.split("---")[0]
            # Consider only text before the first period.
            output_answer = output_answer.split(".")[0]
            # Consider only text before the first comma.
            output_answer = output_answer.split(",")[0]
            output_answer = output_answer.strip()
            # Ignore case.
            output_answer = output_answer.lower()
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
            )
            output_answer = prediction["answer"]
            # Consider only first line.
            output_answer = output_answer.split("\n")[0]
            # Consider only text before the first divider (consecutive dashes).
            output_answer = output_answer.split("---")[0]
            # Consider only text before the first period.
            output_answer = output_answer.split(".")[0]
            # Consider only text before the first comma.
            output_answer = output_answer.split(",")[0]
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
            )
            output_answer = prediction["answer"]
            # Consider only text before the first double newline.
            output_answer = output_answer.split("\n\n")[0]
            # Consider only text before the first divider (consecutive dashes).
            output_answer = output_answer.split("---")[0]
            items = [item.strip() for item in output_answer.split("\n")]
            # Consider only text before the first period.
            items = [item.split(".")[0] for item in items if len(item) > 0]
            # Consider only text before the first comma.
            items = [item.split(",")[0] for item in items if len(item) > 0]
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
    _summary_predict: SummaryPredict
    _exact_predict: ExactPredict

    def __init__(
        self,
        summary_predict: SummaryPredict,
        exact_predict: ExactPredict,
    ) -> None:
        self._summary_predict = summary_predict
        self._exact_predict = exact_predict

    def optimize(
        self,
        trial: Trial,
        answers: Sequence[Answer],
    ) -> Self:
        self._summary_predict.optimize(trial, answers)
        self._exact_predict.optimize(trial, answers)
        return self

    def forward(self, context: PartialAnswer) -> Prediction:
        summary_prediction = self._summary_predict.forward(
            context=context,
        )
        summary: PubMedReferencesSummary = summary_prediction.summary
        exact_prediction = self._exact_predict.forward(
            context=context,
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
