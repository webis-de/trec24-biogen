from functools import cache
from math import inf
from pathlib import Path
from re import compile as re_compile
from spacy import load as spacy_load
from typing import Annotated, Literal, Self, Sequence, TypeAlias

from annotated_types import Ge
from pydantic import (
    BaseModel,
    AfterValidator,
    PlainValidator,
    WithJsonSchema,
    TypeAdapter,
    Field,
    model_validator,
    ConfigDict,
)
from pydantic.networks import EmailStr, HttpUrl, Url
from tqdm import tqdm

from trec_biogen import PROJECT_DIR

# Generic PubMed model definitions.


_PUBMED_ID_PATTERN = re_compile(r"[1-9][0-9]*")


def _validate_pubmed_id(url: HttpUrl) -> HttpUrl:
    if _PUBMED_ID_PATTERN.fullmatch(f"{url}") is None:
        raise ValueError(f"Not a valid PubMed ID: {url}")
    return url


PubMedId: TypeAlias = Annotated[
    str,
    AfterValidator(_validate_pubmed_id),
]


_PUBMED_URL_PATTERN = re_compile(
    r"http:\/\/www\.ncbi\.nlm\.nih\.gov\/pubmed\/[1-9][0-9]*"
)


def _validate_pubmed_url(url: HttpUrl) -> HttpUrl:
    if _PUBMED_URL_PATTERN.fullmatch(f"{url}") is None:
        raise ValueError(f"Not a valid PubMed URL: {url}")
    return url


PubMedUrl: TypeAlias = Annotated[
    HttpUrl,
    AfterValidator(_validate_pubmed_url),
]


# Internal model definitions.


QuestionType: TypeAlias = Literal[
    "yes-no",
    "factual",
    "list",
    "reasoning",
    "definition",
]


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    type: QuestionType | None
    query: str | None
    narrative: str | None


ExactAnswer: TypeAlias = Literal["yes", "no"] | str | Sequence[str]


class Snippet(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    start_section: str
    start_offset: Annotated[int, Ge(0)]
    end_section: str
    end_offset: Annotated[int, Ge(0)]


class PubMedReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    pubmed_id: PubMedId
    snippet: Snippet | None


    def as_unranked(self) -> "PubMedReference":
        return PubMedReference(
            pubmed_id=self.pubmed_id,
            snippet=self.snippet,
        )

class RankedPubMedReference(PubMedReference):
    model_config = ConfigDict(frozen=True)

    rank: Annotated[int, Ge(0)] | None


class PubMedReferenceSentence(BaseModel):
    model_config = ConfigDict(frozen=True)

    sentence: str
    references: Sequence[PubMedReference]

    def as_trec_bio_gen_answer_sentence(self) -> "TrecBioGenAnswerString":
        reference_string: str
        if len(self.references) > 0:
            reference_ids = []
            for reference in self.references:
                if reference.pubmed_id not in reference_ids:
                    reference_ids.append(reference.pubmed_id)
            references = ",".join(reference_ids)
            reference_string = f" [{references}]"
        else:
            reference_string = ""
        sentence = self.sentence.strip()
        if len(sentence) > 1 and sentence[-1] in (".", ":", "!", "?"):
            return f"{sentence[:-1]}{reference_string}{sentence[-1]}"
        else:
            return f"{sentence}{reference_string}."


PubMedReferencesSummary: TypeAlias = Sequence[PubMedReferenceSentence]


class PartialAnswer(Question):
    model_config = ConfigDict(frozen=True)

    summary: PubMedReferencesSummary | None
    exact: ExactAnswer | None
    references: Sequence[RankedPubMedReference] | None

    @model_validator(mode="after")
    def check_references_match(self) -> Self:
        text_references = (
            {
                reference.as_unranked()
                for sentence in self.summary
                for reference in sentence.references  # type: ignore
            }
            if self.summary is not None
            else set()
        )
        references = {
            reference.as_unranked()
            for reference in self.references # type: ignore
        } if self.references is not None else set()
        if not text_references.issubset(references):
            raise ValueError(
                f"In-text references must be a subset of explicit references. Found {len(text_references - references)} missing references: {text_references - references}"
            )
        return self


class RetrievalAnswer(PartialAnswer, Question):
    model_config = ConfigDict(frozen=True)

    references: Sequence[RankedPubMedReference]


class GenerationAnswer(PartialAnswer, Question):
    model_config = ConfigDict(frozen=True)

    summary: PubMedReferencesSummary
    exact: ExactAnswer | None


class Answer(RetrievalAnswer, GenerationAnswer, PartialAnswer, Question):
    model_config = ConfigDict(frozen=True)

    def as_clef_bio_asq_answer(self) -> "ClefBioAsqAnswer":
        type: ClefBioAsqQuestionType
        exact_answer: ClefBioAsqExactAnswer
        if self.type == "yes-no":
            type = "yesno"
            if self.exact == "yes":
                exact_answer = "yes"
            elif self.exact == "no":
                exact_answer = "no"
            else:
                raise ValueError(
                    f"Invalid exact answer for yes-no question: {self.exact}"
                )
        elif self.type == "factual":
            type = "factoid"
            if isinstance(self.exact, str):
                exact_answer = [self.exact]
            else:
                raise ValueError(
                    f"Invalid exact answer for factual question: {self.exact}"
                )
        elif self.type == "list":
            type = "list"
            if isinstance(self.exact, Sequence) and all(
                isinstance(item, str) for item in self.exact
            ):
                exact_answer = [[item] for item in self.exact]
            else:
                raise ValueError(
                    f"Invalid exact answer for list question: {self.exact}"
                )
        elif self.type == "definition" or self.type == "reasoning":
            exact_answer = "n/a"
        else:
            raise ValueError(f"Unknown question type: {self.type}")
        references = sorted(
            self.references,
            key=lambda reference: reference.rank if reference.rank is not None else inf,
        )
        documents = [
            Url(f"http://www.ncbi.nlm.nih.gov/pubmed/{reference.pubmed_id}")
            for reference in references
            if reference.snippet is None
        ]
        snippets = [
            ClefBioAsqSnippet(
                document=Url(
                    f"http://www.ncbi.nlm.nih.gov/pubmed/{reference.pubmed_id}"
                ),
                text=reference.snippet.text,
                beginSection=reference.snippet.start_section,
                offsetInBeginSection=reference.snippet.start_offset,
                endSection=reference.snippet.end_section,
                offsetInEndSection=reference.snippet.end_offset,
            )
            for reference in references
            if reference.snippet is not None
        ]
        return ClefBioAsqAnswer(
            id=self.id,
            type=type,
            body=self.text,
            documents=documents,
            snippets=snippets,
            ideal_answer=" ".join(sentence.sentence for sentence in self.summary),
            exact_answer=exact_answer,
        )

    def as_trec_bio_gen_answer(self) -> "TrecBioGenAnswer":
        return TrecBioGenAnswer(
            topic_id=self.id,
            answer=" ".join(
                sentence.as_trec_bio_gen_answer_sentence() for sentence in self.summary
            ),
            references={
                reference.pubmed_id
                for sentence in self.summary
                for reference in sentence.references
            },
        )


# CLEF BioASQ model definitions.


ClefBioAsqQuestionType: TypeAlias = Literal[
    "yesno",
    "factoid",
    "list",
    "summary",
]


class ClefBioAsqQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: ClefBioAsqQuestionType
    body: str

    def as_question(self) -> Question:
        type: QuestionType
        if self.type == "yesno":
            type = "yes-no"
        elif self.type == "factoid":
            type = "factual"
        elif self.type == "list":
            type = "list"
        elif self.type == "summary":
            if self.body.casefold().startswith("what"):
                type = "definition"
            else:
                type = "reasoning"
        else:
            raise ValueError(f"Unknown question type: {self.type}")
        return Question(
            id=self.id,
            text=self.body,
            type=type,
            query=None,
            narrative=None,
        )


ClefBioAsqDocument: TypeAlias = PubMedUrl

ClefBioAsqDocuments: TypeAlias = Sequence[ClefBioAsqDocument]


def _clamp_positive(value: int) -> int:
    return 0 if value == -1 else value


class ClefBioAsqSnippet(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: ClefBioAsqDocument
    text: str
    beginSection: str
    offsetInBeginSection: Annotated[
        int,
        Ge(0),
        PlainValidator(_clamp_positive),
        WithJsonSchema(
            TypeAdapter(int).json_schema(),
            mode="validation",
        ),
        WithJsonSchema(
            TypeAdapter(Annotated[int, Ge(0)]).json_schema(),  # type: ignore
            mode="serialization",
        ),
    ]
    endSection: str
    offsetInEndSection: Annotated[
        int,
        Ge(0),
    ]


ClefBioAsqSnippets: TypeAlias = Sequence[ClefBioAsqSnippet]


ClefBioAsqIdealAnswer: TypeAlias = Sequence[str]

ClefBioAsqYesNoExactAnswer: TypeAlias = Literal["yes", "no"]


ClefBioAsqFactoidExactAnswer: TypeAlias = Sequence[str]


ClefBioAsqListExactAnswerItem: TypeAlias = Sequence[str]


ClefBioAsqListExactAnswer: TypeAlias = Sequence[ClefBioAsqListExactAnswerItem]


ClefBioAsqSummaryExactAnswer: TypeAlias = None


ClefBioAsqExactAnswer: TypeAlias = Annotated[
    # Attention! The order of this union matters for serialization.
    ClefBioAsqYesNoExactAnswer
    | ClefBioAsqSummaryExactAnswer
    | ClefBioAsqListExactAnswer
    | ClefBioAsqFactoidExactAnswer,
    Field(union_mode="left_to_right"),
]


def _pubmed_url_to_pubmed_id(pubmed_url: ClefBioAsqDocument) -> PubMedId:
    path = pubmed_url.path
    if path is None:
        raise ValueError(f"Invalid PubMed URL: {pubmed_url}")
    return Path(path).name


@cache
def _nlp():
    return spacy_load("en_core_web_sm")


class ClefBioAsqAnswer(ClefBioAsqQuestion):
    model_config = ConfigDict(frozen=True)

    documents: ClefBioAsqDocuments
    snippets: ClefBioAsqSnippets
    ideal_answer: ClefBioAsqIdealAnswer
    exact_answer: ClefBioAsqExactAnswer = None

    def as_answer(self) -> Answer:
        question = self.as_question()

        if len(self.ideal_answer) <= 0:
            raise ValueError(f"Invalid ideal answer: {self.ideal_answer}")

        document_references = [
            RankedPubMedReference(
                pubmed_id=_pubmed_url_to_pubmed_id(document),
                rank=i + 1,
                snippet=None,
            )
            for i, document in enumerate(self.documents)
        ]
        snippet_references = [
            RankedPubMedReference(
                pubmed_id=_pubmed_url_to_pubmed_id(snippet.document),
                rank=i + 1,
                snippet=Snippet(
                    text=snippet.text,
                    start_section=snippet.beginSection,
                    start_offset=snippet.offsetInBeginSection,
                    end_section=snippet.endSection,
                    end_offset=snippet.offsetInEndSection,
                ),
            )
            for i, snippet in enumerate(self.snippets)
        ]

        references = snippet_references + document_references

        ideal_answer = self.ideal_answer[0]
        doc = _nlp()(ideal_answer)

        exact: ExactAnswer | None
        if self.type == "yesno":
            if self.exact_answer == "yes":
                exact = "yes"
            elif self.exact_answer == "no":
                exact = "no"
            else:
                raise ValueError(
                    f"Invalid exact answer for yes-no question: {self.exact_answer}"
                )
        elif self.type == "factoid":
            if (
                isinstance(self.exact_answer, Sequence)
                and len(self.exact_answer) > 0
                and isinstance(self.exact_answer[0], str)
            ):
                exact = self.exact_answer[0]
            else:
                raise ValueError(
                    f"Invalid exact answer for factual question: {self.exact_answer}"
                )
        elif self.type == "list":
            if isinstance(self.exact_answer, Sequence) and all(
                isinstance(item, Sequence)
                and len(item) > 0
                and isinstance(item[0], str)
                for item in self.exact_answer
            ):
                exact = [item[0] for item in self.exact_answer]
            else:
                raise ValueError(
                    f"Invalid exact answer for list question: {self.exact_answer}"
                )
        elif self.type == "summary":
            exact = None
        else:
            raise ValueError(f"Unknown question type: {self.type}")

        return Answer(
            id=question.id,
            text=question.text,
            type=question.type,
            query=question.query,
            narrative=question.narrative,
            summary=[
                PubMedReferenceSentence(
                    sentence=sentence.text,
                    references=references,
                )
                for sentence in doc.sents
            ],
            exact=exact,
            references=references,
        )


class ClefBioAsqQuestions(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: Sequence[ClefBioAsqQuestion]


class ClefBioAsqAnswers(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: Sequence[ClefBioAsqAnswer]


# TREC BioGen model definitions.


class TrecBioGenQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    topic: str
    question: str
    narrative: str

    def as_question(self) -> Question:
        return Question(
            id=f"{self.id}",
            text=self.question,
            type=None,
            query=self.topic,
            narrative=self.narrative,
        )


TrecBioGenAnswerString: TypeAlias = str  # TODO: Add validator.


class TrecBioGenAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    topic_id: str
    answer: TrecBioGenAnswerString
    references: set[PubMedId]


class TrecBioGenQuestions(BaseModel):
    model_config = ConfigDict(frozen=True)

    topics: Sequence[TrecBioGenQuestion]


class TrecBioGenAnswers(BaseModel):
    model_config = ConfigDict(frozen=True)

    team_id: str
    run_name: str
    contact_email: EmailStr
    results: Sequence[TrecBioGenAnswer]
