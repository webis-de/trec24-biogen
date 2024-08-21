from pathlib import Path
from typing import Iterable, Sequence

from pydantic.networks import EmailStr
from tqdm import tqdm

from trec_biogen.model import (
    Answer,
    ClefBioAsqAnswer,
    ClefBioAsqQuestion,
    ClefBioAsqQuestions,
    TrecBioGenQuestion,
    TrecBioGenQuestions,
    ClefBioAsqAnswers,
    TrecBioGenAnswers,
    Question,
)


def load_clef_bioasq_questions(
    path: Path,
    progress: bool = False,
) -> Sequence[Question]:
    with path.open("rb") as file:
        input = ClefBioAsqQuestions.model_validate_json(file.read())
    questions: Iterable[ClefBioAsqQuestion] = input.questions
    if progress:
        questions = tqdm(questions, desc="Convert questions", unit="question")
    return [question.as_question() for question in questions]


def load_trec_biogen_questions(
    path: Path,
    progress: bool = False,
) -> Sequence[Question]:
    with path.open("rb") as file:
        input = TrecBioGenQuestions.model_validate_json(file.read())
    topics: Iterable[TrecBioGenQuestion] = input.topics
    if progress:
        topics = tqdm(topics, desc="Convert questions", unit="question")
    return [question.as_question() for question in topics]


def load_clef_bioasq_answers(
    path: Path,
    progress: bool = False,
) -> Sequence[Answer]:
    with path.open("rb") as file:
        input = ClefBioAsqAnswers.model_validate_json(file.read())
    questions: Iterable[ClefBioAsqAnswer] = input.questions
    if progress:
        questions = tqdm(questions, desc="Convert answers", unit="answer")
    return [question.as_answer() for question in input.questions]


def save_clef_bioasq_answers(
    answers: Iterable[Answer],
    path: Path,
    progress: bool = False,
) -> None:
    if progress:
        answers = tqdm(answers, desc="Convert answers", unit="answer")
    output = ClefBioAsqQuestions(
        questions=[answer.as_clef_bio_asq_answer() for answer in answers],
    )
    with path.open("wt", encoding="utf8") as file:
        file.write(output.model_dump_json(indent=2))


def save_trec_biogen_answers(
    answers: Iterable[Answer],
    team_id: str,
    run_name: str,
    contact_email: EmailStr,
    path: Path,
    progress: bool = False,
) -> None:
    if progress:
        answers = tqdm(answers, desc="Convert answers", unit="answer")
    output = TrecBioGenAnswers(
        team_id=team_id,
        run_name=run_name,
        contact_email=contact_email,
        results=[answer.as_trec_bio_gen_answer() for answer in answers],
    )
    with path.open("wt", encoding="utf8") as file:
        file.write(output.model_dump_json(indent=2))
