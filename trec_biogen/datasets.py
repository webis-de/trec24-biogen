from pathlib import Path
from random import Random
from typing import Annotated, Iterable, Sequence

from annotated_types import Interval
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

_random = Random(0)

def load_clef_bioasq_questions(
    path: Path,
    sample: Annotated[float, Interval(ge=0, le=1)] = 1,
    progress: bool = False,
) -> Sequence[Question]:
    with path.open("rb") as file:
        input = ClefBioAsqQuestions.model_validate_json(file.read())
    questions = input.questions
    if sample < 1:
        questions = _random.sample(questions, k=round(len(questions) * sample))
    questions_iterable: Iterable[ClefBioAsqQuestion] = questions
    if progress:
        questions_iterable = tqdm(
            questions_iterable, desc="Convert questions", unit="question"
        )
    return [question.as_question() for question in questions_iterable]


def load_trec_biogen_questions(
    path: Path,
    sample: Annotated[float, Interval(ge=0, le=1)] = 1,
    progress: bool = False,
) -> Sequence[Question]:
    with path.open("rb") as file:
        input = TrecBioGenQuestions.model_validate_json(file.read())
    topics = input.topics
    if sample < 1:
        topics = _random.sample(topics, k=round(len(topics) * sample))
    topics_iterable: Iterable[TrecBioGenQuestion] = topics
    if progress:
        topics_iterable = tqdm(
            topics_iterable, desc="Convert questions", unit="question"
        )
    return [question.as_question() for question in topics_iterable]


def load_questions(
    path: Path,
    sample: Annotated[float, Interval(ge=0, le=1)] = 1,
    progress: bool = False,
) -> Sequence[Question]:
    if path.name.startswith("BioASQ") and path.name.endswith(".json"):
        return load_clef_bioasq_questions(path, sample=sample, progress=progress)
    elif path.name.startswith("training") and path.name.endswith("_new.json"):
        return load_clef_bioasq_questions(path, sample=sample, progress=progress)
    elif path.name.startswith("BioGen") and path.name.endswith("-json.txt"):
        return load_trec_biogen_questions(path, sample=sample, progress=progress)
    else:
        raise RuntimeError(f"Could not guess question format from file: {path}")


def load_clef_bioasq_answers(
    path: Path,
    sample: Annotated[float, Interval(ge=0, le=1)] = 1,
    progress: bool = False,
) -> Sequence[Answer]:
    with path.open("rb") as file:
        input = ClefBioAsqAnswers.model_validate_json(file.read())
    questions = input.questions
    if sample < 1:
        questions = _random.sample(questions, k=round(len(questions) * sample))
    questions_iterable: Iterable[ClefBioAsqAnswer] = questions
    if progress:
        questions_iterable = tqdm(
            questions_iterable, desc="Convert answers", unit="answer"
        )
    return [question.as_answer() for question in questions_iterable]


def load_answers(
    path: Path,
    sample: Annotated[float, Interval(ge=0, le=1)] = 1,
    progress: bool = False,
) -> Sequence[Answer]:
    if path.name.startswith("training") and path.name.endswith("_new.json"):
        return load_clef_bioasq_answers(path, sample=sample, progress=progress)
    else:
        raise RuntimeError(f"Could not guess answer format from file: {path}")


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
