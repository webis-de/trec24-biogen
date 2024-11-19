from typing import Annotated

from annotated_types import Ge, Gt, Interval
from cyclopts import App, Parameter
from cyclopts.types import ResolvedExistingDirectory, ResolvedFile, ResolvedExistingFile
from cyclopts.validators import Number
from dotenv import load_dotenv, find_dotenv
from email_validator import validate_email

from trec_biogen.evaluation import GenerationMeasure, RetrievalMeasure

app = App()


@app.meta.default
def launcher(*tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)]):
    if find_dotenv():
        load_dotenv()
    command, bound, _ = app.parse_args(tokens)
    return command(*bound.args, **bound.kwargs)


@app.command()
def index_pubmed_full_texts(
    *,
    dry_run: bool = False,
    refetch: bool = False,
    sample: (
        Annotated[
            float, Interval(ge=0, le=1), Parameter(validator=Number(gte=0, lte=1))
        ]
        | None
    ) = None,
) -> None:
    from trec_biogen.jobs.index_pubmed_full_text import (
        index_pubmed_full_texts as _index_pubmed_full_texts,
    )

    _index_pubmed_full_texts(
        dry_run=dry_run,
        refetch=refetch,
        sample=sample,
    )


@app.command()
def index_pubmed_trec_ids(
    trec_ids_path: ResolvedExistingFile,
    *,
    dry_run: bool = False,
) -> None:
    from trec_biogen.jobs.index_pubmed_trec_ids import (
        index_pubmed_trec_ids as _index_pubmed_trec_ids,
    )

    _index_pubmed_trec_ids(
        trec_ids_path=trec_ids_path,
        dry_run=dry_run,
    )


parse_app = App(name="parse")
app.command(parse_app)


@parse_app.command()
def questions(
    path: ResolvedExistingFile,
) -> None:
    from trec_biogen.datasets import load_questions

    questions = load_questions(path, progress=True)
    print(f"Found {len(questions)} questions.")


@parse_app.command()
def answers(
    path: ResolvedExistingFile,
) -> None:
    from trec_biogen.datasets import load_answers

    answers = load_answers(path, progress=True)
    print(f"Found {len(answers)} answers.")


convert_app = App(name="convert")
app.command(convert_app)


def _email_validator(type_, value: str) -> None:
    validate_email(value)


@convert_app.command()
def clef_bioasq_answers_to_trec_biogen_answers(
    input_path: ResolvedExistingFile,
    output_path: ResolvedFile,
    *,
    team_id: str,
    run_name: str,
    contact_email: Annotated[str, Parameter(validator=_email_validator)],
) -> None:
    from trec_biogen.datasets import load_clef_bioasq_answers, save_trec_biogen_answers

    answers = load_clef_bioasq_answers(input_path, progress=True)
    print(f"Found {len(answers)} answers in the CLEF BioASQ file.")
    save_trec_biogen_answers(
        answers=answers,
        team_id=team_id,
        run_name=run_name,
        contact_email=contact_email,
        path=output_path,
        progress=True,
    )


def _retrieval_measures(type_, *values):
    from ir_measures import parse_measure

    return [parse_measure(value) for value in values]


@app.command()
def optimize(
    answers_path: ResolvedExistingFile,
    study_storage_path: ResolvedFile,
    study_name: str,
    *,
    sample: Annotated[
        float, Interval(ge=0, le=1), Parameter(validator=Number(gte=0, lte=1))
    ] | Annotated[
        int, Interval(gt=0), Parameter(validator=Number(gt=0))
    ] = 1.0,
    retrieval_measures: Annotated[
        list[RetrievalMeasure], Parameter(converter=_retrieval_measures)
    ] = [],
    generation_measures: list[GenerationMeasure] = [],
    trials: Annotated[int, Ge(1), Parameter(validator=Number(gte=1))] | None = None,
    timeout: Annotated[float, Gt(0), Parameter(validator=Number(gt=0))] | None = None,
    parallelism: Annotated[int, Ge(1), Parameter(validator=Number(gte=1))] = 1,
    progress: bool = False,
    wandb: bool = False,
    ray: bool = False,
    resume: bool = False,
) -> None:
    if find_dotenv():
        load_dotenv()
        
    from trec_biogen.datasets import load_answers
    from trec_biogen.optimization import optimize_answering_module

    answers = load_answers(
        path=answers_path,
        sample=sample,
        progress=True,
    )
    print(f"Found {len(answers)} answers.")
    optimize_answering_module(
        study_storage_path=study_storage_path,
        study_name=study_name,
        answers=answers,
        retrieval_measures=retrieval_measures,
        generation_measures=generation_measures,
        trials=trials,
        timeout=timeout,
        parallelism=parallelism,
        progress=progress,
        wandb=wandb,
        ray=ray,
        resume=resume,
    )
    

@app.command()
def num_trials(
    study_storage_path: ResolvedFile,
    study_name: str,
) -> None:
    from trec_biogen.optimization import num_trials
    print(num_trials(
        study_storage_path=study_storage_path,
        study_name=study_name,
    ))

@app.command()
def prepare_trec_submissions(
    answers_path: ResolvedExistingFile,
    study_storage_path: ResolvedFile,
    study_name: str,
    submissions_path: ResolvedExistingDirectory,
    questions_path: ResolvedExistingFile,
    *,
    team_id: str,
    contact_email: str,
    sample: Annotated[
        float, Interval(ge=0, le=1), Parameter(validator=Number(gte=0, lte=1))
    ] | Annotated[
        int, Interval(gt=0), Parameter(validator=Number(gt=0))
    ] = 1.0,
    retrieval_measures: Annotated[
        list[RetrievalMeasure], Parameter(converter=_retrieval_measures)
    ] = [],
    generation_measures: list[GenerationMeasure] = [],
    top_k: int | None = None,
) -> None:
    if find_dotenv():
        load_dotenv()
        
    from trec_biogen.datasets import load_answers, load_questions
    from trec_biogen.trec_submissions import prepare_trec_submissions as _prepare_trec_submissions

    answers = load_answers(
        path=answers_path,
        sample=sample,
        progress=True,
    )
    print(f"Found {len(answers)} answers.")

    questions = load_questions(
        path=questions_path,
        progress=True,
    )
    print(f"Found {len(questions)} questions.")

    _prepare_trec_submissions(
        study_storage_path=study_storage_path,
        study_name=study_name,
        submissions_path=submissions_path,
        answers=answers,
        retrieval_measures=retrieval_measures,
        generation_measures=generation_measures,
        questions=questions,
        team_id=team_id,
        contact_email=contact_email,
        top_k=top_k,
    )