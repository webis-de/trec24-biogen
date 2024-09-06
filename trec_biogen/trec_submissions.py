from json import dumps
from pathlib import Path
from statistics import mean
from typing import Any, Sequence, TypedDict
from warnings import catch_warnings, simplefilter

from optuna import load_study
from optuna.exceptions import ExperimentalWarning
from optuna.trial import BaseTrial
from optuna.storages import JournalStorage, JournalFileOpenLock, JournalFileStorage
from optuna.trial._frozen import create_trial
from pydantic.networks import EmailStr

from trec_biogen.datasets import save_trec_biogen_answers
from trec_biogen.evaluation import GenerationMeasure, RetrievalMeasure, evaluate_generation, evaluate_retrieval
from trec_biogen.model import Answer, Question
from trec_biogen.optimization import build_answering_module, suggest_language_model_name

class TrialDict(TypedDict):
    values: list[float]
    params: dict[str, Any]


def _objective(
    answers: Sequence[Answer],
    retrieval_measures: Sequence[RetrievalMeasure],
    generation_measures: Sequence[GenerationMeasure],
    trial: BaseTrial,
) -> Sequence[float]:
    print("Re-evaluating trial...")
    questions = [answer.as_question() for answer in answers]
    contexts = [question.as_partial_answer() for question in questions]
    module = build_answering_module(answers, trial=trial)
    predictions = module.answer_many(contexts)
    retrieval_metrics = [
        evaluate_retrieval(
            ground_truth=answers,
            predictions=predictions,
            measure=measure,
        )
        for measure in retrieval_measures
    ]
    generation_metrics = [
        evaluate_generation(
            ground_truth=answers,
            predictions=predictions,
            measure=measure,
            language_model_name=suggest_language_model_name(
                trial=trial,
                name="language_model_name",
            ),
        )
        for measure in generation_measures
    ]
    print(f"Metrics: {retrieval_metrics + generation_metrics}")
    return (*retrieval_metrics, *generation_metrics)

def prepare_trec_submissions(
    study_storage_path: Path,
    study_name: str, 
    submissions_path: Path,
    answers: Sequence[Answer],
    retrieval_measures: Sequence[RetrievalMeasure],
    generation_measures: Sequence[GenerationMeasure],
    questions: Sequence[Question],
    team_id: str,
    contact_email: EmailStr,
    top_k: int | None = None,
) -> None:
    submissions_study_path = submissions_path / study_name
    submissions_study_path.mkdir(exist_ok=True)
    if sum(1 for _ in submissions_study_path.iterdir()) > 0:
        print("Clear previous submission files.")
        input("To abort, press Ctrl+C. Or hit Enter to proceed.")
        for file_path in submissions_study_path.iterdir():
            file_path.unlink()

    print("Load study.")
    with catch_warnings():
        simplefilter(action="ignore", category=ExperimentalWarning)
        storage = JournalStorage(
            log_storage=JournalFileStorage(
                file_path=str(study_storage_path), 
                lock_obj=JournalFileOpenLock(str(study_storage_path)),
            ),
        )
    study = load_study(
        storage=storage,
        study_name=study_name,
    )

    best_trials = study.best_trials

    if top_k is not None:
        print("Sort best trials (based on previous objective values).")
        best_trials = sorted(
            best_trials, 
            key=lambda trial: mean(trial.values),
            reverse=True,
        )
        best_trials = best_trials[:top_k]

    print("Re-measure (top-)trial values w.r.t. new measures.")
    best_trials_reevaluated = [
        create_trial(
            state=trial.state,
            value=None,
            values=_objective(
                answers=answers,
                retrieval_measures=retrieval_measures,
                generation_measures=generation_measures,
                trial=trial,
            ),
            params=trial.params,
            distributions=trial.distributions,
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs,
            intermediate_values=trial.intermediate_values,
        )
        for trial in best_trials
    ]

    print("Sort best trials.")
    best_trials_reevaluated = sorted(
        best_trials_reevaluated, 
        key=lambda trial: mean(trial.values),
        reverse=True,
    )
    contexts = [question.as_partial_answer() for question in questions]
    for i, trial in enumerate(best_trials_reevaluated):
        run_number = i + 1
        run_name = f"{team_id}-{run_number}"
        run_path = submissions_study_path / f"{run_name}.json"
        run_param_path = submissions_study_path / f"{run_name}.params.json"

        print(f"Computing run '{run_name}'..")
        module = build_answering_module(answers, trial=trial)
        predictions = module.answer_many(contexts)

        print(f"Saving run '{run_name}' to: {run_path}")
        save_trec_biogen_answers(
            answers=predictions,
            team_id=team_id,
            run_name=run_name,
            contact_email=contact_email,
            path=run_path,
            progress=True,
        )

        print(f"Saving params of run '{run_name}' to: {run_param_path}")
        with run_param_path.open("wt", encoding="utf8") as file:
            file.write(dumps({
                "params": trial.params,
                "values": trial.values,
            }, indent=2))


