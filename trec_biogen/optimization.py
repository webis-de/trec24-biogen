from typing import Iterable, Literal, Sequence

from dspy.teleprompt import LabeledFewShot
from optuna import Study, Trial, create_study
from optuna.trial import FrozenTrial

from trec_biogen.answering import IndependentAnsweringModule, RecurrentAnsweringModule
from trec_biogen.evaluation import (
    GenerationMeasure,
    RetrievalMeasure,
    evaluate_generation,
    evaluate_retrieval,
)
from trec_biogen.generation import DspyGenerationModule, RetrievalThenGenerationModule
from trec_biogen.model import Answer
from trec_biogen.modules import AnsweringModule, GenerationModule, RetrievalModule
from trec_biogen.retrieval import (
    GenerationThenRetrievalModule,
    PyterrierRetrievalModule,
)


def build_retrieval_module(
    trial: Trial,
) -> RetrievalModule:
    """
    Build a simple retrieval module based on hyperparameters drawn from the trial.
    """

    # TODO: Define all hyperparameters needed for the retrieval module.
    todo = trial.suggest_float(
        name="todo",
        low=-10,
        high=10,
    )
    retrieval_module = PyterrierRetrievalModule(
        todo=todo,
    )

    # Optimization:
    optimize_retrieval = trial.suggest_categorical(
        name="optimize_retrieval",
        choices=[False, True],
    )
    if optimize_retrieval:
        # TODO (later): Optimize the PyTerrier pipeline with `pipeline.fit()`.
        pass

    return retrieval_module


def build_generation_module(
    trial: Trial,
) -> GenerationModule:
    """
    Build a simple generation module based on hyperparameters drawn from the trial.
    """

    # TODO: Define all hyperparameters needed for the generation module.
    todo = trial.suggest_float(
        name="todo",
        low=-10,
        high=10,
    )
    generation_module = DspyGenerationModule(
        todo=todo,
    )

    # Optimization:
    optimize_generation = trial.suggest_categorical(
        name="optimize_generation",
        choices=[False, True],
    )
    if optimize_generation:
        # TODO (later): Add other DSPy optimizers.
        optimizer_type: Literal["labeled-few-shot"]
        optimizer_type = "labeled-few-shot"
        if optimizer_type == "labeled-few-shot":

            # Tune the generation module with DSPy (to select few-shot examples).
            few_shot_k = trial.suggest_int(
                name="few_shot_k",
                low=1,
                high=10,
            )
            optimizer = LabeledFewShot(k=few_shot_k)
            generation_module = optimizer.compile(
                student=generation_module,
                trainset=NotImplemented,
                sample=True,
            )
        else:
            raise ValueError(f"Unkown optimizer type: {optimizer_type}")

    return generation_module


def build_generation_augmented_retrieval_module(
    trial: Trial,
) -> RetrievalModule:
    """
    Build a generation-augmented retrieval module based on hyperparameters drawn from the trial.
    """

    # Build simple retrieval and generation modules.
    retrieval_module = build_retrieval_module(trial=trial)
    generation_module = build_generation_module(trial=trial)

    # How often should we "cycle" the generation augmented retrieval?
    num_augmentations = trial.suggest_int(
        name="generation_augmented_retrieval_num_augmentations",
        low=1,
        high=10,
    )
    # Should we also augment the generation module after augmenting the retrieval module or keep the generation module fixed?
    back_augment = trial.suggest_categorical(
        name="generation_augmented_retrieval_back_augment", choices=[False, True]
    )

    for _ in range(num_augmentations):
        retrieval_module = GenerationThenRetrievalModule(
            retrieval_module=retrieval_module,
            generation_module=generation_module,
        )
        if back_augment:
            generation_module = RetrievalThenGenerationModule(
                generation_module=generation_module,
                retrieval_module=retrieval_module,
            )

    return retrieval_module


def build_retrieval_augmented_generation_module(
    trial: Trial,
) -> GenerationModule:
    """
    Build a retrieval-augmented generation module based on hyperparameters drawn from the trial.
    """

    # Build simple generation and retrieval modules.
    generation_module = build_generation_module(trial=trial)
    retrieval_module = build_retrieval_module(trial=trial)

    # How often should we "cycle" the generation augmented retrieval?
    num_augmentations = trial.suggest_int(
        name="retrieval_augmented_generation_num_augmentations",
        low=1,
        high=10,
    )
    # Should we also augment the retrieval module after augmenting the generation module or keep the retrieval module fixed?
    back_augment = trial.suggest_categorical(
        name="retrieval_augmented_generation_back_augment", choices=[False, True]
    )

    for _ in range(num_augmentations):
        generation_module = RetrievalThenGenerationModule(
            generation_module=generation_module,
            retrieval_module=retrieval_module,
        )
        if back_augment:
            retrieval_module = GenerationThenRetrievalModule(
                retrieval_module=retrieval_module,
                generation_module=generation_module,
            )

    return generation_module


def build_answering_module_no_augmentation(
    trial: Trial,
) -> AnsweringModule:
    """
    Build a answering module that uses generation and retrieval modules independently without any augmentation.
    """

    # Build simple generation and retrieval modules.
    generation_module = build_generation_module(trial=trial)
    retrieval_module = build_retrieval_module(trial=trial)

    # Compose answering module.
    return IndependentAnsweringModule(
        generation_module=generation_module,
        retrieval_module=retrieval_module,
    )


def build_answering_module_independent_augmentation(
    trial: Trial,
) -> AnsweringModule:
    """
    Build a answering module that uses generation and retrieval modules independently while augmenting generation and retrieval individually.
    """

    # Build augmented generation and retrieval modules.
    generation_module = build_retrieval_augmented_generation_module(trial=trial)
    retrieval_module = build_generation_augmented_retrieval_module(trial=trial)

    # Compose answering module.
    return IndependentAnsweringModule(
        generation_module=generation_module,
        retrieval_module=retrieval_module,
    )


def build_answering_module_cross_augmentation(
    trial: Trial,
) -> AnsweringModule:
    """
    Build a answering module that uses generation and retrieval modules recurrently while feeding back the outputs from the generation module to the retrieval module and vice-versa.
    """

    # Build simple generation and retrieval modules.
    generation_module = build_generation_module(trial=trial)
    retrieval_module = build_retrieval_module(trial=trial)

    # Compose answering module.
    answering_module = IndependentAnsweringModule(
        generation_module=generation_module,
        retrieval_module=retrieval_module,
    )

    # Recurrently cross-augment the generation and retrieval.
    steps = trial.suggest_int(
        name="answering_module_cross_augmentation_steps",
        low=1,
        high=10,
    )
    return RecurrentAnsweringModule(
        answering_module=answering_module,
        steps=steps,
    )


def build_answering_module(
    trial: Trial,
) -> AnsweringModule:
    """
    Build a answering module with or without augmenting the generation and retrieval modules.
    """

    # Which augmentation type/strategy should we apply?
    augmentation_type = trial.suggest_categorical(
        name="answering_module_type",
        choices=[
            "no augmentation",
            "independent augmentation",
            "cross augmentation",
        ],
    )
    if augmentation_type == "no augmentation":
        return build_answering_module_no_augmentation(trial)
    elif augmentation_type == "independent augmentation":
        return build_answering_module_independent_augmentation(trial)
    elif augmentation_type == "cross augmentation":
        return build_answering_module_cross_augmentation(trial)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def optimize_answering_module(
    ground_truth: Sequence[Answer],
    retrieval_measures: Iterable[RetrievalMeasure],
    generation_measures: Iterable[GenerationMeasure],
    trials: int | None = None,
    timeout: float | None = None,
    parallelism: int = 1,
    progress: bool = False,
) -> FrozenTrial:
    questions = [answer.as_question() for answer in ground_truth]
    contexts = [question.as_partial_answer() for question in questions]

    def objective(trial: Trial) -> Sequence[float]:
        module = build_answering_module(trial)
        predictions = module.answer_many(contexts)
        retrieval_metrics = (
            evaluate_retrieval(
                ground_truth=ground_truth,
                predictions=predictions,
                measure=measure,
            )
            for measure in retrieval_measures
        )
        generation_metrics = (
            evaluate_generation(
                ground_truth=ground_truth,
                predictions=predictions,
                measure=measure,
            )
            for measure in generation_measures
        )
        return (*retrieval_metrics, *generation_metrics)

    study: Study = create_study()
    study.optimize(
        func=objective,
        n_trials=trials,
        timeout=timeout,
        n_jobs=parallelism,
        show_progress_bar=progress,
    )
    return study.best_trial


# if __name__ == "main":
#     trial = optimize_retrieval_and_generation_module(
#         retrieval_dataset_name="bioasq-task-b",
#         retrieval_measures=[
#             Precision@1,  # type: ignore
#             nDCG,
#         ],
#         generation_dataset_name="bioasq-task-b",
#         generation_measures=[
#             "todo"  # TODO: Use the implemented measures.
#         ],
#     )
#     print(trial)
