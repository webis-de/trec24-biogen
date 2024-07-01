from statistics import mean
from typing import Iterable, Literal

from dspy.teleprompt import LabeledFewShot
from ir_measures import Precision, nDCG
from optuna import Study, Trial, create_study
from optuna.trial import FrozenTrial

from trec_biogen.datasets import GenerationDatasetName, RetrievalDatasetName, load_generation_dataset, load_retrieval_dataset
from trec_biogen.evaluation import GenerationMeasure, RetrievalMeasure, evaluate_generation_module, evaluate_retrieval_module
from trec_biogen.generation import DspyGenerationModule, RetrievalAugmentedGenerationModule
from trec_biogen.modules import GenerationModule, RetrievalModule
from trec_biogen.retrieval import GenerationAugmentedRetrievalModule, PyterrierRetrievalModule


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

    # How often should we "loop" the generation augmented retrieval?
    num_augmentations = trial.suggest_int(
        name="num_augmentations",
        low=0,
        high=10,
    )
    # Should we also augment the generation module after augmenting the retrieval module or keep the generation module fixed?
    back_augment = trial.suggest_categorical(
        name="back_augment",
        choices=[False, True]
    )
    if num_augmentations > 0:
        # TODO: Define all hyperparameters needed for the augmentations.
        todo1 = trial.suggest_float(name="todo", low=-10, high=10)
        # TODO: Define all hyperparameters needed for the back augmentations.
        if back_augment:
            todo2 = trial.suggest_float(name="todo", low=-10, high=10)
        for _ in range(num_augmentations):
            retrieval_module = GenerationAugmentedRetrievalModule(
                retrieval_module=retrieval_module,
                generation_module=generation_module,
                todo=todo1,
            )
            if back_augment:
                generation_module = RetrievalAugmentedGenerationModule(
                    generation_module=generation_module,
                    retrieval_module=retrieval_module,
                    todo=todo2,
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

    # How often should we "loop" the retrieval augmented generation?
    num_augmentations = trial.suggest_int(
        name="num_augmentations",
        low=0,
        high=10,
    )
    # Should we also augment the retrieval module after augmenting the generation module or keep the retrieval module fixed?
    back_augment = trial.suggest_categorical(
        name="back_augment",
        choices=[False, True]
    )
    if num_augmentations > 0:
        # TODO: Define all hyperparameters needed for the augmentations.
        todo1 = trial.suggest_float(name="todo", low=-10, high=10)
        # TODO: Define all hyperparameters needed for the back augmentations.
        if back_augment:
            todo2 = trial.suggest_float(name="todo", low=-10, high=10)
        for _ in range(num_augmentations):
            generation_module = RetrievalAugmentedGenerationModule(
                generation_module=generation_module,
                retrieval_module=retrieval_module,
                todo=todo1,
            )
            if back_augment:
                retrieval_module = GenerationAugmentedRetrievalModule(
                    retrieval_module=retrieval_module,
                    generation_module=generation_module,
                    todo=todo2,
                )

    return generation_module


def optimize_retrieval_module(
    dataset_name: RetrievalDatasetName,
    measures: Iterable[RetrievalMeasure],
) -> FrozenTrial:

    def objective(trial: Trial) -> float:
        module = build_generation_augmented_retrieval_module(
            trial=trial,
        )
        dataset = load_retrieval_dataset(
            dataset_name=dataset_name,
            split="dev",
        )
        metrics = (
            evaluate_retrieval_module(
                module=module,
                dataset=dataset,
                measure=measure,
            )
            for measure in measures
        )
        return mean(metrics)

    study: Study = create_study()
    study.optimize(
        func=objective,
        n_trials=100,
    )
    return study.best_trial


def optimize_generation_module(
    dataset_name: GenerationDatasetName,
    measures: Iterable[GenerationMeasure],
) -> FrozenTrial:

    def objective(trial: Trial) -> float:
        module = build_retrieval_augmented_generation_module(
            trial=trial,
        )
        dataset = load_generation_dataset(
            dataset_name=dataset_name,
            split="dev",
        )
        metrics = (
            evaluate_generation_module(
                module=module,
                dataset=dataset,
                measure=measure,
            )
            for measure in measures
        )
        return mean(metrics)

    study: Study = create_study()
    study.optimize(
        func=objective,
        n_trials=100,
    )
    return study.best_trial


def optimize_retrieval_and_generation_module(
    retrieval_dataset_name: RetrievalDatasetName,
    retrieval_measures: Iterable[RetrievalMeasure],
    generation_dataset_name: GenerationDatasetName,
    generation_measures: Iterable[GenerationMeasure],
) -> FrozenTrial:

    def objective(trial: Trial) -> float:
        retrieval_module = build_generation_augmented_retrieval_module(
            trial=trial,
        )
        retrieval_dataset = load_retrieval_dataset(
            dataset_name=retrieval_dataset_name,
            split="dev",
        )
        retrieval_metrics = (
            evaluate_retrieval_module(
                module=retrieval_module,
                dataset=retrieval_dataset,
                measure=measure,
            )
            for measure in retrieval_measures
        )

        generation_module = build_retrieval_augmented_generation_module(
            trial=trial,
        )
        generation_dataset = load_generation_dataset(
            dataset_name=generation_dataset_name,
            split="dev",
        )
        generation_metrics = (
            evaluate_generation_module(
                module=generation_module,
                dataset=generation_dataset,
                measure=measure,
            )
            for measure in generation_measures
        )

        return mean([*retrieval_metrics, *generation_metrics])

    study: Study = create_study()
    study.optimize(
        func=objective,
        n_trials=100,
    )
    return study.best_trial


if __name__ == "main":
    trial = optimize_retrieval_and_generation_module(
        retrieval_dataset_name="bioasq-task-b",
        retrieval_measures=[
            Precision@1,  # type: ignore
            nDCG,
        ],
        generation_dataset_name="bioasq-task-b",
        generation_measures=[
            "todo"  # TODO: Use the implemented measures.
        ],
    )
    print(trial)
