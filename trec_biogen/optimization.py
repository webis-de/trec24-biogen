from pathlib import Path
from typing import Callable, Literal, Sequence, TypeAlias
from warnings import catch_warnings, simplefilter, filterwarnings

from dspy import settings as dspy_settings
from joblib import parallel_backend
from optuna import Study, create_study
from optuna.exceptions import TrialPruned
from optuna.study import StudyDirection
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.study.study import load_study
from optuna.trial import FrozenTrial, BaseTrial, TrialState
from optuna.storages import JournalStorage, JournalFileOpenLock, JournalFileStorage
from optuna_integration import WeightsAndBiasesCallback
from pyterrier.transformer import Transformer
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
from pyterrier_dr import TasB, TctColBert, Ance
from ray import init as ray_init
from ray.util.joblib import register_ray

from trec_biogen.answering import IndependentAnsweringModule, RecurrentAnsweringModule
from trec_biogen.dspy_generation import (
    ExactPredict,
    GenerationAnswerPredict,
    PredictType,
    SummaryPredict,
)
from trec_biogen.evaluation import (
    GenerationMeasure,
    RetrievalMeasure,
    evaluate_generation,
    evaluate_retrieval,
)
from trec_biogen.generation import DspyGenerationModule, RetrievalThenGenerationModule
from trec_biogen.language_models import LanguageModelName, get_dspy_language_model
from trec_biogen.model import Answer
from trec_biogen.modules import AnsweringModule, GenerationModule, RetrievalModule
from trec_biogen.pyterrier import CutoffRerank
from trec_biogen.pyterrier_pubmed import (
    PubMedElasticsearchRetrieve,
    PubMedSentencePassager,
)
from trec_biogen.pyterrier_query import (
    ContextElasticsearchQueryTransformer,
    ContextQueryTransformer,
)
from trec_biogen.retrieval import (
    GenerationThenRetrievalModule,
    PyterrierRetrievalModule,
)


def suggest_must_should(
    trial: BaseTrial, name: str
) -> Literal["must", "should"] | None:
    must_should = trial.suggest_categorical(
        name=name,
        choices=["must", "should", None],
    )
    if must_should == "must":
        return "must"
    elif must_should == "should":
        return "should"
    elif must_should is None:
        return None
    else:
        raise ValueError(f"Illegal value: {must_should}")


PointwiseRerankerModel: TypeAlias = Literal[
    "castorini/monot5-base-msmarco",
    "castorini/monot5-3b-msmarco",
    "castorini/monot5-3b-med-msmarco",
    "sentence-transformers/msmarco-distilbert-base-tas-b",
    "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "castorini/tct_colbert-v2-hnp-msmarco",
]
PairwiseRerankerModel: TypeAlias = Literal[
    "castorini/duot5-base-msmarco",
    "castorini/duot5-3b-msmarco",
    "castorini/duot5-3b-med-msmarco",
]


def build_retrieval_module(
    trial: BaseTrial,
) -> RetrievalModule:
    """
    Build a simple retrieval module based on hyperparameters drawn from the trial.
    """
    pipeline: Transformer = Transformer.identity()
    include_question = trial.suggest_categorical(
        name="context_query_include_question",
        choices=[
            # False,
            True,
        ],
    )
    include_query = trial.suggest_categorical(
        name="context_query_include_query",
        choices=[
            False,
            True,
        ],
    )
    include_narrative = trial.suggest_categorical(
        name="context_query_include_narrative",
        choices=[
            False,
            True,
        ],
    )
    include_summary = trial.suggest_categorical(
        name="context_query_include_summary",
        choices=[
            False,
            True,
        ],
    )
    include_exact = trial.suggest_categorical(
        name="context_query_include_exact",
        choices=[
            False,
            True,
        ],
    )

    # Extract/expand the query string from the context.
    context_query = ContextQueryTransformer(
        include_question=include_question,
        include_query=include_query,
        include_narrative=include_narrative,
        include_summary=include_summary,
        include_exact=include_exact,
        progress=True,
    )
    pipeline = pipeline >> context_query

    # Extract the structured Elasticsearch query from the context.
    match_title = suggest_must_should(
        trial=trial,
        name="context_elasticsearch_query_match_title",
    )
    match_abstract = suggest_must_should(
        trial=trial,
        name="context_elasticsearch_query_match_abstract",
    )
    match_full_text = suggest_must_should(
        trial=trial,
        name="context_elasticsearch_query_match_full_text",
    )
    if not any(
        (
            match_title == "must",
            match_abstract == "must",
            match_full_text == "must",
        )
    ):
        raise TrialPruned("Must match at least on title or abstract (or both).")
    context_elasticsearch_query = ContextElasticsearchQueryTransformer(
        require_title=trial.suggest_categorical(
            name="context_elasticsearch_query_require_title",
            choices=[
                False,
                True,
            ],
        ),
        require_abstract=trial.suggest_categorical(
            name="context_elasticsearch_query_require_abstract",
            choices=[
                False,
                True,
            ],
        ),
        filter_publication_types=trial.suggest_categorical(
            name="context_elasticsearch_query_filter_publication_types",
            choices=[
                False,
                True,
            ],
        ),
        remove_stopwords=trial.suggest_categorical(
            name="context_elasticsearch_query_remove_stopwords",
            choices=[
                False,
                True,
            ],
        ),
        match_title=match_title,
        match_abstract=match_abstract,
        match_full_text=match_full_text,
        match_mesh_terms=suggest_must_should(
            trial=trial,
            name="context_elasticsearch_query_match_mesh_terms",
        ),
        progress=True,
    )
    pipeline = pipeline >> context_elasticsearch_query

    # Retrieve PubMed articles from Elasticsearch.
    pubmed_elasticsearch_retrieve = PubMedElasticsearchRetrieve(
        include_title_text=trial.suggest_categorical(
            name="pubmed_elasticsearch_retrieve_include_title_text",
            choices=[False, True],
        ),
        include_abstract_text=trial.suggest_categorical(
            name="pubmed_elasticsearch_retrieve_include_abstract_text",
            choices=[False, True],
        ),
    )
    pipeline = pipeline >> pubmed_elasticsearch_retrieve

    # Passage the documents into snippets.
    passaging_enabled = trial.suggest_categorical(
        name="passaging_enabled",
        choices=[
            # False,
            True,
        ],
    )
    if passaging_enabled:
        pubmed_sentence_passager = PubMedSentencePassager(
            include_title_snippets=trial.suggest_categorical(
                name="pubmed_sentence_passager_include_title_snippets",
                choices=[
                    False,
                    True,
                ],
            ),
            include_abstract_snippets=trial.suggest_categorical(
                name="pubmed_sentence_passager_include_abstract_snippets",
                choices=[
                    False,
                    True,
                ],
            ),
            max_sentences=trial.suggest_int(
                name="pubmed_sentence_passager_max_sentences",
                low=1,
                high=3,
            ),
        )
        pipeline = pipeline >> pubmed_sentence_passager

    # Pointwise re-ranking.
    pointwise_reranker_model: PointwiseRerankerModel | None = trial.suggest_categorical(
        name="pointwise_reranker_model",
        choices=[
            "castorini/monot5-base-msmarco",
            # "castorini/monot5-3b-msmarco",
            # "castorini/monot5-3b-med-msmarco",
            "sentence-transformers/msmarco-distilbert-base-tas-b",
            "sentence-transformers/msmarco-roberta-base-ance-firstp",
            "castorini/tct_colbert-v2-hnp-msmarco",
            None,
        ],
    )  # type: ignore
    pointwise_reranker: Transformer | None
    if pointwise_reranker_model in (
        "castorini/monot5-base-msmarco",
        "castorini/monot5-3b-msmarco",
        "castorini/monot5-3b-med-msmarco",
    ):
        with catch_warnings():
            simplefilter(action="ignore", category=FutureWarning)
            pointwise_reranker = MonoT5ReRanker(
                model=pointwise_reranker_model,
                batch_size=4,
                verbose=True,
            )
    elif pointwise_reranker_model in (
        "sentence-transformers/msmarco-distilbert-base-tas-b",
    ):
        with catch_warnings():
            filterwarnings(
                action="ignore",
                message="TypedStorage is deprecated",
                category=UserWarning,
            )
            simplefilter(action="ignore", category=FutureWarning)
            pointwise_reranker = TasB(
                model_name=pointwise_reranker_model,
                batch_size=8,
                verbose=True,
            )
    elif pointwise_reranker_model in ("castorini/tct_colbert-v2-hnp-msmarco",):
        with catch_warnings():
            simplefilter(action="ignore", category=FutureWarning)
            pointwise_reranker = TctColBert(
                model_name=pointwise_reranker_model,
                batch_size=8,
                verbose=True,
            )
    elif pointwise_reranker_model in (
        "sentence-transformers/msmarco-roberta-base-ance-firstp",
    ):
        with catch_warnings():
            simplefilter(action="ignore", category=FutureWarning)
            pointwise_reranker = Ance(
                model_name=pointwise_reranker_model,
                batch_size=8,
                verbose=True,
            )
    else:
        pointwise_reranker = None
    if pointwise_reranker is not None:
        pointwise_reranker_cutoff = trial.suggest_categorical(
            name="pointwise_reranker_cutoff",
            choices=[
                10,
                50,
            ],
        )
        pipeline = CutoffRerank(
            candidates=pipeline,
            reranker=pointwise_reranker,
            cutoff=pointwise_reranker_cutoff,
        )

    # Pairwise re-ranking.
    pairwise_reranker_model: PairwiseRerankerModel | None = trial.suggest_categorical(
        name="pairwise_reranker_model",
        choices=[
            "castorini/duot5-base-msmarco",
            # "castorini/duot5-3b-msmarco",
            # "castorini/duot5-3b-med-msmarco",
            None,
        ],
    )  # type: ignore
    pairwise_reranker: Transformer | None
    if pairwise_reranker_model in (
        "castorini/duot5-base-msmarco",
        "castorini/duot5-3b-msmarco",
        "castorini/duot5-3b-med-msmarco",
    ):
        with catch_warnings():
            filterwarnings(
                action="ignore",
                message="TypedStorage is deprecated",
                category=UserWarning,
            )
            simplefilter(action="ignore", category=FutureWarning)
            pairwise_reranker = DuoT5ReRanker(
                model=pairwise_reranker_model,
                batch_size=4,
                verbose=True,
            )
    else:
        pairwise_reranker = None
    if pairwise_reranker is not None:
        pairwise_reranker_cutoff = trial.suggest_categorical(
            name="pairwise_reranker_cutoff",
            choices=[
                3,
                5,
            ],
        )
        pipeline = CutoffRerank(
            candidates=pipeline,
            reranker=pairwise_reranker,
            cutoff=pairwise_reranker_cutoff,
        )

    retrieval_module = PyterrierRetrievalModule(pipeline, progress=True)

    # TODO (later): Optimize the PyTerrier pipeline with `pipeline.fit()` if applicable.

    return retrieval_module


def suggest_language_model_name(trial: BaseTrial, name: str) -> LanguageModelName:
    language_model_name: LanguageModelName = trial.suggest_categorical(
        name="language_model_name",
        choices=[
            "blablador:Mistral-7B-Instruct-v0.3",
            "blablador:Mixtral-8x7B-Instruct-v0.1",
            # "blablador:Llama3.1-8B-Instruct",
            # "openai:gpt-4o-mini-2024-07-18",
        ],
    )  # type: ignore
    return language_model_name


def build_generation_module(
    answers: Sequence[Answer],
    trial: BaseTrial,
) -> GenerationModule:
    """
    Build a simple generation module based on hyperparameters drawn from the trial.
    """

    context_references_cutoff = trial.suggest_categorical(
        name="context_references_cutoff",
        choices=[
            3,
            5,
            10,
        ],
    )

    summary_predict_type: PredictType = trial.suggest_categorical(
        name="summary_predict_type",
        choices=[
            "predict",
            "chain-of-thought",
        ],
    )  # type: ignore
    summary_predict = SummaryPredict(
        predict_type=summary_predict_type,
        references_cutoff=context_references_cutoff,
    )

    exact_predict_type: PredictType = trial.suggest_categorical(
        name="exact_predict_type",
        choices=[
            "predict",
            "chain-of-thought",
        ],
    )  # type: ignore
    exact_predict = ExactPredict(
        predict_type=exact_predict_type,
        references_cutoff=context_references_cutoff,
    )

    language_model_name: LanguageModelName = suggest_language_model_name(
        trial=trial,
        name="language_model_name",
    )
    language_model = get_dspy_language_model(language_model_name)

    predict = GenerationAnswerPredict(
        summary_predict=summary_predict,
        exact_predict=exact_predict,
    )

    # Optimization:
    with dspy_settings.context(
        lm=language_model,
    ):
        predict.optimize(trial, answers)

    return DspyGenerationModule(
        predict=predict,
        language_model=language_model,
        progress=True,
    )


def build_generation_augmented_retrieval_module(
    answers: Sequence[Answer],
    trial: BaseTrial,
) -> RetrievalModule:
    """
    Build a generation-augmented retrieval module based on hyperparameters drawn from the trial.
    """

    # Build simple retrieval and generation modules.
    retrieval_module = build_retrieval_module(trial)
    generation_module = build_generation_module(answers, trial)

    # How often should we "cycle" the generation augmented retrieval?
    num_augmentations = trial.suggest_int(
        name="generation_augmented_retrieval_num_augmentations",
        low=1,
        high=3,
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
    answers: Sequence[Answer],
    trial: BaseTrial,
) -> GenerationModule:
    """
    Build a retrieval-augmented generation module based on hyperparameters drawn from the trial.
    """

    # Build simple generation and retrieval modules.
    generation_module = build_generation_module(answers, trial)
    retrieval_module = build_retrieval_module(trial)

    # How often should we "cycle" the generation augmented retrieval?
    num_augmentations = trial.suggest_int(
        name="retrieval_augmented_generation_num_augmentations",
        low=1,
        high=3,
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
    answers: Sequence[Answer],
    trial: BaseTrial,
) -> AnsweringModule:
    """
    Build a answering module that uses generation and retrieval modules independently without any augmentation.
    """

    # Build simple retrieval and generation modules.
    retrieval_module = build_retrieval_module(trial)
    generation_module = build_generation_module(answers, trial)

    # Compose answering module.
    return IndependentAnsweringModule(
        generation_module=generation_module,
        retrieval_module=retrieval_module,
    )


def build_answering_module_independent_augmentation(
    answers: Sequence[Answer],
    trial: BaseTrial,
) -> AnsweringModule:
    """
    Build a answering module that uses generation and retrieval modules independently while augmenting generation and retrieval individually.
    """

    # Build augmented retrieval and generation modules.
    retrieval_module = build_generation_augmented_retrieval_module(answers, trial)
    generation_module = build_retrieval_augmented_generation_module(answers, trial)

    # Compose answering module.
    return IndependentAnsweringModule(
        generation_module=generation_module,
        retrieval_module=retrieval_module,
    )


def build_answering_module_cross_augmentation(
    answers: Sequence[Answer],
    trial: BaseTrial,
) -> AnsweringModule:
    """
    Build a answering module that uses generation and retrieval modules recurrently while feeding back the outputs from the generation module to the retrieval module and vice-versa.
    """

    # Build simple retrieval and generation modules.
    retrieval_module = build_retrieval_module(trial)
    generation_module = build_generation_module(answers, trial)

    # Compose answering module.
    answering_module = IndependentAnsweringModule(
        generation_module=generation_module,
        retrieval_module=retrieval_module,
    )

    # Recurrently cross-augment the generation and retrieval.
    steps = trial.suggest_int(
        name="answering_module_cross_augmentation_steps",
        low=1,
        high=3,
    )
    return RecurrentAnsweringModule(
        answering_module=answering_module,
        steps=steps,
    )


def build_answering_module(
    answers: Sequence[Answer],
    trial: BaseTrial,
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
        return build_answering_module_no_augmentation(answers, trial)
    elif augmentation_type == "independent augmentation":
        return build_answering_module_independent_augmentation(answers, trial)
    elif augmentation_type == "cross augmentation":
        return build_answering_module_cross_augmentation(answers, trial)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def optimize_answering_module(
    study_storage_path: Path,
    study_name: str,
    answers: Sequence[Answer],
    retrieval_measures: Sequence[RetrievalMeasure],
    generation_measures: Sequence[GenerationMeasure],
    trials: int | None = None,
    timeout: float | None = None,
    parallelism: int = 1,
    progress: bool = False,
    wandb: bool = False,
    ray: bool = False,
    resume: bool = False,
) -> Sequence[FrozenTrial]:
    if ray:
        ray_init()
    register_ray()

    questions = [answer.as_question() for answer in answers]
    contexts = [question.as_partial_answer() for question in questions]

    def objective(trial: BaseTrial) -> Sequence[float]:
        module = build_answering_module(answers, trial)
        predictions = module.answer_many(contexts)
        retrieval_metrics = (
            evaluate_retrieval(
                ground_truth=answers,
                predictions=predictions,
                measure=measure,
            )
            for measure in retrieval_measures
        )
        generation_metrics = (
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
        )
        return (*retrieval_metrics, *generation_metrics)

    with catch_warnings():
        simplefilter(action="ignore", category=ExperimentalWarning)
        storage = JournalStorage(
            log_storage=JournalFileStorage(
                file_path=str(study_storage_path), 
                lock_obj=JournalFileOpenLock(str(study_storage_path)),
            ),
        )
    study: Study = create_study(
        storage=storage,
        study_name=study_name,
        directions=[StudyDirection.MAXIMIZE]
        * (len(retrieval_measures) + len(generation_measures)),
        sampler=TPESampler(),
        load_if_exists=resume,
    )
    metric_names = [
        *(str(measure) for measure in retrieval_measures),
        *generation_measures,
    ]
    with catch_warnings():
        simplefilter(action="ignore", category=ExperimentalWarning)
        study.set_metric_names(metric_names)
    callbacks: list[Callable[[Study, FrozenTrial], None]] = []
    if wandb:
        callbacks.append(
            WeightsAndBiasesCallback(
                metric_name=metric_names,
                wandb_kwargs=dict(
                    project="trec-biogen",
                ),
            )
        )
    if ray:
        with parallel_backend(backend="ray", n_jobs=-1):
            study.optimize(
                func=objective,
                n_trials=trials,
                timeout=timeout,
                n_jobs=parallelism,
                show_progress_bar=progress,
                callbacks=callbacks,
                catch=[],
            )
    else:
        study.optimize(
            func=objective,
            n_trials=trials,
            timeout=timeout,
            n_jobs=parallelism,
            show_progress_bar=progress,
            callbacks=callbacks,
            catch=[],
        )
    return study.best_trials


def num_trials(
    study_storage_path: Path,
    study_name: str,
) -> int:
    with catch_warnings():
        simplefilter(action="ignore", category=ExperimentalWarning)
        storage = JournalStorage(
            log_storage=JournalFileStorage(
                file_path=str(study_storage_path), 
                lock_obj=JournalFileOpenLock(str(study_storage_path)),
            ),
        )
    study: Study = load_study(
        storage=storage,
        study_name=study_name,
    )
    trials = study.trials
    trials = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    return len(trials)