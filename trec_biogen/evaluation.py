from math import inf, isnan
from typing import Literal, Mapping, Sequence, TypeAlias
from warnings import catch_warnings, filterwarnings

from datasets import Dataset
from ir_measures import Measure as IrMeasure, Qrel, ScoredDoc
from numpy import array
from pandas import DataFrame
from ragas import evaluate as ragas_evaluate
from ragas.evaluation import Result
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_utilization,
    context_recall,
    answer_similarity,
    answer_correctness,
    summarization_score,
)
from ragas.metrics.base import Metric as RagasMeasure
from sklearn.metrics import accuracy_score

from trec_biogen.language_models import LanguageModelName, get_langchain_language_model
from trec_biogen.model import GenerationAnswer, RankedPubMedReference, RetrievalAnswer


RetrievalMeasure: TypeAlias = IrMeasure


def _as_qrels(ground_truth: Sequence[RetrievalAnswer]) -> list[Qrel]:
    """
    Build fake qrels where any given ground-truth reference is deemed relevant.
    No judgments are returned for any reference not included in the ground truth.
    Consequentially, using these qrels only makes sense when unjudged documents are deemed irrelevant.
    """
    return [
        Qrel(
            query_id=answer.id,
            doc_id=reference.pubmed_id,
            relevance=1,
        )
        for answer in ground_truth
        for reference in answer.references
    ]


def _rank_or_negative_inf(reference: RankedPubMedReference) -> float:
    if reference.rank is not None:
        return reference.rank
    else:
        return -inf


def _as_run(predictions: Sequence[RetrievalAnswer]) -> list[ScoredDoc]:
    """
    Build a fake run containing the predicted references ordered by rank.
    No judgments are returned for any reference not included in the ground truth.
    Consequentially, using these qrels only makes sense when unjudged documents are deemed irrelevant.
    """
    run = []
    for answer in predictions:
        doc_ids = set()
        for i, reference in enumerate(
            sorted(
                answer.references,
                key=_rank_or_negative_inf,
            )
        ):
            if reference.pubmed_id in doc_ids:
                continue
            run.append(
                ScoredDoc(
                    query_id=answer.id,
                    doc_id=reference.pubmed_id,
                    score=-i,
                )
            )
            doc_ids.add(reference.pubmed_id)
    return run


def evaluate_retrieval(
    predictions: Sequence[RetrievalAnswer],
    ground_truth: Sequence[RetrievalAnswer],
    measure: RetrievalMeasure,
) -> float:
    qrels = _as_qrels(ground_truth)
    run = _as_run(predictions)
    return measure.calc_aggregate(
        qrels=qrels,
        run=run,
    )


GenerationMeasure: TypeAlias = Literal[
    "yes-no-accuracy",
    # TODO (later): Measures for remaining exact answers.
    "faithfulness",
    "answer-relevance",
    "context-precision",
    "context-utilization",
    "context-recall",
    "answer-similarity",
    "answer-correctness",
    "summarization-score",
]


_RAGAS_MEASURES: Mapping[GenerationMeasure, RagasMeasure] = {
    "faithfulness": faithfulness,
    "answer-relevance": answer_relevancy,
    "context-precision": context_precision,
    "context-utilization": context_utilization,
    "context-recall": context_recall,
    "answer-similarity": answer_similarity,
    "answer-correctness": answer_correctness,
    "summarization-score": summarization_score,
}


def _as_ragas_dataset(
    predictions: Sequence[GenerationAnswer],
    ground_truth: Sequence[GenerationAnswer],
) -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": answer_ground_truth.text,
                "answer": " ".join(
                    sentence.sentence for sentence in answer_prediction.summary
                ),
                "contexts": (
                    [
                        reference.snippet.text
                        for reference in answer_ground_truth.references
                        if reference.snippet is not None
                    ]
                    if answer_ground_truth.references is not None
                    else []
                ),
                "groun_truth": " ".join(
                    sentence.sentence for sentence in answer_ground_truth.summary
                ),
            }
            for answer_prediction, answer_ground_truth in zip(predictions, ground_truth)
        ]
    )


def evaluate_generation(
    predictions: Sequence[GenerationAnswer],
    ground_truth: Sequence[GenerationAnswer],
    measure: GenerationMeasure,
    language_model_name: LanguageModelName,
) -> float:
    # TODO: Implement some useful measures, e.g.: Perplexity, BLEU, ROUGE, RAGAS (https://docs.ragas.io/en/stable/concepts/metrics/index.html), DeepEval (https://docs.confident-ai.com/docs/metrics-introduction), Tonic (https://docs.tonic.ai/validate/about-rag-metrics/tonic-validate-rag-metrics-summary), DSPy (https://dspy-docs.vercel.app/docs/building-blocks/metrics), etc.
    if measure == "yes-no-accuracy":
        yes_no_predictions_ground_truth = [
            (answer_prediction.exact, answer_ground_truth.exact)
            for answer_prediction, answer_ground_truth in zip(predictions, ground_truth)
            if answer_ground_truth.type == "yes-no"
        ]
        with catch_warnings():
            filterwarnings(
                action="ignore",
                category=RuntimeWarning,
                message=r".*Mean of empty slice.*",
            )
            filterwarnings(
                action="ignore",
                category=RuntimeWarning,
                message=r".*invalid value encountered in scalar divide.*",
            )
            accuracy = float(
                accuracy_score(
                    y_true=array(
                        [
                            yes_no_ground_truth
                            for _, yes_no_ground_truth in yes_no_predictions_ground_truth
                        ]
                    ),
                    y_pred=array(
                        [
                            yes_no_prediction
                            for yes_no_prediction, _ in yes_no_predictions_ground_truth
                        ]
                    ),
                )
            )
        return accuracy if not isnan(accuracy) else 0
    elif measure in (
        "faithfulness",
        "answer-relevance",
        "context-precision",
        "context-utilization",
        "context-recall",
        "answer-similarity",
        "answer-correctness",
        "summarization-score",
    ):
        ragas_measure = _RAGAS_MEASURES[measure]
        dataset = _as_ragas_dataset(predictions, ground_truth)
        language_model = get_langchain_language_model(language_model_name)
        result: Result = ragas_evaluate(
            dataset=dataset,
            metrics=[ragas_measure],
            llm=language_model,
        )
        result_df: DataFrame = result.to_pandas()  # type: ignore
        return float(result_df[ragas_measure.name].mean())
    else:
        raise ValueError(f"Invalid generation measure: {measure}")
