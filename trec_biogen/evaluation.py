from typing import Literal, Sequence, TypeAlias

from ir_measures import Measure

from trec_biogen.model import GenerationAnswer, RetrievalAnswer


RetrievalMeasure: TypeAlias = Measure


def evaluate_retrieval(
    predictions: Sequence[RetrievalAnswer],
    ground_truth: Sequence[RetrievalAnswer],
    measure: RetrievalMeasure,
) -> float:
    # TODO: Implement some useful measures using `ir_measures`, e.g., P@1, nDCG@1, nDCG@3, nDCG@10, etc.
    return NotImplemented


GenerationMeasure: TypeAlias = Literal["todo"]


def evaluate_generation(
    predictions: Sequence[GenerationAnswer],
    ground_truth: Sequence[GenerationAnswer],
    measure: GenerationMeasure,
) -> float:
    # TODO: Implement some useful measures, e.g.: Perplexity, BLEU, ROUGE, RAGAS (https://docs.ragas.io/en/stable/concepts/metrics/index.html), DeepEval (https://docs.confident-ai.com/docs/metrics-introduction), Tonic (https://docs.tonic.ai/validate/about-rag-metrics/tonic-validate-rag-metrics-summary), DSPy (https://dspy-docs.vercel.app/docs/building-blocks/metrics), etc.
    return NotImplemented
