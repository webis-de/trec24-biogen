from typing import Literal, TypeAlias

from ir_measures import Measure

from trec_biogen.datasets import GenerationDataset, RetrievalDataset
from trec_biogen.modules import GenerationModule, RetrievalModule


RetrievalMeasure: TypeAlias = Measure


def evaluate_retrieval_module(
    module: RetrievalModule,
    dataset: RetrievalDataset,
    measure: RetrievalMeasure,
) -> float:
    # TODO: Implement some useful measures using `ir_measures`, e.g., P@1, nDCG@1, nDCG@3, nDCG@10, etc.
    return NotImplemented


GenerationMeasure: TypeAlias = Literal["todo"]


def evaluate_generation_module(
    module: GenerationModule,
    dataset: GenerationDataset,
    measure: GenerationMeasure,
) -> float:
    # TODO: Implement some useful measures, e.g.: Perplexity, BLEU, ROUGE, RAGAS (https://docs.ragas.io/en/stable/concepts/metrics/index.html), DeepEval (https://docs.confident-ai.com/docs/metrics-introduction), Tonic (https://docs.tonic.ai/validate/about-rag-metrics/tonic-validate-rag-metrics-summary), DSPy (https://dspy-docs.vercel.app/docs/building-blocks/metrics), etc.
    return NotImplemented