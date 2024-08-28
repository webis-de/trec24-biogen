from dataclasses import dataclass, field

from pandas import DataFrame
from pyterrier.transformer import Transformer


@dataclass(frozen=True)
class CachableTransformer(Transformer):
    wrapped: Transformer = field(repr=False)
    key: str

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self.wrapped.transform(topics_or_res)


@dataclass(frozen=True)
class CutoffRerank(Transformer):
    candidates: Transformer
    reranker: Transformer
    cutoff: int

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res = self.candidates.transform(topics_or_res)
        pipeline = Transformer.from_df(
            input=topics_or_res,
            uniform=True,
        )
        pipeline = ((pipeline % self.cutoff) >> self.reranker) ^ pipeline
        return pipeline.transform(topics_or_res)
