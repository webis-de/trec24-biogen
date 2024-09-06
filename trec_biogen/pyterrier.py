from dataclasses import dataclass, field

from pandas import DataFrame
from pyterrier import Transformer



@dataclass(frozen=True)
class CutoffRerank(Transformer):
    candidates: Transformer
    reranker: Transformer
    cutoff: int

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res = self.candidates.transform(topics_or_res)
        if len(topics_or_res) == 0:
            return topics_or_res
        pipeline = Transformer.from_df(
            input=topics_or_res,
            uniform=True,
        )
        pipeline = ((pipeline % self.cutoff) >> self.reranker) ^ pipeline
        return pipeline.transform(topics_or_res)
