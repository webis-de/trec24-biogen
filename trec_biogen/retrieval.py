from dataclasses import dataclass
from typing import Any, Sequence

from pandas import DataFrame
from pyterrier.transformer import Transformer

from trec_biogen.model import PartialAnswer, RankedPubMedReference, RetrievalAnswer, Snippet
from trec_biogen.modules import GenerationModule, RetrievalModule


@dataclass(frozen=True)
class PyterrierRetrievalModule(RetrievalModule, Transformer):
    """
    Retrieve relevant context using the PyTerrier framework.
    """

    todo: Any  # TODO: Define hyper-parameters.

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return NotImplemented

    def retrieve(self, context: PartialAnswer) -> RetrievalAnswer:
        answers = self.retrieve_many([context])
        return answers[0]
    
    @staticmethod
    def _build_query(context: PartialAnswer) -> str:
        # TODO: Build query.
        return context.text
    
    @staticmethod
    def _parse_res(
        context: PartialAnswer, 
        res: DataFrame | None,
    ) -> RetrievalAnswer:
        references: Sequence[RankedPubMedReference]
        if res is None:
            references = []
        else:
            references = [
                RankedPubMedReference(
                    pubmed_id=row["docno"],
                    snippet=Snippet(
                        text=row["text"],
                        start_section=row["start_section"],
                        start_offset=row["start_offset"],
                        end_section=row["end_section"],
                        end_offset=row["end_offset"],
                    ),
                    rank=row["rank"],
                )
                for _, row in res.iterrows()
            ]

        summary = context.summary
        # TODO: Filter summary for missing references.

        return RetrievalAnswer(
            id=context.id,
            text=context.text,
            type=context.type,
            query=context.query,
            narrative=context.narrative,
            summary=summary,
            exact=context.exact,
            references=references,
        )

    def retrieve_many(self, contexts: Sequence[PartialAnswer]) -> Sequence[RetrievalAnswer]:
        res: DataFrame = DataFrame([
            {
                "qid": qid,
                "query": self._build_query(context)
            }
            for qid, context in enumerate(contexts)
        ], columns=["qid", "query"])
        res = self.transform(res)
        res.sort_values(
            ["qid", "rank"],
              ascending=[True,True], 
              inplace=True,
        )
        qid_res = {
            int(qid): group # type: ignore
            for qid, group in res.groupby(
                "qid",
                  sort=False,
                    group_keys=False,
            )
        }
        return [
            self._parse_res(
                context=context,
                res=qid_res[qid]
            )
            for qid, context in enumerate(contexts)
        ]



@dataclass(frozen=True)
class GenerationThenRetrievalModule(RetrievalModule):
    """
    Retrieve relevant context based on the generated answer from some generation module, known as Generation-augmented Retrieval.
    """

    generation_module: GenerationModule
    retrieval_module: RetrievalModule

    def retrieve(self, context: PartialAnswer) -> RetrievalAnswer:
        context = self.generation_module.generate(context)
        return self.retrieval_module.retrieve(context)

    def retrieve_many(
        self, contexts: Sequence[PartialAnswer]
    ) -> Sequence[RetrievalAnswer]:
        contexts = self.generation_module.generate_many(contexts)
        return self.retrieval_module.retrieve_many(contexts)
