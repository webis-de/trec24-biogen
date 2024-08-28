from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

from pandas import DataFrame, Series
from pyterrier.transformer import Transformer
from tqdm.auto import tqdm

from trec_biogen.model import (
    PartialAnswer,
    RankedPubMedReference,
    RetrievalAnswer,
    Snippet,
    PubMedReferenceSentence,
)
from trec_biogen.modules import GenerationModule, RetrievalModule


def _remove_dangling_references(
    sentence: PubMedReferenceSentence,
    references: Sequence[RankedPubMedReference],
) -> PubMedReferenceSentence:
    sentence_references = [
        sentence_reference
        for sentence_reference in sentence.references
        if any(sentence_reference in reference for reference in references)
    ]
    return PubMedReferenceSentence(
        sentence=sentence.sentence,
        references=sentence_references,
    )


@dataclass(frozen=True)
class PyterrierRetrievalModule(RetrievalModule, Transformer):
    """
    Retrieve relevant context using the PyTerrier framework.
    """

    transformer: Transformer
    progress: bool = False

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self.transformer.transform(topics_or_res)

    def retrieve(self, context: PartialAnswer) -> RetrievalAnswer:
        answers = self.retrieve_many([context])
        return answers[0]

    def _parse_res(
        self,
        context: PartialAnswer,
        res: DataFrame | None,
    ) -> RetrievalAnswer:
        references: Sequence[RankedPubMedReference]
        if res is None:
            references = []
        else:
            references = [
                RankedPubMedReference(
                    pubmed_id=row["docno"].split("%")[0],
                    snippet=(
                        Snippet(
                            text=row["text"],
                            start_section=row["start_section"],
                            start_offset=row["start_offset"],
                            end_section=row["end_section"],
                            end_offset=row["end_offset"],
                        )
                        if row["text"]
                        and row["start_section"]
                        and row["start_offset"]
                        and row["end_section"]
                        and row["end_offset"]
                        else None
                    ),
                    rank=row["rank"],
                )
                for _, row in res.iterrows()
            ]

        # Remove dangling references from the summary.
        summary = context.summary
        if summary is not None:
            summary = [
                _remove_dangling_references(sentence, references)
                for sentence in summary
            ]

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

    def retrieve_many(
        self, contexts: Sequence[PartialAnswer]
    ) -> Sequence[RetrievalAnswer]:
        res: DataFrame = DataFrame(
            [
                {
                    "qid": qid,
                    "context": context,
                }
                for qid, context in enumerate(contexts)
            ],
            columns=["qid", "context"],
        )
        res = self.transform(res)
        res.sort_values(
            ["qid", "rank"],
            ascending=[True, True],
            inplace=True,
        )
        qid_res = {
            int(qid): group  # type: ignore
            for qid, group in res.groupby(
                "qid",
                sort=False,
                group_keys=False,
            )
        }
        contexts_iterable: Iterable[PartialAnswer] = contexts
        if self.progress:
            contexts_iterable = tqdm(
                contexts_iterable,
                desc="Parse retrieved results",
                unit="question",
            )
        return [
            self._parse_res(context=context, res=qid_res.get(qid))
            for qid, context in enumerate(contexts_iterable)
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
