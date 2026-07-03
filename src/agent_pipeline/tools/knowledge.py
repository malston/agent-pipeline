"""Semantic memory: the retrieval corpus A1 draws evidence from.

Wraps a LangChain vector store behind a domain method (``search``) that returns
typed ``Passage`` objects. The in-memory store is the dev/test default; the same
seam accepts a pgvector/Qdrant-backed store in production.
"""
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from agent_pipeline.contracts.retrieval import Passage


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _metadata_filter(
    filters: dict[str, object] | None,
) -> Callable[[Document], bool] | None:
    if not filters:
        return None
    return lambda doc: all(doc.metadata.get(k) == v for k, v in filters.items())


class KnowledgeStore:
    def __init__(self, embeddings: Embeddings) -> None:
        self._store = InMemoryVectorStore(embeddings)

    def index(self, documents: list[Document]) -> None:
        self._store.add_documents(documents)

    def search(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, object] | None = None,
    ) -> list[Passage]:
        results = self._store.similarity_search_with_score(
            query, k=k, filter=_metadata_filter(filters)
        )
        return [
            Passage(
                source_id=doc.id,
                text=doc.page_content,
                score=_clamp_unit(score),
            )
            for doc, score in results
        ]
