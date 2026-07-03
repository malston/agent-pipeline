"""Shared fixtures.

These use REAL local embeddings (fastembed) over an in-memory vector store --
no mocks. The embedding model downloads once on first run, then is cached.
"""
import pytest

from agent_pipeline.contracts.retrieval import RetrievalRequest


@pytest.fixture(scope="session")
def local_embeddings():
    from agent_pipeline.tools.embeddings import LocalEmbeddings

    return LocalEmbeddings()


@pytest.fixture(scope="session")
def knowledge(local_embeddings):
    """A KnowledgeStore seeded with a small, topically-distinct corpus."""
    from langchain_core.documents import Document

    from agent_pipeline.tools.knowledge import KnowledgeStore

    store = KnowledgeStore(local_embeddings)
    store.index(
        [
            Document(
                id="mito",
                page_content=(
                    "The mitochondrion is the powerhouse of the cell, "
                    "producing ATP that fuels cellular activity."
                ),
                metadata={"topic": "biology"},
            ),
            Document(
                id="photo",
                page_content=(
                    "Photosynthesis converts sunlight into chemical energy "
                    "stored as sugars inside plant chloroplasts."
                ),
                metadata={"topic": "biology"},
            ),
            Document(
                id="econ",
                page_content=(
                    "Central banks adjust interest rates to influence "
                    "inflation and the pace of economic growth."
                ),
                metadata={"topic": "economics"},
            ),
        ]
    )
    return store


@pytest.fixture
def sample_request():
    return RetrievalRequest(request_id="r1", raw_query="how do cells produce energy?")
