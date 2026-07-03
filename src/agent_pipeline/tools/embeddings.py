"""On-device embeddings that need no API key.

Implements the LangChain ``Embeddings`` seam so the pipeline can swap in a
provider-backed embedder (Voyage, OpenAI, ...) by config without touching
callers.
"""
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()
