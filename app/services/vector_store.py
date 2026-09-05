from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain.docstore.document import Document as LCDocument
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings
from app.core.exceptions import ServiceConfigurationError


def build_embeddings(settings=None):
    """Construct the configured embedding backend.

    Embeddings are deliberately independent of the chat provider: the default
    chat provider (Groq) exposes no embeddings endpoint, so reusing its
    base URL and key - as this used to - could never work.
    """
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "fastembed":
        # Runs a small quantised ONNX model in-process. No API key, no network
        # call, and identical behaviour locally and in a container.
        return FastEmbedEmbeddings(model_name=settings.embedding_model)

    if provider == "ollama":
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url or settings.ollama_base_url,
        )

    api_key = settings.embedding_api_key.strip() or settings.llm_api_key.strip()
    if not api_key:
        raise ServiceConfigurationError(
            f"Embedding provider is set to hosted '{settings.embedding_provider}', but no API key is "
            "configured. Set EMBEDDING_API_KEY in .env, or use EMBEDDING_PROVIDER=fastembed to run "
            "embeddings locally with no key."
        )
    return OpenAIEmbeddings(
        api_key=api_key,
        model=settings.embedding_model,
        base_url=settings.embedding_base_url or None,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Build the embedding backend once per process.

    fastembed loads ONNX weights on construction (~2.7s). Services are
    instantiated per request, so without this cache every upload and every
    retrieval paid that cost again.
    """
    return build_embeddings(get_settings())


class VectorStoreService:
    def __init__(self) -> None:
        settings = get_settings()
        self.index_path = Path(settings.faiss_index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = get_embeddings()

    def _load_store(self) -> FAISS | None:
        if not self.index_path.exists():
            return None
        return FAISS.load_local(
            folder_path=str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> None:
        documents = [LCDocument(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas, strict=False)]
        store = self._load_store()
        if store is None:
            store = FAISS.from_documents(documents, self.embeddings)
        else:
            store.add_documents(documents)
        store.save_local(str(self.index_path))

    def delete_document(self, document_id: int) -> int:
        """Remove every chunk belonging to one document. Returns the count.

        Without this, deleting a document would leave its text retrievable -
        and quotable in answers - long after the user removed it.
        """
        store = self._load_store()
        if store is None:
            return 0

        doomed = [
            docstore_id
            for docstore_id in store.index_to_docstore_id.values()
            if str(getattr(store.docstore.search(docstore_id), "metadata", {}).get("document_id", ""))
            == str(document_id)
        ]
        if not doomed:
            return 0

        store.delete(doomed)
        store.save_local(str(self.index_path))
        return len(doomed)

    def search_chunks(
        self,
        query: str,
        k: int = 4,
        owner_id: int | None = None,
        document_id: int | None = None,
    ) -> list[dict[str, Any]]:
        store = self._load_store()
        if store is None:
            return []

        raw_results = store.similarity_search_with_score(query, k=max(k * 3, 8))
        results = [
            {"document": doc, "score": score}
            for doc, score in raw_results
        ]

        if owner_id is not None:
            results = [
                item
                for item in results
                if str(item["document"].metadata.get("owner_id", "")) == str(owner_id)
            ]
        if document_id is not None:
            results = [
                item
                for item in results
                if str(item["document"].metadata.get("document_id", "")) == str(document_id)
            ]

        trimmed = results[:k]
        return [
            {
                "document_id": int(item["document"].metadata.get("document_id", 0) or 0),
                "file_name": item["document"].metadata.get("file_name", "unknown"),
                "content": item["document"].page_content,
                "score": float(item["score"]),
            }
            for item in trimmed
        ]

