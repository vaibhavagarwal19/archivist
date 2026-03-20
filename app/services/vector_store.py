from pathlib import Path
from typing import Any

from langchain.docstore.document import Document as LCDocument
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.core.exceptions import ServiceConfigurationError
from app.core.config import get_settings


class VectorStoreService:
    def __init__(self) -> None:
        settings = get_settings()
        self.index_path = Path(settings.faiss_index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if settings.embedding_provider.lower() == "ollama":
            self.embeddings = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.embedding_base_url,
            )
        else:
            if not settings.llm_api_key.strip():
                raise ServiceConfigurationError(
                    f"Embedding provider is set to hosted {settings.embedding_provider}, but no API key is configured. "
                    "Set the provider API key in .env, for example LLM_API_KEY."
                )
            self.embeddings = OpenAIEmbeddings(
                api_key=settings.llm_api_key,
                model=settings.embedding_model,
                base_url=settings.llm_base_url,
            )

    def _load_store(self) -> FAISS | None:
        if not self.index_path.exists():
            return None
        return FAISS.load_local(
            folder_path=str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> None:
        documents = [LCDocument(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
        store = self._load_store()
        if store is None:
            store = FAISS.from_documents(documents, self.embeddings)
        else:
            store.add_documents(documents)
        store.save_local(str(self.index_path))

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

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        owner_id: int | None = None,
        document_id: int | None = None,
    ) -> list[str]:
        trimmed = self.search_chunks(query, k=k, owner_id=owner_id, document_id=document_id)
        return [
            (
                f"Document: {item['file_name']}\n"
                f"Content: {item['content']}"
            )
            for item in trimmed
        ]
