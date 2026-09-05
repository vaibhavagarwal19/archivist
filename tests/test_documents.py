"""Document upload, chunking, and per-user isolation tests.

The vector store is faked so the suite never loads the embedding model.
"""
import io

import pytest
from fastapi.testclient import TestClient
from langchain.docstore.document import Document as LCDocument

from app.services.vector_store import VectorStoreService
from app.utils.text import chunk_text


@pytest.fixture(autouse=True)
def fake_vector_store(monkeypatch):
    """Record indexed chunks instead of embedding them."""
    indexed: list[dict] = []

    def fake_add_documents(self, texts, metadatas):
        indexed.extend({"text": t, "metadata": m} for t, m in zip(texts, metadatas, strict=False))

    monkeypatch.setattr(VectorStoreService, "__init__", lambda self: None)
    monkeypatch.setattr(VectorStoreService, "add_documents", fake_add_documents)
    return indexed


def upload(client: TestClient, headers: dict, name: str, body: str):
    return client.post(
        "/api/v1/upload-doc",
        headers=headers,
        files={"file": (name, io.BytesIO(body.encode()), "text/plain")},
    )


def test_upload_stores_and_indexes_document(client, auth_headers, fake_vector_store):
    response = upload(client, auth_headers, "handbook.txt", "Leave is requested via the HR portal.")
    assert response.status_code == 201
    assert response.json()["document"]["file_name"] == "handbook.txt"
    assert len(fake_vector_store) == 1
    assert fake_vector_store[0]["metadata"]["file_name"] == "handbook.txt"


def test_indexed_chunks_carry_owner_metadata(client, auth_headers, fake_vector_store):
    """Owner id must reach the index, or retrieval cannot scope by user."""
    upload(client, auth_headers, "notes.txt", "Quarterly planning notes.")
    assert fake_vector_store[0]["metadata"]["owner_id"] is not None


def test_empty_upload_is_rejected(client, auth_headers):
    response = upload(client, auth_headers, "empty.txt", "")
    assert response.status_code == 400


def test_whitespace_only_upload_is_rejected(client, auth_headers):
    response = upload(client, auth_headers, "blank.txt", "   \n\t  ")
    assert response.status_code == 400


def test_upload_requires_authentication(client):
    response = client.post(
        "/api/v1/upload-doc",
        files={"file": ("x.txt", io.BytesIO(b"data"), "text/plain")},
    )
    assert response.status_code == 401


def test_users_cannot_see_each_others_documents(client, auth_headers, fake_vector_store):
    """The core multi-tenancy guarantee."""
    upload(client, auth_headers, "private.txt", "Confidential salary bands.")

    client.post(
        "/api/v1/auth/register",
        json={"email": "other@example.com", "full_name": "Other", "password": "StrongPass123"},
    )
    other_token = client.post(
        "/api/v1/auth/login",
        json={"email": "other@example.com", "password": "StrongPass123"},
    ).json()["access_token"]
    other_headers = {"Authorization": f"Bearer {other_token}"}

    mine = client.get("/api/v1/upload-doc", headers=auth_headers).json()["documents"]
    theirs = client.get("/api/v1/upload-doc", headers=other_headers).json()["documents"]

    assert [d["file_name"] for d in mine] == ["private.txt"]
    assert theirs == []


def test_uploaded_filenames_are_sanitised(client, auth_headers, fake_vector_store):
    """A traversal-style name must not escape the upload directory."""
    from app.services.document_service import DocumentService

    safe = DocumentService._safe_file_name("../../etc/passwd")
    assert "/" not in safe and ".." not in safe


class TestChunking:
    def test_short_text_is_a_single_chunk(self):
        assert chunk_text("short text") == ["short text"]

    def test_long_text_splits_into_overlapping_chunks(self):
        text = "x" * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=150)
        assert len(chunks) > 1
        assert all(len(c) <= 1000 for c in chunks)

    def test_chunks_cover_the_whole_document(self):
        text = "".join(str(i % 10) for i in range(3000))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert chunks[0][0] == text[0]
        assert chunks[-1][-1] == text[-1]

    def test_empty_text_produces_no_chunks(self):
        assert chunk_text("") == []


class TestRetrievalScoping:
    """search_chunks must filter results by owner before returning them."""

    def _service_with_results(self, monkeypatch, docs):
        monkeypatch.setattr(VectorStoreService, "__init__", lambda self: None)
        service = VectorStoreService()

        class FakeStore:
            def similarity_search_with_score(self, query, k):
                return [(d, 0.1) for d in docs]

        monkeypatch.setattr(VectorStoreService, "_load_store", lambda self: FakeStore())
        return service

    def test_results_are_filtered_to_the_requesting_owner(self, monkeypatch):
        docs = [
            LCDocument(page_content="mine", metadata={"owner_id": 1, "document_id": 10, "file_name": "a.txt"}),
            LCDocument(page_content="theirs", metadata={"owner_id": 2, "document_id": 20, "file_name": "b.txt"}),
        ]
        service = self._service_with_results(monkeypatch, docs)
        results = service.search_chunks("anything", k=4, owner_id=1)
        assert [r["content"] for r in results] == ["mine"]

    def test_results_can_be_narrowed_to_one_document(self, monkeypatch):
        docs = [
            LCDocument(page_content="first", metadata={"owner_id": 1, "document_id": 10, "file_name": "a.txt"}),
            LCDocument(page_content="second", metadata={"owner_id": 1, "document_id": 11, "file_name": "b.txt"}),
        ]
        service = self._service_with_results(monkeypatch, docs)
        results = service.search_chunks("anything", k=4, owner_id=1, document_id=11)
        assert [r["content"] for r in results] == ["second"]

    def test_missing_index_returns_no_results(self, monkeypatch):
        monkeypatch.setattr(VectorStoreService, "__init__", lambda self: None)
        monkeypatch.setattr(VectorStoreService, "_load_store", lambda self: None)
        assert VectorStoreService().search_chunks("anything", owner_id=1) == []


class TestDeletion:
    def test_owner_can_delete_their_document(self, client, auth_headers, fake_vector_store):
        doc_id = upload(client, auth_headers, "temp.txt", "Disposable notes.").json()["document"]["id"]

        response = client.delete(f"/api/v1/upload-doc/{doc_id}", headers=auth_headers)
        assert response.status_code == 204
        assert client.get("/api/v1/upload-doc", headers=auth_headers).json()["documents"] == []

    def test_deleting_another_users_document_is_not_found(self, client, auth_headers, fake_vector_store):
        """404, not 403: the response must not confirm the document exists."""
        doc_id = upload(client, auth_headers, "private.txt", "Secret.").json()["document"]["id"]

        client.post(
            "/api/v1/auth/register",
            json={"email": "intruder@example.com", "full_name": "Intruder", "password": "StrongPass123"},
        )
        token = client.post(
            "/api/v1/auth/login",
            json={"email": "intruder@example.com", "password": "StrongPass123"},
        ).json()["access_token"]

        response = client.delete(
            f"/api/v1/upload-doc/{doc_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404
        # The owner still has it.
        assert len(client.get("/api/v1/upload-doc", headers=auth_headers).json()["documents"]) == 1

    def test_deleting_a_missing_document_is_not_found(self, client, auth_headers):
        assert client.delete("/api/v1/upload-doc/99999", headers=auth_headers).status_code == 404

    def test_delete_requires_authentication(self, client):
        assert client.delete("/api/v1/upload-doc/1").status_code == 401

    def test_delete_removes_vectors_for_that_document_only(self, monkeypatch, tmp_path):
        """Chunks from other documents must survive the delete."""
        from langchain.docstore.document import Document as LCDocument

        from app.services.vector_store import VectorStoreService

        monkeypatch.setattr(VectorStoreService, "__init__", lambda self: None)
        service = VectorStoreService()
        service.index_path = tmp_path / "index"

        saved = {}

        class FakeDocstore:
            def __init__(self, docs):
                self.docs = docs

            def search(self, key):
                return self.docs[key]

        class FakeStore:
            def __init__(self):
                self.index_to_docstore_id = {0: "a", 1: "b", 2: "c"}
                self.docstore = FakeDocstore(
                    {
                        "a": LCDocument(page_content="1", metadata={"document_id": 7}),
                        "b": LCDocument(page_content="2", metadata={"document_id": 8}),
                        "c": LCDocument(page_content="3", metadata={"document_id": 7}),
                    }
                )

            def delete(self, ids):
                saved["deleted"] = sorted(ids)

            def save_local(self, path):
                saved["saved"] = True

        monkeypatch.setattr(VectorStoreService, "_load_store", lambda self: FakeStore())

        removed = service.delete_document(7)
        assert removed == 2
        assert saved["deleted"] == ["a", "c"]
        assert saved["saved"] is True

    def test_delete_without_an_index_is_a_noop(self, monkeypatch):
        from app.services.vector_store import VectorStoreService

        monkeypatch.setattr(VectorStoreService, "__init__", lambda self: None)
        monkeypatch.setattr(VectorStoreService, "_load_store", lambda self: None)
        assert VectorStoreService().delete_document(1) == 0
