"""Rebuild the FAISS index from documents stored in the database.

Needed whenever:
  * the embedding model or provider changes (vectors of different dimensions
    cannot share an index),
  * the index file is lost, which happens on every restart on hosts with an
    ephemeral filesystem.

Document text lives in the `documents` table, so the index is always
reproducible from the database and never needs to be backed up.

Usage:
    python -m scripts.reindex          # rebuild for every user
    python -m scripts.reindex --user 3 # rebuild one user's documents
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.document import Document
from app.services.vector_store import VectorStoreService
from app.utils.text import chunk_text


def reindex(user_id: int | None = None, drop_existing: bool = True) -> tuple[int, int]:
    """Re-embed stored documents. Returns (documents, chunks)."""
    settings = get_settings()
    index_path = Path(settings.faiss_index_path)

    if drop_existing and index_path.exists():
        shutil.rmtree(index_path)
        print(f"Removed existing index at {index_path}")

    store = VectorStoreService()

    with SessionLocal() as db:
        query = select(Document).order_by(Document.id)
        if user_id is not None:
            query = query.where(Document.owner_id == user_id)
        documents = list(db.scalars(query))

    if not documents:
        print("No documents found; nothing to index.")
        return (0, 0)

    total_chunks = 0
    for document in documents:
        chunks = chunk_text(document.content)
        if not chunks:
            continue
        store.add_documents(
            chunks,
            [
                {
                    "document_id": document.id,
                    "file_name": document.file_name,
                    "owner_id": document.owner_id,
                }
                for _ in chunks
            ],
        )
        total_chunks += len(chunks)
        print(f"  indexed {document.file_name} ({len(chunks)} chunks)")

    print(
        f"\nDone: {len(documents)} document(s), {total_chunks} chunk(s) "
        f"using {settings.embedding_provider}/{settings.embedding_model}."
    )
    return (len(documents), total_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", type=int, default=None, help="Only reindex this owner's documents")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to the current index instead of rebuilding it",
    )
    args = parser.parse_args()
    reindex(user_id=args.user, drop_existing=not args.keep_existing)


if __name__ == "__main__":
    main()
