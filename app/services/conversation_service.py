"""Conversation history.

Kept deliberately small: the agent needs the last few turns to resolve
references like "summarize that", not an unbounded transcript. Sending the
whole history would grow cost linearly with conversation length for no gain.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.conversation import ConversationMessage
from app.models.user import User

# Turns of context handed to the model. Each turn is one user + one assistant
# message, so this is 6 messages.
HISTORY_TURNS = 3
MAX_STORED_CHARS = 8000


def new_thread_id() -> str:
    return uuid4().hex


class ConversationService:
    def recent_messages(
        self, db: Session, owner: User, thread_id: str, turns: int = HISTORY_TURNS
    ) -> list[dict[str, str]]:
        """Return the last `turns` exchanges, oldest first."""
        rows = db.scalars(
            select(ConversationMessage)
            .where(
                ConversationMessage.owner_id == owner.id,
                ConversationMessage.thread_id == thread_id,
            )
            .order_by(ConversationMessage.id.desc())
            .limit(turns * 2)
        ).all()
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    def append(
        self, db: Session, owner: User, thread_id: str, role: str, content: str
    ) -> ConversationMessage:
        message = ConversationMessage(
            thread_id=thread_id,
            role=role,
            # A pathological answer should not bloat the row or later prompts.
            content=content[:MAX_STORED_CHARS],
            owner_id=owner.id,
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    def clear(self, db: Session, owner: User, thread_id: str) -> int:
        rows = db.scalars(
            select(ConversationMessage).where(
                ConversationMessage.owner_id == owner.id,
                ConversationMessage.thread_id == thread_id,
            )
        ).all()
        for row in rows:
            db.delete(row)
        db.commit()
        return len(rows)
