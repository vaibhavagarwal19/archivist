from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ConversationMessage(Base):
    """One turn in a conversation.

    History is stored server-side and keyed by thread so a follow-up like
    "summarize that" can resolve against what was actually said, and so a
    conversation survives a restart or a page reload.
    """

    __tablename__ = "conversation_messages"
    __table_args__ = (
        # Every read is "the last N messages of one thread for one user".
        Index("ix_conversation_owner_thread", "owner_id", "thread_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )

    owner = relationship("User")
