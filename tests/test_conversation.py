"""Conversation memory.

Before this existed every chat request was independent, so a follow-up like
"summarize that" had no referent. History is stored server-side and scoped to
the caller, which also means a client cannot fabricate turns it never had.
"""
from app.services.conversation_service import ConversationService, new_thread_id


def test_thread_ids_are_unique():
    assert new_thread_id() != new_thread_id()


class TestPersistence:
    def _user(self):
        from sqlalchemy import select

        from app.db.session import SessionLocal
        from app.models.user import User

        with SessionLocal() as db:
            return db.scalar(select(User))

    def test_messages_round_trip_in_order(self, client, auth_headers):
        from app.db.session import SessionLocal

        service = ConversationService()
        thread = new_thread_id()
        user = self._user()

        with SessionLocal() as db:
            service.append(db, user, thread, "user", "what is the SLA?")
            service.append(db, user, thread, "assistant", "Tier 1 is 15 minutes.")
            service.append(db, user, thread, "user", "who owns it?")
            history = service.recent_messages(db, user, thread)

        assert [m["role"] for m in history] == ["user", "assistant", "user"]
        assert history[0]["content"] == "what is the SLA?"

    def test_history_is_bounded_to_recent_turns(self, client, auth_headers):
        from app.db.session import SessionLocal

        service = ConversationService()
        thread = new_thread_id()
        user = self._user()

        with SessionLocal() as db:
            for i in range(20):
                service.append(db, user, thread, "user", f"question {i}")
                service.append(db, user, thread, "assistant", f"answer {i}")
            history = service.recent_messages(db, user, thread, turns=3)

        assert len(history) == 6
        assert history[-1]["content"] == "answer 19", "must keep the newest, not the oldest"

    def test_threads_are_isolated_from_each_other(self, client, auth_headers):
        from app.db.session import SessionLocal

        service = ConversationService()
        a, b = new_thread_id(), new_thread_id()
        user = self._user()

        with SessionLocal() as db:
            service.append(db, user, a, "user", "in thread A")
            service.append(db, user, b, "user", "in thread B")
            assert len(service.recent_messages(db, user, a)) == 1
            assert service.recent_messages(db, user, a)[0]["content"] == "in thread A"

    def test_a_users_thread_is_invisible_to_another_user(self, client, auth_headers):
        """Thread ids are guessable strings; ownership must be enforced too."""
        from sqlalchemy import select

        from app.db.session import SessionLocal
        from app.models.user import User

        service = ConversationService()
        thread = new_thread_id()
        owner = self._user()

        client.post(
            "/api/v1/auth/register",
            json={"email": "other@example.com", "full_name": "Other", "password": "StrongPass123"},
        )
        with SessionLocal() as db:
            other = db.scalar(select(User).where(User.email == "other@example.com"))
            service.append(db, owner, thread, "user", "confidential")
            assert service.recent_messages(db, other, thread) == []

    def test_clear_removes_only_that_thread(self, client, auth_headers):
        from app.db.session import SessionLocal

        service = ConversationService()
        a, b = new_thread_id(), new_thread_id()
        user = self._user()

        with SessionLocal() as db:
            service.append(db, user, a, "user", "one")
            service.append(db, user, b, "user", "two")
            removed = service.clear(db, user, a)
            assert removed == 1
            assert service.recent_messages(db, user, a) == []
            assert len(service.recent_messages(db, user, b)) == 1

    def test_very_long_content_is_truncated(self, client, auth_headers):
        from app.db.session import SessionLocal
        from app.services.conversation_service import MAX_STORED_CHARS

        service = ConversationService()
        thread = new_thread_id()
        user = self._user()

        with SessionLocal() as db:
            stored = service.append(db, user, thread, "assistant", "x" * (MAX_STORED_CHARS + 5000))
            assert len(stored.content) == MAX_STORED_CHARS


class TestAgentUsesHistory:
    def test_query_is_prefixed_with_prior_turns(self):
        from app.services.langgraph_agent import AgentState, LangGraphAgentService

        state = AgentState(
            query="who owns it?",
            history=[
                {"role": "user", "content": "what is the payroll SLA?"},
                {"role": "assistant", "content": "Tier 1 acknowledges in 15 minutes."},
            ],
        )
        prompt = LangGraphAgentService._query_with_history(state)
        assert "payroll SLA" in prompt
        assert "15 minutes" in prompt
        assert "who owns it?" in prompt

    def test_no_history_leaves_the_query_unchanged(self):
        from app.services.langgraph_agent import AgentState, LangGraphAgentService

        state = AgentState(query="hello")
        assert LangGraphAgentService._query_with_history(state) == "User query:\nhello"
