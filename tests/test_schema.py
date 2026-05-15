"""
Unit Tests - Schema Validation
Tests that Pydantic models enforce the non-negotiable schema.
"""
import pytest
from pydantic import ValidationError

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    Recommendation,
)


class TestMessageSchema:
    def test_valid_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_assistant_message(self):
        msg = Message(role="assistant", content="How can I help?")
        assert msg.role == "assistant"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="system", content="test")

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="")

    def test_whitespace_only_content_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="   ")


class TestChatRequestSchema:
    def test_valid_request(self):
        req = ChatRequest(messages=[Message(role="user", content="Hello")])
        assert len(req.messages) == 1

    def test_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_multi_message_valid(self):
        req = ChatRequest(messages=[
            Message(role="user", content="Hiring a developer"),
            Message(role="assistant", content="What level?"),
            Message(role="user", content="Senior"),
        ])
        assert len(req.messages) == 3


class TestChatResponseSchema:
    def test_valid_empty_recommendations(self):
        resp = ChatResponse(
            reply="What role?",
            recommendations=[],
            end_of_conversation=False,
        )
        assert resp.recommendations == []

    def test_valid_with_recommendations(self):
        recs = [
            Recommendation(name="Java Test", url="https://www.shl.com/test", test_type="K")
        ]
        resp = ChatResponse(
            reply="Here are results",
            recommendations=recs,
            end_of_conversation=True,
        )
        assert len(resp.recommendations) == 1

    def test_more_than_10_recommendations_rejected(self):
        recs = [
            Recommendation(
                name=f"Test {i}",
                url=f"https://www.shl.com/test{i}",
                test_type="K"
            )
            for i in range(11)
        ]
        with pytest.raises(ValidationError):
            ChatResponse(
                reply="Too many",
                recommendations=recs,
                end_of_conversation=False,
            )

    def test_end_of_conversation_must_be_bool(self):
        resp = ChatResponse(
            reply="Done",
            recommendations=[],
            end_of_conversation=True,
        )
        assert isinstance(resp.end_of_conversation, bool)

    def test_schema_always_has_three_fields(self):
        resp = ChatResponse(
            reply="test",
            recommendations=[],
            end_of_conversation=False,
        )
        d = resp.model_dump()
        assert "reply" in d
        assert "recommendations" in d
        assert "end_of_conversation" in d
