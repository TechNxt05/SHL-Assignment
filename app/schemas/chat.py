"""
SHL Assessment Recommender - Chat API Schemas
Strict Pydantic models for the /chat endpoint. Schema MUST NEVER break.
"""
from typing import List, Literal
from pydantic import BaseModel, field_validator


class Message(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class ChatRequest(BaseModel):
    """Request body for POST /chat. Contains the full stateless conversation history."""

    messages: List[Message]

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages(cls, v: list) -> list:
        if not v:
            raise ValueError("messages list cannot be empty")
        if len(v) > 16:
            raise ValueError("messages list exceeds maximum length of 16")
        return v


class Recommendation(BaseModel):
    """A single SHL assessment recommendation. All fields come directly from catalog."""

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.
    CRITICAL: This schema must NEVER break — even on errors, refusals, or timeouts.
    - recommendations is [] when clarifying or refusing
    - recommendations has 1-10 items when committing to a shortlist
    - end_of_conversation is True only when the agent considers the task complete
    """

    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

    @field_validator("recommendations", mode="before")
    @classmethod
    def validate_rec_count(cls, v: list) -> list:
        if v and not (1 <= len(v) <= 10):
            raise ValueError(
                f"recommendations must be empty or 1-10 items, got {len(v)}"
            )
        return v
