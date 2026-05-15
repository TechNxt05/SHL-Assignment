"""
Unit Tests - Conversation State Machine
Tests state determination logic without requiring LLM or catalog.
"""
import pytest

from app.schemas.chat import Message
from app.services.conversation_state import (
    ConversationState,
    determine_state,
    get_missing_slots,
)


def make_messages(count: int) -> list:
    """Generate a fake conversation with `count` messages."""
    msgs = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(Message(role=role, content=f"Message {i}"))
    return msgs


class TestStateDetermination:
    def test_guard_blocked_takes_priority(self):
        intent = {"completeness_score": 0.9, "is_comparison_request": False}
        state = determine_state(intent, make_messages(2), guardrail_blocked=True)
        assert state == ConversationState.GUARD_BLOCKED

    def test_comparison_detected(self):
        intent = {
            "completeness_score": 0.2,
            "is_comparison_request": True,
            "comparison_targets": ["OPQ", "GSA"],
        }
        state = determine_state(intent, make_messages(2), guardrail_blocked=False)
        assert state == ConversationState.COMPARE

    def test_clarify_when_low_completeness(self):
        intent = {
            "completeness_score": 0.35,
            "is_comparison_request": False,
            "is_refinement": False,
        }
        state = determine_state(intent, make_messages(2), guardrail_blocked=False)
        assert state == ConversationState.CLARIFY

    def test_recommend_when_high_completeness(self):
        intent = {
            "completeness_score": 0.80,
            "is_comparison_request": False,
            "is_refinement": False,
        }
        state = determine_state(intent, make_messages(2), guardrail_blocked=False)
        assert state == ConversationState.RECOMMEND

    def test_force_recommend_near_turn_limit(self):
        """At turn 7, must force RECOMMEND even with low completeness."""
        intent = {
            "completeness_score": 0.20,
            "is_comparison_request": False,
            "is_refinement": False,
        }
        # 7 messages = nearing 8-turn limit
        state = determine_state(intent, make_messages(7), guardrail_blocked=False)
        assert state == ConversationState.RECOMMEND

    def test_refine_with_prior_recommendations(self):
        intent = {
            "completeness_score": 0.80,
            "is_comparison_request": False,
            "is_refinement": True,
        }
        prior_recs = [{"name": "Test A", "url": "https://www.shl.com/a", "test_type": "K"}]
        state = determine_state(
            intent, make_messages(4),
            guardrail_blocked=False,
            prior_recommendations=prior_recs
        )
        assert state == ConversationState.REFINE


class TestMissingSlots:
    def test_role_missing(self):
        intent = {"role": "", "seniority": "senior", "technical_skills": ["Java"]}
        missing = get_missing_slots(intent)
        assert "role" in missing

    def test_seniority_missing_when_role_present(self):
        intent = {"role": "Developer", "seniority": "", "technical_skills": []}
        missing = get_missing_slots(intent)
        assert "seniority" in missing

    def test_nothing_missing_with_full_intent(self):
        intent = {
            "role": "Java Developer",
            "seniority": "mid",
            "technical_skills": ["Java"],
            "soft_skills": [],
        }
        missing = get_missing_slots(intent)
        assert missing == []
