"""
SHL Assessment Recommender - FastAPI Routes
Implements GET /health and POST /chat endpoints.
"""
import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services import state_machine as sm
from app.services import intent_parser
from app.services import response_generator as rg
from app.services.guardrails import check_input
from app.services.constraint_merger import merge_constraints, extract_prior_recommendations
from app.config import TOTAL_REQUEST_TIMEOUT
from utils.logger import RequestContext

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """Readiness probe. Returns 200 OK when service is ready."""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.
    """
    ctx = RequestContext()
    ctx.log(logging.INFO, f"Incoming request: {len(request.messages)} messages")
    
    try:
        async with asyncio.timeout(TOTAL_REQUEST_TIMEOUT):
            response = await _process_conversation(request, ctx)
            ctx.log(logging.INFO, "Request processed successfully", metadata={"reply_len": len(response.reply)})
            return response
    except asyncio.TimeoutError:
        ctx.log(logging.ERROR, "Request timed out", stage="timeout")
        return rg.safe_response(
            "I'm taking too long to respond. Please try again with a more specific query."
        )
    except Exception as e:
        ctx.log(logging.ERROR, f"Unhandled error: {str(e)}", stage="error")
        return rg.safe_response(
            "An unexpected error occurred. Please try again."
        )


async def _process_conversation(request: ChatRequest, ctx: RequestContext) -> ChatResponse:
    """
    Core conversation processing pipeline.
    """
    messages = request.messages

    # Get the latest user message for guardrail check
    last_user_message = ""
    for msg in reversed(messages):
        if msg.role == "user":
            last_user_message = msg.content
            break

    if not last_user_message:
        return rg.safe_response("I didn't receive your message. Could you try again?")

    # Step 1: Guardrail check
    is_blocked, threat_type, refusal_message = check_input(last_user_message)
    if is_blocked:
        ctx.log(logging.INFO, f"Guardrail blocked: {threat_type.value}", stage="guardrail")
        return await rg.generate_refusal_response(refusal_message)

    # Step 2: Parse raw intent from full conversation history
    ctx.log(logging.INFO, "Parsing intent...", stage="intent")
    raw_intent = await intent_parser.parse_intent(messages)
    
    # Step 3: Reconstruct full constraint state (Stateless Merge)
    ctx.log(logging.INFO, "Merging constraints...", stage="merger")
    intent = merge_constraints(messages, raw_intent)
    prior_recs = extract_prior_recommendations(messages)
    
    ctx.log(logging.INFO, f"Intent extracted", stage="intent", metadata={
        "role": intent.get('role'),
        "seniority": intent.get('seniority'),
        "completeness": intent.get('completeness_score', 0)
    })

    # Step 4: Determine conversation stage (FSM)
    stage = sm.determine_stage(
        intent=intent,
        history=messages,
        guardrail_blocked=False,
        prior_recommendations_count=len(prior_recs)
    )
    ctx.log(logging.INFO, f"FSM Transition: {stage.value}", stage="fsm")

    # Step 5: Generate response based on stage
    ctx.log(logging.INFO, "Generating response...", stage="generation")
    if stage == sm.ConversationStage.REFUSE:
        return await rg.generate_refusal_response(
            "I can only assist with SHL assessment recommendations."
        )

    elif stage == sm.ConversationStage.COMPARE:
        return await rg.generate_comparison_response(intent, messages)

    elif stage in (sm.ConversationStage.RECOMMEND, sm.ConversationStage.REFINE):
        return await rg.generate_recommendation_response(intent, stage)

    elif stage == sm.ConversationStage.CLARIFY:
        return await rg.generate_clarification_response(intent, messages)

    else:
        return rg.safe_response("How can I help you find the right SHL assessment?")
