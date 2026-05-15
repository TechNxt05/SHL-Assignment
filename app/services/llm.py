"""
SHL Assessment Recommender - LLM Provider
Async wrapper for Google Gemini 2.0 Flash.
Provides deterministic, low-temperature generation with retry logic.
"""
import asyncio
import logging
import os
from typing import Optional, Type

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import PRIMARY_MODEL, FALLBACK_MODEL, LLM_TIMEOUT, GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Lazy client singleton
_client = None

def _get_client():
    """Lazy-initialize the Gemini client."""
    global _client
    if _client is None:
        import google.genai as genai
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"Gemini client initialized with model: {PRIMARY_MODEL}")
    return _client


async def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
    response_mime_type: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Async LLM call with timeout and fallback logic.
    """
    client = _get_client()
    model_id = model or PRIMARY_MODEL

    from google.genai import types

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "system_instruction": system_prompt,
    }
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    generation_config = types.GenerateContentConfig(**config_kwargs)

    try:
        # Run in thread pool to avoid blocking, with a hard timeout
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _call_with_retry(client, model_id, user_message, generation_config),
            ),
            timeout=LLM_TIMEOUT
        )
        return response
    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out after {LLM_TIMEOUT}s")
        raise
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


def _call_with_retry(client, model_id: str, message: str, config) -> str:
    """Synchronous call with internal retry and model fallback."""
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=message,
            config=config,
        )
        return response.text or ""
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "503" in error_str:
            # Try fallback model immediately on rate limit or server error
            if model_id != FALLBACK_MODEL:
                logger.warning(f"Primary model failed ({e}), trying fallback: {FALLBACK_MODEL}")
                return _call_with_retry(client, FALLBACK_MODEL, message, config)
        raise


async def call_llm_json(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.1,
) -> str:
    """Convenience wrapper for JSON-mode LLM calls."""
    return await call_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        response_mime_type="application/json",
        max_output_tokens=2048,
    )
