"""
SHL Assessment Recommender - Local Replay Harness
Simulates multi-turn conversations against the /chat endpoint.
Reads trace files and measures Recall@10, schema compliance, hallucination.

Usage:
    python -m evaluation.replay_harness --url http://localhost:8000 --traces evaluation/traces/
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import httpx

from evaluation.metrics import (
    TraceResult,
    mean_recall_at_k,
    print_evaluation_report,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Valid SHL catalog URL prefix
SHL_URL_PREFIX = "https://www.shl.com/"


def _is_valid_catalog_url(url: str) -> bool:
    """Check if a URL looks like a valid SHL catalog URL."""
    return url.startswith(SHL_URL_PREFIX) and len(url) > len(SHL_URL_PREFIX)


def _validate_schema(response: dict) -> bool:
    """Validate response strictly matches the required schema."""
    if not isinstance(response, dict):
        return False
    if "reply" not in response or not isinstance(response["reply"], str):
        return False
    if "recommendations" not in response or not isinstance(response["recommendations"], list):
        return False
    if "end_of_conversation" not in response or not isinstance(response["end_of_conversation"], bool):
        return False

    recs = response["recommendations"]
    # Validate recommendation count
    if recs and not (1 <= len(recs) <= 10):
        return False

    # Validate each recommendation
    for rec in recs:
        if not all(k in rec for k in ["name", "url", "test_type"]):
            return False

    return True


async def run_trace(
    client: httpx.AsyncClient,
    base_url: str,
    trace: dict,
    max_turns: int = 8,
) -> TraceResult:
    """
    Run a single conversation trace against the /chat endpoint.
    
    The harness simulates a user by replaying the trace's pre-defined turns.
    For each user turn, it calls /chat with the accumulated history.
    """
    trace_id = trace.get("id", "unknown")
    expected = trace.get("expected_assessments", [])
    turns = trace.get("turns", [])

    messages = []
    final_recommendations = []
    turns_used = 0
    schema_compliant = True
    hallucinated_urls = []

    logger.info(f"Running trace: {trace_id}")

    for turn in turns:
        if turns_used >= max_turns:
            logger.warning(f"Trace {trace_id}: Hit turn limit")
            break

        # Add user message
        user_content = turn.get("user", "")
        if not user_content:
            continue

        messages.append({"role": "user", "content": user_content})
        turns_used += 1

        # Call the API
        try:
            response = await client.post(
                f"{base_url}/chat",
                json={"messages": messages},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            logger.error(f"Trace {trace_id}: Request timed out on turn {turns_used}")
            schema_compliant = False
            break
        except Exception as e:
            logger.error(f"Trace {trace_id}: API error: {e}")
            schema_compliant = False
            break

        # Validate schema
        if not _validate_schema(data):
            logger.error(f"Trace {trace_id}: Schema violation on turn {turns_used}: {data}")
            schema_compliant = False

        # Check for hallucinated URLs
        for rec in data.get("recommendations", []):
            url = rec.get("url", "")
            if not _is_valid_catalog_url(url):
                hallucinated_urls.append(url)
                logger.warning(f"Hallucinated URL: {url}")

        # Add assistant response to history
        assistant_reply = data.get("reply", "")
        messages.append({"role": "assistant", "content": assistant_reply})
        turns_used += 1

        # Capture recommendations
        if data.get("recommendations"):
            final_recommendations = [r["name"] for r in data["recommendations"]]

        # Check if conversation ended
        if data.get("end_of_conversation"):
            logger.info(f"Trace {trace_id}: Completed in {turns_used} turns")
            break

    result = TraceResult(
        trace_id=trace_id,
        relevant_assessments=expected,
        retrieved_assessments=final_recommendations,
        turns_used=turns_used,
        schema_compliant=schema_compliant,
        hallucinated_urls=hallucinated_urls,
    )

    from evaluation.metrics import recall_at_k
    score = recall_at_k(expected, final_recommendations)
    logger.info(
        f"Trace {trace_id}: Recall@10={score:.3f}, "
        f"turns={turns_used}, schema={'OK' if schema_compliant else 'FAIL'}"
    )
    return result


async def run_evaluation(
    base_url: str,
    traces_dir: str,
    max_turns: int = 8,
) -> List[TraceResult]:
    """Run all traces in a directory and collect results."""
    traces_path = Path(traces_dir)
    trace_files = list(traces_path.glob("*.json"))

    if not trace_files:
        logger.error(f"No trace files found in {traces_dir}")
        return []

    logger.info(f"Found {len(trace_files)} traces in {traces_dir}")

    # Check health first
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{base_url}/health", timeout=120)
            if health.json().get("status") != "ok":
                raise ValueError("Health check failed")
            logger.info("Health check passed")
        except Exception as e:
            logger.error(f"Service not healthy: {e}")
            return []

        results = []
        for trace_file in sorted(trace_files):
            with open(trace_file) as f:
                trace = json.load(f)

            result = await run_trace(client, base_url, trace, max_turns)
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="SHL Recommender Replay Harness")
    parser.add_argument("--url", default="http://localhost:8000", help="Service base URL")
    parser.add_argument(
        "--traces", default="evaluation/traces", help="Directory with trace JSON files"
    )
    parser.add_argument("--max-turns", type=int, default=8, help="Max turns per trace")
    args = parser.parse_args()

    results = asyncio.run(
        run_evaluation(args.url, args.traces, args.max_turns)
    )

    if results:
        print_evaluation_report(results)
        mr10 = mean_recall_at_k(results)
        sys.exit(0 if mr10 > 0.5 else 1)
    else:
        print("No results — check service logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
