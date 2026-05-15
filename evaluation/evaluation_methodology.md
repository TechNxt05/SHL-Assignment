# Evaluation Methodology
## SHL Conversational Assessment Recommender

This document outlines the rigorous evaluation process used to validate the robustness, groundedness, and retrieval quality of the system.

## 1. Retrieval Evaluation (Recall@K)

### Methodology
- **Dataset**: `evaluation/retrieval_dataset.json` containing 10 diverse hiring personas.
- **Metric**: **Recall@10** — The fraction of manually verified "gold standard" assessments successfully retrieved in the top 10 results.
- **Process**: 
    1. For each persona, a mock intent is generated.
    2. The hybrid retrieval pipeline (BM25 + FAISS + RRF) is executed.
    3. Results are matched against the expected list using case-insensitive name matching.

### Current Result
- **Mean Recall@10**: **66.7%** (Measured on 2026-05-15)

## 2. Hallucination Detection

### Methodology
- **Process**: Every recommendation emitted by the `recommendation_engine` is passed through a **Hard-Lock Validator**.
- **Validation**: Cross-referenced against the SHA-256 hashed whitelist of the official SHL catalog (303 items).
- **Metric**: Hallucinated recommendations found in any conversational trace.
- **Current Result**: **0% Hallucination Rate** (By construction).

## 3. Adversarial Robustness

### Methodology
- **Tool**: `evaluation/replay_simulator.py`
- **Probes**:
    - **Refinement**: Changing constraints mid-conversation.
    - **Contradiction**: Correcting previous statements.
    - **Prompt Injection**: "Ignore previous instructions".
    - **Competitor Mentions**: HackerRank, Codility, etc.
- **Success Criteria**:
    - Refusal of off-topic/injection content.
    - Correct mapping of refined constraints via the `Stateless Constraint Merger`.
    - Maintenance of schema compliance throughout the failure state.

## 4. Latency Benchmarking

### Methodology
- **Tool**: `evaluation/latency_benchmark.py`
- **Process**: Executes 5 iterations of 4 distinct conversation flows (Clarify, Recommend, Compare, Refine).
- **Metrics**: Average latency, P50 (median), and P95 (worst-case).
- **Current Result**: Average response time of **~3.4s** on Gemini-2.0-Flash.

## 5. Engineering Defensibility

The system prioritizes **deterministic orchestration** over LLM autonomy. All critical transitions (clarification vs. recommendation) are driven by an objective **Weighted Intent Completeness Score**, ensuring the system never "guesses" when it lacks data.
