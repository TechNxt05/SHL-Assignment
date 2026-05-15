"""
SHL Assessment Recommender - Evaluation Metrics
Implements Recall@K and Mean Recall@K as defined in the assignment spec.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TraceResult:
    """Result of running a single conversation trace."""
    trace_id: str
    relevant_assessments: List[str]  # Ground truth: expected assessment names
    retrieved_assessments: List[str]  # What the agent recommended
    turns_used: int
    schema_compliant: bool
    hallucinated_urls: List[str] = field(default_factory=list)


def recall_at_k(relevant: List[str], retrieved: List[str], k: int = 10) -> float:
    """
    Recall@K: fraction of relevant assessments appearing in top-K retrieved.
    
    Recall@K = (# relevant in top K) / (total relevant)
    
    Args:
        relevant: Ground truth list of relevant assessment names
        retrieved: Agent's recommended assessment names (ordered)
        k: Cutoff (default 10 per assignment spec)
    
    Returns:
        Float in [0.0, 1.0]
    """
    if not relevant:
        return 1.0  # No relevant items = trivially satisfied

    def normalize(s: str) -> str:
        return s.strip().lower()

    top_k = {normalize(name) for name in retrieved[:k]}
    relevant_set = {normalize(name) for name in relevant}
    
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mean_recall_at_k(traces: List[TraceResult], k: int = 10) -> float:
    """
    Mean Recall@K across all traces.
    
    MR@K = (1/N) * sum(Recall@K_i)
    """
    if not traces:
        return 0.0
    scores = [
        recall_at_k(t.relevant_assessments, t.retrieved_assessments, k)
        for t in traces
    ]
    return sum(scores) / len(scores)


def schema_compliance_rate(traces: List[TraceResult]) -> float:
    """Fraction of traces with fully schema-compliant responses."""
    if not traces:
        return 0.0
    compliant = sum(1 for t in traces if t.schema_compliant)
    return compliant / len(traces)


def hallucination_rate(traces: List[TraceResult]) -> float:
    """Fraction of traces containing at least one hallucinated URL."""
    if not traces:
        return 0.0
    hallucinated = sum(1 for t in traces if t.hallucinated_urls)
    return hallucinated / len(traces)


def turn_efficiency(traces: List[TraceResult], max_turns: int = 8) -> dict:
    """
    Statistics on turn count usage.
    Ideal: recommendations given by turn 4-6 (leaving room for refinement).
    """
    if not traces:
        return {}
    counts = [t.turns_used for t in traces]
    within_limit = sum(1 for c in counts if c <= max_turns)
    return {
        "avg_turns": sum(counts) / len(counts),
        "max_turns": max(counts),
        "min_turns": min(counts),
        "within_limit_rate": within_limit / len(counts),
    }


def print_evaluation_report(traces: List[TraceResult]) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 60)
    print("SHL RECOMMENDER EVALUATION REPORT")
    print("=" * 60)
    print(f"Total traces evaluated: {len(traces)}")
    print(f"Mean Recall@10:         {mean_recall_at_k(traces):.3f}")
    print(f"Schema compliance:      {schema_compliance_rate(traces):.3f}")
    print(f"Hallucination rate:     {hallucination_rate(traces):.3f}")

    eff = turn_efficiency(traces)
    if eff:
        print(f"Avg turns to answer:    {eff['avg_turns']:.1f}")
        print(f"Within 8-turn limit:    {eff['within_limit_rate']:.3f}")

    print("\nPer-trace results:")
    print("-" * 60)
    for trace in traces:
        r = recall_at_k(trace.relevant_assessments, trace.retrieved_assessments)
        status = "PASS" if trace.schema_compliant else "FAIL"
        hallu = " [HALLUCINATION]" if trace.hallucinated_urls else ""
        print(
            f"  {status} {trace.trace_id}: Recall@10={r:.3f}, "
            f"turns={trace.turns_used}{hallu}"
        )
    print("=" * 60)
