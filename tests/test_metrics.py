"""
Unit Tests - Evaluation Metrics
Tests Recall@K and related metrics computations.
"""
import pytest
from evaluation.metrics import (
    TraceResult,
    hallucination_rate,
    mean_recall_at_k,
    recall_at_k,
    schema_compliance_rate,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        relevant = ["A", "B", "C"]
        retrieved = ["A", "B", "C", "D", "E"]
        assert recall_at_k(relevant, retrieved, k=10) == 1.0

    def test_zero_recall(self):
        relevant = ["A", "B", "C"]
        retrieved = ["D", "E", "F"]
        assert recall_at_k(relevant, retrieved, k=10) == 0.0

    def test_partial_recall(self):
        relevant = ["A", "B", "C", "D"]
        retrieved = ["A", "B", "X", "Y"]
        assert recall_at_k(relevant, retrieved, k=10) == 0.5

    def test_k_cutoff_respected(self):
        relevant = ["A"]
        retrieved = ["B", "C", "D", "A"]  # A is at position 4
        assert recall_at_k(relevant, retrieved, k=3) == 0.0
        assert recall_at_k(relevant, retrieved, k=4) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k([], ["A", "B"], k=10) == 1.0

    def test_empty_retrieved(self):
        assert recall_at_k(["A", "B"], [], k=10) == 0.0


class TestMeanRecallAtK:
    def test_mean_calculation(self):
        traces = [
            TraceResult("t1", ["A", "B"], ["A", "B"], 4, True),
            TraceResult("t2", ["C", "D"], ["C"], 6, True),
        ]
        # t1: 1.0, t2: 0.5 → mean = 0.75
        assert abs(mean_recall_at_k(traces) - 0.75) < 0.001

    def test_empty_traces(self):
        assert mean_recall_at_k([]) == 0.0


class TestSchemaComplianceRate:
    def test_all_compliant(self):
        traces = [
            TraceResult("t1", [], [], 4, True),
            TraceResult("t2", [], [], 4, True),
        ]
        assert schema_compliance_rate(traces) == 1.0

    def test_partial_compliance(self):
        traces = [
            TraceResult("t1", [], [], 4, True),
            TraceResult("t2", [], [], 4, False),
        ]
        assert schema_compliance_rate(traces) == 0.5


class TestHallucinationRate:
    def test_no_hallucinations(self):
        traces = [TraceResult("t1", [], [], 4, True, [])]
        assert hallucination_rate(traces) == 0.0

    def test_with_hallucinations(self):
        traces = [
            TraceResult("t1", [], [], 4, True, []),
            TraceResult("t2", [], [], 4, True, ["bad_url"]),
        ]
        assert hallucination_rate(traces) == 0.5
