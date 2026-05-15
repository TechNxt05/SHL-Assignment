import json
import logging
import numpy as np
from typing import List, Dict, Any
from app.services.retrieval import retrieve, initialize_retrieval
from app.services.intent_parser import _heuristic_intent_extraction
from app.schemas.chat import Message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recall_at_k(expected: List[str], actual: List[str], k: int) -> float:
    """Compute Recall@K."""
    actual_k = actual[:k]
    actual_lower = [a.lower() for a in actual_k]
    found = 0
    for e in expected:
        if e.lower() in actual_lower:
            found += 1
    return found / len(expected) if expected else 0.0

def run_retrieval_benchmark(dataset_path: str):
    """Run benchmark over the retrieval dataset."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    initialize_retrieval()
    
    recalls_5 = []
    recalls_10 = []
    
    print("\n" + "="*60)
    print("RETRIEVAL BENCHMARK REPORT")
    print("="*60)
    print(f"{'Query':<40} | R@5   | R@10")
    print("-" * 60)
    
    for item in dataset:
        query_text = item["query"]
        expected = item["expected"]
        
        # Build a better mock intent
        intent = {
            "role": query_text if " " not in query_text else "",
            "technical_skills": [],
            "soft_skills": [],
            "seniority": "",
        }
        
        # Simple extraction for benchmark
        q_low = query_text.lower()
        if "java" in q_low: intent["technical_skills"].append("Java")
        if "python" in q_low: intent["technical_skills"].append("Python")
        if "manager" in q_low: intent["role"] = "Manager"
        if "cashier" in q_low: intent["role"] = "Cashier"
        if "financial" in q_low: intent["role"] = "Financial Analyst"
        if "microservices" in q_low: intent["technical_skills"].append("Microservices")
        
        # Ensure role is set if still empty
        if not intent["role"] and not intent["technical_skills"]:
            intent["role"] = query_text
        
        # Retrieve
        results = retrieve(intent, top_k=15)
        actual_names = [r["name"] for r in results]
        
        r5 = recall_at_k(expected, actual_names, 5)
        r10 = recall_at_k(expected, actual_names, 10)
        
        recalls_5.append(r5)
        recalls_10.append(r10)
        
        print(f"{query_text[:40]:<40} | {r5:.3f} | {r10:.3f}")
        
    print("-" * 60)
    print(f"{'MEAN':<40} | {np.mean(recalls_5):.3f} | {np.mean(recalls_10):.3f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_retrieval_benchmark("evaluation/retrieval_dataset.json")
