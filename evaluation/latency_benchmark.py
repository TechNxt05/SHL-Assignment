import asyncio
import httpx
import time
import json
import statistics
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENARIOS = [
    {
        "name": "Single Keyword Clarification",
        "messages": [{"role": "user", "content": "assessments"}]
    },
    {
        "name": "Rich Intent Recommendation",
        "messages": [
            {"role": "user", "content": "I need a Java developer assessment"},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user", "content": "Senior level with leadership skills"}
        ]
    },
    {
        "name": "Comparison Request",
        "messages": [{"role": "user", "content": "Compare OPQ32r and GSA"}]
    },
    {
        "name": "Refinement Flow",
        "messages": [
            {"role": "user", "content": "Hiring a Python dev"},
            {"role": "assistant", "content": "Recommendations for Python..."},
            {"role": "user", "content": "Actually, make it Ruby instead"}
        ]
    }
]

async def benchmark_request(client: httpx.AsyncClient, url: str, scenario: Dict[str, Any]) -> float:
    start_time = time.time()
    try:
        response = await client.post(f"{url}/chat", json={"messages": scenario["messages"]})
        response.raise_for_status()
        latency = time.time() - start_time
        return latency
    except Exception as e:
        logger.error(f"Request failed in {scenario['name']}: {e}")
        return None

async def run_latency_benchmark(url: str, iterations: int = 5):
    print("\n" + "="*60)
    print("SHL RECOMMENDER LATENCY BENCHMARK")
    print("="*60)
    
    results = {}
    all_latencies = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for scenario in SCENARIOS:
            print(f"Benchmarking: {scenario['name']} ({iterations} iterations)...")
            latencies = []
            for _ in range(iterations):
                lat = await benchmark_request(client, url, scenario)
                if lat:
                    latencies.append(lat)
                    all_latencies.append(lat)
                await asyncio.sleep(0.5) # Prevent rate limiting
            
            if latencies:
                avg = sum(latencies) / len(latencies)
                p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else latencies[0]
                results[scenario["name"]] = {
                    "avg": avg,
                    "p95": p95,
                    "min": min(latencies),
                    "max": max(latencies)
                }
                print(f"  -> Avg: {avg:.2f}s | P95: {p95:.2f}s")
            else:
                print(f"  -> FAILED")

    # Overall stats
    if all_latencies:
        overall = {
            "p50": statistics.median(all_latencies),
            "p95": statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 2 else all_latencies[0],
            "avg": sum(all_latencies) / len(all_latencies),
            "count": len(all_latencies)
        }
        
        report = {
            "scenarios": results,
            "overall": overall,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        with open("evaluation/latency_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print("\n" + "="*60)
        print(f"OVERALL P50: {overall['p50']:.2f}s")
        print(f"OVERALL P95: {overall['p95']:.2f}s")
        print("Report saved to evaluation/latency_report.json")
        print("="*60 + "\n")

if __name__ == "__main__":
    # Ensure server is running before executing
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    asyncio.run(run_latency_benchmark(url))
