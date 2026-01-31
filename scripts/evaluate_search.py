"""
Search Quality Evaluation Script
Benchmarks search relevance and quality
"""

import sys
import time
from typing import List, Dict

# Add parent to path
sys.path.append(".")

from backend.core.search_engine import SemanticSearchEngine, SearchParams
from backend.core.embeddings import EmbeddingService
from backend.core.endee_client import EndeeVectorDB
from backend.config import get_settings

TEST_QUERIES = [
    "deep learning in medical imaging",
    "large language model optimization",
    "climate change impact on crop yield",
    "quantum error correction",
    "reinforcement learning for robotics",
    "asdfjkl"  # Noise query
]

def evaluate_search():
    print("🧪 Starting Search Quality Evaluation")
    print("-" * 50)
    
    settings = get_settings()
    
    # Initialize components
    # Using mocks where necessary if real services aren't live
    embedding_service = EmbeddingService()
    endee_client = EndeeVectorDB(url=settings.ENDEE_URL)
    engine = SemanticSearchEngine(endee_client, embedding_service)
    
    stats = {
        "total_queries": 0,
        "zero_results": 0,
        "avg_latency_ms": 0,
        "avg_score": 0
    }
    
    total_latency = 0
    total_score = 0
    
    for query in TEST_QUERIES:
        print(f"\nQuery: '{query}'")
        
        # Test 1: Basic Search
        params = SearchParams(
            query=query,
            top_k=5,
            expand_query=False,
            rerank_by_recency=False
        )
        
        start = time.time()
        try:
            # We use advanced_search to test the full pipeline
            # Note: advanced_search expects params
            response = engine.advanced_search(params)
            duration = (time.time() - start) * 1000
            
            count = len(response.results)
            top_score = response.results[0].similarity if count > 0 else 0
            
            print(f"  Results: {count} | Top Score: {top_score:.4f} | Time: {duration:.2f}ms")
            
            stats["total_queries"] += 1
            if count == 0:
                stats["zero_results"] += 1
            total_latency += duration
            total_score += top_score
            
        except Exception as e:
            print(f"  Error: {e}")

    # Aggregates
    if stats["total_queries"] > 0:
        stats["avg_latency_ms"] = total_latency / stats["total_queries"]
        stats["avg_score"] = total_score / stats["total_queries"]
        
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Zero Results:  {stats['zero_results']}")
    print(f"Avg Latency:   {stats['avg_latency_ms']:.2f} ms")
    print(f"Avg Relevance: {stats['avg_score']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    evaluate_search()
