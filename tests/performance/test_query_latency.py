"""
Performance tests for query latency
Measure and validate search performance
"""

import pytest
import time
import statistics
from typing import List
import httpx


class TestQueryLatency:
    """Test query latency performance"""

    @pytest.fixture
    def api_client(self):
        """Create HTTP client for API testing"""
        return httpx.Client(base_url="http://localhost:8000", timeout=30.0)

    def measure_latency(self, func, iterations: int = 10) -> List[float]:
        """
        Measure function latency over multiple iterations
        
        Args:
            func: Function to measure
            iterations: Number of iterations
            
        Returns:
            List of latencies in milliseconds
        """
        latencies = []
        for _ in range(iterations):
            start = time.time()
            func()
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        return latencies

    def test_semantic_search_latency(self, api_client, sample_queries):
        """Test semantic search latency"""
        # Arrange
        query = sample_queries[0]
        target_p95 = 500  # ms
        target_p99 = 800  # ms

        # Act
        latencies = self.measure_latency(
            lambda: api_client.post(
                "/api/v1/search/semantic",
                json={"query": query, "top_k": 10}
            ),
            iterations=20
        )

        # Assert
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)

        print(f"\nSemantic Search Latency:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")

        assert p95 < target_p95, f"P95 latency {p95:.2f}ms exceeds target {target_p95}ms"
        assert p99 < target_p99, f"P99 latency {p99:.2f}ms exceeds target {target_p99}ms"

    def test_hybrid_search_latency(self, api_client, sample_queries):
        """Test hybrid search latency"""
        # Arrange
        query = sample_queries[0]
        target_p95 = 600  # ms

        # Act
        latencies = self.measure_latency(
            lambda: api_client.post(
                "/api/v1/search/hybrid",
                json={"query": query, "top_k": 10, "alpha": 0.7, "beta": 0.3}
            ),
            iterations=20
        )

        # Assert
        p95 = statistics.quantiles(latencies, n=20)[18]
        print(f"\nHybrid Search Latency P95: {p95:.2f}ms")
        assert p95 < target_p95, f"P95 latency {p95:.2f}ms exceeds target {target_p95}ms"

    def test_rag_answer_latency(self, api_client):
        """Test RAG answer generation latency"""
        # Arrange
        question = "What are transformers in deep learning?"
        target_p95 = 3000  # ms (3 seconds)

        # Act
        latencies = self.measure_latency(
            lambda: api_client.post(
                "/api/v1/rag/ask",
                json={"question": question, "top_k": 5}
            ),
            iterations=10
        )

        # Assert
        p95 = statistics.quantiles(latencies, n=10)[8]
        print(f"\nRAG Answer Latency P95: {p95:.2f}ms")
        assert p95 < target_p95, f"P95 latency {p95:.2f}ms exceeds target {target_p95}ms"

    def test_recommendation_latency(self, api_client):
        """Test recommendation latency"""
        # Arrange
        paper_id = "arxiv_1706_03762"
        target_p95 = 400  # ms

        # Act
        latencies = self.measure_latency(
            lambda: api_client.post(
                "/api/v1/recommend/similar",
                json={"paper_id": paper_id, "top_k": 10, "diversity": 0.5}
            ),
            iterations=20
        )

        # Assert
        p95 = statistics.quantiles(latencies, n=20)[18]
        print(f"\nRecommendation Latency P95: {p95:.2f}ms")
        assert p95 < target_p95, f"P95 latency {p95:.2f}ms exceeds target {target_p95}ms"

    def test_cache_hit_performance(self, api_client, sample_queries):
        """Test cache hit improves performance"""
        # Arrange
        query = sample_queries[0]

        # Act - First request (cache miss)
        start = time.time()
        api_client.post("/api/v1/search/semantic", json={"query": query, "top_k": 10})
        first_latency = (time.time() - start) * 1000

        # Act - Second request (cache hit)
        start = time.time()
        api_client.post("/api/v1/search/semantic", json={"query": query, "top_k": 10})
        second_latency = (time.time() - start) * 1000

        # Assert
        print(f"\nCache Performance:")
        print(f"  First request: {first_latency:.2f}ms")
        print(f"  Second request: {second_latency:.2f}ms")
        print(f"  Speedup: {first_latency / second_latency:.2f}x")

        # Cache hit should be faster
        assert second_latency < first_latency, "Cache hit should be faster than cache miss"

    def test_latency_consistency(self, api_client, sample_queries):
        """Test latency consistency (low variance)"""
        # Arrange
        query = sample_queries[0]
        max_std_dev = 100  # ms

        # Act
        latencies = self.measure_latency(
            lambda: api_client.post(
                "/api/v1/search/semantic",
                json={"query": query, "top_k": 10}
            ),
            iterations=30
        )

        # Assert
        std_dev = statistics.stdev(latencies)
        print(f"\nLatency Standard Deviation: {std_dev:.2f}ms")
        assert std_dev < max_std_dev, f"Latency variance {std_dev:.2f}ms too high"

    @pytest.mark.slow
    def test_query_latency_under_load(self, api_client, large_query_batch):
        """Test query latency under sustained load"""
        # Arrange
        target_avg = 300  # ms
        num_queries = 50

        # Act
        latencies = []
        for query in large_query_batch[:num_queries]:
            start = time.time()
            api_client.post("/api/v1/search/semantic", json={"query": query, "top_k": 10})
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        # Assert
        avg_latency = statistics.mean(latencies)
        print(f"\nAverage Latency Under Load: {avg_latency:.2f}ms")
        assert avg_latency < target_avg, f"Average latency {avg_latency:.2f}ms exceeds target"
