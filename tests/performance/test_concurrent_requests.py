"""
Performance tests for concurrent requests
Measure system behavior under concurrent load
"""

import pytest
import time
import statistics
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


class TestConcurrentRequests:
    """Test concurrent request handling"""

    @pytest.fixture
    def api_base_url(self):
        """API base URL"""
        return "http://localhost:8000"

    def make_request(self, url: str, payload: dict) -> Tuple[int, float]:
        """
        Make a single HTTP request
        
        Returns:
            Tuple of (status_code, latency_ms)
        """
        start = time.time()
        try:
            response = httpx.post(url, json=payload, timeout=30.0)
            latency = (time.time() - start) * 1000
            return (response.status_code, latency)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return (500, latency)

    def test_concurrent_search_requests(self, api_base_url, sample_queries):
        """Test concurrent search requests"""
        # Arrange
        num_concurrent = 10
        url = f"{api_base_url}/api/v1/search/semantic"
        payloads = [{"query": q, "top_k": 10} for q in sample_queries[:num_concurrent]]

        # Act
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.make_request, url, p) for p in payloads]
            results = [f.result() for f in as_completed(futures)]

        # Assert
        status_codes = [r[0] for r in results]
        latencies = [r[1] for r in results]

        success_rate = sum(1 for s in status_codes if s == 200) / len(status_codes)
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        print(f"\nConcurrent Search ({num_concurrent} requests):")
        print(f"  Success Rate: {success_rate * 100:.1f}%")
        print(f"  Avg Latency: {avg_latency:.2f}ms")
        print(f"  Max Latency: {max_latency:.2f}ms")

        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
        assert avg_latency < 1000, f"Average latency {avg_latency:.2f}ms too high"

    def test_concurrent_rag_requests(self, api_base_url):
        """Test concurrent RAG requests"""
        # Arrange
        num_concurrent = 5  # Lower for RAG due to LLM latency
        url = f"{api_base_url}/api/v1/rag/ask"
        questions = [
            "What are transformers?",
            "Explain attention mechanisms",
            "What is BERT?",
            "How does GPT work?",
            "What are neural networks?",
        ]
        payloads = [{"question": q, "top_k": 5} for q in questions]

        # Act
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.make_request, url, p) for p in payloads]
            results = [f.result() for f in as_completed(futures)]

        # Assert
        status_codes = [r[0] for r in results]
        latencies = [r[1] for r in results]

        success_rate = sum(1 for s in status_codes if s == 200) / len(status_codes)
        avg_latency = statistics.mean(latencies)

        print(f"\nConcurrent RAG ({num_concurrent} requests):")
        print(f"  Success Rate: {success_rate * 100:.1f}%")
        print(f"  Avg Latency: {avg_latency:.2f}ms")

        assert success_rate >= 0.90, f"Success rate {success_rate:.2%} below 90%"

    def test_concurrent_recommendation_requests(self, api_base_url):
        """Test concurrent recommendation requests"""
        # Arrange
        num_concurrent = 10
        url = f"{api_base_url}/api/v1/recommend/similar"
        paper_ids = [f"arxiv_test_{i:03d}" for i in range(num_concurrent)]
        payloads = [{"paper_id": pid, "top_k": 10, "diversity": 0.5} for pid in paper_ids]

        # Act
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.make_request, url, p) for p in payloads]
            results = [f.result() for f in as_completed(futures)]

        # Assert
        status_codes = [r[0] for r in results]
        success_rate = sum(1 for s in status_codes if s == 200) / len(status_codes)

        print(f"\nConcurrent Recommendations ({num_concurrent} requests):")
        print(f"  Success Rate: {success_rate * 100:.1f}%")

        assert success_rate >= 0.90, "Success rate should be at least 90%"

    @pytest.mark.slow
    def test_sustained_concurrent_load(self, api_base_url, sample_queries):
        """Test sustained concurrent load"""
        # Arrange
        num_concurrent = 20
        duration_seconds = 30
        url = f"{api_base_url}/api/v1/search/semantic"

        # Act
        start_time = time.time()
        total_requests = 0
        successful_requests = 0
        latencies = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            while time.time() - start_time < duration_seconds:
                query = sample_queries[total_requests % len(sample_queries)]
                payload = {"query": query, "top_k": 10}
                
                future = executor.submit(self.make_request, url, payload)
                status, latency = future.result()
                
                total_requests += 1
                if status == 200:
                    successful_requests += 1
                latencies.append(latency)

        # Assert
        elapsed = time.time() - start_time
        throughput = total_requests / elapsed
        success_rate = successful_requests / total_requests
        avg_latency = statistics.mean(latencies)

        print(f"\nSustained Load ({duration_seconds}s):")
        print(f"  Total Requests: {total_requests}")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Success Rate: {success_rate * 100:.1f}%")
        print(f"  Avg Latency: {avg_latency:.2f}ms")

        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
        assert throughput >= 10, f"Throughput {throughput:.2f} req/s below target"

    def test_mixed_workload_concurrency(self, api_base_url, sample_queries):
        """Test mixed workload (search + RAG + recommendations)"""
        # Arrange
        num_concurrent = 15
        search_url = f"{api_base_url}/api/v1/search/semantic"
        rag_url = f"{api_base_url}/api/v1/rag/ask"
        rec_url = f"{api_base_url}/api/v1/recommend/similar"

        requests = []
        # 60% search, 30% recommendations, 10% RAG
        for i in range(num_concurrent):
            if i < 9:  # Search
                requests.append((search_url, {"query": sample_queries[i % len(sample_queries)], "top_k": 10}))
            elif i < 14:  # Recommendations
                requests.append((rec_url, {"paper_id": f"arxiv_test_{i}", "top_k": 10, "diversity": 0.5}))
            else:  # RAG
                requests.append((rag_url, {"question": "What are transformers?", "top_k": 5}))

        # Act
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.make_request, url, payload) for url, payload in requests]
            results = [f.result() for f in as_completed(futures)]

        # Assert
        status_codes = [r[0] for r in results]
        success_rate = sum(1 for s in status_codes if s == 200) / len(status_codes)

        print(f"\nMixed Workload ({num_concurrent} requests):")
        print(f"  Success Rate: {success_rate * 100:.1f}%")

        assert success_rate >= 0.90, f"Success rate {success_rate:.2%} below 90%"

    def test_error_rate_under_load(self, api_base_url, sample_queries):
        """Test error rate under high concurrent load"""
        # Arrange
        num_concurrent = 50  # High load
        url = f"{api_base_url}/api/v1/search/semantic"
        payloads = [{"query": sample_queries[i % len(sample_queries)], "top_k": 10} 
                    for i in range(num_concurrent)]

        # Act
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.make_request, url, p) for p in payloads]
            results = [f.result() for f in as_completed(futures)]

        # Assert
        status_codes = [r[0] for r in results]
        error_rate = sum(1 for s in status_codes if s >= 400) / len(status_codes)

        print(f"\nError Rate Under High Load ({num_concurrent} concurrent):")
        print(f"  Error Rate: {error_rate * 100:.1f}%")

        assert error_rate < 0.10, f"Error rate {error_rate:.2%} exceeds 10%"

    def test_latency_degradation_under_load(self, api_base_url, sample_queries):
        """Test latency degradation as concurrency increases"""
        # Arrange
        url = f"{api_base_url}/api/v1/search/semantic"
        concurrency_levels = [1, 5, 10, 20]
        query = sample_queries[0]

        # Act
        results = {}
        for concurrency in concurrency_levels:
            payloads = [{"query": query, "top_k": 10}] * concurrency
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(self.make_request, url, p) for p in payloads]
                latencies = [f.result()[1] for f in as_completed(futures)]
            
            results[concurrency] = statistics.mean(latencies)

        # Assert
        print(f"\nLatency vs Concurrency:")
        for concurrency, avg_latency in results.items():
            print(f"  {concurrency} concurrent: {avg_latency:.2f}ms")

        # Latency should not degrade more than 3x
        degradation = results[max(concurrency_levels)] / results[1]
        assert degradation < 3.0, f"Latency degradation {degradation:.2f}x too high"
