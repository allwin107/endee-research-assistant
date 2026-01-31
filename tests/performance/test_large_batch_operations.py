"""
Performance tests for large batch operations
Test system behavior with large data volumes
"""

import pytest
import time
import statistics
from typing import List


class TestLargeBatchOperations:
    """Test large batch operation performance"""

    def test_batch_embedding_generation(self, large_query_batch):
        """Test batch embedding generation performance"""
        from backend.core.embeddings import EmbeddingService

        # Arrange
        embedding_service = EmbeddingService()
        batch_sizes = [16, 32, 64]
        target_throughput = 50  # embeddings/second

        # Act & Assert
        print("\nBatch Embedding Performance:")
        for batch_size in batch_sizes:
            texts = large_query_batch[:batch_size]
            
            start = time.time()
            embeddings = embedding_service.embed_batch(texts, batch_size=batch_size)
            elapsed = time.time() - start
            
            throughput = len(texts) / elapsed
            avg_time_per_embedding = (elapsed / len(texts)) * 1000  # ms

            print(f"  Batch size {batch_size}:")
            print(f"    Throughput: {throughput:.2f} embeddings/s")
            print(f"    Avg time: {avg_time_per_embedding:.2f}ms per embedding")

            assert len(embeddings) == len(texts), "Should return embedding for each text"
            assert throughput >= target_throughput, f"Throughput {throughput:.2f} below target"

    def test_large_search_result_set(self, large_query_batch):
        """Test handling large search result sets"""
        from backend.core.search_engine import SemanticSearchEngine
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService

        # Arrange
        endee_client = EndeeVectorDB()
        embedding_service = EmbeddingService()
        search_engine = SemanticSearchEngine(endee_client, embedding_service)
        
        query = large_query_batch[0]
        top_k_values = [10, 50, 100]

        # Act & Assert
        print("\nLarge Result Set Performance:")
        for top_k in top_k_values:
            start = time.time()
            results = search_engine.search(query=query, top_k=top_k)
            elapsed = (time.time() - start) * 1000  # ms

            print(f"  Top-{top_k}: {elapsed:.2f}ms")

            assert len(results) <= top_k, f"Should return at most {top_k} results"
            # Latency should scale sub-linearly
            if top_k == 100:
                assert elapsed < 1000, f"Latency {elapsed:.2f}ms too high for top-{top_k}"

    def test_bulk_paper_ingestion(self, large_paper_batch):
        """Test bulk paper ingestion performance"""
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService

        # Arrange
        endee_client = EndeeVectorDB()
        embedding_service = EmbeddingService()
        batch_size = 100
        papers = large_paper_batch[:batch_size]

        # Act
        start = time.time()
        
        # Generate embeddings
        texts = [p["abstract"] for p in papers]
        embeddings = embedding_service.embed_batch(texts, batch_size=32)
        
        # Upsert to Endee (mock)
        # In real implementation: endee_client.upsert_batch(...)
        
        elapsed = time.time() - start
        throughput = len(papers) / elapsed

        # Assert
        print(f"\nBulk Ingestion ({batch_size} papers):")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} papers/s")

        assert throughput >= 10, f"Throughput {throughput:.2f} papers/s too low"

    def test_large_user_interaction_batch(self, large_paper_batch):
        """Test processing large batch of user interactions"""
        from backend.core.recommendation import RecommendationEngine
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService

        # Arrange
        endee_client = EndeeVectorDB()
        embedding_service = EmbeddingService()
        rec_engine = RecommendationEngine(endee_client, embedding_service)
        
        user_id = "user_batch_test"
        num_interactions = 100
        interactions = [
            (user_id, f"arxiv_test_{i:05d}", "view")
            for i in range(num_interactions)
        ]

        # Act
        start = time.time()
        for user_id, paper_id, action in interactions:
            rec_engine.track_interaction(user_id, paper_id, action)
        elapsed = time.time() - start

        throughput = num_interactions / elapsed

        # Assert
        print(f"\nUser Interaction Batch ({num_interactions} interactions):")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} interactions/s")

        assert throughput >= 50, f"Throughput {throughput:.2f} interactions/s too low"

    def test_pagination_performance(self, large_query_batch):
        """Test pagination performance with large result sets"""
        from backend.core.search_engine import SemanticSearchEngine, SearchParams
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService

        # Arrange
        endee_client = EndeeVectorDB()
        embedding_service = EmbeddingService()
        search_engine = SemanticSearchEngine(endee_client, embedding_service)
        
        query = large_query_batch[0]
        page_size = 10
        num_pages = 5

        # Act
        latencies = []
        for page in range(1, num_pages + 1):
            params = SearchParams(
                query=query,
                page=page,
                page_size=page_size,
                top_k=page_size * num_pages
            )
            
            start = time.time()
            response = search_engine.advanced_search(params)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        # Assert
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        print(f"\nPagination Performance ({num_pages} pages):")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")

        assert avg_latency < 300, f"Average pagination latency {avg_latency:.2f}ms too high"

    @pytest.mark.slow
    def test_memory_usage_large_batch(self, large_paper_batch):
        """Test memory usage with large batch operations"""
        import psutil
        from backend.core.embeddings import EmbeddingService

        # Arrange
        embedding_service = EmbeddingService()
        process = psutil.Process()
        batch_size = 500

        # Measure initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Act
        texts = [p["abstract"] for p in large_paper_batch[:batch_size]]
        embeddings = embedding_service.embed_batch(texts, batch_size=32)

        # Measure final memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before

        # Assert
        print(f"\nMemory Usage (batch size {batch_size}):")
        print(f"  Before: {mem_before:.2f} MB")
        print(f"  After: {mem_after:.2f} MB")
        print(f"  Delta: {mem_delta:.2f} MB")

        # Memory increase should be reasonable (< 1GB for 500 embeddings)
        assert mem_delta < 1000, f"Memory increase {mem_delta:.2f} MB too high"

    def test_cache_efficiency_large_batch(self, large_query_batch):
        """Test cache efficiency with large batch of queries"""
        from backend.utils.cache import get_cache

        # Arrange
        cache = get_cache()
        cache.clear()
        num_queries = 100
        queries = large_query_batch[:num_queries]

        # Act - First pass (all cache misses)
        for query in queries:
            cache.set(f"query:{query}", {"result": "mock"})

        # Act - Second pass (all cache hits)
        hits = 0
        for query in queries:
            if cache.get(f"query:{query}") is not None:
                hits += 1

        hit_rate = hits / num_queries

        # Assert
        print(f"\nCache Efficiency ({num_queries} queries):")
        print(f"  Hit rate: {hit_rate * 100:.1f}%")

        assert hit_rate >= 0.95, f"Cache hit rate {hit_rate:.2%} below 95%"

    def test_concurrent_batch_operations(self, large_query_batch):
        """Test concurrent batch operations"""
        from concurrent.futures import ThreadPoolExecutor
        from backend.core.embeddings import EmbeddingService

        # Arrange
        embedding_service = EmbeddingService()
        num_batches = 5
        batch_size = 20
        batches = [large_query_batch[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

        # Act
        start = time.time()
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            futures = [
                executor.submit(embedding_service.embed_batch, batch, batch_size=batch_size)
                for batch in batches
            ]
            results = [f.result() for f in futures]
        elapsed = time.time() - start

        total_embeddings = sum(len(r) for r in results)
        throughput = total_embeddings / elapsed

        # Assert
        print(f"\nConcurrent Batch Operations:")
        print(f"  Total embeddings: {total_embeddings}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} embeddings/s")

        assert throughput >= 30, f"Throughput {throughput:.2f} embeddings/s too low"
