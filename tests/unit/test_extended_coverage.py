# Additional unit tests for improved coverage

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestEmbeddingServiceExtended:
    """Extended tests for EmbeddingService to improve coverage"""

    def test_embed_with_gpu(self):
        """Test embedding generation with GPU"""
        from backend.core.embeddings import EmbeddingService
        
        # Mock torch.cuda.is_available to return True
        with patch('torch.cuda.is_available', return_value=True):
            service = EmbeddingService(device='cuda')
            embedding = service.embed("test query")
            
            assert len(embedding) == 384
            assert isinstance(embedding, list)

    def test_embed_batch_empty_list(self):
        """Test embed_batch with empty list"""
        from backend.core.embeddings import EmbeddingService
        
        service = EmbeddingService()
        embeddings = service.embed_batch([])
        
        assert embeddings == []

    def test_embed_batch_single_item(self):
        """Test embed_batch with single item"""
        from backend.core.embeddings import EmbeddingService
        
        service = EmbeddingService()
        embeddings = service.embed_batch(["single text"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    def test_cache_hit(self):
        """Test cache hit for repeated embeddings"""
        from backend.core.embeddings import EmbeddingService
        
        service = EmbeddingService()
        text = "test query for caching"
        
        # First call - cache miss
        embedding1 = service.embed(text)
        
        # Second call - cache hit
        embedding2 = service.embed(text)
        
        assert embedding1 == embedding2


class TestCacheExtended:
    """Extended tests for Cache to improve coverage"""

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration"""
        import time
        from backend.utils.cache import SimpleCache
        
        cache = SimpleCache(max_size=100, default_ttl=1)
        cache.set("key", "value", ttl=1)
        
        # Should exist immediately
        assert cache.get("key") == "value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("key") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        from backend.utils.cache import SimpleCache
        
        cache = SimpleCache(max_size=3)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") is not None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_cache_invalidate_pattern(self):
        """Test pattern-based cache invalidation"""
        from backend.utils.cache import SimpleCache
        
        cache = SimpleCache()
        cache.set("user:123:profile", "data1")
        cache.set("user:123:settings", "data2")
        cache.set("user:456:profile", "data3")
        
        # Invalidate all user:123 keys
        cache.invalidate_pattern("user:123:*")
        
        assert cache.get("user:123:profile") is None
        assert cache.get("user:123:settings") is None
        assert cache.get("user:456:profile") is not None

    def test_cache_stats(self):
        """Test cache statistics tracking"""
        from backend.utils.cache import SimpleCache
        
        cache = SimpleCache()
        cache.clear()
        
        # Generate some cache activity
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit
        
        stats = cache.get_stats()
        
        assert stats["hits"] >= 2
        assert stats["misses"] >= 1
        assert stats["size"] >= 1


class TestPerformanceMonitorExtended:
    """Extended tests for performance monitoring"""

    def test_performance_decorator_with_exception(self):
        """Test performance monitor with exception"""
        from backend.utils.performance import performance_monitor
        
        @performance_monitor("test_function")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()

    def test_track_query_time(self):
        """Test query time tracking"""
        import time
        from backend.utils.performance import track_query_time
        
        with track_query_time("test_query") as timer:
            time.sleep(0.1)
        
        assert timer.elapsed_ms >= 100

    def test_performance_metrics_collection(self):
        """Test performance metrics collection"""
        from backend.utils.performance import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        
        metrics.record_query("search", 150.5)
        metrics.record_query("search", 200.0)
        metrics.record_query("rag", 1500.0)
        
        stats = metrics.get_stats()
        
        assert "search" in stats
        assert stats["search"]["count"] == 2
        assert stats["search"]["avg"] > 0


class TestSearchEngineExtended:
    """Extended tests for SearchEngine"""

    def test_search_with_filters(self):
        """Test search with metadata filters"""
        from backend.core.search_engine import SemanticSearchEngine, SearchParams
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService
        
        endee_client = Mock(spec=EndeeVectorDB)
        embedding_service = Mock(spec=EmbeddingService)
        search_engine = SemanticSearchEngine(endee_client, embedding_service)
        
        # Mock responses
        embedding_service.embed.return_value = [0.1] * 384
        endee_client.search.return_value = []
        
        params = SearchParams(
            query="test",
            filters={"year": {"$gte": 2020}},
            top_k=10
        )
        
        results = search_engine.advanced_search(params)
        
        assert isinstance(results, dict)
        endee_client.search.assert_called_once()

    def test_search_pagination(self):
        """Test search with pagination"""
        from backend.core.search_engine import SemanticSearchEngine, SearchParams
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService
        
        endee_client = Mock(spec=EndeeVectorDB)
        embedding_service = Mock(spec=EmbeddingService)
        search_engine = SemanticSearchEngine(endee_client, embedding_service)
        
        # Mock responses
        embedding_service.embed.return_value = [0.1] * 384
        endee_client.search.return_value = []
        
        params = SearchParams(
            query="test",
            page=2,
            page_size=10,
            top_k=50
        )
        
        results = search_engine.advanced_search(params)
        
        assert isinstance(results, dict)


class TestRAGPipelineExtended:
    """Extended tests for RAG Pipeline"""

    def test_rag_with_empty_context(self):
        """Test RAG when no context is found"""
        from backend.core.rag_pipeline import RAGPipeline
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService
        from backend.core.groq_client import GroqClient
        
        endee_client = Mock(spec=EndeeVectorDB)
        embedding_service = Mock(spec=EmbeddingService)
        groq_client = Mock(spec=GroqClient)
        
        rag = RAGPipeline(endee_client, embedding_service, groq_client)
        
        # Mock empty search results
        embedding_service.embed.return_value = [0.1] * 384
        endee_client.search.return_value = []
        groq_client.generate.return_value = {
            "answer": "I don't have enough information to answer that.",
            "tokens_used": 50
        }
        
        response = rag.ask("What is X?")
        
        assert "answer" in response
        assert response["sources"] == []

    def test_rag_conversation_context(self):
        """Test RAG with conversation context"""
        from backend.core.rag_pipeline import RAGPipeline
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService
        from backend.core.groq_client import GroqClient
        
        endee_client = Mock(spec=EndeeVectorDB)
        embedding_service = Mock(spec=EmbeddingService)
        groq_client = Mock(spec=GroqClient)
        
        rag = RAGPipeline(endee_client, embedding_service, groq_client)
        
        # Mock responses
        embedding_service.embed.return_value = [0.1] * 384
        endee_client.search.return_value = []
        groq_client.generate.return_value = {
            "answer": "Test answer",
            "tokens_used": 100
        }
        
        # First question
        response1 = rag.ask("What is X?")
        conv_id = response1["conversation_id"]
        
        # Follow-up question
        response2 = rag.continue_conversation(conv_id, "Tell me more")
        
        assert response2["conversation_id"] == conv_id


class TestRecommendationEngineExtended:
    """Extended tests for Recommendation Engine"""

    def test_recommendation_diversity(self):
        """Test recommendation diversity filtering"""
        from backend.core.recommendation import RecommendationEngine
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService
        
        endee_client = Mock(spec=EndeeVectorDB)
        embedding_service = Mock(spec=EmbeddingService)
        
        rec_engine = RecommendationEngine(endee_client, embedding_service)
        
        # Mock responses
        endee_client.get_by_id.return_value = {"embedding": [0.1] * 384}
        endee_client.search.return_value = [
            {"id": f"paper_{i}", "score": 0.9 - i*0.1}
            for i in range(20)
        ]
        
        recommendations = rec_engine.get_similar_papers(
            paper_id="test_paper",
            top_k=10,
            diversity=0.8
        )
        
        assert len(recommendations) <= 10

    def test_cold_start_recommendations(self):
        """Test recommendations for new users (cold start)"""
        from backend.core.recommendation import RecommendationEngine
        from backend.core.endee_client import EndeeVectorDB
        from backend.core.embeddings import EmbeddingService
        
        endee_client = Mock(spec=EndeeVectorDB)
        embedding_service = Mock(spec=EmbeddingService)
        
        rec_engine = RecommendationEngine(endee_client, embedding_service)
        
        # Mock empty user history
        endee_client.search.return_value = []
        
        recommendations = rec_engine.get_personalized_recommendations(
            user_id="new_user",
            top_k=10
        )
        
        # Should handle gracefully
        assert isinstance(recommendations, list)
