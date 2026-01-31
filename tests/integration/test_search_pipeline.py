"""
Integration tests for Search Pipeline
Tests end-to-end search functionality
"""

import pytest
from typing import List
from backend.core.search_engine import SemanticSearchEngine, SearchParams
from backend.core.endee_client import EndeeVectorDB
from backend.core.embeddings import EmbeddingService


class TestSearchPipeline:
    """Integration tests for semantic search pipeline"""

    @pytest.fixture
    def endee_client(self):
        """Create Endee client for testing"""
        return EndeeVectorDB(url="http://localhost:8080", timeout=30)

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing"""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")

    @pytest.fixture
    def search_engine(self, endee_client, embedding_service):
        """Create search engine instance"""
        return SemanticSearchEngine(
            endee_client=endee_client,
            embedding_service=embedding_service,
            index_name="research_papers_dense",
        )

    def test_semantic_search_end_to_end(self, search_engine):
        """Test complete semantic search flow"""
        # Arrange
        query = "transformer models for natural language processing"
        top_k = 10

        # Act
        results = search_engine.search(query=query, top_k=top_k)

        # Assert
        assert isinstance(results, list)
        assert len(results) <= top_k
        
        # Verify result structure
        if results:
            result = results[0]
            assert hasattr(result, 'id')
            assert hasattr(result, 'title')
            assert hasattr(result, 'similarity')
            assert 0.0 <= result.similarity <= 1.0

    def test_search_with_filters(self, search_engine):
        """Test search with metadata filters"""
        # Arrange
        query = "deep learning"
        filters = {
            "year": {"$gte": 2020, "$lte": 2024},
            "category": {"$in": ["cs.AI", "cs.LG"]}
        }

        # Act
        results = search_engine.search(query=query, top_k=10, filters=filters)

        # Assert
        assert isinstance(results, list)
        
        # Verify filters are applied
        for result in results:
            if result.year:
                assert 2020 <= result.year <= 2024
            if result.category:
                assert result.category in ["cs.AI", "cs.LG"]

    def test_pagination(self, search_engine):
        """Test search with pagination"""
        # Arrange
        query = "machine learning"
        params_page1 = SearchParams(
            query=query,
            page=1,
            page_size=5,
            top_k=20
        )
        params_page2 = SearchParams(
            query=query,
            page=2,
            page_size=5,
            top_k=20
        )

        # Act
        response_page1 = search_engine.advanced_search(params_page1)
        response_page2 = search_engine.advanced_search(params_page2)

        # Assert
        assert response_page1.page == 1
        assert response_page2.page == 2
        assert len(response_page1.results) <= 5
        assert len(response_page2.results) <= 5
        
        # Verify different results on different pages
        page1_ids = {r.id for r in response_page1.results}
        page2_ids = {r.id for r in response_page2.results}
        assert page1_ids != page2_ids  # Different results

    def test_invalid_queries(self, search_engine):
        """Test handling of invalid queries"""
        # Test empty query
        with pytest.raises(Exception):
            search_engine.search(query="", top_k=10)

        # Test invalid top_k
        with pytest.raises(Exception):
            search_engine.search(query="test", top_k=-1)

        # Test invalid top_k (too large)
        with pytest.raises(Exception):
            search_engine.search(query="test", top_k=10000)

    def test_search_performance(self, search_engine):
        """Test search performance metrics"""
        # Arrange
        query = "neural networks"

        # Act
        response = search_engine.advanced_search(
            SearchParams(query=query, top_k=10)
        )

        # Assert
        assert response.query_time_ms > 0
        assert response.query_time_ms < 5000  # Should be under 5 seconds

    def test_minimum_similarity_filter(self, search_engine):
        """Test minimum similarity threshold"""
        # Arrange
        query = "artificial intelligence"
        params = SearchParams(
            query=query,
            top_k=20,
            min_similarity=0.7
        )

        # Act
        response = search_engine.advanced_search(params)

        # Assert
        for result in response.results:
            assert result.similarity >= 0.7

    def test_cache_functionality(self, search_engine):
        """Test that caching works for repeated queries"""
        # Arrange
        query = "computer vision"

        # Act - First query
        results1 = search_engine.search(query=query, top_k=5)
        
        # Act - Second query (should hit cache)
        results2 = search_engine.search(query=query, top_k=5)

        # Assert
        assert len(results1) == len(results2)
        assert results1[0].id == results2[0].id if results1 else True

    def test_rerank_by_recency(self, search_engine):
        """Test re-ranking by recency"""
        # Arrange
        query = "transformers"
        params_normal = SearchParams(query=query, top_k=10, rerank_by_recency=False)
        params_reranked = SearchParams(query=query, top_k=10, rerank_by_recency=True)

        # Act
        response_normal = search_engine.advanced_search(params_normal)
        response_reranked = search_engine.advanced_search(params_reranked)

        # Assert
        assert len(response_normal.results) > 0
        assert len(response_reranked.results) > 0
        
        # Results should potentially be in different order
        # (unless all papers have same year)
