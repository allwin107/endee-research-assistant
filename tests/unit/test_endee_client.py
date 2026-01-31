"""
Unit tests for Endee Client
Tests index creation, vector operations, and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.core.endee_client import EndeeVectorDB


class TestEndeeVectorDB:
    """Test cases for EndeeVectorDB class"""

    @pytest.fixture
    def endee_client(self):
        """Create EndeeVectorDB instance for testing"""
        return EndeeVectorDB(url="http://localhost:8080", timeout=30)

    @pytest.fixture
    def mock_endee_sdk(self):
        """Mock Endee SDK client"""
        with patch('backend.core.endee_client.EndeeClient') as mock:
            yield mock

    def test_index_creation(self, endee_client):
        """Test creating a dense index"""
        # Arrange
        index_name = "test_index"
        dimension = 384
        space_type = "cosine"
        precision = "INT8D"

        # Act
        result = endee_client.create_dense_index(
            name=index_name,
            dimension=dimension,
            space_type=space_type,
            precision=precision
        )

        # Assert
        assert result is True

    def test_hybrid_index_creation(self, endee_client):
        """Test creating a hybrid index"""
        # Arrange
        index_name = "test_hybrid_index"
        dimension = 384
        sparse_dim = 30000

        # Act
        result = endee_client.create_hybrid_index(
            name=index_name,
            dimension=dimension,
            sparse_dim=sparse_dim,
            space_type="cosine",
            precision="INT8D"
        )

        # Assert
        assert result is True

    def test_vector_upsert(self, endee_client):
        """Test upserting vectors to an index"""
        # Arrange
        index_name = "test_index"
        vectors_batch = [
            {
                "id": "vec_1",
                "vector": [0.1] * 384,
                "meta": {"title": "Test Paper"},
                "filter": {"year": 2024}
            },
            {
                "id": "vec_2",
                "vector": [0.2] * 384,
                "meta": {"title": "Another Paper"},
                "filter": {"year": 2023}
            }
        ]

        # Act
        result = endee_client.upsert_vectors(
            index_name=index_name,
            vectors_batch=vectors_batch
        )

        # Assert
        assert result is True

    def test_search_operations(self, endee_client):
        """Test dense vector search"""
        # Arrange
        index_name = "test_index"
        query_vector = [0.1] * 384
        top_k = 10
        filters = {"year": {"$gte": 2020}}

        # Act
        results = endee_client.search_dense(
            index_name=index_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters
        )

        # Assert
        assert isinstance(results, list)

    def test_hybrid_search(self, endee_client):
        """Test hybrid search with dense and sparse vectors"""
        # Arrange
        index_name = "test_hybrid_index"
        dense_vector = [0.1] * 384
        sparse_vector = {0: 0.5, 100: 0.3, 500: 0.8}
        top_k = 5
        alpha = 0.7
        beta = 0.3

        # Act
        results = endee_client.search_hybrid(
            index_name=index_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            alpha=alpha,
            beta=beta
        )

        # Assert
        assert isinstance(results, list)

    def test_delete_index(self, endee_client):
        """Test deleting an index"""
        # Arrange
        index_name = "test_index"

        # Act
        result = endee_client.delete_index(index_name)

        # Assert
        assert result is True

    def test_get_index_stats(self, endee_client):
        """Test retrieving index statistics"""
        # Arrange
        index_name = "test_index"

        # Act
        stats = endee_client.get_index_stats(index_name)

        # Assert
        assert isinstance(stats, dict)
        assert "name" in stats
        assert stats["name"] == index_name

    def test_error_handling(self, endee_client):
        """Test error handling for invalid operations"""
        # Arrange
        invalid_index = ""

        # Act & Assert
        with pytest.raises(Exception):
            # This should raise an error due to empty index name
            endee_client.create_dense_index(
                name=invalid_index,
                dimension=384,
                space_type="cosine",
                precision="INT8D"
            )

    def test_connection_initialization(self):
        """Test EndeeVectorDB initialization"""
        # Arrange
        url = "http://test-endee:8080"
        timeout = 60

        # Act
        client = EndeeVectorDB(url=url, timeout=timeout)

        # Assert
        assert client.url == url
        assert client.timeout == timeout

    def test_search_with_no_filters(self, endee_client):
        """Test search without metadata filters"""
        # Arrange
        index_name = "test_index"
        query_vector = [0.5] * 384
        top_k = 20

        # Act
        results = endee_client.search_dense(
            index_name=index_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=None
        )

        # Assert
        assert isinstance(results, list)
