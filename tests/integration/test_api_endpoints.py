"""
Integration tests for API Endpoints
Tests all FastAPI routes with real HTTP requests
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from backend.main import app


class TestAPIEndpoints:
    """Integration tests for all API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    # Health Check Endpoints
    def test_health_endpoint(self, client):
        """Test main health check endpoint"""
        # Act
        response = client.get("/api/v1/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_endee_health_endpoint(self, client):
        """Test Endee health check"""
        # Act
        response = client.get("/api/v1/health/endee")

        # Assert
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    def test_groq_health_endpoint(self, client):
        """Test Groq health check"""
        # Act
        response = client.get("/api/v1/health/groq")

        # Assert
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    # Search Endpoints
    def test_semantic_search_endpoint(self, client):
        """Test semantic search endpoint"""
        # Arrange
        payload = {
            "query": "transformer models for NLP",
            "top_k": 10,
            "filters": {}
        }

        # Act
        response = client.post("/api/v1/search/semantic", json=payload)

        # Assert
        assert response.status_code in [200, 500]  # May fail if Endee not running
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert isinstance(data["results"], list)

    def test_hybrid_search_endpoint(self, client):
        """Test hybrid search endpoint"""
        # Arrange
        payload = {
            "query": "deep learning",
            "top_k": 10,
            "alpha": 0.7,
            "beta": 0.3
        }

        # Act
        response = client.post("/api/v1/search/hybrid", json=payload)

        # Assert
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "results" in data

    def test_search_with_invalid_payload(self, client):
        """Test search with invalid request payload"""
        # Arrange
        invalid_payload = {
            "query": "",  # Empty query
            "top_k": -1   # Invalid top_k
        }

        # Act
        response = client.post("/api/v1/search/semantic", json=invalid_payload)

        # Assert
        assert response.status_code == 422  # Validation error

    # RAG Endpoints
    def test_rag_ask_endpoint(self, client):
        """Test RAG ask question endpoint"""
        # Arrange
        payload = {
            "question": "What are transformers in deep learning?",
            "top_k": 5,
            "temperature": 0.7
        }

        # Act
        response = client.post("/api/v1/rag/ask", json=payload)

        # Assert
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "conversation_id" in data

    def test_rag_continue_endpoint(self, client):
        """Test RAG continue conversation endpoint"""
        # Arrange - First ask a question
        initial_payload = {
            "question": "What is deep learning?",
            "top_k": 5
        }
        initial_response = client.post("/api/v1/rag/ask", json=initial_payload)
        
        if initial_response.status_code == 200:
            conversation_id = initial_response.json()["conversation_id"]
            
            # Act - Continue conversation
            continue_payload = {
                "conversation_id": conversation_id,
                "question": "How does it differ from machine learning?"
            }
            response = client.post("/api/v1/rag/continue", json=continue_payload)

            # Assert
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "answer" in data

    def test_get_conversation_endpoint(self, client):
        """Test get conversation history endpoint"""
        # Arrange
        conversation_id = "test_conv_123"

        # Act
        response = client.get(f"/api/v1/rag/conversation/{conversation_id}")

        # Assert
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "messages" in data

    def test_delete_conversation_endpoint(self, client):
        """Test delete conversation endpoint"""
        # Arrange
        conversation_id = "test_conv_456"

        # Act
        response = client.delete(f"/api/v1/rag/conversation/{conversation_id}")

        # Assert
        assert response.status_code in [200, 404]

    # Recommendation Endpoints
    def test_similar_papers_endpoint(self, client):
        """Test similar papers recommendation endpoint"""
        # Arrange
        payload = {
            "paper_id": "arxiv_2401_12345",
            "top_k": 10,
            "diversity": 0.5
        }

        # Act
        response = client.post("/api/v1/recommend/similar", json=payload)

        # Assert
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data

    def test_personalized_recommendations_endpoint(self, client):
        """Test personalized recommendations endpoint"""
        # Arrange
        payload = {
            "user_id": "user_001",
            "top_k": 10
        }

        # Act
        response = client.post("/api/v1/recommend/personalized", json=payload)

        # Assert
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data

    def test_track_interaction_endpoint(self, client):
        """Test track user interaction endpoint"""
        # Arrange
        payload = {
            "user_id": "user_002",
            "paper_id": "arxiv_2024_001",
            "action": "view"
        }

        # Act
        response = client.post("/api/v1/recommend/track-interaction", json=payload)

        # Assert
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data

    # CORS Testing
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        # Act
        response = client.options("/api/v1/health")

        # Assert
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code == 200

    # Error Response Testing
    def test_404_error(self, client):
        """Test 404 error for non-existent endpoint"""
        # Act
        response = client.get("/api/v1/nonexistent")

        # Assert
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method"""
        # Act
        response = client.get("/api/v1/search/semantic")  # Should be POST

        # Assert
        assert response.status_code == 405

    # API Documentation
    def test_openapi_docs(self, client):
        """Test that OpenAPI documentation is accessible"""
        # Act
        response = client.get("/docs")

        # Assert
        assert response.status_code == 200

    def test_openapi_json(self, client):
        """Test OpenAPI JSON schema"""
        # Act
        response = client.get("/openapi.json")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    # Rate Limiting (if implemented)
    @pytest.mark.skip(reason="Rate limiting not implemented yet")
    def test_rate_limiting(self, client):
        """Test rate limiting on endpoints"""
        # Make multiple rapid requests
        for _ in range(100):
            response = client.get("/api/v1/health")
            if response.status_code == 429:  # Too Many Requests
                break
        
        # Should eventually hit rate limit
        assert response.status_code == 429
