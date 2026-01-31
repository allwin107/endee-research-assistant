"""
Unit tests for Embedding Service
"""

import pytest
from backend.core.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService"""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance"""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")

    def test_embedding_generation(self, embedding_service):
        """Test single text embedding generation"""
        text = "This is a test sentence"
        embedding = embedding_service.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embedding_dimension(self, embedding_service):
        """Test embedding dimension"""
        dimension = embedding_service.get_embedding_dimension()
        assert dimension == 384

    def test_batch_processing(self, embedding_service):
        """Test batch embedding generation"""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedding_service.embed_batch(texts, batch_size=2)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_cache_functionality(self, embedding_service):
        """Test embedding caching"""
        text = "Cached text"
        
        # First call
        emb1 = embedding_service.embed_text(text)
        
        # Second call (should be cached)
        emb2 = embedding_service.embed_text(text)
        
        assert emb1 == emb2
