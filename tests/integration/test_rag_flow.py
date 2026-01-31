"""
Integration tests for RAG Flow
Tests end-to-end RAG question answering
"""

import pytest
from unittest.mock import Mock, patch
from backend.core.rag_pipeline import RAGPipeline
from backend.core.endee_client import EndeeVectorDB
from backend.core.embeddings import EmbeddingService
from backend.core.groq_client import GroqClient


class TestRAGFlow:
    """Integration tests for RAG pipeline"""

    @pytest.fixture
    def endee_client(self):
        """Create Endee client for testing"""
        return EndeeVectorDB(url="http://localhost:8080")

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing"""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")

    @pytest.fixture
    def mock_groq_client(self):
        """Create mocked Groq client"""
        mock_client = Mock(spec=GroqClient)
        
        # Mock generate response
        mock_client.generate.return_value = {
            "answer": "Transformers are neural network architectures based on self-attention mechanisms [Attention Is All You Need, 2017].",
            "tokens_used": 150
        }
        
        return mock_client

    @pytest.fixture
    def rag_pipeline(self, endee_client, embedding_service, mock_groq_client):
        """Create RAG pipeline instance"""
        return RAGPipeline(
            endee_client=endee_client,
            embedding_service=embedding_service,
            groq_client=mock_groq_client,
            index_name="research_papers_dense",
        )

    def test_single_question(self, rag_pipeline):
        """Test asking a single question"""
        # Arrange
        question = "What are transformers in deep learning?"

        # Act
        response = rag_pipeline.ask(
            question=question,
            conversation_id=None,
            top_k=5,
        )

        # Assert
        assert response is not None
        assert hasattr(response, 'answer')
        assert hasattr(response, 'sources')
        assert hasattr(response, 'conversation_id')
        assert len(response.answer) > 0

    def test_multi_turn_conversation(self, rag_pipeline):
        """Test multi-turn conversation"""
        # Arrange
        question1 = "What are transformers?"
        question2 = "How do they differ from RNNs?"

        # Act - First question
        response1 = rag_pipeline.ask(
            question=question1,
            conversation_id=None,
            top_k=5,
        )
        
        conversation_id = response1.conversation_id

        # Act - Follow-up question
        response2 = rag_pipeline.continue_conversation(
            conversation_id=conversation_id,
            question=question2,
        )

        # Assert
        assert response1.conversation_id == response2.conversation_id
        assert len(response2.answer) > 0
        
        # Verify conversation history
        conversation = rag_pipeline.get_conversation(conversation_id)
        assert conversation is not None
        assert len(conversation.messages) >= 4  # 2 questions + 2 answers

    def test_citation_extraction(self, rag_pipeline):
        """Test that citations are properly extracted"""
        # Arrange
        question = "Explain attention mechanisms"

        # Act
        response = rag_pipeline.ask(
            question=question,
            top_k=5,
        )

        # Assert
        assert hasattr(response, 'sources')
        assert isinstance(response.sources, list)
        
        # Verify source structure
        if response.sources:
            source = response.sources[0]
            assert hasattr(source, 'title')
            assert hasattr(source, 'year')

    def test_context_window_management(self, rag_pipeline):
        """Test context window management for long conversations"""
        # Arrange
        conversation_id = None
        questions = [
            "What is deep learning?",
            "What are neural networks?",
            "What is backpropagation?",
            "What are activation functions?",
            "What is gradient descent?",
        ]

        # Act - Ask multiple questions
        for question in questions:
            if conversation_id is None:
                response = rag_pipeline.ask(question=question)
                conversation_id = response.conversation_id
            else:
                response = rag_pipeline.continue_conversation(
                    conversation_id=conversation_id,
                    question=question,
                )

        # Assert
        conversation = rag_pipeline.get_conversation(conversation_id)
        
        # Verify conversation history is managed
        assert conversation is not None
        assert len(conversation.messages) > 0
        
        # Context should be limited (not storing infinite history)
        # Exact limit depends on implementation

    def test_empty_context_handling(self, rag_pipeline, mock_groq_client):
        """Test handling when no relevant context is found"""
        # Arrange
        question = "What is the meaning of life?"  # Unlikely to find in research papers
        
        # Mock Groq to return appropriate response
        mock_groq_client.generate.return_value = {
            "answer": "I don't have enough information in the provided context to answer this question.",
            "tokens_used": 50
        }

        # Act
        response = rag_pipeline.ask(question=question, top_k=5)

        # Assert
        assert response is not None
        assert len(response.answer) > 0

    def test_conversation_deletion(self, rag_pipeline):
        """Test deleting a conversation"""
        # Arrange
        response = rag_pipeline.ask(question="Test question")
        conversation_id = response.conversation_id

        # Act
        rag_pipeline.delete_conversation(conversation_id)

        # Assert
        # Attempting to get deleted conversation should return None or raise error
        conversation = rag_pipeline.get_conversation(conversation_id)
        assert conversation is None

    def test_temperature_control(self, rag_pipeline):
        """Test temperature parameter affects generation"""
        # Arrange
        question = "Explain neural networks"

        # Act
        response_low_temp = rag_pipeline.ask(
            question=question,
            top_k=5,
            temperature=0.1,
        )
        
        response_high_temp = rag_pipeline.ask(
            question=question,
            top_k=5,
            temperature=0.9,
        )

        # Assert
        assert response_low_temp is not None
        assert response_high_temp is not None
        # Both should return valid answers

    def test_source_tracking(self, rag_pipeline):
        """Test that sources are properly tracked"""
        # Arrange
        question = "What are convolutional neural networks?"

        # Act
        response = rag_pipeline.ask(question=question, top_k=5)

        # Assert
        assert hasattr(response, 'sources')
        assert len(response.sources) <= 5  # Should not exceed top_k
        
        # Verify each source has required fields
        for source in response.sources:
            assert hasattr(source, 'id')
            assert hasattr(source, 'title')
            assert hasattr(source, 'similarity')
