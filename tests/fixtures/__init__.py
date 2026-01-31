"""
Test fixtures for unit and integration tests
Provides sample data, mock responses, and test utilities
"""

import pytest
from typing import List, Dict, Any
from datetime import datetime


# Sample Papers
@pytest.fixture
def sample_papers() -> List[Dict[str, Any]]:
    """Sample research papers for testing"""
    return [
        {
            "id": "arxiv_1706_03762",
            "title": "Attention Is All You Need",
            "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
            "year": 2017,
            "category": "cs.CL",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            "url": "https://arxiv.org/abs/1706.03762",
            "embedding": [0.1] * 384,  # Mock embedding
        },
        {
            "id": "arxiv_1810_04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": ["Devlin, J.", "Chang, M.", "Lee, K."],
            "year": 2018,
            "category": "cs.CL",
            "abstract": "We introduce a new language representation model called BERT...",
            "url": "https://arxiv.org/abs/1810.04805",
            "embedding": [0.2] * 384,
        },
        {
            "id": "arxiv_2005_14165",
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "authors": ["Brown, T.", "Mann, B.", "Ryder, N."],
            "year": 2020,
            "category": "cs.CL",
            "abstract": "Recent work has demonstrated substantial gains on many NLP tasks...",
            "url": "https://arxiv.org/abs/2005.14165",
            "embedding": [0.3] * 384,
        },
        {
            "id": "arxiv_1512_03385",
            "title": "Deep Residual Learning for Image Recognition",
            "authors": ["He, K.", "Zhang, X.", "Ren, S."],
            "year": 2015,
            "category": "cs.CV",
            "abstract": "Deeper neural networks are more difficult to train...",
            "url": "https://arxiv.org/abs/1512.03385",
            "embedding": [0.4] * 384,
        },
        {
            "id": "arxiv_1412_6980",
            "title": "Adam: A Method for Stochastic Optimization",
            "authors": ["Kingma, D.P.", "Ba, J."],
            "year": 2014,
            "category": "cs.LG",
            "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization...",
            "url": "https://arxiv.org/abs/1412.6980",
            "embedding": [0.5] * 384,
        },
    ]


@pytest.fixture
def sample_paper() -> Dict[str, Any]:
    """Single sample paper for testing"""
    return {
        "id": "arxiv_test_001",
        "title": "Test Paper: A Comprehensive Study",
        "authors": ["Test, A.", "Author, B."],
        "year": 2024,
        "category": "cs.AI",
        "abstract": "This is a test paper for unit testing purposes.",
        "url": "https://arxiv.org/abs/test.001",
        "embedding": [0.1] * 384,
    }


# Mock Endee Responses
@pytest.fixture
def mock_endee_search_response() -> Dict[str, Any]:
    """Mock Endee search response"""
    return {
        "results": [
            {
                "id": "arxiv_1706_03762",
                "score": 0.92,
                "meta": {
                    "title": "Attention Is All You Need",
                    "abstract": "The dominant sequence transduction models...",
                    "url": "https://arxiv.org/abs/1706.03762",
                },
                "filter": {
                    "year": 2017,
                    "category": "cs.CL",
                    "authors": ["Vaswani, A."],
                },
            },
            {
                "id": "arxiv_1810_04805",
                "score": 0.87,
                "meta": {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "abstract": "We introduce a new language representation model...",
                    "url": "https://arxiv.org/abs/1810.04805",
                },
                "filter": {
                    "year": 2018,
                    "category": "cs.CL",
                    "authors": ["Devlin, J."],
                },
            },
        ],
        "total": 2,
        "query_time_ms": 45,
    }


@pytest.fixture
def mock_groq_response() -> Dict[str, Any]:
    """Mock Groq API response"""
    return {
        "answer": "Transformers are neural network architectures based on self-attention mechanisms [Attention Is All You Need, 2017]. They process sequences in parallel rather than sequentially, enabling better parallelization and handling of long-range dependencies.",
        "tokens_used": 245,
        "model": "llama-3.1-70b-versatile",
    }


# Mock Embeddings
@pytest.fixture
def mock_embedding_vector() -> List[float]:
    """Mock 384-dimensional embedding vector"""
    return [0.1] * 384


@pytest.fixture
def mock_embedding_batch() -> List[List[float]]:
    """Mock batch of embedding vectors"""
    return [[0.1 * i] * 384 for i in range(1, 6)]


# User Interaction Data
@pytest.fixture
def sample_user_interactions() -> List[Dict[str, Any]]:
    """Sample user interaction data"""
    return [
        {
            "user_id": "user_001",
            "paper_id": "arxiv_1706_03762",
            "action": "view",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0),
        },
        {
            "user_id": "user_001",
            "paper_id": "arxiv_1810_04805",
            "action": "save",
            "timestamp": datetime(2024, 1, 16, 14, 20, 0),
        },
        {
            "user_id": "user_001",
            "paper_id": "arxiv_2005_14165",
            "action": "cite",
            "timestamp": datetime(2024, 1, 17, 9, 15, 0),
        },
    ]


# Conversation Data
@pytest.fixture
def sample_conversation() -> Dict[str, Any]:
    """Sample RAG conversation"""
    return {
        "conversation_id": "conv_test_123",
        "messages": [
            {
                "role": "user",
                "content": "What are transformers in deep learning?",
                "timestamp": "2024-01-30T10:00:00Z",
            },
            {
                "role": "assistant",
                "content": "Transformers are neural network architectures...",
                "timestamp": "2024-01-30T10:00:05Z",
                "sources": [
                    {
                        "id": "arxiv_1706_03762",
                        "title": "Attention Is All You Need",
                        "similarity": 0.92,
                    }
                ],
            },
            {
                "role": "user",
                "content": "How do they differ from RNNs?",
                "timestamp": "2024-01-30T10:01:00Z",
            },
            {
                "role": "assistant",
                "content": "Unlike RNNs which process sequences sequentially...",
                "timestamp": "2024-01-30T10:01:05Z",
                "sources": [],
            },
        ],
        "created_at": "2024-01-30T10:00:00Z",
    }


# Test Queries
@pytest.fixture
def sample_queries() -> List[str]:
    """Sample search queries"""
    return [
        "transformer models for natural language processing",
        "attention mechanisms in deep learning",
        "computer vision with convolutional neural networks",
        "reinforcement learning for robotics",
        "generative adversarial networks",
    ]


# API Request/Response Fixtures
@pytest.fixture
def api_search_request() -> Dict[str, Any]:
    """Sample API search request"""
    return {
        "query": "transformer models for NLP",
        "top_k": 10,
        "filters": {
            "year": {"$gte": 2017, "$lte": 2024},
            "category": {"$in": ["cs.CL", "cs.AI"]},
        },
    }


@pytest.fixture
def api_rag_request() -> Dict[str, Any]:
    """Sample API RAG request"""
    return {
        "question": "What are transformers in deep learning?",
        "top_k": 5,
        "temperature": 0.7,
    }


@pytest.fixture
def api_recommendation_request() -> Dict[str, Any]:
    """Sample API recommendation request"""
    return {
        "paper_id": "arxiv_1706_03762",
        "top_k": 10,
        "diversity": 0.5,
    }


# Error Fixtures
@pytest.fixture
def api_error_responses() -> Dict[str, Dict[str, Any]]:
    """Sample API error responses"""
    return {
        "400": {"detail": "Invalid request parameters"},
        "404": {"detail": "Resource not found"},
        "422": {
            "detail": [
                {
                    "loc": ["body", "query"],
                    "msg": "field required",
                    "type": "value_error.missing",
                }
            ]
        },
        "500": {"detail": "Internal server error occurred"},
    }


# Performance Test Data
@pytest.fixture
def large_query_batch() -> List[str]:
    """Large batch of queries for performance testing"""
    return [f"test query {i}" for i in range(100)]


@pytest.fixture
def large_paper_batch() -> List[Dict[str, Any]]:
    """Large batch of papers for performance testing"""
    return [
        {
            "id": f"arxiv_test_{i:05d}",
            "title": f"Test Paper {i}",
            "authors": [f"Author {i}"],
            "year": 2020 + (i % 5),
            "category": ["cs.AI", "cs.CL", "cs.CV", "cs.LG"][i % 4],
            "abstract": f"This is test paper {i} for performance testing.",
            "url": f"https://arxiv.org/abs/test.{i}",
            "embedding": [0.01 * i] * 384,
        }
        for i in range(1000)
    ]
