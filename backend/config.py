"""
Configuration management using Pydantic Settings
Loads environment variables and provides validated configuration
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Groq API Configuration
    GROQ_API_KEY: str = Field(..., description="Groq API key for LLM")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")
    GROQ_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    GROQ_MAX_TOKENS: int = Field(default=2048, gt=0)

    # Endee Vector Database
    ENDEE_URL: str = Field(default="http://localhost:8080", description="Endee server URL")
    ENDEE_TIMEOUT: int = Field(default=30, description="Endee request timeout in seconds")

    # Application Settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    CACHE_SIZE: int = Field(default=1000, description="LRU cache size")
    MAX_WORKERS: int = Field(default=4, description="Max worker threads")
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")

    # Embedding Model
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    EMBEDDING_DIMENSION: int = Field(default=384, description="Embedding dimension")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, description="Batch size for embeddings")

    # Search Settings
    DEFAULT_TOP_K: int = Field(default=10, description="Default number of results")
    MAX_TOP_K: int = Field(default=100, description="Maximum number of results")

    # RAG Settings
    RAG_CONTEXT_WINDOW: int = Field(default=8000, description="RAG context window size")
    RAG_MAX_SOURCES: int = Field(default=5, description="Maximum sources for RAG")
    RAG_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)

    # Performance
    ENABLE_CACHE: bool = Field(default=True, description="Enable caching")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Returns singleton Settings object
    """
    return Settings()
