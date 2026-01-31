"""
Input and output validation utilities
Provides validation helpers and custom validators
"""

from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field, validator, root_validator
import re

from backend.utils.exceptions import ValidationError


class QueryValidator(BaseModel):
    """Validator for search queries"""
    
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    
    @validator("query")
    def validate_query(cls, v):
        """Validate query string"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        
        # Remove excessive whitespace
        v = " ".join(v.split())
        
        return v
    
    @validator("filters")
    def validate_filters(cls, v):
        """Validate filter structure"""
        if v is None:
            return v
        
        allowed_operators = ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]
        
        for field, condition in v.items():
            if isinstance(condition, dict):
                for operator in condition.keys():
                    if operator not in allowed_operators:
                        raise ValueError(f"Invalid filter operator: {operator}")
        
        return v


class RAGQueryValidator(BaseModel):
    """Validator for RAG queries"""
    
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    max_sources: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    @validator("query")
    def validate_query(cls, v):
        """Validate query string"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        return v.strip()
    
    @validator("conversation_id")
    def validate_conversation_id(cls, v):
        """Validate conversation ID format"""
        if v is None:
            return v
        
        # UUID format validation
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, v, re.IGNORECASE):
            raise ValueError("Invalid conversation ID format (must be UUID)")
        
        return v


class PaperValidator(BaseModel):
    """Validator for paper metadata"""
    
    title: str = Field(..., min_length=1, max_length=500)
    abstract: str = Field(..., min_length=10, max_length=5000)
    authors: List[str] = Field(..., min_items=1)
    year: Optional[int] = Field(None, ge=1900, le=2100)
    url: Optional[str] = None
    
    @validator("title", "abstract")
    def validate_text_fields(cls, v):
        """Validate text fields"""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        
        return v.strip()
    
    @validator("authors")
    def validate_authors(cls, v):
        """Validate authors list"""
        if not v:
            raise ValueError("At least one author is required")
        
        # Remove empty strings
        v = [author.strip() for author in v if author.strip()]
        
        if not v:
            raise ValueError("At least one valid author is required")
        
        return v
    
    @validator("url")
    def validate_url(cls, v):
        """Validate URL format"""
        if v is None:
            return v
        
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, v, re.IGNORECASE):
            raise ValueError("Invalid URL format")
        
        return v


class ConfigValidator(BaseModel):
    """Validator for configuration"""
    
    groq_api_key: str = Field(..., min_length=10)
    endee_url: str = Field(...)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    cache_size: int = Field(default=1000, ge=0, le=100000)
    
    @validator("groq_api_key")
    def validate_api_key(cls, v):
        """Validate API key format"""
        if not v or v.startswith("your-") or v == "":
            raise ValueError("Invalid or missing Groq API key")
        
        return v
    
    @validator("endee_url")
    def validate_endee_url(cls, v):
        """Validate Endee URL"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Endee URL must start with http:// or https://")
        
        return v.rstrip("/")


# Validation helper functions

def validate_pagination(page: int, page_size: int) -> None:
    """
    Validate pagination parameters
    
    Args:
        page: Page number (1-indexed)
        page_size: Items per page
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if page < 1:
        raise ValidationError(
            message="Page number must be >= 1",
            field="page",
            details={"value": page}
        )
    
    if page_size < 1 or page_size > 100:
        raise ValidationError(
            message="Page size must be between 1 and 100",
            field="page_size",
            details={"value": page_size}
        )


def validate_embedding_dimension(embedding: List[float], expected_dim: int = 384) -> None:
    """
    Validate embedding dimension
    
    Args:
        embedding: Embedding vector
        expected_dim: Expected dimension
        
    Raises:
        ValidationError: If dimension is incorrect
    """
    if len(embedding) != expected_dim:
        raise ValidationError(
            message=f"Embedding dimension must be {expected_dim}",
            field="embedding",
            details={
                "expected": expected_dim,
                "actual": len(embedding)
            }
        )


def validate_score_range(score: float, min_score: float = 0.0, max_score: float = 1.0) -> None:
    """
    Validate score is within range
    
    Args:
        score: Score value
        min_score: Minimum allowed score
        max_score: Maximum allowed score
        
    Raises:
        ValidationError: If score is out of range
    """
    if not min_score <= score <= max_score:
        raise ValidationError(
            message=f"Score must be between {min_score} and {max_score}",
            field="score",
            details={
                "value": score,
                "min": min_score,
                "max": max_score
            }
        )


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Raises:
        ValidationError: If input is invalid
    """
    if not text:
        raise ValidationError(message="Input cannot be empty")
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Check length
    if len(text) > max_length:
        raise ValidationError(
            message=f"Input exceeds maximum length of {max_length} characters",
            details={"length": len(text), "max_length": max_length}
        )
    
    return text


def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> None:
    """
    Validate JSON structure has required fields
    
    Args:
        data: JSON data
        required_fields: List of required field names
        
    Raises:
        ValidationError: If required fields are missing
    """
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValidationError(
            message="Missing required fields",
            details={"missing_fields": missing_fields}
        )


# Output validation

def validate_search_results(results: List[Dict[str, Any]]) -> None:
    """
    Validate search results structure
    
    Args:
        results: Search results
        
    Raises:
        ValidationError: If results are invalid
    """
    for i, result in enumerate(results):
        required_fields = ["id", "score"]
        missing = [f for f in required_fields if f not in result]
        
        if missing:
            raise ValidationError(
                message=f"Invalid search result at index {i}",
                details={"missing_fields": missing}
            )
        
        # Validate score
        if not isinstance(result["score"], (int, float)):
            raise ValidationError(
                message=f"Invalid score type at index {i}",
                details={"type": type(result["score"]).__name__}
            )


def validate_rag_response(response: Dict[str, Any]) -> None:
    """
    Validate RAG response structure
    
    Args:
        response: RAG response
        
    Raises:
        ValidationError: If response is invalid
    """
    required_fields = ["answer", "sources"]
    validate_json_structure(response, required_fields)
    
    if not isinstance(response["sources"], list):
        raise ValidationError(
            message="Sources must be a list",
            field="sources"
        )
