"""
Custom exceptions for the AI Research Assistant
Provides specific exception types for different error scenarios
"""

from typing import Optional, Dict, Any


class BaseAPIException(Exception):
    """
    Base exception for all API errors
    
    Attributes:
        message: Error message
        status_code: HTTP status code
        error_code: Internal error code
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response"""
        return {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


# Service-specific exceptions

class EndeeConnectionError(BaseAPIException):
    """Raised when unable to connect to Endee vector database"""

    def __init__(
        self,
        message: str = "Failed to connect to Endee vector database",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=503,
            error_code="ENDEE_CONNECTION_ERROR",
            details=details,
        )


class EndeeIndexError(BaseAPIException):
    """Raised when Endee index operation fails"""

    def __init__(
        self,
        message: str = "Endee index operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="ENDEE_INDEX_ERROR",
            details=details,
        )


class EmbeddingGenerationError(BaseAPIException):
    """Raised when embedding generation fails"""

    def __init__(
        self,
        message: str = "Failed to generate embeddings",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="EMBEDDING_GENERATION_ERROR",
            details=details,
        )


class GroqAPIError(BaseAPIException):
    """Raised when Groq API call fails"""

    def __init__(
        self,
        message: str = "Groq API request failed",
        status_code: int = 503,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_code="GROQ_API_ERROR",
            details=details,
        )


class GroqRateLimitError(GroqAPIError):
    """Raised when Groq API rate limit is exceeded"""

    def __init__(
        self,
        message: str = "Groq API rate limit exceeded",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=429,
            details=details,
        )
        self.error_code = "GROQ_RATE_LIMIT_ERROR"


class RAGPipelineError(BaseAPIException):
    """Raised when RAG pipeline fails"""

    def __init__(
        self,
        message: str = "RAG pipeline execution failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="RAG_PIPELINE_ERROR",
            details=details,
        )


class SearchError(BaseAPIException):
    """Raised when search operation fails"""

    def __init__(
        self,
        message: str = "Search operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="SEARCH_ERROR",
            details=details,
        )


class RecommendationError(BaseAPIException):
    """Raised when recommendation generation fails"""

    def __init__(
        self,
        message: str = "Recommendation generation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="RECOMMENDATION_ERROR",
            details=details,
        )


# Validation exceptions

class ValidationError(BaseAPIException):
    """Raised when input validation fails"""

    def __init__(
        self,
        message: str = "Input validation failed",
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if field:
            details = details or {}
            details["field"] = field
        
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class ConfigurationError(BaseAPIException):
    """Raised when configuration is invalid"""

    def __init__(
        self,
        message: str = "Invalid configuration",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details=details,
        )


# Resource exceptions

class ResourceNotFoundError(BaseAPIException):
    """Raised when requested resource is not found"""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"{resource_type} with ID '{resource_id}' not found"
        details = details or {}
        details.update({
            "resource_type": resource_type,
            "resource_id": resource_id,
        })
        
        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details=details,
        )


class ResourceExistsError(BaseAPIException):
    """Raised when resource already exists"""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"{resource_type} with ID '{resource_id}' already exists"
        details = details or {}
        details.update({
            "resource_type": resource_type,
            "resource_id": resource_id,
        })
        
        super().__init__(
            message=message,
            status_code=409,
            error_code="RESOURCE_EXISTS",
            details=details,
        )


# Cache exceptions

class CacheError(BaseAPIException):
    """Raised when cache operation fails"""

    def __init__(
        self,
        message: str = "Cache operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CACHE_ERROR",
            details=details,
        )


# Timeout exceptions

class TimeoutError(BaseAPIException):
    """Raised when operation times out"""

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"Operation '{operation}' timed out after {timeout_seconds}s"
        details = details or {}
        details.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds,
        })
        
        super().__init__(
            message=message,
            status_code=504,
            error_code="TIMEOUT_ERROR",
            details=details,
        )


# Circuit breaker exception

class CircuitBreakerOpenError(BaseAPIException):
    """Raised when circuit breaker is open"""

    def __init__(
        self,
        service: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"Service '{service}' is temporarily unavailable (circuit breaker open)"
        details = details or {}
        details["service"] = service
        details["suggestion"] = "Please try again later"
        
        super().__init__(
            message=message,
            status_code=503,
            error_code="CIRCUIT_BREAKER_OPEN",
            details=details,
        )


# Helper function to create user-friendly error responses

def create_error_response(
    exception: Exception,
    request_id: Optional[str] = None,
    include_traceback: bool = False,
) -> Dict[str, Any]:
    """
    Create a user-friendly error response
    
    Args:
        exception: The exception that occurred
        request_id: Optional request ID for tracking
        include_traceback: Whether to include traceback (dev only)
        
    Returns:
        Error response dictionary
    """
    if isinstance(exception, BaseAPIException):
        response = exception.to_dict()
    else:
        response = {
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "status_code": 500,
            "details": {},
        }
    
    if request_id:
        response["request_id"] = request_id
    
    if include_traceback:
        import traceback
        response["traceback"] = traceback.format_exc()
    
    # Add helpful suggestions based on error type
    if isinstance(exception, EndeeConnectionError):
        response["suggestion"] = "Check if Endee service is running and accessible"
    elif isinstance(exception, GroqRateLimitError):
        response["suggestion"] = "Please wait a moment before retrying"
    elif isinstance(exception, ValidationError):
        response["suggestion"] = "Please check your input parameters"
    elif isinstance(exception, TimeoutError):
        response["suggestion"] = "The operation took too long. Try with a smaller dataset or simpler query"
    
    return response
