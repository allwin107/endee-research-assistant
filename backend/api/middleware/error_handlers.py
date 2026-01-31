"""
Global exception handlers for FastAPI
Provides centralized error handling with logging and metrics
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
import structlog

from backend.utils.exceptions import (
    BaseAPIException,
    create_error_response,
    ValidationError,
)
from backend.utils.logger import get_request_id
from backend.utils.monitoring import get_error_tracker, get_metrics_collector

logger = structlog.get_logger(__name__)


async def base_api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """
    Handle custom API exceptions
    
    Args:
        request: FastAPI request
        exc: Custom exception
        
    Returns:
        JSON error response
    """
    request_id = get_request_id()
    
    # Log error
    logger.error(
        "api_exception",
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method,
        request_id=request_id,
        details=exc.details,
    )
    
    # Track error metrics
    error_tracker = get_error_tracker()
    error_tracker.track_error(
        exc,
        context=f"{request.method} {request.url.path}",
        severity="error" if exc.status_code >= 500 else "warning",
    )
    
    # Create response
    response_data = create_error_response(exc, request_id=request_id)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle Pydantic validation errors
    
    Args:
        request: FastAPI request
        exc: Validation error
        
    Returns:
        JSON error response
    """
    request_id = get_request_id()
    
    # Extract validation errors
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })
    
    # Log validation error
    logger.warning(
        "validation_error",
        path=request.url.path,
        method=request.method,
        request_id=request_id,
        errors=errors,
    )
    
    # Track error
    metrics = get_metrics_collector()
    metrics.increment_counter(
        "validation_errors_total",
        labels={"endpoint": request.url.path}
    )
    
    # Create response
    response_data = {
        "error": "VALIDATION_ERROR",
        "message": "Request validation failed",
        "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "request_id": request_id,
        "details": {
            "errors": errors,
        },
        "suggestion": "Please check your request parameters and try again",
    }
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions
    
    Args:
        request: FastAPI request
        exc: Exception
        
    Returns:
        JSON error response
    """
    request_id = get_request_id()
    
    # Log error with traceback
    logger.exception(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        request_id=request_id,
    )
    
    # Track error
    error_tracker = get_error_tracker()
    error_tracker.track_error(
        exc,
        context=f"{request.method} {request.url.path}",
        severity="critical",
    )
    
    # Create response (don't expose internal details in production)
    response_data = {
        "error": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred. Please try again later.",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "request_id": request_id,
        "details": {},
        "suggestion": "If the problem persists, please contact support with the request ID",
    }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data,
    )


def register_exception_handlers(app):
    """
    Register all exception handlers with FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    # Custom API exceptions
    app.add_exception_handler(BaseAPIException, base_api_exception_handler)
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(PydanticValidationError, validation_exception_handler)
    
    # Generic exceptions (catch-all)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("exception_handlers_registered")
