"""
Enhanced logging configuration with structured logging, rotation, and request tracking
Provides production-ready logging with JSON format, log levels, and component separation
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import structlog
from contextvars import ContextVar

# Context variable for request ID tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get current request ID from context"""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set request ID in context"""
    request_id_var.set(request_id)


def clear_request_id() -> None:
    """Clear request ID from context"""
    request_id_var.set(None)


class RequestIDProcessor:
    """Processor to add request ID to log entries"""

    def __call__(self, logger, method_name, event_dict):
        request_id = get_request_id()
        if request_id:
            event_dict["request_id"] = request_id
        return event_dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_data["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_json: bool = True,
    enable_rotation: bool = True,
) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        enable_json: Enable JSON formatting
        enable_rotation: Enable log rotation
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            RequestIDProcessor(),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if enable_json else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    if enable_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)8s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(console_handler)

    # File handlers with rotation
    if enable_rotation:
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(JSONFormatter() if enable_json else logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s - %(message)s"
        ))
        root_logger.addHandler(app_handler)

        # Error log
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "error.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter() if enable_json else logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s - %(message)s"
        ))
        root_logger.addHandler(error_handler)

        # Performance log
        perf_handler = logging.handlers.RotatingFileHandler(
            log_path / "performance.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.addFilter(lambda record: "performance" in record.name.lower())
        perf_handler.setFormatter(JSONFormatter() if enable_json else logging.Formatter(
            "%(asctime)s - %(message)s"
        ))
        root_logger.addHandler(perf_handler)

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


# Component-specific loggers
def get_api_logger() -> structlog.BoundLogger:
    """Get logger for API routes"""
    return get_logger("api")


def get_core_logger() -> structlog.BoundLogger:
    """Get logger for core services"""
    return get_logger("core")


def get_performance_logger() -> structlog.BoundLogger:
    """Get logger for performance metrics"""
    return get_logger("performance")


def get_security_logger() -> structlog.BoundLogger:
    """Get logger for security events"""
    return get_logger("security")


# Initialize logging on module import
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs"),
    enable_json=os.getenv("LOG_JSON", "true").lower() == "true",
    enable_rotation=os.getenv("LOG_ROTATION", "true").lower() == "true",
)


# Example usage
if __name__ == "__main__":
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("Debug message", extra_field="value")
    logger.info("Info message", user_id="123")
    logger.warning("Warning message", threshold=0.8)
    logger.error("Error message", error_code="E001")
    
    # Test with request ID
    set_request_id("req-12345")
    logger.info("Request processed", duration_ms=150)
    clear_request_id()
    
    # Test exception logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception("Exception occurred", error_type=type(e).__name__)
