"""
FastAPI Backend for AI Research Assistant
Main application entry point
"""

import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from backend.config import get_settings
from backend.api.routes import search, rag, health, autocomplete
from backend.api.routes.prometheus import router as prometheus_router
from backend.api.middleware.error_handlers import register_exception_handlers
from backend.utils.logger import set_request_id, clear_request_id, get_logger

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="AI Research Assistant API",
    description="Production-ready API for semantic search, RAG, and recommendations using Endee Vector Database",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add request ID to all requests"""
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    clear_request_id()
    return response


# Register exception handlers
register_exception_handlers(app)


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])

app.include_router(prometheus_router, tags=["Monitoring"])
app.include_router(autocomplete.router, prefix="/api/v1/search/autocomplete", tags=["Autocomplete"])





@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    settings = get_settings()
    logger.info("application_startup", env=settings.LOG_LEVEL)
    
    # Initialize Memory Monitor
    from backend.utils.memory import get_memory_optimizer
    # Set limit based on container/machine logic ideally
    mem_opt = get_memory_optimizer()
    mem_opt.start_monitoring()
    # Initialize connections, load models, etc.
    
    # Seed data if empty (Background task ideally, but for now await it)
    try:
        from backend.utils.seeding import seed_data_if_empty
        await seed_data_if_empty()
    except Exception as e:
        logger.error("seeding_failed", error=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("application_shutdown")
    
    # Stop Memory Monitor
    from backend.utils.memory import get_memory_optimizer
    get_memory_optimizer().stop_monitoring()
    # Close connections, cleanup resources


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
