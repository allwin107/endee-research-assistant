"""
Auto-completion API endpoints
Provides query suggestions for search
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List
import structlog

from backend.core.query_completion import QueryCompletionEngine
from backend.core.embeddings import EmbeddingService
from backend.core.endee_client import EndeeVectorDB
from backend.utils.exceptions import SearchError
from backend.utils.monitoring import PerformanceTracker

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["autocomplete"])

# Initialize services (singleton pattern)
_completion_engine: QueryCompletionEngine = None


def get_completion_engine() -> QueryCompletionEngine:
    """Get or create query completion engine"""
    global _completion_engine
    
    if _completion_engine is None:
        from backend.config import get_settings
        settings = get_settings()
        endee_client = EndeeVectorDB(url=settings.ENDEE_URL)
        embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
        _completion_engine = QueryCompletionEngine(
            endee_client=endee_client,
            embedding_service=embedding_service,
        )
    
    return _completion_engine


class AutocompleteResponse(BaseModel):
    """Auto-completion response model"""
    suggestions: List[str] = Field(..., description="List of query suggestions")
    count: int = Field(..., description="Number of suggestions returned")


class TrackQueryRequest(BaseModel):
    """Track query request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="Query to track")


@router.get("", response_model=AutocompleteResponse)
@router.get("/", response_model=AutocompleteResponse)
async def autocomplete(
    q: str = Query(..., min_length=1, max_length=100, description="Partial query string"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of suggestions"),
    min_similarity: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold"),
):
    """
    Get query auto-completion suggestions
    
    Returns suggestions based on:
    - Semantic similarity to popular queries
    - Query frequency
    - Recency (with decay)
    
    Args:
        q: Partial query string
        limit: Maximum number of suggestions (1-20)
        min_similarity: Minimum similarity threshold (0.0-1.0)
        
    Returns:
        List of suggested query completions
        
    Example:
        GET /api/v1/search/autocomplete?q=machine+learn&limit=5
    """
    with PerformanceTracker("autocomplete_api", labels={"limit": str(limit)}):
        try:
            logger.info("autocomplete_request", query=q, limit=limit)
            
            # Get completion engine
            engine = get_completion_engine()
            
            # Get suggestions
            suggestions = await engine.suggest(
                partial_query=q,
                limit=limit,
                min_similarity=min_similarity,
            )
            
            logger.info(
                "autocomplete_success",
                query=q,
                suggestions_count=len(suggestions)
            )
            
            return AutocompleteResponse(
                suggestions=suggestions,
                count=len(suggestions),
            )
        
        except Exception as e:
            logger.error("autocomplete_failed", query=q, error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Auto-completion failed: {str(e)}"
            )


@router.post("/track")
async def track_query(request: TrackQueryRequest):
    """
    Track query usage for future suggestions
    
    Call this endpoint when a user submits a search query
    to improve future auto-completion suggestions.
    
    Args:
        request: Query tracking request
        
    Returns:
        Success message
        
    Example:
        POST /api/v1/search/autocomplete/track
        {"query": "machine learning"}
    """
    try:
        logger.info("track_query_request", query=request.query)
        
        # Get completion engine
        engine = get_completion_engine()
        
        # Track query
        engine.track_query(request.query)
        
        logger.info("track_query_success", query=request.query)
        
        return {
            "status": "success",
            "message": "Query tracked successfully",
            "query": request.query,
        }
    
    except Exception as e:
        logger.error("track_query_failed", query=request.query, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Query tracking failed: {str(e)}"
        )


@router.get("/stats")
async def get_stats():
    """
    Get auto-completion statistics
    
    Returns statistics about query patterns and cache performance.
    
    Returns:
        Statistics dictionary
    """
    try:
        engine = get_completion_engine()
        stats = engine.get_stats()
        
        return {
            "status": "success",
            "stats": stats,
        }
    
    except Exception as e:
        logger.error("get_stats_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.post("/decay")
async def decay_old_queries(days_threshold: int = Query(90, ge=1, le=365)):
    """
    Decay or remove old queries
    
    Removes queries that haven't been used in the specified number of days.
    This helps keep the suggestion database fresh and relevant.
    
    Args:
        days_threshold: Days after which to remove queries (default: 90)
        
    Returns:
        Number of queries removed
    """
    try:
        logger.info("decay_queries_request", threshold_days=days_threshold)
        
        engine = get_completion_engine()
        removed_count = engine.decay_old_queries(days_threshold)
        
        logger.info("decay_queries_success", removed_count=removed_count)
        
        return {
            "status": "success",
            "removed_count": removed_count,
            "threshold_days": days_threshold,
        }
    
    except Exception as e:
        logger.error("decay_queries_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Query decay failed: {str(e)}"
        )
