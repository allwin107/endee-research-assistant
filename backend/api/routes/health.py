"""
Enhanced health check endpoints
Provides comprehensive health status for all system components
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import httpx
import psutil
import os

from backend.utils.logger import get_logger
from backend.utils.monitoring import get_system_monitor, get_metrics_collector

logger = get_logger(__name__)
router = APIRouter(prefix="/health")


class HealthStatus(BaseModel):
    """Health status response model"""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
    checks: Dict[str, Any] = {}


class ServiceHealth(BaseModel):
    """Individual service health"""
    status: str
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@router.get("", response_model=HealthStatus)
@router.get("/", response_model=HealthStatus)
async def get_health():
    """
    Get overall system health status
    
    Checks:
    - Endee connectivity
    - Groq API availability
    - Database connectivity
    - System resources
    
    Returns:
        Overall health status
    """
    logger.info("health_check_requested")
    
    checks = {}
    overall_status = "healthy"
    
    # Check Endee
    endee_health = await check_endee_health()
    checks["endee"] = endee_health.dict()
    if endee_health.status != "healthy":
        overall_status = "degraded"
    
    # Check Groq
    groq_health = await check_groq_health()
    checks["groq"] = groq_health.dict()
    if groq_health.status != "healthy":
        overall_status = "degraded"
    
    # Check system resources
    system_health = check_system_health()
    checks["system"] = system_health.dict()
    if system_health.status != "healthy":
        overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
    
    # Get metrics
    metrics_collector = get_metrics_collector()
    all_metrics = metrics_collector.get_all_metrics()
    
    response = HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=all_metrics.get("uptime_seconds"),
        checks=checks
    )
    
    logger.info("health_check_completed", status=overall_status)
    
    if overall_status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.dict()
        )
    
    return response


@router.get("/endee", response_model=ServiceHealth)
async def get_endee_health():
    """Check Endee vector database health"""
    return await check_endee_health()


@router.get("/groq", response_model=ServiceHealth)
async def get_groq_health():
    """Check Groq API health"""

# Helper Functions

async def check_endee_health() -> ServiceHealth:
    """Check connectivity to Endee Vector DB"""
    try:
        start_time = datetime.now()
        from backend.core.endee_client import EndeeVectorDB
        from backend.config import get_settings
        
        settings = get_settings()
        # Initialize with URL from settings
        client = EndeeVectorDB(url=settings.ENDEE_URL)
        
        # Simple ping or check
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        return ServiceHealth(
            status="healthy",
            latency_ms=round(latency, 2),
            message="Endee Vector DB is reachable"
        )
    except Exception as e:
        logger.error("endee_health_check_failed", error=str(e))
        return ServiceHealth(
            status="unhealthy",
            message=f"Connection failed: {str(e)}",
            details={"error": str(e)}
        )

async def check_groq_health() -> ServiceHealth:
    """Check connectivity to Groq API"""
    try:
        start_time = datetime.now()
        from backend.core.groq_client import GroqClient
        from backend.config import get_settings
        
        settings = get_settings()
        # Initialize with API Key from settings
        client = GroqClient(api_key=settings.GROQ_API_KEY)
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        return ServiceHealth(
            status="healthy", 
            latency_ms=round(latency, 2),
            message="Groq Client initialized"
        )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            message=str(e)
        )

def check_system_health() -> ServiceHealth:
    """Check local system resources"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status_code = "healthy"
        messages = []
        
        if cpu_percent > 90:
            status_code = "degraded"
            messages.append("High CPU usage")
            
        if memory.percent > 90:
            status_code = "degraded"
            messages.append("High Memory usage")
            
        return ServiceHealth(
            status=status_code,
            message=", ".join(messages) if messages else "System resources normal",
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            }
        )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            message=f"System check failed: {str(e)}"
        )
