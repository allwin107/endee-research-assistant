"""
RAG (Retrieval-Augmented Generation) endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import structlog

from backend.api.models.requests import RAGRequest
from backend.api.models.responses import RAGResponse, Conversation

logger = structlog.get_logger()

router = APIRouter()


from functools import lru_cache
from fastapi import Depends
from backend.config import get_settings
from backend.core.endee_client import EndeeVectorDB
from backend.core.embeddings import EmbeddingService
from backend.core.groq_client import GroqClient
from backend.core.rag_pipeline import RAGPipeline

@lru_cache()
def get_rag_pipeline():
    settings = get_settings()
    
    endee = EndeeVectorDB(settings.ENDEE_URL)
    
    embeddings = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL
    )
    
    groq = GroqClient(
        api_key=settings.GROQ_API_KEY, 
        model=settings.GROQ_MODEL
    )
    
    return RAGPipeline(
        endee_client=endee,
        embedding_service=embeddings,
        groq_client=groq,
        index_name="research_papers_dense"
    )

@router.post("/ask", response_model=RAGResponse)
async def ask_question(
    request: RAGRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Ask a question using RAG pipeline
    
    Args:
        request: RAGRequest with question and optional conversation_id
        rag_pipeline: Injected RAG pipeline service

    Returns:
        RAGResponse with answer, sources, and metadata
    """
    try:
        logger.info("rag_ask_request", question=request.question, conversation_id=request.conversation_id)

        result = rag_pipeline.ask(
            question=request.question,
            conversation_id=request.conversation_id,
            top_k=request.top_k or 5,
            temperature=request.temperature or 0.7,
            filter_ids=request.filter_ids
        )

        return RAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id=result["conversation_id"],
            tokens_used=result["tokens_used"],
            query_time_ms=0.0, # We could measure this if needed
        )

    except Exception as e:
        logger.error("rag_ask_failed", error=str(e), question=request.question)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/continue", response_model=RAGResponse)
async def continue_conversation(
    request: RAGRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Continue an existing conversation

    Args:
        request: RAGRequest with question and conversation_id
        rag_pipeline: Injected RAG pipeline service

    Returns:
        RAGResponse with answer maintaining conversation context
    """
    if not request.conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    try:
        # The rag_pipeline.ask method already handles history internally if conversation_id is provided
        return await ask_question(request, rag_pipeline)

    except Exception as e:
        logger.error("continue_conversation_failed", error=str(e), conversation_id=request.conversation_id)
        raise HTTPException(status_code=500, detail=f"Failed to continue conversation: {str(e)}")


@router.get("/conversation/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: str,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Retrieve conversation history from the in-memory store

    Args:
        conversation_id: Conversation identifier
        rag_pipeline: Injected RAG pipeline service

    Returns:
        Conversation object with message history
    """
    try:
        history = rag_pipeline.get_conversation(conversation_id)
        
        if not history:
            # Return empty conversation if not found
            return Conversation(
                id=conversation_id, 
                messages=[], 
                created_at="N/A", 
                updated_at="N/A"
            )

        # Convert simple dicts to Message models
        from backend.api.models.responses import Message
        from datetime import datetime
        
        formatted_messages = []
        for msg in history:
            formatted_messages.append(Message(
                role=msg["role"],
                content=msg["content"],
                timestamp=datetime.now().isoformat(), # We don't store time in pipeline list yet
                sources=None
            ))

        return Conversation(
            id=conversation_id,
            messages=formatted_messages,
            created_at="N/A",
            updated_at="N/A"
        )

    except Exception as e:
        logger.error("get_conversation_failed", error=str(e), conversation_id=conversation_id)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Delete a conversation from the in-memory store

    Args:
        conversation_id: Conversation identifier
        rag_pipeline: Injected RAG pipeline service

    Returns:
        Success message
    """
    try:
        success = rag_pipeline.delete_conversation(conversation_id)
        return {"success": success, "conversation_id": conversation_id}

    except Exception as e:
        logger.error("delete_conversation_failed", error=str(e), conversation_id=conversation_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")
