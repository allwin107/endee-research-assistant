"""
Search endpoints - Semantic and Hybrid search
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, List
import structlog
import time

from backend.api.models.requests import SearchRequest, HybridSearchRequest
from backend.api.models.responses import SearchResponse, Paper
from backend.config import get_settings

logger = structlog.get_logger()

router = APIRouter()


from functools import lru_cache
from fastapi import Depends

@lru_cache()
def get_search_engine():
    from backend.core.search_engine import SemanticSearchEngine as SearchEngine
    from backend.core.endee_client import EndeeVectorDB
    from backend.core.embeddings import EmbeddingService
    
    settings = get_settings()
    
    # Initialize dependencies
    endee_client = EndeeVectorDB(url=settings.ENDEE_URL)
    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL
    )
    
    return SearchEngine(
        endee_client=endee_client,
        embedding_service=embedding_service,
        index_name="research_papers_dense", # Default index
        enable_cache=True
    )

@router.post("", response_model=SearchResponse)
@router.post("/", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    search_engine = Depends(get_search_engine)
):
    """
    Perform semantic search on research papers
    """
    # Force cache disabled for debugging
    if search_engine:
        search_engine.enable_cache = False
        if hasattr(search_engine, "cache") and search_engine.cache:
            search_engine.cache.clear()
            
    query = request.query
    original_query = query
    # target_lang = request.target_language or "en" # Removed
    original_query = query


    # 1.5 Live Fetch from arXiv (The "Real-World" implementation)
    try:
        from backend.utils.arxiv import fetch_arxiv_papers
        from backend.utils.semantic_scholar import fetch_semantic_scholar_papers
        import uuid
        import random
        import asyncio
        
        print(f"DEBUG: Starting Live Fetch for '{query}'")
        logger.info("live_fetch_start", query=query)
        
        # Parallel fetch (naive asyncio.gather)
        # Since these are synchronous requests wrapped in async endpoints, we can run them in threads
        # or just call them sequentially for now to be safe and simple
        
        arxiv_papers = fetch_arxiv_papers(query=query, max_results=5)
        semanticscholar_papers = fetch_semantic_scholar_papers(query=query, max_results=5)
        
        # Merge results
        new_papers = arxiv_papers + semanticscholar_papers
        logger.info("live_fetch_complete", arxiv_count=len(arxiv_papers), semanticscholar_count=len(semanticscholar_papers))
        print(f"DEBUG: Fetched {len(new_papers) if new_papers else 0} papers")
        
        if new_papers:
            # Upsert into In-Memory DB immediately
            batch_papers = []
            for p in new_papers:
                # Deduplication: Use deterministic UUID based on Paper Title
                # URLs can vary (http/https, v1/v2), but Title is stable
                paper_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, p['title'].lower().strip()))
                text_to_embed = f"{p['title']} {p['abstract']}"
                vector = search_engine.embeddings.embed_text(text_to_embed)
                
                paper_doc = {
                    "id": paper_id,
                    "vector": vector,
                    "meta": {
                        "title": p["title"],
                        "abstract": p["abstract"],
                        "authors": p["authors"],
                        "url": p["url"]
                    },
                    "filter": {
                        "year": p["year"],
                        "category": p["category"],
                        "citations": random.randint(0, 100)
                    }
                }
                batch_papers.append(paper_doc)
            
            search_engine.endee.upsert("research_papers_dense", batch_papers)
            logger.info("live_fetch_ingest_success", count=len(batch_papers))
            
    except Exception as e:
        # Don't fail the search if live fetch fails (e.g. arXiv down)
        logger.error("live_fetch_failed", error=str(e))

    # 2. Perform search
    try:
        start_time = time.time()
        results = await search_engine.search(
            query=query,
            top_k=request.top_k,
            filters=request.filters,
        )
        duration = (time.time() - start_time) * 1000
        


        # Convert to Pydantic models (Paper) validation
        paper_results = []
        for r in results:
            # Safely get values regardless of whether r is a dict or a SearchResult object
            is_dict = isinstance(r, dict)
            
            # Map attributes
            p_id = r.get("id") if is_dict else r.id
            p_title = r.get("title") if is_dict else r.title
            p_abstract = r.get("abstract") if is_dict else r.abstract
            p_authors = r.get("authors", []) if is_dict else r.authors
            p_year = r.get("year", 0) if is_dict else r.year
            p_category = r.get("category") if is_dict else r.category
            p_similarity = r.get("similarity", 0.0) if is_dict else r.similarity
            p_url = r.get("url", "") if is_dict else r.url
            p_citations = r.get("citations") if is_dict else r.citations
            
            paper_results.append(
                Paper(
                    id=p_id,
                    title=p_title,
                    abstract=p_abstract,
                    authors=p_authors,
                    year=p_year,
                    category=p_category,
                    similarity=p_similarity,
                    url=p_url,
                    citations=p_citations
                )
            )

        return SearchResponse(
            results=paper_results,
            total=len(results),
            query_time_ms=duration,
            query=original_query
        )

    except Exception as e:
        logger.error("search_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    search_engine = Depends(get_search_engine)
):
    """
    Hybrid search endpoint
    Combines dense and sparse vector search

    Args:
        request: HybridSearchRequest with query, top_k, alpha, beta
        search_engine: Injected search engine service

    Returns:
        SearchResponse with fused results
    """
    try:
        start_time = time.time()
        
        results = await search_engine.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            alpha=request.alpha
        )
        
        duration = (time.time() - start_time) * 1000

        # Map to Paper objects
        paper_results = []
        for r in results:
            paper_results.append(
                Paper(
                    id=r.id,
                    title=r.title,
                    abstract=r.abstract,
                    authors=r.authors,
                    year=r.year,
                    category=r.category,
                    similarity=r.similarity,
                    url=r.url,
                    citations=r.citations
                )
            )

        return SearchResponse(
            results=paper_results,
            total=len(results),
            query_time_ms=duration,
            query=request.query
        )

    except Exception as e:
        logger.error("hybrid_search_failed", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


