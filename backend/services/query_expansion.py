"""
Query Expansion Service
Expands search queries with synonyms and related terms using LLM
"""

from typing import List
import structlog
from pydantic import BaseModel

from backend.core.groq_client import GroqClient
from backend.utils.cache import SimpleCache
from backend.utils.monitoring import PerformanceTracker

logger = structlog.get_logger(__name__)


class QueryExpansionService:
    """
    Service to expand queries for better recall
    """

    def __init__(self):
        from backend.config import get_settings
        settings = get_settings()
        self.groq_client = GroqClient(api_key=settings.GROQ_API_KEY)
        self.cache = SimpleCache(max_size=1000, default_ttl=3600)  # Cache for 1 hour
        self.metrics = PerformanceTracker

    async def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms
        
        Args:
            query: Original query
            
        Returns:
            Expanded query string
        """
        if not query or len(query.strip()) < 3:
            return query

        cache_key = f"expand:{query.lower().strip()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        with self.metrics("expand_query"):
            try:
                # Prompt specific for academic/technical synonym expansion
                prompt = f"""
                You are a search query optimizer for a research assistant.
                Expand the following search query by adding 2-3 highly relevant academic synonyms or related technical terms.
                Keep it concise. valid JSON format list of strings only.
                
                Query: "{query}"
                Synonyms:
                """
                
                # We ask for JSON list to be robust
                response = self.groq_client.generate(prompt, temperature=0.3, max_tokens=60)
                
                # Extract text from response dict
                response_text = response.get("text", "")
                # Assuming LLM returns a list or comma separated string
                # For now, let's treat the output as additional terms
                
                # Cleanup response
                expanded_terms = response_text.strip().replace('"', '').replace('[', '').replace(']', '').split(',')
                expanded_terms = [t.strip() for t in expanded_terms if t.strip()]
                
                # Combine with original query
                # "original query term1 term2"
                expanded_query = f"{query} {' '.join(expanded_terms)}"
                
                logger.info("query_expanded", original=query, expansion=expanded_terms)
                
                self.cache.set(cache_key, expanded_query)
                return expanded_query

            except Exception as e:
                logger.warning("query_expansion_failed", error=str(e), query=query)
                return query

# Singleton
_query_expansion_service = None

def get_query_expansion_service() -> QueryExpansionService:
    global _query_expansion_service
    if _query_expansion_service is None:
        _query_expansion_service = QueryExpansionService()
    return _query_expansion_service
