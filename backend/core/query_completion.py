"""
Query auto-completion using vector similarity
Provides intelligent query suggestions based on popular queries and semantic similarity
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import structlog

from backend.core.embeddings import EmbeddingService
from backend.core.endee_client import EndeeVectorDB
from backend.utils.cache import SimpleCache
from backend.utils.monitoring import PerformanceTracker, get_usage_analytics
from backend.utils.exceptions import SearchError

logger = structlog.get_logger(__name__)


class QueryCompletionEngine:
    """
    Query auto-completion engine using vector similarity
    
    Stores popular queries in Endee and suggests completions based on:
    - Semantic similarity (vector search)
    - Query frequency
    - Recency
    """

    def __init__(
        self,
        endee_client: EndeeVectorDB,
        embedding_service: EmbeddingService,
        index_name: str = "query_patterns",
        cache_ttl: int = 300,  # 5 minutes
        decay_days: int = 30,
    ):
        """
        Initialize query completion engine
        
        Args:
            endee_client: Endee vector database client
            embedding_service: Embedding generation service
            index_name: Name of Endee index for query patterns
            cache_ttl: Cache TTL in seconds
            decay_days: Days after which to decay query scores
        """
        self.endee_client = endee_client
        self.embedding_service = embedding_service
        self.index_name = index_name
        self.decay_days = decay_days
        
        # Cache for suggestions
        self.cache = SimpleCache(max_size=1000, default_ttl=cache_ttl)
        
        # In-memory query frequency tracking
        self.query_frequencies: Dict[str, int] = defaultdict(int)
        self.query_last_seen: Dict[str, datetime] = {}
        
        # Initialize index
        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        """Ensure query patterns index exists in Endee"""
        try:
            # Check if index exists, create if not
            # This is a placeholder - actual implementation depends on Endee API
            logger.info("query_completion_index_ready", index=self.index_name)
        except Exception as e:
            logger.error("query_completion_index_setup_failed", error=str(e))

    async def suggest(
        self,
        partial_query: str,
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> List[str]:
        """
        Get query completion suggestions
        
        Args:
            partial_query: Partial query string
            limit: Maximum number of suggestions
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of suggested query completions
        """
        with PerformanceTracker("query_completion", labels={"limit": str(limit)}):
            # Normalize input
            partial_query = partial_query.strip().lower()
            
            if not partial_query:
                return await self._get_popular_queries(limit)
            
            # Check cache
            cache_key = f"suggest:{partial_query}:{limit}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug("query_completion_cache_hit", query=partial_query)
                return cached
            
            try:
                # Get suggestions
                suggestions = await self._get_suggestions(partial_query, limit, min_similarity)
                
                # Cache results
                self.cache.set(cache_key, suggestions)
                
                logger.info(
                    "query_completion_success",
                    query=partial_query,
                    suggestions_count=len(suggestions)
                )
                
                return suggestions
            
            except Exception as e:
                logger.error(
                    "query_completion_failed",
                    query=partial_query,
                    error=str(e)
                )
                # Fallback to prefix matching
                return self._prefix_match_fallback(partial_query, limit)

    async def _get_suggestions(
        self,
        partial_query: str,
        limit: int,
        min_similarity: float,
    ) -> List[str]:
        """
        Get suggestions using vector similarity
        
        Args:
            partial_query: Partial query
            limit: Max suggestions
            min_similarity: Min similarity threshold
            
        Returns:
            List of suggestions
        """
        # Generate embedding for partial query
        query_embedding = self.embedding_service.embed(partial_query)
        
        # Search in Endee for similar queries
        results = self.endee_client.search_dense(
            index_name=self.index_name,
            query_vector=query_embedding,
            top_k=limit * 2,  # Get more to filter
            filters=None,
        )
        
        # Filter and rank results
        suggestions = []
        for result in results:
            # Check similarity threshold
            if result.get("score", 0) < min_similarity:
                continue
            
            query_text = result.get("query", "")
            
            # Skip if same as input
            if query_text.lower() == partial_query.lower():
                continue
            
            # Apply frequency boost
            frequency = result.get("frequency", 1)
            last_seen = result.get("last_seen")
            
            # Calculate decay factor
            decay_factor = self._calculate_decay(last_seen)
            
            # Boosted score
            boosted_score = result["score"] * (1 + 0.1 * frequency) * decay_factor
            
            suggestions.append({
                "query": query_text,
                "score": boosted_score,
            })
        
        # Sort by boosted score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top suggestions
        return [s["query"] for s in suggestions[:limit]]

    async def _get_popular_queries(self, limit: int) -> List[str]:
        """
        Get most popular queries
        
        Args:
            limit: Max queries to return
            
        Returns:
            List of popular queries
        """
        try:
            # Query Endee for most frequent queries
            # Sort by frequency, apply decay
            results = self.endee_client.search_dense(
                index_name=self.index_name,
                query_vector=[0.0] * 384,  # Endee requires a vector
                top_k=limit,
                filters=None,
            )
            
            # Sort by frequency with decay
            queries_with_scores = []
            for result in results:
                frequency = result.get("frequency", 1)
                last_seen = result.get("last_seen")
                decay_factor = self._calculate_decay(last_seen)
                score = frequency * decay_factor
                
                queries_with_scores.append({
                    "query": result.get("query", ""),
                    "score": score,
                })
            
            queries_with_scores.sort(key=lambda x: x["score"], reverse=True)
            
            return [q["query"] for q in queries_with_scores[:limit]]
        
        except Exception as e:
            logger.error("get_popular_queries_failed", error=str(e))
            return []

    def _prefix_match_fallback(self, partial_query: str, limit: int) -> List[str]:
        """
        Fallback to simple prefix matching
        
        Args:
            partial_query: Partial query
            limit: Max suggestions
            
        Returns:
            List of matching queries
        """
        matches = []
        
        for query, freq in sorted(
            self.query_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if query.lower().startswith(partial_query.lower()):
                matches.append(query)
                if len(matches) >= limit:
                    break
        
        return matches

    def _calculate_decay(self, last_seen: Optional[str]) -> float:
        """
        Calculate decay factor based on recency
        
        Args:
            last_seen: ISO timestamp of last seen
            
        Returns:
            Decay factor (0.0 to 1.0)
        """
        if not last_seen:
            return 0.5  # Default for unknown
        
        try:
            last_seen_dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            days_ago = (datetime.utcnow() - last_seen_dt.replace(tzinfo=None)).days
            
            # Exponential decay
            decay = max(0.1, 1.0 - (days_ago / self.decay_days))
            return decay
        
        except Exception:
            return 0.5

    def track_query(self, query: str) -> None:
        """
        Track query usage for future suggestions
        
        Args:
            query: Query string
        """
        query = query.strip().lower()
        
        if not query or len(query) < 2:
            return
        
        try:
            # Update in-memory tracking
            self.query_frequencies[query] += 1
            self.query_last_seen[query] = datetime.utcnow()
            
            # Update in Endee
            self._update_query_in_endee(query)
            
            # Track analytics
            analytics = get_usage_analytics()
            analytics.track_query("autocomplete_track", duration_ms=0)
            
            logger.debug("query_tracked", query=query, frequency=self.query_frequencies[query])
        
        except Exception as e:
            logger.error("query_tracking_failed", query=query, error=str(e))

    def _update_query_in_endee(self, query: str) -> None:
        """
        Update query pattern in Endee
        
        Args:
            query: Query string
        """
        try:
            # Generate embedding
            embedding = self.embedding_service.embed(query)
            
            # Prepare document
            doc = {
                "id": f"query_{hash(query)}",
                "query": query,
                "embedding": embedding,
                "frequency": self.query_frequencies[query],
                "last_seen": datetime.utcnow().isoformat() + "Z",
            }
            
            # Upsert to Endee
            self.endee_client.upsert(
                index_name=self.index_name,
                documents=[doc]
            )
        
        except Exception as e:
            logger.error("endee_query_update_failed", query=query, error=str(e))

    def decay_old_queries(self, days_threshold: int = 90) -> int:
        """
        Remove or decay very old queries
        
        Args:
            days_threshold: Days after which to remove queries
            
        Returns:
            Number of queries removed
        """
        removed_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        
        try:
            # Remove from in-memory tracking
            queries_to_remove = [
                query for query, last_seen in self.query_last_seen.items()
                if last_seen < cutoff_date
            ]
            
            for query in queries_to_remove:
                del self.query_frequencies[query]
                del self.query_last_seen[query]
                removed_count += 1
            
            logger.info(
                "old_queries_decayed",
                removed_count=removed_count,
                threshold_days=days_threshold
            )
            
            return removed_count
        
        except Exception as e:
            logger.error("query_decay_failed", error=str(e))
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get query completion statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_queries": len(self.query_frequencies),
            "cache_size": len(self.cache._cache),
            "most_popular": sorted(
                self.query_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }
