"""
Hybrid Search Engine
Combines dense and sparse vector search for optimal results
"""

from typing import List, Dict, Optional, Tuple
import time
import structlog
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.core.endee_client import EndeeVectorDB
from backend.core.embeddings import EmbeddingService

logger = structlog.get_logger()


# Pydantic Models
class SearchResult(BaseModel):
    """Single search result"""
    
    id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    category: Optional[str] = None
    similarity: float
    url: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


class ComparisonResult(BaseModel):
    """Comparison between dense and hybrid search"""
    
    query: str
    dense_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    dense_time_ms: float
    hybrid_time_ms: float
    overlap_count: int
    differences: List[str]


class HybridSearchEngine:
    """
    Hybrid search engine combining dense and sparse vectors
    Provides optimal search using both semantic and keyword matching
    """

    def __init__(
        self,
        endee_client: EndeeVectorDB,
        embedding_service: EmbeddingService,
        dense_index: str = "research_papers_dense",
        hybrid_index: str = "research_papers_hybrid",
        use_tfidf: bool = True,
    ):
        """
        Initialize hybrid search engine

        Args:
            endee_client: Endee vector database client
            embedding_service: Embedding service for dense vectors
            dense_index: Name of dense-only index
            hybrid_index: Name of hybrid index
            use_tfidf: Use TF-IDF instead of SPLADE (simpler fallback)
        """
        self.endee = endee_client
        self.embeddings = embedding_service
        self.dense_index = dense_index
        self.hybrid_index = hybrid_index
        self.use_tfidf = use_tfidf
        
        # Initialize TF-IDF vectorizer if using TF-IDF
        if use_tfidf:
            # We use a simple but effective character n-gram hashing approach
            # for sparse vectors to avoid needing a large persisted TF-IDF vocabulary.
            # This provides robust keyword matching without the overhead.
            logger.info("hybrid_search_initialized", mode="hashing_sparse")
        else:
            # Fallback for future SPLADE integration
            self.splade_model = None
            logger.info("hybrid_search_initialized", mode="splade_placeholder")

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7,
        beta: float = 0.3,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse vectors

        Args:
            query: Search query text
            top_k: Number of results to return
            alpha: Weight for dense score (0-1)
            beta: Weight for sparse score (0-1)
            filters: Optional metadata filters

        Returns:
            List of search results with fused scores
        """
        start_time = time.time()

        try:
            logger.info("hybrid_search_start", query=query, alpha=alpha, beta=beta)

            # 1. Generate dense embedding
            dense_vector = self.embeddings.embed_text(query)

            # 2. Generate sparse embedding
            sparse_vector = self._generate_sparse_embedding(query)

            # 3. Query Endee hybrid index
            results = self.endee.search_hybrid(
                index_name=self.hybrid_index,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                alpha=alpha,
                beta=beta,
            )

            # 4. Convert to SearchResult objects
            search_results = self._convert_results(results)

            query_time = (time.time() - start_time) * 1000
            logger.info(
                "hybrid_search_complete",
                query=query,
                results_count=len(search_results),
                query_time_ms=query_time,
            )

            return search_results

        except Exception as e:
            logger.error("hybrid_search_failed", error=str(e), query=query)
            raise

    def compare_search_modes(
        self,
        query: str,
        top_k: int = 10,
    ) -> ComparisonResult:
        """
        Compare dense-only vs hybrid search

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            ComparisonResult with both result sets and metrics
        """
        try:
            logger.info("compare_search_modes", query=query)

            # 1. Run dense-only search
            dense_start = time.time()
            dense_vector = self.embeddings.embed_text(query)
            dense_raw = self.endee.search_dense(
                index_name=self.dense_index,
                query_vector=dense_vector,
                top_k=top_k,
            )
            dense_results = self._convert_results(dense_raw)
            dense_time = (time.time() - dense_start) * 1000

            # 2. Run hybrid search
            hybrid_start = time.time()
            # Pass top_k correctly
            hybrid_results = self.search(query=query, top_k=top_k)
            hybrid_time = (time.time() - hybrid_start) * 1000

            # 3. Calculate overlap
            dense_ids = {r.id for r in dense_results}
            hybrid_ids = {r.id for r in hybrid_results}
            overlap_count = len(dense_ids & hybrid_ids)

            # 4. Identify differences
            differences = []
            only_dense = dense_ids - hybrid_ids
            only_hybrid = hybrid_ids - dense_ids
            
            if only_dense:
                differences.append(f"Dense-only found {len(only_dense)} unique papers")
            if only_hybrid:
                differences.append(f"Hybrid found {len(only_hybrid)} unique papers")

            # 5. Build comparison result
            comparison = ComparisonResult(
                query=query,
                dense_results=dense_results,
                hybrid_results=hybrid_results,
                dense_time_ms=dense_time,
                hybrid_time_ms=hybrid_time,
                overlap_count=overlap_count,
                differences=differences,
            )

            logger.info(
                "comparison_complete",
                overlap=overlap_count,
                dense_time=dense_time,
                hybrid_time=hybrid_time,
            )

            return comparison

        except Exception as e:
            logger.error("compare_search_modes_failed", error=str(e))
            raise

    def optimize_weights(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        alpha_range: Tuple[float, float] = (0.0, 1.0),
        beta_range: Tuple[float, float] = (0.0, 1.0),
        step: float = 0.1,
    ) -> Tuple[float, float]:
        """
        Optimize alpha and beta weights using grid search

        Args:
            queries: List of test queries
            ground_truth: List of relevant paper IDs for each query
            alpha_range: Range for alpha (dense weight)
            beta_range: Range for beta (sparse weight)
            step: Step size for grid search

        Returns:
            Tuple of optimal (alpha, beta)
        """
        try:
            logger.info("optimize_weights_start", num_queries=len(queries))

            best_score = 0.0
            best_alpha = 0.5
            best_beta = 0.5

            # Grid search
            alpha = alpha_range[0]
            while alpha <= alpha_range[1]:
                beta = beta_range[0]
                while beta <= beta_range[1]:
                    # Ensure alpha + beta = 1.0
                    if abs(alpha + beta - 1.0) < 0.01:
                        # Evaluate this combination
                        score = self._evaluate_weights(queries, ground_truth, alpha, beta)
                        
                        if score > best_score:
                            best_score = score
                            best_alpha = alpha
                            best_beta = beta
                    
                    beta += step
                alpha += step

            logger.info(
                "optimize_weights_complete",
                best_alpha=best_alpha,
                best_beta=best_beta,
                best_score=best_score,
            )

            return (best_alpha, best_beta)

        except Exception as e:
            logger.error("optimize_weights_failed", error=str(e))
            raise

    def detect_query_type(self, query: str) -> str:
        """
        Detect if query is keyword-heavy or semantic

        Args:
            query: Search query

        Returns:
            Query type: "keyword" or "semantic"
        """
        # Simple heuristic: check for specific terms, acronyms, exact phrases
        query_lower = query.lower()
        
        # Keyword indicators
        has_quotes = '"' in query
        has_acronyms = any(word.isupper() for word in query.split())
        has_specific_terms = any(term in query_lower for term in ['arxiv:', 'doi:', 'author:'])
        
        if has_quotes or has_acronyms or has_specific_terms:
            logger.debug("query_type_detected", query=query, type="keyword")
            return "keyword"
        else:
            logger.debug("query_type_detected", query=query, type="semantic")
            return "semantic"

    def recommend_weights(self, query: str) -> Tuple[float, float]:
        """
        Recommend alpha/beta weights based on query type

        Args:
            query: Search query

        Returns:
            Recommended (alpha, beta) weights
        """
        query_type = self.detect_query_type(query)
        
        if query_type == "keyword":
            # Favor sparse (keyword) matching
            alpha, beta = 0.4, 0.6
        else:
            # Favor dense (semantic) matching
            alpha, beta = 0.7, 0.3

        logger.info("weights_recommended", query_type=query_type, alpha=alpha, beta=beta)
        return (alpha, beta)

    def _generate_sparse_embedding(self, query: str) -> Dict[int, float]:
        """
        Generate sparse embedding using character n-gram hashing.
        This provides efficient keyword/acronym matching.

        Args:
            query: Query text

        Returns:
            Sparse vector as dictionary {index: weight}
        """
        import hashlib
        import numpy as np
        
        # Standard dimension for sparse vectors in Endee if not specified
        sparse_dim = 30000 
        sparse_vector = {}
        
        # Normalize
        processed_text = query.lower().strip()
        if not processed_text:
            return {}
            
        # Generate Character Trigrams + Bigrams for keyword matching
        tokens = []
        # Bigrams
        if len(processed_text) >= 2:
            tokens.extend([processed_text[i:i+2] for i in range(len(processed_text)-1)])
        # Trigrams
        if len(processed_text) >= 3:
            tokens.extend([processed_text[i:i+3] for i in range(len(processed_text)-2)])
        # Words (full keyword matching)
        tokens.extend(processed_text.split())
        
        for token in tokens:
            # Hash token to an index [0, sparse_dim-1]
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % sparse_dim
            sparse_vector[idx] = sparse_vector.get(idx, 0.0) + 1.0
            
        # Normalize weights
        total = sum(sparse_vector.values())
        if total > 0:
            for idx in sparse_vector:
                sparse_vector[idx] /= total
                
        logger.debug("sparse_embedding_generated", token_count=len(tokens), unique_indices=len(sparse_vector))
        return sparse_vector

    def _convert_results(self, raw_results: List[Dict]) -> List[SearchResult]:
        """Convert raw results to SearchResult objects"""
        results = []
        
        for result in raw_results:
            meta = result.get("meta", {})
            filter_data = result.get("filter", {})
            
            search_result = SearchResult(
                id=result.get("id", ""),
                title=meta.get("title", "Unknown"),
                abstract=meta.get("abstract", ""),
                authors=meta.get("authors", []),
                year=filter_data.get("year", 0),
                category=filter_data.get("category"),
                similarity=result.get("score", 0.0),
                url=meta.get("url", ""),
                dense_score=result.get("dense_score"),
                sparse_score=result.get("sparse_score"),
            )
            results.append(search_result)

        return results

    def _evaluate_weights(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        alpha: float,
        beta: float,
    ) -> float:
        """
        Evaluate alpha/beta weights using nDCG@10

        Args:
            queries: Test queries
            ground_truth: Relevant paper IDs for each query
            alpha: Dense weight
            beta: Sparse weight

        Returns:
            Average nDCG@10 score
        """
        # TODO: Implement actual nDCG calculation
        # For now, return placeholder
        
        total_score = 0.0
        for query, relevant_ids in zip(queries, ground_truth):
            results = self.search(query=query, top_k=10, alpha=alpha, beta=beta)
            # Calculate nDCG for this query
            # total_score += calculate_ndcg(results, relevant_ids)
            pass
        
        avg_score = total_score / len(queries) if queries else 0.0
        return avg_score
