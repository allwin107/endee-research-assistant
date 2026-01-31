"""
Endee Vector Database Client Wrapper
Provides high-level interface for Endee operations using the official SDK
"""

from typing import List, Dict, Optional, Any
import structlog
import endee
from endee import Precision

logger = structlog.get_logger()


class EndeeVectorDB:
    """
    Wrapper class for Endee Vector Database operations
    Integrates with the official endee-python SDK
    """

    def __init__(self, url: str, timeout: int = 30):
        """
        Initialize Endee client

        Args:
            url: Endee server URL
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip('/')
        if not self.url.endswith("/api/v1"):
            self.url = f"{self.url}/api/v1"
            
        self.client = endee.Endee()
        self.client.set_base_url(self.url)
        # The SDK uses its own session management, but we could pass a custom one if needed
        logger.info("endee_client_initialized", url=self.url)

    def upsert(self, index_name: str, vectors_batch: List[Dict[str, Any]]) -> bool:
        """
        Upsert vectors to an index
        """
        try:
            logger.info("upserting_vectors", index_name=index_name, count=len(vectors_batch))
            index = self.client.get_index(index_name)
            
            # The SDK expects a list of dictionaries with 'id', 'vector', 'meta', 'filter'
            # Our input format already matches this based on seeding.py and ingest_data.py
            index.upsert(vectors_batch)
            return True
            
        except Exception as e:
            logger.error("upsert_vectors_failed", error=str(e), index_name=index_name)
            raise

    def _get_precision(self, precision: str) -> Any:
        """Map precision string to SDK enum"""
        mapping = {
            "INT8D": Precision.INT8D,
            "INT16D": Precision.INT16D,
            "FLOAT16": Precision.FLOAT16,
            "FLOAT32": Precision.FLOAT32,
            "BINARY2": Precision.BINARY2,
        }
        # SDK is case-sensitive and prefers enum or lowercase
        return mapping.get(precision.upper(), Precision.INT8D)

    def create_dense_index(
        self, 
        name: str, 
        dimension: int, 
        space_type: str = "cosine", 
        precision: str = "INT8D",
        M: int = 16,
        ef_construct: int = 100
    ) -> bool:
        """Create a dense vector index"""
        try:
            logger.info("creating_dense_index", name=name)
            
            # Map space_type if needed
            if space_type == "inner_product":
                space_type = "ip"
            
            self.client.create_index(
                name=name,
                dimension=dimension,
                space_type=space_type,
                M=M,
                ef_con=ef_construct,
                precision=self._get_precision(precision)
            )
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            logger.error("create_dense_index_failed", error=str(e), name=name)
            raise

    def create_hybrid_index(
        self, 
        name: str, 
        dimension: int, 
        sparse_dim: int, 
        space_type: str = "cosine", 
        precision: str = "INT8D"
    ) -> bool:
        """Create a hybrid index"""
        try:
            logger.info("creating_hybrid_index", name=name)
            self.client.create_index(
                name=name,
                dimension=dimension,
                space_type=space_type,
                sparse_dim=sparse_dim,
                precision=self._get_precision(precision)
            )
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            # Fallback to dense if hybrid fails or is not supported
            return self.create_dense_index(name, dimension, space_type, precision)

    def _normalize_filter(self, filters: Any) -> Optional[List]:
        """
        Normalize filters to the format expected by Endee:
        List[Dict[str, Dict[str, Any]]] e.g. [{"field": {"$eq": "value"}}]
        """
        if not filters:
            return None
        
        # If it's already a list, trust it for now but check if empty
        if isinstance(filters, list):
            return filters if len(filters) > 0 else None
            
        # If it's a dict (common from Swagger/UI), check if it's empty or has placeholders
        if isinstance(filters, dict):
            # Check for Swagger placeholders or empty dicts
            clean_filters = {k: v for k, v in filters.items() if v and v != {}}
            if not clean_filters:
                return None
            
            # Convert dict {"field": condition} to [{"field": condition}]
            # Endee expects a list of conditions
            normalized = []
            for field, condition in clean_filters.items():
                normalized.append({field: condition})
            
            logger.info("normalized_filter_format", original=filters, normalized=normalized)
            return normalized

        return None

    def search_dense(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Search using dense vector"""
        try:
            logger.info("dense_search", index_name=index_name, top_k=top_k)
            index = self.client.get_index(index_name)
            
            normalized_filters = self._normalize_filter(filters)
            
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                filter=normalized_filters
            )
            
            # Normalize SDK results to the format expected by our SearchEngine
            normalized_results = []
            for r in results:
                normalized_results.append({
                    "id": r.get("id") if isinstance(r, dict) else getattr(r, "id", ""),
                    "score": r.get("score", r.get("similarity", 0.0)) if isinstance(r, dict) else getattr(r, "score", getattr(r, "similarity", 0.0)),
                    "meta": r.get("meta") if isinstance(r, dict) else getattr(r, "meta", {}),
                    "filter": r.get("filter") if isinstance(r, dict) else getattr(r, "filter", {})
                })
            
            return normalized_results
            
        except Exception as e:
            logger.error("dense_search_failed", error=str(e), index_name=index_name)
            return []

    def search_hybrid(
        self, 
        index_name: str, 
        dense_vector: List[float], 
        sparse_vector: Dict[int, float], 
        top_k: int = 10, 
        alpha: float = 0.7, 
        beta: float = 0.3,
        filters: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid search"""
        try:
            logger.info("hybrid_search", index_name=index_name, top_k=top_k)
            index = self.client.get_index(index_name)
            
            normalized_filters = self._normalize_filter(filters)
            
            # SDK query supports hybrid if sparse data is provided
            results = index.query(
                vector=dense_vector,
                sparse_indices=list(sparse_vector.keys()) if sparse_vector else None,
                sparse_values=list(sparse_vector.values()) if sparse_vector else None,
                top_k=top_k,
                alpha=alpha,
                filter=normalized_filters
            )
            
            normalized_results = []
            for r in results:
                normalized_results.append({
                    "id": r.get("id") if isinstance(r, dict) else getattr(r, "id", ""),
                    "score": r.get("similarity", r.get("score", 0.0)) if isinstance(r, dict) else getattr(r, "similarity", getattr(r, "score", 0.0)),
                    "meta": r.get("meta") if isinstance(r, dict) else getattr(r, "meta", {}),
                    "filter": r.get("filter") if isinstance(r, dict) else getattr(r, "filter", {})
                })
            
            return normalized_results
        except Exception as e:
            logger.error("hybrid_search_failed", error=str(e))
            # Fallback to dense search
            return self.search_dense(index_name, dense_vector, top_k)

    def delete_index(self, index_name: str) -> bool:
        """Delete an index"""
        try:
            self.client.delete_index(index_name)
            return True
        except Exception:
            return False

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            # Try to get stats via SDK method if available, else use describe
            if hasattr(self.client, "get_stats"):
                return self.client.get_stats(index_name)
            
            index = self.client.get_index(index_name)
            return index.describe()
        except Exception:
            return {}

    def get_item(self, index_name: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID"""
        try:
            index = self.client.get_index(index_name)
            # Assuming SDK supports get or list with ID filter
            # LanceDB/Endee typically support: index.to_pandas() or similar query
            # We'll try a filter query for exact ID match
            # Note: Index requires vector of correct dimension even for filtering
            # We use the dimension from settings or default to 384
            from backend.config import get_settings
            settings = get_settings()
            dim = settings.EMBEDDING_DIMENSION or 384
            
            results = index.query(
                vector=[0.0] * dim, 
                top_k=1,
                filter=[{"id": item_id}]
            )
            
            if results and len(results) > 0:
                # return the first match
                r = results[0]
                return {
                    "id": r.get("id") if isinstance(r, dict) else getattr(r, "id", ""),
                    "vector": r.get("vector") if isinstance(r, dict) else getattr(r, "vector", []),
                    "meta": r.get("meta") if isinstance(r, dict) else getattr(r, "meta", {}),
                }
            return None
        except Exception as e:
            logger.error("get_item_failed", error=str(e), index_name=index_name, item_id=item_id)
            return None
