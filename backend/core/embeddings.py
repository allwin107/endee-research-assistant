"""
Embedding Service using sentence-transformers
Generates vector embeddings for text
"""

from typing import List
from functools import lru_cache
import structlog

import torch
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = structlog.get_logger()


class EmbeddingService:
    """
    Service for generating text embeddings
    Uses sentence-transformers models with optimization
    """

    def __init__(self, model_name: str = None, cache_size: int = 1000):
        """
        Initialize embedding service

        Args:
            model_name: Name of the sentence-transformer model
            cache_size: Size of LRU cache for embeddings
        """
        from backend.config import get_settings
        self.settings = get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL
        try:
            from backend.utils.monitoring import PerformanceTracker
            self.metrics = PerformanceTracker
            
            # Default to CPU
            self.device = "cpu"
            
            # Check for GPU
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                
            if HAS_TRANSFORMERS:
                logger.info("loading_embedding_model", model=self.model_name, device=self.device)
                # Load model to specific device
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
                # Warn if using large model in dev
                if "large" in self.model_name.lower():
                    logger.warning(
                        "large_model_detected", 
                        msg="Consider using 'all-MiniLM-L6-v2' for faster development"
                    )
            else:
                logger.warning("sentence_transformers_not_found", msg="Using mock embeddings")
                self.model = None
                
        except Exception as e:
            logger.error("embedding_model_load_failed", error=str(e))
            self.model = None
            self.device = "cpu"

    @lru_cache(maxsize=1000)
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        with self.metrics("embed_text", labels={"model": self.model_name, "device": self.device}):
            try:
                if self.model:
                    embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                    return embedding.tolist()
                
                return [0.0] * self.get_embedding_dimension()
            except Exception as e:
                # Fallback to Sparse Hashing (Bag-of-Words like)
                # This ensures similar texts have similar vectors
                import hashlib
                import numpy as np
                import re
                
                dim = self.get_embedding_dimension()
                vec = np.zeros(dim)
                
                # Fallback to Character N-Gram Embedding (Robust to suffixes/stemming)
                import hashlib
                import numpy as np
                import re
                
                dim = self.get_embedding_dimension()
                vec = np.zeros(dim)
                
                # Normalize
                processed_text = text.lower()
                
                # Generate Character Trigrams
                n = 3
                if len(processed_text) < n:
                    tokens = [processed_text]
                else:
                    tokens = [processed_text[i:i+n] for i in range(len(processed_text)-n+1)]
                
                for token in tokens:
                    # Hash token to an index [0, dim-1]
                    hash_object = hashlib.md5(token.encode())
                    idx = int(hash_object.hexdigest(), 16) % dim
                    # We can weigh them? simple count is fine.
                    vec[idx] += 1.0
                
                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                    
                return vec.tolist()
            except Exception as e:
                logger.error("embed_text_failed", error=str(e))
                raise

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        """
        if not texts:
            return []

        try:
            with self.metrics("embed_batch", labels={"batch_size": str(batch_size), "count": str(len(texts))}) as tracker:
                if self.model:
                    # Actual batch generation
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device=self.device
                    )
                    
                    # Log performance
                    duration = tracker.elapsed_ms
                    ms_per_doc = duration / len(texts) if texts else 0
                    logger.info(
                        "batch_embedding_complete", 
                        count=len(texts), 
                        ms_per_doc=ms_per_doc,
                        device=self.device
                    )
                    
                    return embeddings.tolist()
                
                return [[0.0] * self.get_embedding_dimension() for _ in texts]
                
        except Exception as e:
            logger.error("embed_batch_failed", error=str(e), count=len(texts))
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return self.settings.MULTILINGUAL_DIMENSION if self.settings.ENABLE_MULTILINGUAL else self.settings.EMBEDDING_DIMENSION
