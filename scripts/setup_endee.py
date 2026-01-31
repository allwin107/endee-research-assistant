"""
Setup Endee Vector Database
Creates indexes for the research assistant
"""

import sys
import structlog

# Add parent directory to path
sys.path.append("..")

from backend.core.endee_client import EndeeVectorDB
from backend.config import get_settings

logger = structlog.get_logger()


def setup_endee_indexes():
    """Create all required Endee indexes"""
    settings = get_settings()
    
    # Initialize Endee client
    endee = EndeeVectorDB(url=settings.ENDEE_URL, timeout=settings.ENDEE_TIMEOUT)
    
    logger.info("setting_up_endee_indexes")
    
    # Index 1: Dense Research Papers
    logger.info("creating_dense_index")
    endee.create_dense_index(
        name="research_papers_dense",
        dimension=384,
        space_type="cosine",
        precision="INT8D"
    )
    
    # Index 2: Hybrid Research Papers
    logger.info("creating_hybrid_index")
    endee.create_hybrid_index(
        name="research_papers_hybrid",
        dimension=384,
        sparse_dim=30000,
        space_type="cosine",
        precision="INT8D"
    )
    
    logger.info("endee_setup_complete")
    print("Endee indexes created successfully!")


if __name__ == "__main__":
    setup_endee_indexes()
