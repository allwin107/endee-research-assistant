
import structlog
import uuid
import random

from backend.core.endee_client import EndeeVectorDB
from backend.core.embeddings import EmbeddingService
from backend.utils.arxiv import fetch_arxiv_papers
from backend.config import get_settings

logger = structlog.get_logger()

async def seed_data_if_empty():
    """
    Check if EndeeDB is empty, if so, seed with real arXiv data.
    """
    settings = get_settings()
    endee = EndeeVectorDB(url=settings.ENDEE_URL)
    
    # Check if index exists and has data
    stats = endee.get_index_stats("research_papers_dense")
    existing_count = stats.get("vector_count", 0) if stats else 0
    
    if existing_count > 0:
        logger.info("db_already_seeded", count=existing_count)
        return

    logger.info("seeding_db_start", source="arXiv")
    
    # Ensure index exists
    endee.create_dense_index("research_papers_dense", dimension=settings.EMBEDDING_DIMENSION)
    
    # Fetch real data
    # We mix a few queries to get diverse data
    queries = ["machine learning", "large language models", "retrieval augmented generation", "generative ai", "transformers"]
    all_papers = []
    
    for q in queries:
        papers = fetch_arxiv_papers(query=q, max_results=5)
        all_papers.extend(papers)
        
    embeddings = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    batch_papers = []
    
    for p in all_papers:
        paper_id = str(uuid.uuid4())
        text_to_embed = f"{p['title']} {p['abstract']}"
        vector = embeddings.embed_text(text_to_embed)
        
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
                "citations": random.randint(0, 500) # Real arXiv API doesn't give citation count easily
            }
        }
        batch_papers.append(paper_doc)
        
    endee.upsert("research_papers_dense", batch_papers)
    logger.info("seeding_db_complete", count=len(batch_papers))
