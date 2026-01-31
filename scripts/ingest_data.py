"""
Data Ingestion Script
Downloads and processes arXiv papers, generates embeddings, and uploads to Endee
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import structlog

# Add parent directory to path
sys.path.append("..")

from backend.core.endee_client import EndeeVectorDB
from backend.core.embeddings import EmbeddingService
from backend.config import get_settings

logger = structlog.get_logger()


def ingest_data(input_file: str, num_papers: int, index_name: str, batch_size: int):
    """
    Ingest research papers into Endee

    Args:
        input_file: Path to input CSV file
        num_papers: Number of papers to process
        index_name: Target Endee index
        batch_size: Batch size for embeddings
    """
    logger.info("starting_data_ingestion", input_file=input_file, num_papers=num_papers)
    
    settings = get_settings()
    
    # Initialize services
    endee = EndeeVectorDB(url=settings.ENDEE_URL)
    embeddings = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    
    # Load and process data
    try:
        import pandas as pd
        import numpy as np
        import json
        
        logger.info("loading_data", file=input_file)
        df = pd.read_csv(input_file)
        
        # Limit papers
        if num_papers:
            df = df.head(num_papers)
            
        papers = []
        logger.info("processing_papers", count=len(df))
        
        # Create mapping of indices to IDs for citation generation
        paper_ids = [f"paper_{i}" for i in range(len(df))]
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            paper_id = paper_ids[i]
            
            # Generate mock citations if not present
            # In a real scenario, this would come from the dataset
            num_citations = np.random.randint(0, min(20, len(df)))
            references = []
            if len(paper_ids) > 1:
                references = list(np.random.choice(paper_ids, num_citations, replace=False))
                # Remove self-reference
                if paper_id in references:
                    references.remove(paper_id)
            
            paper = {
                "id": paper_id,
                "title": str(row.get("title", "Untitled")),
                "abstract": str(row.get("abstract", "No abstract")),
                "authors": str(row.get("authors", "Unknown")).split(","),
                "year": int(row.get("year", 2023)),
                "category": str(row.get("categories", "cs.AI")).split(" ")[0],
                "url": f"https://arxiv.org/abs/{row.get('id', '0000.0000')}",
                "references": references,
                "citation_count": len(references)  # Mock count
            }
            papers.append(paper)
            
        # Generate embeddings in batches
        logger.info("generating_embeddings")
        batch_papers = []
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            texts = [f"{p['title']} {p['abstract']}" for p in batch]
            
            embeddings_batch = embeddings.embed_batch(texts)
            
            for j, paper in enumerate(batch):
                paper["embedding"] = embeddings_batch[j]
                batch_papers.append(paper)
                
            # Upsert batch to Endee
            if len(batch_papers) >= batch_size * 5:
                endee.upsert(index_name, batch_papers)
                batch_papers = []
                
        # Upsert remaining
        if batch_papers:
            endee.upsert(index_name, batch_papers)
            
    except Exception as e:
        logger.error("ingestion_failed", error=str(e))
        raise
    
    print(f"✅ Ingested {len(papers)} papers with citations to {index_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into Endee")
    parser.add_argument("--input-file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--num-papers", type=int, default=10000, help="Number of papers to process")
    parser.add_argument("--index-name", type=str, default="research_papers_dense", help="Endee index name")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    
    args = parser.parse_args()
    
    ingest_data(
        input_file=args.input_file,
        num_papers=args.num_papers,
        index_name=args.index_name,
        batch_size=args.batch_size
    )
