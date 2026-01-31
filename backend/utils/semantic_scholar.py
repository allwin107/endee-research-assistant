
import requests
import structlog
from typing import List, Dict, Any
import time

logger = structlog.get_logger()

# Semantic Scholar Graph API
# Using the public API endpoint (rate limited)
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def fetch_semantic_scholar_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch papers from Semantic Scholar API.
    
    Args:
        query: Search query
        max_results: Max papers to return
    
    Returns:
        List of paper dictionaries in internal format
    """
    if not query:
        return []

    # Fields to retrieve
    fields = "paperId,title,abstract,authors,year,url,venue,citationCount,openAccessPdf"
    
    params = {
        "query": query,
        "limit": max_results,
        "fields": fields
    }
    
    try:
        logger.info("fetching_semantic_scholar_data", query=query)
        # 1 second sleep to be nice to public API rate limits if called frequently
        # In production, use an API key and rate limiter
        # time.sleep(0.5) 
        
        response = requests.get(SEMANTIC_SCHOLAR_API_URL, params=params, timeout=10)
        
        if response.status_code == 429:
            logger.warning("semantic_scholar_rate_limit")
            return []
            
        response.raise_for_status()
        data = response.json()
        
        papers = []
        for item in data.get("data", []):
            # Extract basic info
            paper_id = item.get("paperId")
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            year = item.get("year")
            
            # Authors
            authors = [a.get("name") for a in item.get("authors", [])]
            
            # URL - prioritize open access PDF, fall back to semantic scholar page
            url = item.get("url")
            open_access_pdf = item.get("openAccessPdf")
            if open_access_pdf and open_access_pdf.get("url"):
                url = open_access_pdf.get("url")
                
            # Category (Semantic Scholar doesn't always provide a clean category like arXiv)
            # We'll default to General or infer from venue if needed
            category = item.get("venue") or "General"
            
            # Citations
            citations = item.get("citationCount", 0)

            # Clean blank abstracts
            if not abstract:
                abstract = "No abstract available."

            papers.append({
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year or 0,
                "category": category,
                "url": url,
                "citations": citations,
                "source": "semantic_scholar"
            })
            
        logger.info("semantic_scholar_fetch_success", count=len(papers))
        return papers
        
    except Exception as e:
        logger.error("semantic_scholar_fetch_failed", error=str(e))
        return []

if __name__ == "__main__":
    # Test run
    results = fetch_semantic_scholar_papers("generative ai", 5)
    for p in results:
        print(f"[{p['year']}] {p['title']} ({p['url']})")
