
import requests
import xml.etree.ElementTree as ET
import structlog
from typing import List, Dict, Any

logger = structlog.get_logger()

ARXIV_API_URL = "http://export.arxiv.org/api/query"

def fetch_arxiv_papers(query: str = "machine learning", max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch papers from the arXiv API.
    
    Args:
        query: Search query (e.g. "cat:cs.AI")
        max_results: Max papers to return
    
    Returns:
        List of paper dictionaries
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        logger.info("fetching_arxiv_data", query=query, max_results=max_results)
        response = requests.get(ARXIV_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        # Namespace for Atom
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        
        papers = []
        
        for entry in root.findall('atom:entry', ns):
            paper_id = entry.find('atom:id', ns).text
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns).text
                authors.append(name)
                
            links = []
            url = paper_id
            for link in entry.findall('atom:link', ns):
                if link.get('rel') == 'alternate':
                    url = link.get('href')
            
            category = "cs.AI"
            cat_elem = entry.find('arxiv:primary_category', ns)
            if cat_elem is not None:
                category = cat_elem.get('term')

            papers.append({
                "title": title,
                "abstract": summary,
                "authors": authors,
                "year": int(published[:4]),
                "category": category,
                "url": url,
                "id": paper_id
            })
            
        logger.info("arxiv_fetch_success", count=len(papers))
        return papers
        
    except Exception as e:
        logger.error("arxiv_fetch_failed", error=str(e))
        return []

if __name__ == "__main__":
    # Test run
    papers = fetch_arxiv_papers("machine learning", 5)
    print(f"Fetched {len(papers)} papers")
    for p in papers:
        print(f"- {p['title']}")
