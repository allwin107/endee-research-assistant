"""
Semantic Search Page
"""

import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any

# Configure page
st.set_page_config(
    page_title="Semantic Search",
    layout="wide"
)

# Constants
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = os.getenv("API_URL", f"{API_BASE_URL}/api/v1")


def search_api(query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
    """Call search API"""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "filters": filters
        }
        response = requests.post(f"{API_URL}/search", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []


def get_autocomplete_suggestions(query: str) -> List[str]:
    """Get autocomplete suggestions"""
    try:
        if not query or len(query) < 2:
            return []
            
        response = requests.get(
            f"{API_URL}/search/autocomplete",
            params={"q": query, "limit": 5}
        )
        if response.status_code == 200:
            return response.json().get("suggestions", [])
        return []
    except Exception:
        return []


def track_query(query: str):
    """Track query usage"""
    try:
        requests.post(
            f"{API_URL}/search/autocomplete/track",
            json={"query": query}
        )
    except Exception:
        pass  # Fail silently


def render_search_result(result: Dict):
    """Render a single search result"""
    with st.container():
        # Display translated title if available
        st.markdown(f"### {result.get('title', 'Untitled')}")
            
        st.markdown(f"**Score:** {result.get('score', 0.0):.4f}")
        
        # Metadata
        meta_cols = st.columns(3)
        with meta_cols[0]:
            if result.get('year'):
                st.markdown(f"**Year:** {result.get('year')}")
        with meta_cols[1]:
            if result.get('authors'):
                st.markdown(f"**Authors:** {', '.join(result.get('authors')[:3])}")
        with meta_cols[2]:
            if result.get('category'):
                st.markdown(f"**Category:** {result.get('category')}")
        
        # Abstract
        st.markdown(f"_{result.get('abstract', '')[:300]}..._")
        
        # Actions
        if result.get('url'):
            st.markdown(f"[Read Paper]({result.get('url')})")
        
        st.divider()


def main():
    st.title("Semantic Search")
    st.markdown("Search for research papers using natural language.")

    # Sidebar parameters
    with st.sidebar:
        st.header("Search Parameters")
        top_k = st.slider("Number of results", min_value=1, max_value=50, value=10)
        
        st.subheader("Filters")
        year_min = st.number_input("Min Year", min_value=1900, max_value=2024, value=2010)
        year_max = st.number_input("Max Year", min_value=1900, max_value=2026, value=2024)
        
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: left'>
                <p>Built using <a href='https://endee.ai' target='_blank'>Endee Vector Database</a></p>
                <p><a href='https://github.com/allwin107/endee-research-assistant' target='_blank'>GitHub</a> | 
                <a href='http://localhost:8000/docs' target='_blank'>Documentation</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        


    # Search interface with autocomplete
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    # Search input
    # Note: Streamlit doesn't support real-time typing events natively for autocomplete
    # We use a selectbox workaround for suggestions when available, or simple text input
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query",
            value=st.session_state.search_query,
            placeholder="e.g., machine learning in healthcare",
            key="query_input"
        )
        
        # Show suggestions below input if query exists but not submitted
        if query and query != st.session_state.get("last_submitted", ""):
            suggestions = get_autocomplete_suggestions(query)
            if suggestions:
                cols = st.columns(len(suggestions))
                st.caption("Suggestions:")
                for i, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"sugg_{i}"):
                        st.session_state.search_query = suggestion
                        st.experimental_rerun()

    with col2:
        st.write("") # Spacing
        st.write("") 
        search_button = st.button("Search", type="primary", use_container_width=True)

    # Execute search
    if search_button or (query and query != st.session_state.get("last_submitted", "")):
        if query:
            st.session_state.last_submitted = query
            
            # Track query
            track_query(query)
            
            # Prepare filters
            filters = {}
            if year_min or year_max:
                filters["year"] = {"$gte": year_min, "$lte": year_max}
            
            with st.spinner("Searching..."):
                api_response = search_api(query, top_k, filters)
            
            # Handle response structure
            results = []
            total = 0
            
            if isinstance(api_response, dict):
                results = api_response.get("results", [])
                total = api_response.get("total", len(results))
            elif isinstance(api_response, list):
                results = api_response
                total = len(results)
                
            if results:
                # Debugging: Check structure
                if isinstance(results, dict):
                    st.error(f"Error: Results is a dictionary, expected list. Keys: {list(results.keys())}")
                    st.json(results)
                    results = []
                elif results and isinstance(results[0], str):
                    st.error(f"Error: Result items are strings: {results[:3]}")
                    results = []
                
                st.info(f"Found {total} papers")
                
                # Cache results for RAG Context
                if results:
                    st.session_state.context_papers = results
                    st.session_state.context_paper_ids = [r.get("id") for r in results if r.get("id")]
                    # Optional: visual indicator
                    st.toast(f"Cached {len(results)} papers for RAG context.")
                
                # Export Controls


                st.write("---")
                for result in results:
                    render_search_result(result)
            else:
                st.warning("No results found. Try different keywords.")


if __name__ == "__main__":
    main()
