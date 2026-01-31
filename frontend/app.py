"""
AI Research Assistant - Streamlit Frontend
Main application entry point
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main content
st.title("AI Research Assistant")
st.markdown("### Research Platform with Vector Search & RAG")

# Feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        #### Semantic Search
        Intelligent search that understands meaning, not just keywords.
        Find relevant research papers using natural language queries.
        """
    )
    if st.button("Try Semantic Search"):
        st.switch_page("pages/1_semantic_search.py")

with col2:
    st.markdown(
        """
        #### RAG Assistant
        Ask questions to your research papers.
        Get answers with direct citations to the source material.
        """
    )
    if st.button("Try RAG Assistant"):
        st.switch_page("pages/2_rag_assistant.py")
    

    


# Quick stats


# Getting started
st.markdown("### Getting Started")

st.markdown(
    """
    1. **Search Papers**: Use semantic search to find relevant research.
    2. **Ask Questions**: Use the RAG Assistant to chat with found papers.
    """
)

# Sidebar footer
with st.sidebar:
    # st.markdown("---")
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
