"""
RAG Assistant Page - Chat Interface
"""

import streamlit as st
import os

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("RAG Research Assistant")
st.markdown("Ask questions and get answers with citations from research papers")

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = os.getenv("API_URL", f"{API_BASE_URL}/api/v1")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Sidebar settings
with st.sidebar:
    st.markdown("### Chat Settings")
    top_k = st.slider("Sources to Retrieve", 1, 10, 5)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    
    st.markdown("---")
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Paper Context")
    
    # Retrieve cached papers from session state
    context_ids = st.session_state.get("context_paper_ids", [])
    
    if context_ids:
        st.info(f"{len(context_ids)} papers available from search.")
        
        # Allow user to toggle papers
        selected_papers = []
        st.markdown("Select papers to use:")
        for pid in context_ids:
            # We only have IDs here, ideally we'd have titles too.
            # For now, show ID or generic label.
            # Enhanced: Semantic Search should verify_titles and store dicts not just IDs.
            is_selected = st.checkbox(f"Paper {pid[:8]}...", value=True, key=f"chk_{pid}")
            if is_selected:
                selected_papers.append(pid)
        
        st.session_state.active_rag_papers = selected_papers
    else:
        st.caption("No papers selected from Semantic Search.")
        st.caption("Go to 'Semantic Search', find papers, then return here.")
    
    st.markdown("---")
    st.markdown("### Example Questions")
    examples = [
        "What are transformers in NLP?",
        "Explain attention mechanisms",
        "Recent advances in computer vision",
    ]
    for example in examples:
        if st.button(example, key=example):
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()
            
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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- **{source.get('title', 'Unknown')}** ({source.get('year', 'N/A')})")

# Chat input
if prompt := st.chat_input("Ask a question about research papers..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                import requests
                
                response = requests.post(
                    f"{API_URL}/rag/ask",
                    json={
                        "question": prompt,
                        "conversation_id": st.session_state.conversation_id,
                        "top_k": top_k,
                        "temperature": temperature,
                        "filter_ids": st.session_state.get("active_rag_papers", st.session_state.get("context_paper_ids", []))
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer generated.")
                    sources = data.get("sources", [])
                    st.session_state.conversation_id = data.get("conversation_id")
                else:
                    answer = f"Error: {response.text}"
                    sources = []
                
                st.markdown(answer)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Failed to generate answer: {str(e)}")
