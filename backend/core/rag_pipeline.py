"""
RAG Pipeline - Retrieval-Augmented Generation
Combines vector search with LLM generation
"""

from typing import List, Dict, Optional
import structlog

from backend.core.endee_client import EndeeVectorDB
from backend.core.embeddings import EmbeddingService
from backend.core.groq_client import GroqClient

logger = structlog.get_logger()


class RAGPipeline:
    """
    RAG Pipeline for question answering
    Retrieves relevant context and generates answers
    """

    def __init__(
        self,
        endee_client: EndeeVectorDB,
        embedding_service: EmbeddingService,
        groq_client: GroqClient,
        index_name: str = "research_papers_dense",
    ):
        """
        Initialize RAG pipeline

        Args:
            endee_client: Endee vector database client
            embedding_service: Embedding service
            groq_client: Groq API client
            index_name: Index to search for context
        """
        self.endee = endee_client
        self.embeddings = embedding_service
        self.groq = groq_client
        self.index_name = index_name
        self.conversations: Dict[str, List[Dict]] = {}
        logger.info("rag_pipeline_initialized", index=index_name)

    def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0.7,
        filter_ids: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Ask a question and get RAG-powered answer

        Args:
            question: User question
            conversation_id: Optional conversation ID for context
            top_k: Number of sources to retrieve
            temperature: LLM temperature

        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            logger.info("rag_ask", question=question, top_k=top_k)

            # 1. Embed question
            query_vector = self.embeddings.embed_text(question)

            # 2. Retrieve relevant chunks
            search_filter = None
            if filter_ids:
                # Filter by list of IDs
                # Endee/LanceDB usually support 'in' operator or list for ID
                # We'll use a standard MongoDB-like style if supported, or multiple ORs
                # Assuming Endee client handles list as 'in'
                search_filter = {"id": filter_ids}
                
            results = self.endee.search_dense(
                index_name=self.index_name, 
                query_vector=query_vector, 
                top_k=top_k,
                filters=search_filter
            )

            # 3. Assemble context
            context = self._assemble_context(results)

            # 4. Get conversation history if available
            history = None
            if conversation_id and conversation_id in self.conversations:
                history = self.conversations[conversation_id]

            # 5. Generate answer
            response = self.groq.generate_with_context(
                question=question,
                context=context,
                conversation_history=history,
                temperature=temperature,
            )

            # 6. Store conversation
            if conversation_id:
                self._update_conversation(conversation_id, question, response["text"])

            # 7. Extract sources
            sources = self._extract_sources(results)

            return {
                "answer": response["text"],
                "sources": sources,
                "conversation_id": conversation_id or "new",
                "tokens_used": response["tokens_used"],
            }

        except Exception as e:
            logger.error("rag_ask_failed", error=str(e), question=question)
            raise

    def _assemble_context(self, results: List[Dict]) -> str:
        """Assemble context from search results with paper titles and years"""
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result.get("meta", {})
            # Prefer chunk_text (specific context) over full abstract for RAG
            text = meta.get("chunk_text", meta.get("abstract", ""))
            if not text:
                continue
                
            title = meta.get("title", "Unknown")
            year = meta.get("year", "N/A")
            
            # Use a clean format that the LLM can easily cite
            context_parts.append(f"SOURCE [{i}]: {title} ({year})\nCONTENT: {text}\n")
            
        if not context_parts:
            return "No relevant information found in source papers."
            
        return "\n".join(context_parts)

    def _extract_sources(self, results: List[Dict]) -> List[Dict]:
        """Extract unique source papers from results for the UI"""
        sources = []
        seen_titles = set()
        
        for result in results:
            meta = result.get("meta", {})
            title = meta.get("title", "").strip()
            title_norm = title.lower()
            
            if not title or title_norm in seen_titles:
                continue
                
            seen_titles.add(title_norm)
            sources.append(
                {
                    "id": result.get("id", ""),
                    "title": title,
                    "abstract": meta.get("abstract", ""),
                    "authors": meta.get("authors", []),
                    "year": meta.get("year", 0),
                    "similarity": result.get("score", result.get("similarity", 0.0)),
                    "url": meta.get("url", ""),
                    "category": meta.get("category", ""),
                    "citations": meta.get("citations", 0),
                }
            )
        return sources

    def _update_conversation(self, conversation_id: str, question: str, answer: str):
        """Update conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

        # Keep only last 5 turns
        if len(self.conversations[conversation_id]) > 10:
            self.conversations[conversation_id] = self.conversations[conversation_id][-10:]

    def get_conversation(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
