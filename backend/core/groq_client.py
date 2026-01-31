"""
Groq API Client for LLM interactions
"""

from typing import Optional, List, Dict
import structlog

from groq import Groq

logger = structlog.get_logger()


class GroqClient:
    """
    Client for Groq API interactions
    Handles LLM calls for RAG pipeline
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq client

        Args:
            api_key: Groq API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        logger.info("groq_client_initialized", model=model)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, any]:
        """
        Generate text using Groq LLM
        """
        try:
            logger.info("groq_generate", model=self.model, temperature=temperature)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Incorporate conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return {
                "text": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "model": self.model,
            }

        except Exception as e:
            logger.error("groq_generate_failed", error=str(e))
            raise

    def generate_with_context(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, any]:
        """
        Generate answer with context (for RAG)

        Args:
            question: User question
            context: Retrieved context from vector DB
            conversation_history: Previous conversation turns
            temperature: Sampling temperature

        Returns:
            Generated answer with metadata
        """
        system_prompt = """You are a helpful research assistant. Answer questions based on the provided research papers.
Always cite your sources using [Paper Title, Year] format.
If information is not in the context, say so clearly."""

        # Build prompt with context
        prompt = f"""Context from research papers:
{context}

Question: {question}

Please provide a detailed answer based on the context above, with proper citations."""

        return self.generate(
            prompt=prompt, 
            temperature=temperature, 
            system_prompt=system_prompt,
            conversation_history=conversation_history
        )
