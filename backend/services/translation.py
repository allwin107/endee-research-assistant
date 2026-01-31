"""
Translation Service
Handles language detection and translation using Groq
"""

from typing import Optional, List, Dict, Any
from functools import lru_cache
import structlog
from pydantic import BaseModel

from backend.core.groq_client import GroqClient
from backend.config import get_settings
from backend.utils.cache import SimpleCache

logger = structlog.get_logger(__name__)


class TranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: Optional[str] = None


class TranslationService:
    """
    Service for translating text and detecting language
    Uses Groq LLM for high-quality translation
    """

    def __init__(self):
        settings = get_settings()
        self.groq_client = GroqClient(api_key=settings.GROQ_API_KEY)
        self.cache = SimpleCache(max_size=1000, default_ttl=86400)  # Cache for 24h
        
        # Supported languages
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese",
            "ar": "Arabic",
            "he": "Hebrew",
            "hi": "Hindi",
            "pt": "Portuguese"
        }

    async def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Input text
            
        Returns:
            Language code (e.g., 'en', 'es')
        """
        if not text or len(text.strip()) < 2:
            return "en"

        cache_key = f"detect:{text}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            # Simple heuristic for common scripts if LLM is too slow/costly
            # For now, we use a fast LLM prompt
            prompt = f"""
            Detect the language of the following text. Return ONLY the 2-letter ISO code (e.g., en, es, fr).
            Text: "{text[:100]}"
            Language Code:
            """
            
            # Synch call
            response = self.groq_client.generate(prompt, max_tokens=5, temperature=0.0)
            lang_code = response.get("text", "").strip().lower()[:2]
            
            # Default to English if detection fails or returns garbage
            if lang_code not in self.supported_languages:
                lang_code = "en"
                
            self.cache.set(cache_key, lang_code)
            return lang_code
            
        except Exception as e:
            logger.error("language_detection_failed", error=str(e))
            return "en"

    async def translate(self, text: str, target_lang: str, source_lang: str = None) -> str:
        """
        Translate text to target language
        
        Args:
            text: Input text
            target_lang: Target language code
            source_lang: Source language code (optional)
            
        Returns:
            Translated text
        """
        if not text:
            return ""
            
        if source_lang == target_lang:
            return text

        # If source not provided, try to detect or assume it might be needed
        if not source_lang:
            source_lang = await self.detect_language(text)
            if source_lang == target_lang:
                return text

        cache_key = f"trans:{source_lang}:{target_lang}:{text}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            target_name = self.supported_languages.get(target_lang, target_lang)
            
            prompt = f"""
            Translate the following academic/technical text to {target_name}.
            Maintain the original meaning and technical terms where appropriate.
            Return ONLY the translated text, no explanations.
            
            Text: "{text}"
            Translation:
            """
            
            translated = self.groq_client.generate(prompt, temperature=0.3)
            translated_text = translated.get("text", "").strip().strip('"')
            
            self.cache.set(cache_key, translated_text)
            return translated_text
            
        except Exception as e:
            logger.error("translation_failed", error=str(e), target=target_lang)
            return text

    async def translate_search_results(self, results: List[Any], target_lang: str) -> List[Any]:
        """
        Translate a list of search results
        
        Args:
            results: List of paper objects or dictionaries
            target_lang: Target language code
            
        Returns:
            List of translated results
        """
        if target_lang == "en":
            return results

        translated_results = []
        for result in results:
            # We need to handle both Dict and Pydantic models (SearchResult)
            is_dict = isinstance(result, dict)
            
            # Translate title
            title = result.get("title", "") if is_dict else getattr(result, "title", "")
            title_translated = await self.translate(title, target_lang, "en")
            
            if is_dict:
                localized = result.copy()
                localized["title_translated"] = title_translated
                translated_results.append(localized)
            else:
                # If it's a Pydantic model, we can't easily add a new field without modifying the class
                # But the route expecting this might cast it. 
                # For now, let's keep it consistent: if it was an object, return an object or a metadata-enhanced dict
                # However, the SearchResult model doesn't have 'title_translated'.
                # Let's check how the route uses it.
                # Actually, many Pydantic models allow extra fields if Config.extra = 'allow', but SearchResult doesn't.
                # The safest for now is to just update the title attribute if we want to "translate" the result object.
                result.title = title_translated
                translated_results.append(result)
            
        return translated_results

# Singleton instance
_translation_service = None

def get_translation_service() -> TranslationService:
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service
