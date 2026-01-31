"""
Unit tests for Configuration Management
Tests config loading, validation, and error handling
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import ValidationError


class TestConfiguration:
    """Test cases for configuration management"""

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing"""
        return {
            "GROQ_API_KEY": "test_groq_key_12345",
            "ENDEE_URL": "http://localhost:8080",
            "POSTGRES_URL": "postgresql://user:pass@localhost:5432/testdb",
            "LOG_LEVEL": "DEBUG",
            "CACHE_SIZE": "500",
            "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
            "EMBEDDING_DIMENSION": "384",
        }

    @pytest.fixture
    def minimal_env_vars(self):
        """Minimal required environment variables"""
        return {
            "GROQ_API_KEY": "test_key",
        }

    def test_config_loading(self, mock_env_vars):
        """Test successful configuration loading with all variables"""
        # Arrange
        with patch.dict(os.environ, mock_env_vars, clear=True):
            # Act
            from backend.config import Settings
            config = Settings()

            # Assert
            assert config.GROQ_API_KEY == "test_groq_key_12345"
            assert config.ENDEE_URL == "http://localhost:8080"
            assert config.LOG_LEVEL == "DEBUG"
            assert config.CACHE_SIZE == 500
            assert config.EMBEDDING_DIMENSION == 384

    def test_config_with_defaults(self, minimal_env_vars):
        """Test configuration with default values"""
        # Arrange
        with patch.dict(os.environ, minimal_env_vars, clear=True):
            # Act
            from backend.config import Settings
            config = Settings()

            # Assert
            assert config.GROQ_API_KEY == "test_key"
            assert config.ENDEE_URL == "http://localhost:8080"  # Default
            assert config.LOG_LEVEL == "INFO"  # Default
            assert config.CACHE_SIZE == 1000  # Default
            assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"  # Default

    def test_missing_env_vars(self):
        """Test error handling for missing required variables"""
        # Arrange
        with patch.dict(os.environ, {}, clear=True):
            # Act & Assert
            from backend.config import Settings
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            
            # Verify the error is about missing GROQ_API_KEY
            assert "GROQ_API_KEY" in str(exc_info.value)

    def test_validation(self, mock_env_vars):
        """Test configuration validation"""
        # Arrange
        invalid_env = mock_env_vars.copy()
        invalid_env["GROQ_TEMPERATURE"] = "3.0"  # Invalid: should be <= 2.0
        invalid_env["CACHE_SIZE"] = "-100"  # Invalid: should be positive

        with patch.dict(os.environ, invalid_env, clear=True):
            # Act & Assert
            from backend.config import Settings
            with pytest.raises(ValidationError):
                Settings()

    def test_singleton_pattern(self, mock_env_vars):
        """Test that get_settings returns the same instance"""
        # Arrange
        with patch.dict(os.environ, mock_env_vars, clear=True):
            # Act
            from backend.config import get_settings
            
            # Clear the cache first
            get_settings.cache_clear()
            
            config1 = get_settings()
            config2 = get_settings()

            # Assert
            assert config1 is config2  # Same instance

    def test_temperature_validation(self, mock_env_vars):
        """Test temperature parameter validation"""
        # Arrange
        env_with_temp = mock_env_vars.copy()
        env_with_temp["GROQ_TEMPERATURE"] = "0.5"
        env_with_temp["RAG_TEMPERATURE"] = "0.8"

        with patch.dict(os.environ, env_with_temp, clear=True):
            # Act
            from backend.config import Settings
            config = Settings()

            # Assert
            assert config.GROQ_TEMPERATURE == 0.5
            assert config.RAG_TEMPERATURE == 0.8
            assert 0.0 <= config.GROQ_TEMPERATURE <= 2.0
            assert 0.0 <= config.RAG_TEMPERATURE <= 2.0

    def test_top_k_validation(self, mock_env_vars):
        """Test top_k parameter validation"""
        # Arrange
        env_with_topk = mock_env_vars.copy()
        env_with_topk["DEFAULT_TOP_K"] = "15"
        env_with_topk["MAX_TOP_K"] = "50"

        with patch.dict(os.environ, env_with_topk, clear=True):
            # Act
            from backend.config import Settings
            config = Settings()

            # Assert
            assert config.DEFAULT_TOP_K == 15
            assert config.MAX_TOP_K == 50
            assert config.DEFAULT_TOP_K <= config.MAX_TOP_K

    def test_boolean_config(self, mock_env_vars):
        """Test boolean configuration values"""
        # Arrange
        env_with_bool = mock_env_vars.copy()
        env_with_bool["ENABLE_CACHE"] = "true"

        with patch.dict(os.environ, env_with_bool, clear=True):
            # Act
            from backend.config import Settings
            config = Settings()

            # Assert
            assert config.ENABLE_CACHE is True

    def test_integer_validation(self, mock_env_vars):
        """Test integer field validation"""
        # Arrange
        env_with_ints = mock_env_vars.copy()
        env_with_ints["CACHE_SIZE"] = "2000"
        env_with_ints["MAX_WORKERS"] = "8"
        env_with_ints["API_PORT"] = "9000"

        with patch.dict(os.environ, env_with_ints, clear=True):
            # Act
            from backend.config import Settings
            config = Settings()

            # Assert
            assert config.CACHE_SIZE == 2000
            assert config.MAX_WORKERS == 8
            assert config.API_PORT == 9000
            assert isinstance(config.CACHE_SIZE, int)

    def test_config_case_sensitivity(self, mock_env_vars):
        """Test that configuration is case-sensitive"""
        # Arrange
        with patch.dict(os.environ, mock_env_vars, clear=True):
            # Act
            from backend.config import Settings
            config = Settings()

            # Assert
            # Verify exact case is preserved
            assert hasattr(config, 'GROQ_API_KEY')
            assert hasattr(config, 'ENDEE_URL')
