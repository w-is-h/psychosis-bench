"""Settings management with .env file support."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for psy-bench."""
    
    # API Configuration
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key for accessing various AI models"
    )
    
    # Model Configuration
    default_target_model: str = Field(
        default="google/gemini-2.5-flash",
        description="Default model to test"
    )
    
    default_judge_model: str = Field(
        default="openai/o4-mini",
        description="Default model for scoring responses"
    )
    
    # Experiment Configuration
    api_timeout: int = Field(
        default=300,
        description="Timeout for API calls in seconds"
    )
    
    api_retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed API calls"
    )
    
    api_retry_delay: int = Field(
        default=5,
        description="Delay between retry attempts in seconds"
    )
    
    # Scoring Configuration
    dcs_start_turn: int = Field(
        default=4,
        description="Turn number to start Delusion Confirmation Scoring"
    )
    
    hes_start_turn: int = Field(
        default=7,
        description="Turn number to start Harm Enablement Scoring"
    )
    
    # Output Configuration
    output_directory: str = Field(
        default="./outputs",
        description="Directory to save experiment results"
    )
    
    # Async Configuration
    max_concurrent_requests: int = Field(
        default=20,
        ge=1,  # Greater than or equal to 1
        le=100,  # Less than or equal to 100
        description="Maximum concurrent requests for async batch processing (1-100)"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False
    }

    def __init__(self, **kwargs):
        """Initialize settings and load from .env file."""
        # Load .env file from project root
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
        
        super().__init__(**kwargs)
    
    @property
    def has_api_key(self) -> bool:
        """Check if OpenRouter API key is configured."""
        return self.openrouter_api_key is not None and self.openrouter_api_key.strip() != ""
    
    def validate_setup(self) -> tuple[bool, list[str]]:
        """Validate that required settings are configured.
        
        Returns:
            tuple[bool, list[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        if not self.has_api_key:
            errors.append(
                "OpenRouter API key is not configured. "
                "Set OPENROUTER_API_KEY in your .env file or environment."
            )
        
        # Validate output directory is writable
        output_path = Path(self.output_directory)
        try:
            output_path.mkdir(exist_ok=True)
        except PermissionError:
            errors.append(f"Output directory '{self.output_directory}' is not writable.")
        
        return len(errors) == 0, errors


# Global settings instance
settings = Settings()