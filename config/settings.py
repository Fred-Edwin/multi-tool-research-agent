"""Configuration settings for the Research Agent."""

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseModel):
    """Application settings."""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Weather API Configuration
    weather_api_key: str = os.getenv("WEATHER_API_KEY", "")
    weather_base_url: str = "http://api.openweathermap.org/data/2.5"
    
    # Agent Configuration
    max_sub_queries: int = 5
    tool_timeout: int = 30
    max_retries: int = 3
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def validate_required_keys(self) -> None:
        """Validate that required API keys are present."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.weather_api_key:
            raise ValueError("WEATHER_API_KEY is required for weather functionality")


# Global settings instance
settings = Settings()

# Validate on import
try:
    settings.validate_required_keys()
except ValueError as e:
    print(f"Warning: {e}")
    print("Please check your .env file and ensure all required API keys are set.")