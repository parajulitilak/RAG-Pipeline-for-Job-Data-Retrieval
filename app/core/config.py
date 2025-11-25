from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings managed via Pydantic.
    Reads values from system environment variables or a .env file.
    """
    
    # External API Keys
    GOOGLE_API_KEY: str
    COHERE_API_KEY: str
    
    # Vector Database (Qdrant)
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "lf_jobs_hybrid"
    
    # AI Models
    LLM_MODEL: str
    EMBEDDING_MODEL: str = "embed-english-v3.0"
    RERANK_MODEL: str = "rerank-english-v3.0"
    
    # Cache (Redis)
    REDIS_URL: str = "redis://redis:6379"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" 

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings class.
    Prevents re-reading the .env file on every request.
    """
    return Settings()