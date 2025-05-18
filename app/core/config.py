from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "llm-by-zero"
    
    # Database settings
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/llm_training"
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # MinIO settings
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "llm-models"
    
    # Model training settings
    BATCH_SIZE: int = 128
    CONTEXT_LENGTH: int = 128
    D_MODEL: int = 256
    NUM_BLOCKS: int = 4
    NUM_HEADS: int = 4
    LEARNING_RATE: float = 3e-4
    DROPOUT: float = 0.1
    MAX_ITERS: int = 2000
    EVAL_INTERVAL: int = 50
    EVAL_ITERS: int = 10
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings() 