import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "PlanBook AI Service")

    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

    # MongoDB Configuration
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017/planbook_db")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "planbook_db")

    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # Celery Configuration
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: list = ["json"]
    CELERY_TIMEZONE: str = "Asia/Ho_Chi_Minh"
    CELERY_ENABLE_UTC: bool = True

    # RAG Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))

    # File Upload Settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "209715200"))  # 200MB
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1048576"))  # 1MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt"]
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "temp_uploads")

    # OCR Settings
    TESSERACT_CONFIG: str = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6")
    IMAGE_DPI: int = int(os.getenv("IMAGE_DPI", "300"))
    PREPROCESSING_ENABLED: bool = os.getenv("PREPROCESSING_ENABLED", "True").lower() == "true"

    # AI Agent Settings
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))

    # Security
    SECRET_KEY: Optional[str] = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    class Config:
        case_sensitive = True

    def __post_init__(self):
        # Only validate API keys for main application, not for Celery workers
        import sys
        if 'celery' not in ' '.join(sys.argv).lower():
            if not self.GEMINI_API_KEY:
                print("⚠️ Warning: GEMINI_API_KEY not set")
            if not self.SECRET_KEY or self.SECRET_KEY == "your_secret_key_here":
                print("⚠️ Warning: SECRET_KEY not set or using default")

settings = Settings()
