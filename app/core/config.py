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
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    # OpenRouter Configuration
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

    OPENROUTER_SITE_URL: Optional[str] = os.getenv("OPENROUTER_SITE_URL")
    OPENROUTER_SITE_NAME: Optional[str] = os.getenv("OPENROUTER_SITE_NAME")

    # MongoDB Configuration
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017/planbook_db")
    MONGODB_DATABASE: str = os.getenv(
        "MONGODB_DATABASE", "planbook_db"
    )  # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(
        os.getenv("REDIS_DB", "0")
    )  # Celery Configuration (sử dụng DB riêng để tránh conflict)
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/1"
    )
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: list = ["json"]
    CELERY_TIMEZONE: str = "Asia/Ho_Chi_Minh"
    CELERY_ENABLE_UTC: bool = True

    # RAG Settings - Optimized for Vietnamese and performance
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1500"))  # Word-based chunking
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))     # Reduced overlap
    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))

    # Qdrant Settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # File Upload Settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "209715200"))  # 200MB
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1048576"))  # 1MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt"]
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "temp_uploads")

    # Data Processing Settings
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    CONCURRENT_WORKERS: int = int(os.getenv("CONCURRENT_WORKERS", "4"))

    # OCR Settings
    TESSERACT_CONFIG: str = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6")
    IMAGE_DPI: int = int(os.getenv("IMAGE_DPI", "300"))
    PREPROCESSING_ENABLED: bool = (
        os.getenv("PREPROCESSING_ENABLED", "True").lower() == "true"
    )

    # AI Agent Settings
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))

    # Google Drive Settings
    GOOGLE_DRIVE_CREDENTIALS_PATH: Optional[str] = os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH")
    GOOGLE_DRIVE_FOLDER_ID: Optional[str] = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    GOOGLE_DRIVE_AUTO_DELETE_DAYS: int = int(os.getenv("GOOGLE_DRIVE_AUTO_DELETE_DAYS", "7"))
    ENABLE_GOOGLE_DRIVE: bool = os.getenv("ENABLE_GOOGLE_DRIVE", "False").lower() == "true"

    # Supabase Settings
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
    SUPABASE_BUCKET_NAME: str = os.getenv("SUPABASE_BUCKET_NAME", "pdf-documents")

    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPIC_NAME: str = os.getenv("KAFKA_TOPIC_NAME", "planbook")
    KAFKA_CONNECTION_TIMEOUT: int = int(os.getenv("KAFKA_CONNECTION_TIMEOUT", "10"))
    KAFKA_REQUEST_TIMEOUT: int = int(os.getenv("KAFKA_REQUEST_TIMEOUT", "30"))
    KAFKA_AUTO_CREATE_TOPICS: bool = os.getenv("KAFKA_AUTO_CREATE_TOPICS", "True").lower() == "true"

    # Security
    SECRET_KEY: Optional[str] = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )

    class Config:
        case_sensitive = True

    def __post_init__(self):
        # Only validate API keys for main application, not for Celery workers
        import sys

        if "celery" not in " ".join(sys.argv).lower():
            # Check if we have either Gemini or OpenRouter API key
            if not self.GEMINI_API_KEY and not self.OPENROUTER_API_KEY:
                print("⚠️ Warning: Neither GEMINI_API_KEY nor OPENROUTER_API_KEY is set")
            elif self.OPENROUTER_API_KEY:
                print("✅ Using OpenRouter API")
            elif self.GEMINI_API_KEY:
                print("✅ Using Gemini API")

            if not self.SECRET_KEY or self.SECRET_KEY == "your_secret_key_here":
                # Tự động tạo SECRET_KEY nếu không có
                import secrets
                self.SECRET_KEY = secrets.token_urlsafe(32)
                print("⚠️ Warning: SECRET_KEY not set, generated temporary key for this session")
                print("💡 Tip: Set SECRET_KEY environment variable for production")


settings = Settings()
