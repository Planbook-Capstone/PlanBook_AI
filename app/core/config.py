import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "PlanBook AI Service")

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyBcaQNOdHn6xUI7RKKuFm0KynV4GCL1cVU")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your_secret_key_here")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    class Config:
        case_sensitive = True

settings = Settings()
