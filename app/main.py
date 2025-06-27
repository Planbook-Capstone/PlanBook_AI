# Setup UTF-8 logging before importing anything else
from app.core.logging_config import setup_utf8_logging
setup_utf8_logging()

from app.api.api import app

# Export app for uvicorn
__all__ = ["app"]
