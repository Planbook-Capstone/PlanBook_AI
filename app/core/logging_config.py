"""
Logging configuration with UTF-8 encoding support
"""
import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_utf8_logging():
    """Setup logging with UTF-8 encoding to prevent charmap codec errors"""
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Force UTF-8 encoding for console output
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass
    
    root_logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    try:
        file_handler = RotatingFileHandler(
            os.path.join(logs_dir, 'planbook_ai.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return root_logger

def safe_log_text(text: str) -> str:
    """
    Safely encode text for logging to prevent encoding errors
    
    Args:
        text: Text that may contain Unicode characters
        
    Returns:
        ASCII-safe text for logging
    """
    if not text:
        return ""
    
    try:
        # Try to encode as ASCII, replace problematic characters
        return text.encode('ascii', 'replace').decode('ascii')
    except Exception:
        return str(text).encode('ascii', 'replace').decode('ascii')
