#!/usr/bin/env python3
"""
Simple server runner for PlanBook AI
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.main import app
    import uvicorn
    
    if __name__ == "__main__":
        print("Starting PlanBook AI Server...")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to avoid issues
            log_level="info"
        )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error starting server: {e}")
    import traceback
    traceback.print_exc()
