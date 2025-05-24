#!/usr/bin/env python
import sys
import os
import uvicorn

def main():
    """
    CLI tool to run FastAPI applications in development mode.
    Usage: python run.py dev [path_to_app]
    Example: python run.py dev app/api/api.py
    """
    if len(sys.argv) < 2:
        print("Usage: python run.py dev [path_to_app]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "dev":
        if len(sys.argv) < 3:
            # Default to app.api.api:app if no path is provided
            app_path = "app.api.api:app"
        else:
            # Convert file path to module path
            app_path = sys.argv[2].replace("/", ".").replace("\\", ".")
            if app_path.endswith(".py"):
                app_path = app_path[:-3]
            app_path += ":app"
        
        print(f"Starting development server for {app_path}")
        uvicorn.run(
            app_path,
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    else:
        print(f"Unknown command: {command}")
        print("Available commands: dev")
        sys.exit(1)

if __name__ == "__main__":
    main()
