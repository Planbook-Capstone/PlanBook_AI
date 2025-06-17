@echo off
REM Script setup hoàn chỉnh cho PlanBook AI System

echo 🎯 PlanBook AI System Setup
echo ============================
echo.

REM Check Python installation
echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

python --version
echo ✅ Python is available
echo.

REM Check pip
echo 🔍 Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not available
    echo 💡 Please install pip
    pause
    exit /b 1
)

echo ✅ pip is available
echo.

REM Install dependencies
echo 📦 Installing Python dependencies...
echo This may take several minutes...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    echo 💡 Please check the error messages above
    pause
    exit /b 1
)

echo ✅ All Python dependencies installed successfully
echo.

REM Check Redis
echo 🔍 Checking Redis connection...
python -c "import redis; r = redis.Redis.from_url('redis://localhost:6379/1'); r.ping(); print('✅ Redis: Connected')" 2>nul
if errorlevel 1 (
    echo ⚠️  Redis: Not running
    echo 💡 Please start Redis server:
    echo    - Download Redis from: https://redis.io/download
    echo    - Or use Docker: docker run -d -p 6379:6379 redis:alpine
    echo    - Or use Windows Redis: https://github.com/microsoftarchive/redis/releases
    echo.
) else (
    echo ✅ Redis: Connected
    echo.
)

REM Check MongoDB
echo 🔍 Checking MongoDB connection...
python -c "import pymongo; client = pymongo.MongoClient('mongodb://localhost:27017/'); client.admin.command('ping'); print('✅ MongoDB: Connected')" 2>nul
if errorlevel 1 (
    echo ⚠️  MongoDB: Not running
    echo 💡 Please start MongoDB server:
    echo    - Download MongoDB from: https://www.mongodb.com/try/download/community
    echo    - Or use Docker: docker run -d -p 27017:27017 mongo:latest
    echo    - Start service: net start MongoDB (Windows)
    echo.
) else (
    echo ✅ MongoDB: Connected
    echo.
)

REM Test core imports
echo 🧪 Testing core imports...
python -c "
try:
    import fastapi
    import uvicorn
    import celery
    import sentence_transformers
    import qdrant_client
    import tf_keras
    print('✅ All core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if errorlevel 1 (
    echo ❌ Some modules failed to import
    echo 💡 Please check the error messages above
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📋 Next steps:
echo   1. Make sure Redis and MongoDB are running
echo   2. Run: start_planbook_system.bat (to start all services)
echo   3. Or run individual services:
echo      - FastAPI: start_fastapi.bat
echo      - Celery Worker: start_celery_worker.bat  
echo      - Celery Flower: start_celery_flower.bat
echo.
echo 📊 Access Points (after starting services):
echo   - FastAPI Server: http://localhost:8000
echo   - API Documentation: http://localhost:8000/docs
echo   - Celery Flower: http://localhost:5555
echo.

pause
