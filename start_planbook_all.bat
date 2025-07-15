@echo off
REM Script setup va khoi dong hoan chinh cho PlanBook AI System

echo.
echo ============================================
echo    🎯 PlanBook AI System Startup Script
echo ============================================
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
    echo ❓ Continue without Redis? (Y/N)
    set /p continue="Press Y to continue or N to exit: "
    if /i not "%continue%"=="Y" (
        echo 🛑 Startup cancelled
        pause
        exit /b 1
    )
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
    echo ❓ Continue without MongoDB? (Y/N)
    set /p continue="Press Y to continue or N to exit: "
    if /i not "%continue%"=="Y" (
        echo 🛑 Startup cancelled
        pause
        exit /b 1
    )
) else (
    echo ✅ MongoDB: Connected
    echo.
)

REM Test core imports
echo 🧪 Testing core imports...
python -c "import fastapi, uvicorn, celery; print('✅ Core modules imported successfully')" 2>nul
if errorlevel 1 (
    echo ❌ Some core modules failed to import
    echo 💡 Installing missing dependencies...
    pip install fastapi uvicorn celery redis pymongo
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo 🎉 Prerequisites check completed!
echo.

REM ============================================
REM Start Services
REM ============================================

echo 🚀 Starting PlanBook AI Services...
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set CELERY_BROKER_URL=redis://localhost:6379/1
set CELERY_RESULT_BACKEND=redis://localhost:6379/1

echo 📋 Environment Variables:
echo   PYTHONPATH: %PYTHONPATH%
echo   CELERY_BROKER_URL: %CELERY_BROKER_URL%
echo   CELERY_RESULT_BACKEND: %CELERY_RESULT_BACKEND%
echo.

REM Kill any existing services
echo 🧹 Cleaning up existing services...
taskkill /f /im "python.exe" 2>nul >nul
echo ✅ Cleanup completed
echo.

REM Start services in sequence

echo ⚡ Starting Celery Worker (with slide_generation using solo pool)...
start "PlanBook AI - Celery Worker" cmd /k "title PlanBook AI - Celery Worker && echo 🎯 PlanBook AI - Celery Worker && echo ⚡ Starting Celery Worker... && python -m celery -A app.core.celery_app worker --loglevel=info --pool=solo --concurrency=1 --queues=pdf_queue,embeddings_queue,cv_queue,slide_generation_queue,default --hostname=planbook_worker@%%h"
echo ✅ Celery Worker started in separate window
echo.

echo ⏳ Waiting for Celery Worker to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

echo 🌸 Starting Celery Flower (Task Monitor)...
start "PlanBook AI - Celery Flower" cmd /k "title PlanBook AI - Celery Flower && echo 🎯 PlanBook AI - Celery Flower && echo 🌸 Starting Celery Flower... && python -m celery -A app.core.celery_app flower --port=5555"
echo ✅ Celery Flower started in separate window
echo.

echo ⏳ Waiting for Flower to initialize (3 seconds)...
timeout /t 3 /nobreak >nul

echo 🌐 Starting FastAPI Server...
start "PlanBook AI - FastAPI Server" cmd /k "title PlanBook AI - FastAPI Server && echo 🎯 PlanBook AI - FastAPI Server && echo 🌐 Starting FastAPI Server... && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ✅ FastAPI Server started in separate window
echo.

echo ⏳ Waiting for FastAPI to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

REM ============================================
REM Service Status
REM ============================================

echo.
echo ============================================
echo          🎉 ALL SERVICES STARTED!
echo ============================================
echo.

echo 📊 Access Points:
echo   🌐 FastAPI Server:     http://localhost:8000
echo   📚 API Documentation:  http://localhost:8000/docs
echo   📊 Interactive API:    http://localhost:8000/redoc
echo   🌸 Celery Flower:      http://localhost:5555
echo   🔧 Health Check:       http://localhost:8000/health
echo.

echo 🎯 Service Windows (check taskbar):
echo   ⚡ PlanBook AI - Celery Worker
echo   🌸 PlanBook AI - Celery Flower
echo   🌐 PlanBook AI - FastAPI Server
echo.

echo 💡 Management Commands:
echo   ✅ Stop services:      Close the respective windows or Ctrl+C
echo   🔄 Restart all:        Run this script again
echo   📊 Monitor tasks:      http://localhost:5555
echo   🧪 Test API:           http://localhost:8000/docs
echo   📝 View logs:          Check service windows
echo.

echo 🚀 Testing API connection...
timeout /t 2 /nobreak >nul
python -c "import requests; r = requests.get('http://localhost:8000/health', timeout=5); print('✅ API Health:', r.json().get('status', 'Unknown'))" 2>nul
if errorlevel 1 (
    echo ⚠️  API might still be starting up...
)

echo.
echo ⏰ All services should be ready now!
echo 🎉 PlanBook AI is running and ready to use!
echo.
echo 📌 Keep this window open for reference
echo 📌 Check the service windows for detailed logs
echo.

pause
