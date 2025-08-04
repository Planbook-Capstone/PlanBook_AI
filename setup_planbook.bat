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
python -c "import fastapi, uvicorn, celery, sentence_transformers, qdrant_client, tf_keras; print('✅ All core modules imported successfully')"

if errorlevel 1 (
    echo ❌ Some modules failed to import
    echo 💡 Please check the error messages above
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo � Starting PlanBook AI Services...
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set CELERY_BROKER_URL=redis://localhost:6379/1
set CELERY_RESULT_BACKEND=redis://localhost:6379/1

REM Start Celery Worker
echo ⚡ Starting Celery Worker (threads pool, concurrency=8, no singletons)...
start "PlanBook AI - Celery Worker" cmd /k "title PlanBook AI - Celery Worker && python -m celery -A app.core.celery_app worker --loglevel=info --pool=threads --concurrency=8 --queues=pdf_queue,embeddings_queue,cv_queue,slide_generation_queue,default --hostname=planbook_worker@%%h"

REM Wait for worker to start
timeout /t 3 /nobreak >nul

REM Start Celery Flower
echo 🌸 Starting Celery Flower...
start "PlanBook AI - Celery Flower" cmd /k "title PlanBook AI - Celery Flower && python -m celery -A app.core.celery_app flower --port=5555"

REM Wait for flower to start
timeout /t 3 /nobreak >nul

REM Start FastAPI
echo 🌐 Starting FastAPI Server...
start "PlanBook AI - FastAPI Server" cmd /k "title PlanBook AI - FastAPI Server && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo ✅ All services started successfully!
echo.
echo 📊 Access Points:
echo   - FastAPI Server: http://localhost:8000
echo   - API Documentation: http://localhost:8000/docs
echo   - Celery Flower: http://localhost:5555
echo.
echo 🎯 Service Windows:
echo   - Celery Worker: Running in separate window
echo   - Celery Flower: Running in separate window
echo   - FastAPI Server: Running in separate window
echo.
echo 💡 To stop services: Close the respective windows or press Ctrl+C in each
echo.

pause
