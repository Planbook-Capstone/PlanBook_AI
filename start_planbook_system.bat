@echo off
REM Script tổng hợp để khởi động toàn bộ PlanBook AI System

echo 🎯 PlanBook AI System Startup
echo ===============================
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set CELERY_BROKER_URL=redis://localhost:6379/1
set CELERY_RESULT_BACKEND=redis://localhost:6379/1

echo 📋 Environment Configuration:
echo   PYTHONPATH: %PYTHONPATH%
echo   CELERY_BROKER_URL: %CELERY_BROKER_URL%
echo   CELERY_RESULT_BACKEND: %CELERY_RESULT_BACKEND%
echo.

REM Check dependencies
echo 🔍 Checking System Dependencies...

REM Check Redis
echo   Checking Redis...
python -c "import redis; r = redis.Redis.from_url('redis://localhost:6379/1'); r.ping(); print('✅ Redis: OK')" 2>nul
if errorlevel 1 (
    echo   ❌ Redis: NOT RUNNING
    echo   💡 Please start Redis first: redis-server
    echo.
    pause
    exit /b 1
)

REM Check MongoDB
echo   Checking MongoDB...
python -c "import pymongo; client = pymongo.MongoClient('mongodb://localhost:27017/'); client.admin.command('ping'); print('✅ MongoDB: OK')" 2>nul
if errorlevel 1 (
    echo   ❌ MongoDB: NOT RUNNING
    echo   💡 Please start MongoDB first: mongod
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ All dependencies are running!
echo.

REM Start services
echo 🚀 Starting PlanBook AI Services...
echo.

REM Start Celery Worker
echo ⚡ Starting Celery Worker...
start "PlanBook AI - Celery Worker" cmd /k "title PlanBook AI - Celery Worker && python -m celery -A app.core.celery_app worker --loglevel=info --pool=threads --concurrency=4 --include=app.tasks.pdf_tasks --queues=pdf_queue,default --hostname=planbook_worker@%%h"

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
