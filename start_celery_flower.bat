@echo off
REM Script để chạy Celery Flower monitoring trên Windows

echo 🌸 Starting Celery Flower for PlanBook AI...

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set CELERY_BROKER_URL=redis://localhost:6379/1
set CELERY_RESULT_BACKEND=redis://localhost:6379/1

echo 📋 Environment:
echo   PYTHONPATH: %PYTHONPATH%
echo   CELERY_BROKER_URL: %CELERY_BROKER_URL%
echo   CELERY_RESULT_BACKEND: %CELERY_RESULT_BACKEND%
echo.

REM Check Redis connection
echo 🔍 Checking Redis connection...
python -c "import redis; r = redis.Redis.from_url('redis://localhost:6379/1'); r.ping(); print('✅ Redis connection successful')" 2>nul
if errorlevel 1 (
    echo ❌ Redis connection failed. Please start Redis server first.
    pause
    exit /b 1
)

REM Start Celery Flower
echo 🌸 Starting Celery Flower monitoring...
echo 📊 Flower will be available at: http://localhost:5555
echo.
python -m celery -A app.core.celery_app flower --port=5555

echo ✅ Celery Flower stopped
pause
