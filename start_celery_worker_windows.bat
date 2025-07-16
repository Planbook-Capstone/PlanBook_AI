@echo off
REM Windows-compatible Celery worker startup script

echo 🎯 PlanBook AI - Windows Celery Worker
echo =====================================
echo.

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set CELERY_BROKER_URL=redis://localhost:6379/1
set CELERY_RESULT_BACKEND=redis://localhost:6379/1
set FORKED_BY_MULTIPROCESSING=1

echo 📋 Environment Configuration:
echo   PYTHONPATH: %PYTHONPATH%
echo   CELERY_BROKER_URL: %CELERY_BROKER_URL%
echo   CELERY_RESULT_BACKEND: %CELERY_RESULT_BACKEND%
echo   FORKED_BY_MULTIPROCESSING: %FORKED_BY_MULTIPROCESSING%
echo.

REM Check Redis
echo 🔍 Checking Redis...
python -c "import redis; r = redis.Redis.from_url('redis://localhost:6379/1'); r.ping(); print('✅ Redis: OK')" 2>nul
if errorlevel 1 (
    echo   ❌ Redis: NOT RUNNING
    echo   💡 Please start Redis first: redis-server
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Redis is running!
echo.

REM Start Windows-compatible Celery worker
echo 🚀 Starting Windows-compatible Celery worker...
echo.

python start_celery_worker_windows.py

echo.
echo 🛑 Worker stopped
echo.

pause
