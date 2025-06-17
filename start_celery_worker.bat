@echo off
REM Script để chạy Celery Worker trên Windows

echo 🚀 Starting Celery Worker for PlanBook AI...

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set CELERY_BROKER_URL=redis://localhost:6379/1
set CELERY_RESULT_BACKEND=redis://localhost:6379/1

echo 📋 Environment:
echo   PYTHONPATH: %PYTHONPATH%
echo   CELERY_BROKER_URL: %CELERY_BROKER_URL%
echo   CELERY_RESULT_BACKEND: %CELERY_RESULT_BACKEND%
echo.

REM Start với include tasks
echo ⚡ Starting Celery Worker with PDF tasks...
python -m celery -A app.core.celery_app worker ^
    --loglevel=info ^
    --pool=solo ^
    --concurrency=1 ^
    --include=app.tasks.pdf_tasks ^
    --queues=pdf_queue,default ^
    --hostname=planbook_worker@%%h

echo ✅ Celery Worker stopped
pause
