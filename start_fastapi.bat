@echo off
REM Script để chạy FastAPI server cho PlanBook AI

echo 🚀 Starting FastAPI Server for PlanBook AI...

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%

echo 📋 Environment:
echo   PYTHONPATH: %PYTHONPATH%
echo   Server: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.

REM Check dependencies
echo 🔍 Checking dependencies...

REM Check Redis connection
python -c "import redis; r = redis.Redis.from_url('redis://localhost:6379/1'); r.ping(); print('✅ Redis connection successful')" 2>nul
if errorlevel 1 (
    echo ⚠️  Redis connection failed. Some features may not work.
    echo 💡 To start Redis: redis-server
)

REM Check MongoDB connection  
python -c "import pymongo; client = pymongo.MongoClient('mongodb://localhost:27017/'); client.admin.command('ping'); print('✅ MongoDB connection successful')" 2>nul
if errorlevel 1 (
    echo ⚠️  MongoDB connection failed. Database features may not work.
    echo 💡 To start MongoDB: mongod
)

echo.
echo 🌐 Starting FastAPI server...
echo 📊 Access points:
echo   - API Server: http://localhost:8000
echo   - Interactive Docs: http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo.
echo ⏹️  Press Ctrl+C to stop the server
echo.

REM Start FastAPI with uvicorn
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

echo ✅ FastAPI server stopped
pause
