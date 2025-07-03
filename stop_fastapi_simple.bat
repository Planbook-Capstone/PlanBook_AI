@echo off
REM Script đơn giản để tắt FastAPI

echo [STOP] Stopping FastAPI server...

REM Tắt process đang chạy trên port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /PID %%a /F

REM Tắt tất cả process uvicorn
wmic process where "commandline like '%%uvicorn%%'" delete >nul 2>&1

echo [OK] FastAPI stopped!
