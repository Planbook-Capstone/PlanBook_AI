@echo off
REM Script Windows để chạy đồng thời FastAPI và Celery

echo 🎯 PlanBook AI - Khởi động tất cả services
echo ==================================================

REM Kiểm tra và khởi động Docker services
echo 🔍 Kiểm tra Docker services...
docker-compose ps | findstr "Up" >nul
if errorlevel 1 (
    echo ⚠️  Khởi động Docker services...
    docker-compose up -d
    timeout /t 5 /nobreak >nul
)
echo ✅ Docker services đã sẵn sàng

REM Kích hoạt virtual environment nếu có
if exist "venv\Scripts\activate.bat" (
    echo 🐍 Kích hoạt virtual environment...
    call venv\Scripts\activate.bat
)

echo --------------------------------------------------
echo 🚀 Khởi động FastAPI và Celery...

REM Tạo file batch tạm để chạy FastAPI
echo @echo off > temp_fastapi.bat
echo echo 🚀 FastAPI Server đang chạy... >> temp_fastapi.bat
echo uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload >> temp_fastapi.bat

REM Tạo file batch tạm để chạy Celery Worker
echo @echo off > temp_celery.bat
echo echo 🔄 Celery Worker đang chạy... >> temp_celery.bat
echo celery -A app.core.celery_app worker --loglevel=info --pool=solo --concurrency=1 >> temp_celery.bat

REM Chạy FastAPI trong cửa sổ mới
start "FastAPI Server" cmd /k temp_fastapi.bat

REM Đợi 3 giây
timeout /t 3 /nobreak >nul

REM Chạy Celery Worker trong cửa sổ mới
start "Celery Worker" cmd /k temp_celery.bat

REM Nếu có tham số --flower, chạy Celery Flower
if "%1"=="--flower" (
    timeout /t 3 /nobreak >nul
    echo @echo off > temp_flower.bat
    echo echo 🌸 Celery Flower đang chạy... >> temp_flower.bat
    echo celery -A app.core.celery_app flower --port=5555 >> temp_flower.bat
    start "Celery Flower" cmd /k temp_flower.bat
)

echo.
echo 🎉 Tất cả services đã khởi động!
echo 📍 URLs:
echo    - FastAPI: http://localhost:8000
echo    - FastAPI Docs: http://localhost:8000/docs
if "%1"=="--flower" echo    - Celery Flower: http://localhost:5555
echo.
echo 💡 Đóng các cửa sổ terminal để dừng services
echo 💡 Hoặc chạy stop_services.bat để dừng tất cả

pause
