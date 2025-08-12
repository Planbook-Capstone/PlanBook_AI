#!/usr/bin/env python3
"""
Script để chạy đồng thời FastAPI và Celery
"""
import subprocess
import sys
import os
import signal
import time
from threading import Thread

def run_fastapi():
    """Chạy FastAPI server"""
    print("🚀 Đang khởi động FastAPI server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  FastAPI server đã dừng")
    except Exception as e:
        print(f"❌ Lỗi khi chạy FastAPI: {e}")

def run_celery_worker():
    """Chạy Celery worker"""
    print("🔄 Đang khởi động Celery worker...")
    try:
        subprocess.run([
            "celery", "-A", "app.core.celery_app", 
            "worker", 
            "--loglevel=info", 
            "--pool=solo", 
            "--concurrency=1"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Celery worker đã dừng")
    except Exception as e:
        print(f"❌ Lỗi khi chạy Celery worker: {e}")

def run_celery_flower():
    """Chạy Celery flower (monitoring)"""
    print("🌸 Đang khởi động Celery Flower...")
    try:
        subprocess.run([
            "celery", "-A", "app.core.celery_app", 
            "flower", 
            "--port=5555"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Celery Flower đã dừng")
    except Exception as e:
        print(f"❌ Lỗi khi chạy Celery Flower: {e}")

def signal_handler(signum, frame):
    """Xử lý tín hiệu dừng"""
    print("\n🛑 Đang dừng tất cả services...")
    sys.exit(0)

def main():
    """Chạy tất cả services"""
    # Đăng ký signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🎯 PlanBook AI - Khởi động tất cả services")
    print("=" * 50)
    
    # Kiểm tra Docker services
    print("🔍 Kiểm tra Docker services...")
    try:
        result = subprocess.run(["docker-compose", "ps"], 
                              capture_output=True, text=True, check=True)
        if "Up" not in result.stdout:
            print("⚠️  Một số Docker services chưa chạy. Khởi động Docker services...")
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            time.sleep(5)  # Đợi services khởi động
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra Docker: {e}")
        return
    
    print("✅ Docker services đã sẵn sàng")
    print("-" * 50)
    
    # Tạo các thread cho từng service
    threads = []
    
    # FastAPI thread
    fastapi_thread = Thread(target=run_fastapi, daemon=True)
    threads.append(fastapi_thread)
    
    # Celery worker thread
    celery_thread = Thread(target=run_celery_worker, daemon=True)
    threads.append(celery_thread)
    
    # Celery flower thread (tùy chọn)
    if len(sys.argv) > 1 and "--flower" in sys.argv:
        flower_thread = Thread(target=run_celery_flower, daemon=True)
        threads.append(flower_thread)
    
    # Khởi động tất cả threads
    for thread in threads:
        thread.start()
        time.sleep(2)  # Đợi một chút giữa các service
    
    print("\n🎉 Tất cả services đã khởi động!")
    print("📍 URLs:")
    print("   - FastAPI: http://localhost:8000")
    print("   - FastAPI Docs: http://localhost:8000/docs")
    if "--flower" in sys.argv:
        print("   - Celery Flower: http://localhost:5555")
    print("\n💡 Nhấn Ctrl+C để dừng tất cả services")
    
    try:
        # Giữ main thread chạy
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Đang dừng tất cả services...")

if __name__ == "__main__":
    main()
