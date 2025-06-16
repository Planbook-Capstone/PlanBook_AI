#!/usr/bin/env python3
"""
Script tổng hợp để start toàn bộ PlanBookAI với Celery
Cách sử dụng:
    python scripts/start_planbook_celery.py --mode local
    python scripts/start_planbook_celery.py --mode docker
    python scripts/start_planbook_celery.py --mode worker-only
"""

import os
import sys
import time
import argparse
import subprocess
import threading
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_redis():
    """Kiểm tra Redis có chạy không"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        return True
    except:
        return False

def check_mongodb():
    """Kiểm tra MongoDB có chạy không"""
    try:
        import pymongo
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        client.admin.command('ping')
        return True
    except:
        return False

def start_redis():
    """Start Redis server"""
    print("🚀 Starting Redis server...")
    try:
        subprocess.Popen(['redis-server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        if check_redis():
            print("✅ Redis started successfully")
            return True
        else:
            print("❌ Failed to start Redis")
            return False
    except Exception as e:
        print(f"❌ Error starting Redis: {e}")
        return False

def start_mongodb():
    """Start MongoDB server"""
    print("🚀 Starting MongoDB server...")
    try:
        subprocess.Popen(['mongod'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        if check_mongodb():
            print("✅ MongoDB started successfully")
            return True
        else:
            print("❌ Failed to start MongoDB")
            return False
    except Exception as e:
        print(f"❌ Error starting MongoDB: {e}")
        return False

def start_celery_worker(queue, concurrency=1):
    """Start Celery worker trong thread riêng"""
    def run_worker():
        cmd = [
            "celery", 
            "-A", "app.core.celery_app:celery_app",
            "worker",
            "--loglevel", "info",
            "--queues", queue,
            "--concurrency", str(concurrency),
            "--pool", "prefork"
        ]
        
        print(f"🔧 Starting Celery worker for queue: {queue}")
        try:
            subprocess.run(cmd, cwd=project_root)
        except KeyboardInterrupt:
            print(f"⏹️  Worker {queue} stopped")
    
    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()
    return thread

def start_fastapi():
    """Start FastAPI server trong thread riêng"""
    def run_fastapi():
        cmd = ["fastapi", "dev", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
        
        print("🚀 Starting FastAPI server...")
        try:
            subprocess.run(cmd, cwd=project_root)
        except KeyboardInterrupt:
            print("⏹️  FastAPI server stopped")
    
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()
    return thread

def start_local_mode():
    """Start tất cả services locally"""
    print("🚀 Starting PlanBookAI in LOCAL mode...")
    print("=" * 50)
    
    # Check và start Redis
    if not check_redis():
        if not start_redis():
            print("❌ Cannot start Redis. Please install and start Redis manually.")
            return False
    else:
        print("✅ Redis is already running")
    
    # Check và start MongoDB
    if not check_mongodb():
        if not start_mongodb():
            print("❌ Cannot start MongoDB. Please install and start MongoDB manually.")
            return False
    else:
        print("✅ MongoDB is already running")
    
    print("\n🔧 Starting Celery workers...")
    
    # Start workers
    workers = []
    workers.append(start_celery_worker("pdf_queue", 2))
    workers.append(start_celery_worker("embeddings_queue", 1))
    workers.append(start_celery_worker("cv_queue", 1))
    
    time.sleep(3)  # Wait for workers to start
    
    # Start FastAPI
    fastapi_thread = start_fastapi()
    
    print("\n✅ All services started!")
    print("📊 Access points:")
    print("  - FastAPI: http://localhost:8000")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - Celery Health: http://localhost:8000/api/v1/celery/health")
    print("\n⏹️  Press Ctrl+C to stop all services")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping all services...")
        return True

def start_docker_mode():
    """Start với Docker Compose"""
    print("🚀 Starting PlanBookAI in DOCKER mode...")
    print("=" * 50)
    
    cmd = ["docker-compose", "-f", "docker-compose.celery.yml", "up", "--build"]
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping Docker services...")
        subprocess.run(["docker-compose", "-f", "docker-compose.celery.yml", "down"], cwd=project_root)

def start_worker_only():
    """Chỉ start Celery workers"""
    print("🔧 Starting CELERY WORKERS only...")
    print("=" * 50)
    
    # Check dependencies
    if not check_redis():
        print("❌ Redis is not running. Please start Redis first.")
        return False
    
    if not check_mongodb():
        print("❌ MongoDB is not running. Please start MongoDB first.")
        return False
    
    print("✅ Dependencies are running")
    
    # Start workers
    workers = []
    workers.append(start_celery_worker("pdf_queue", 2))
    workers.append(start_celery_worker("embeddings_queue", 1))
    workers.append(start_celery_worker("cv_queue", 1))
    
    print("\n✅ All workers started!")
    print("⏹️  Press Ctrl+C to stop workers")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping workers...")
        return True

def main():
    parser = argparse.ArgumentParser(description="Start PlanBookAI with Celery")
    
    parser.add_argument(
        "--mode", "-m",
        choices=["local", "docker", "worker-only"],
        default="local",
        help="Start mode (default: local)"
    )
    
    args = parser.parse_args()
    
    print("🎯 PlanBookAI Celery Startup Script")
    print(f"📁 Project root: {project_root}")
    print(f"🔧 Mode: {args.mode}")
    print("=" * 50)
    
    if args.mode == "local":
        start_local_mode()
    elif args.mode == "docker":
        start_docker_mode()
    elif args.mode == "worker-only":
        start_worker_only()

if __name__ == "__main__":
    main()
