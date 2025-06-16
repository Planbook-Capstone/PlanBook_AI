#!/usr/bin/env python3
"""
Script để start Celery worker cho PlanBookAI
Cách sử dụng:
    python scripts/start_celery_worker.py
    python scripts/start_celery_worker.py --queue pdf_queue
    python scripts/start_celery_worker.py --concurrency 4
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def start_celery_worker(queue=None, concurrency=2, loglevel="info"):
    """
    Start Celery worker với các tùy chọn
    
    Args:
        queue: Queue name để worker xử lý (default: all queues)
        concurrency: Số worker processes (default: 2)
        loglevel: Log level (default: info)
    """
    
    # Đảm bảo chúng ta ở project root
    os.chdir(project_root)
    
    # Build command
    cmd = [
        "celery", 
        "-A", "app.core.celery_app:celery_app",
        "worker",
        "--loglevel", loglevel,
        "--concurrency", str(concurrency),
    ]
    
    # Thêm queue nếu được chỉ định
    if queue:
        cmd.extend(["--queues", queue])
    
    # Thêm các options khác
    cmd.extend([
        "--pool", "prefork",  # Sử dụng prefork pool cho stability
        "--without-gossip",   # Tắt gossip để giảm network traffic
        "--without-mingle",   # Tắt mingle để start nhanh hơn
        "--without-heartbeat", # Tắt heartbeat nếu không cần
    ])
    
    print("🚀 Starting Celery Worker...")
    print(f"📁 Project root: {project_root}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"📊 Queue: {queue or 'all queues'}")
    print(f"⚡ Concurrency: {concurrency}")
    print(f"📝 Log level: {loglevel}")
    print("-" * 50)
    
    try:
        # Start worker
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Worker stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting worker: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Start Celery Worker for PlanBookAI")
    
    parser.add_argument(
        "--queue", "-q",
        help="Queue name to process (pdf_queue, embeddings_queue, cv_queue)",
        choices=["pdf_queue", "embeddings_queue", "cv_queue", "default"]
    )
    
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=2,
        help="Number of worker processes (default: 2)"
    )
    
    parser.add_argument(
        "--loglevel", "-l",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Kiểm tra Redis connection trước khi start
    try:
        import redis
        from app.core.config import settings
        
        r = redis.from_url(settings.CELERY_BROKER_URL)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Make sure Redis is running: redis-server")
        sys.exit(1)
    
    start_celery_worker(
        queue=args.queue,
        concurrency=args.concurrency,
        loglevel=args.loglevel
    )

if __name__ == "__main__":
    main()
