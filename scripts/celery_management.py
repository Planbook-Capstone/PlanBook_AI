#!/usr/bin/env python3
"""
Script quản lý Celery tasks và workers cho PlanBookAI
Cách sử dụng:
    python scripts/celery_management.py status
    python scripts/celery_management.py inspect active
    python scripts/celery_management.py purge
    python scripts/celery_management.py test
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_celery_app():
    """Get Celery app instance"""
    from app.core.celery_app import celery_app
    return celery_app

def check_worker_status():
    """Kiểm tra trạng thái workers"""
    celery_app = get_celery_app()
    
    print("🔍 Checking Celery worker status...")
    
    # Inspect active workers
    inspect = celery_app.control.inspect()
    
    # Get active workers
    active_workers = inspect.active()
    if active_workers:
        print("✅ Active workers:")
        for worker, tasks in active_workers.items():
            print(f"  📊 {worker}: {len(tasks)} active tasks")
            for task in tasks:
                print(f"    - {task['name']} (ID: {task['id'][:8]}...)")
    else:
        print("❌ No active workers found")
    
    # Get registered tasks
    registered = inspect.registered()
    if registered:
        print("\n📋 Registered tasks:")
        for worker, tasks in registered.items():
            print(f"  🔧 {worker}:")
            for task in sorted(tasks):
                print(f"    - {task}")
    
    # Get worker stats
    stats = inspect.stats()
    if stats:
        print("\n📈 Worker statistics:")
        for worker, stat in stats.items():
            print(f"  📊 {worker}:")
            print(f"    - Pool: {stat.get('pool', {}).get('implementation', 'N/A')}")
            print(f"    - Processes: {stat.get('pool', {}).get('processes', 'N/A')}")
            print(f"    - Total tasks: {stat.get('total', 'N/A')}")

def inspect_tasks(task_type="active"):
    """Inspect tasks"""
    celery_app = get_celery_app()
    inspect = celery_app.control.inspect()
    
    print(f"🔍 Inspecting {task_type} tasks...")
    
    if task_type == "active":
        tasks = inspect.active()
    elif task_type == "scheduled":
        tasks = inspect.scheduled()
    elif task_type == "reserved":
        tasks = inspect.reserved()
    else:
        print(f"❌ Unknown task type: {task_type}")
        return
    
    if tasks:
        for worker, task_list in tasks.items():
            print(f"📊 {worker}: {len(task_list)} {task_type} tasks")
            for task in task_list:
                print(f"  - {task['name']} (ID: {task['id'][:8]}...)")
                if 'args' in task and task['args']:
                    print(f"    Args: {task['args']}")
    else:
        print(f"✅ No {task_type} tasks found")

def purge_tasks():
    """Xóa tất cả pending tasks"""
    celery_app = get_celery_app()
    
    print("⚠️  Purging all pending tasks...")
    response = input("Are you sure? This will delete all pending tasks (y/N): ")
    
    if response.lower() == 'y':
        result = celery_app.control.purge()
        print(f"✅ Purged tasks: {result}")
    else:
        print("❌ Purge cancelled")

def test_celery():
    """Test Celery connection và tasks"""
    celery_app = get_celery_app()
    
    print("🧪 Testing Celery connection...")
    
    # Test 1: Health check task
    try:
        result = celery_app.send_task('app.tasks.health_check')
        print(f"✅ Health check task sent: {result.id}")
        
        # Wait for result (timeout 10s)
        try:
            health_result = result.get(timeout=10)
            print(f"✅ Health check result: {health_result}")
        except Exception as e:
            print(f"⚠️  Health check timeout or error: {e}")
    except Exception as e:
        print(f"❌ Failed to send health check task: {e}")
    
    # Test 2: Simple test task
    try:
        result = celery_app.send_task('app.tasks.test_task', args=['Hello from management script!'])
        print(f"✅ Test task sent: {result.id}")
        
        try:
            test_result = result.get(timeout=10)
            print(f"✅ Test task result: {test_result}")
        except Exception as e:
            print(f"⚠️  Test task timeout or error: {e}")
    except Exception as e:
        print(f"❌ Failed to send test task: {e}")

def show_queues():
    """Hiển thị thông tin queues"""
    print("📋 Configured queues:")
    queues = [
        "default",
        "pdf_queue", 
        "embeddings_queue",
        "cv_queue"
    ]
    
    for queue in queues:
        print(f"  - {queue}")

def monitor_tasks():
    """Monitor tasks in real-time"""
    import time
    
    print("📊 Monitoring tasks (Press Ctrl+C to stop)...")
    
    try:
        while True:
            celery_app = get_celery_app()
            inspect = celery_app.control.inspect()
            
            active = inspect.active()
            if active:
                total_active = sum(len(tasks) for tasks in active.values())
                print(f"⚡ Active tasks: {total_active}")
                
                for worker, tasks in active.items():
                    if tasks:
                        print(f"  📊 {worker}: {len(tasks)} tasks")
            else:
                print("💤 No active tasks")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")

def main():
    parser = argparse.ArgumentParser(description="Celery Management for PlanBookAI")
    
    parser.add_argument(
        "command",
        choices=["status", "inspect", "purge", "test", "queues", "monitor"],
        help="Management command to run"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["active", "scheduled", "reserved"],
        default="active",
        help="Task type for inspect command (default: active)"
    )
    
    args = parser.parse_args()
    
    # Check Redis connection
    try:
        import redis
        from app.core.config import settings
        
        r = redis.from_url(settings.CELERY_BROKER_URL)
        r.ping()
        print("✅ Redis connection successful\n")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Make sure Redis is running: redis-server")
        sys.exit(1)
    
    # Execute command
    if args.command == "status":
        check_worker_status()
    elif args.command == "inspect":
        inspect_tasks(args.type)
    elif args.command == "purge":
        purge_tasks()
    elif args.command == "test":
        test_celery()
    elif args.command == "queues":
        show_queues()
    elif args.command == "monitor":
        monitor_tasks()

if __name__ == "__main__":
    main()
