"""
Celery Health Check Endpoints
Endpoints để monitor và health check Celery workers
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from app.services.celery_task_service import celery_task_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/celery", tags=["Celery Health"])


@router.get("/health", response_model=Dict[str, Any])
async def celery_health_check() -> Dict[str, Any]:
    """
    Health check cho Celery service
    
    Kiểm tra:
    - Redis connection
    - Worker availability  
    - Task dispatch capability
    
    Returns:
        Dict: Health status và chi tiết
    """
    try:
        health_status = await celery_task_service.health_check()
        
        # Set HTTP status code based on health
        if health_status["status"] == "unhealthy":
            # Don't raise exception, just return status
            # Client có thể check status field
            pass
            
        return {
            "success": True,
            "health": health_status,
            "message": "Celery health check completed"
        }
        
    except Exception as e:
        logger.error(f"Error in Celery health check: {e}")
        return {
            "success": False,
            "health": {
                "status": "error",
                "error": str(e)
            },
            "message": "Celery health check failed"
        }


@router.get("/workers", response_model=Dict[str, Any])
async def get_worker_stats() -> Dict[str, Any]:
    """
    Lấy thống kê workers
    
    Returns:
        Dict: Worker statistics
    """
    try:
        stats = await celery_task_service.get_worker_stats()
        
        return {
            "success": True,
            "stats": stats,
            "message": "Worker stats retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting worker stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get worker stats: {str(e)}"
        )


@router.post("/test-task", response_model=Dict[str, Any])
async def create_test_task() -> Dict[str, Any]:
    """
    Tạo test task để kiểm tra Celery worker
    
    Returns:
        Dict: Test task result
    """
    try:
        # Tạo simple test task
        task_data = {
            "message": "Test task from health check endpoint",
            "timestamp": "now"
        }
        
        # Dispatch test task
        from app.core.celery_app import celery_app
        result = celery_app.send_task('app.tasks.test_task', args=['Hello from health check!'])
        
        return {
            "success": True,
            "task_id": result.id,
            "message": "Test task created successfully",
            "note": "Check task status using /api/v1/tasks/{task_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Error creating test task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create test task: {str(e)}"
        )


@router.get("/queues", response_model=Dict[str, Any])
async def get_queue_info() -> Dict[str, Any]:
    """
    Lấy thông tin về queues
    
    Returns:
        Dict: Queue information
    """
    try:
        from app.core.celery_app import celery_app
        
        # Get configured queues
        configured_queues = [
            "default",
            "pdf_queue",
            "embeddings_queue", 
            "cv_queue"
        ]
        
        # Try to get queue lengths (may not work with all brokers)
        queue_info = {}
        for queue in configured_queues:
            queue_info[queue] = {
                "name": queue,
                "configured": True,
                "length": "N/A"  # Redis doesn't easily expose queue length
            }
        
        # Get active tasks per queue
        inspect = celery_app.control.inspect()
        active = inspect.active()
        
        if active:
            for worker, tasks in active.items():
                for task in tasks:
                    # Try to determine queue from routing
                    task_name = task.get("name", "")
                    if "pdf_tasks" in task_name:
                        queue = "pdf_queue"
                    elif "embeddings_tasks" in task_name:
                        queue = "embeddings_queue"
                    elif "cv_tasks" in task_name:
                        queue = "cv_queue"
                    else:
                        queue = "default"
                    
                    if queue in queue_info:
                        if "active_tasks" not in queue_info[queue]:
                            queue_info[queue]["active_tasks"] = 0
                        queue_info[queue]["active_tasks"] += 1
        
        return {
            "success": True,
            "queues": queue_info,
            "total_queues": len(configured_queues),
            "message": "Queue information retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting queue info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get queue info: {str(e)}"
        )


@router.post("/purge-tasks", response_model=Dict[str, Any])
async def purge_pending_tasks() -> Dict[str, Any]:
    """
    Xóa tất cả pending tasks (NGUY HIỂM!)
    
    Returns:
        Dict: Purge result
    """
    try:
        from app.core.celery_app import celery_app
        
        # Purge all pending tasks
        result = celery_app.control.purge()
        
        return {
            "success": True,
            "purged_tasks": result,
            "message": "All pending tasks have been purged",
            "warning": "This action cannot be undone!"
        }
        
    except Exception as e:
        logger.error(f"Error purging tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to purge tasks: {str(e)}"
        )
