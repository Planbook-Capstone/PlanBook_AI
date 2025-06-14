"""
Task Endpoints - API endpoints riêng cho quản lý tasks
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any

from app.services.background_task_processor import background_task_processor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status/{task_id}", response_model=Dict[str, Any])
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Lấy trạng thái của background task

    Args:
        task_id: ID của task cần kiểm tra

    Returns:
        Dict chứa thông tin chi tiết về task

    Example:
        GET /api/v1/tasks/status/abc-123

    Response:
        {
            "task_id": "abc-123",
            "status": "processing",
            "progress": 45,
            "message": "Creating embeddings...",
            "result": null  // Chỉ có khi completed
        }
    """
    try:
        task = background_task_processor.get_task_status(task_id)

        if not task:
            raise HTTPException(
                status_code=404, detail=f"Task with ID '{task_id}' not found"
            )

        # Tạo response với thông tin cần thiết
        response = {
            "task_id": task["task_id"],
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "created_at": task["created_at"],
            "started_at": task["started_at"],
            "completed_at": task["completed_at"],
        }

        # Thêm result nếu task đã hoàn thành
        if task["status"] == "completed":
            response["result"] = task["result"]

        # Thêm error nếu task thất bại
        if task["status"] == "failed":
            response["error"] = task["error"]

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/", response_model=Dict[str, Any])
async def get_all_tasks(
    limit: int = Query(100, ge=1, le=1000, description="Số lượng tasks tối đa"),
    status: str = Query(None, description="Lọc theo status (pending, processing, completed, failed)")
) -> Dict[str, Any]:
    """
    Lấy danh sách tất cả tasks với filtering

    Args:
        limit: Số lượng tasks tối đa (1-1000)
        status: Lọc theo status

    Returns:
        Dict chứa danh sách tasks và thống kê

    Example:
        GET /api/v1/tasks/?limit=50&status=completed
    """
    try:
        # Get all tasks
        result = background_task_processor.get_all_tasks()
        
        # Filter by status if provided
        if status:
            valid_statuses = ["pending", "processing", "completed", "failed"]
            if status not in valid_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Valid values: {valid_statuses}"
                )
            
            # Filter tasks by status
            filtered_tasks = [t for t in result["tasks"] if t["status"] == status]
            result["tasks"] = filtered_tasks[:limit]
        else:
            result["tasks"] = result["tasks"][:limit]
        
        return {
            "success": True,
            "data": result,
            "filters": {
                "limit": limit,
                "status": status
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Hủy task (chỉ có thể hủy task pending)

    Args:
        task_id: ID của task

    Returns:
        Dict xác nhận hủy task

    Example:
        DELETE /api/v1/tasks/abc-123
    """
    try:
        # Check if task exists
        task = background_task_processor.get_task_status(task_id)
        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task with ID {task_id} not found"
            )

        # Try to cancel (for now, just mark as failed)
        if task["status"] not in ["pending"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task {task_id}. Only pending tasks can be cancelled. Current status: {task['status']}"
            )

        # Mark task as cancelled
        background_task_processor.mark_task_failed(task_id, "Cancelled by user")

        return {
            "success": True,
            "message": f"Task {task_id} cancelled successfully",
            "task_id": task_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_tasks(
    max_age_hours: int = Query(24, ge=1, le=168, description="Tuổi tối đa của tasks (giờ)")
) -> Dict[str, Any]:
    """
    Dọn dẹp tasks cũ (completed/failed)

    Args:
        max_age_hours: Tuổi tối đa của tasks tính bằng giờ (1-168)

    Returns:
        Dict xác nhận dọn dẹp

    Example:
        POST /api/v1/tasks/cleanup?max_age_hours=48
    """
    try:
        # Count tasks before cleanup
        before_count = len(background_task_processor.task_service.tasks)

        # Perform cleanup
        background_task_processor.task_service.cleanup_old_tasks(max_age_hours=max_age_hours)

        # Count tasks after cleanup
        after_count = len(background_task_processor.task_service.tasks)
        cleaned_count = before_count - after_count

        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} old tasks",
            "tasks_before": before_count,
            "tasks_after": after_count,
            "tasks_cleaned": cleaned_count,
            "max_age_hours": max_age_hours
        }

    except Exception as e:
        logger.error(f"Error cleaning up tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/statistics", response_model=Dict[str, Any])
async def get_task_statistics() -> Dict[str, Any]:
    """
    Lấy thống kê tổng quan về tasks

    Returns:
        Dict chứa thống kê chi tiết

    Example:
        GET /api/v1/tasks/statistics
    """
    try:
        stats = background_task_processor.task_service.get_task_statistics()

        return {
            "success": True,
            "statistics": stats,
            "timestamp": background_task_processor.task_service.tasks and max(
                task["created_at"] for task in background_task_processor.task_service.tasks.values()
            ) or None
        }

    except Exception as e:
        logger.error(f"Error getting task statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/types", response_model=Dict[str, Any])
async def get_task_types() -> Dict[str, Any]:
    """
    Lấy danh sách các loại task có sẵn

    Returns:
        Dict chứa danh sách task types và statuses
    """
    return {
        "success": True,
        "task_types": ["process_textbook", "process_cv", "create_embeddings", "generate_lesson_plan"],
        "task_statuses": ["pending", "processing", "completed", "failed"],
        "descriptions": {
            "process_textbook": "Xử lý sách giáo khoa PDF với OCR và embeddings",
            "process_cv": "Xử lý CV/Resume với OCR",
            "create_embeddings": "Tạo embeddings cho RAG search",
            "generate_lesson_plan": "Tạo giáo án từ nội dung sách"
        }
    }
