"""
Task Endpoints - API endpoints riêng cho quản lý tasks
"""

import asyncio
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
        }"""
    try:  # Sử dụng method tối ưu hơn nếu có
        if hasattr(background_task_processor, "get_task_status_optimized"):
            task = await background_task_processor.get_task_status_optimized(task_id)
        else:
            task = await background_task_processor.get_task_status(task_id)

        if not task:
            raise HTTPException(
                status_code=404, detail=f"Task with ID '{task_id}' not found"
            )

        # Tạo response với thông tin cần thiết
        response = {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "created_at": task["created_at"],
            "started_at": task["started_at"],
            "completed_at": task["completed_at"],
            "estimated_duration": task.get("estimated_duration", "Unknown"),
        }

        # Thêm result nếu task đã hoàn thành
        if task["status"] == "completed":
            response["result"] = task["result"]

            # Thêm thông tin nhanh cho completed tasks
            if task["result"]:
                response["quick_info"] = {
                    "success": task["result"].get("success", False),
                    "book_id": task["result"].get("book_id"),
                    "filename": task["result"].get("filename"),
                    "embeddings_created": task["result"].get(
                        "embeddings_created", False
                    ),
                    "total_pages": task["result"]
                    .get("statistics", {})
                    .get("total_pages", 0),
                    "total_chapters": task["result"]
                    .get("statistics", {})
                    .get("total_chapters", 0),
                    "total_lessons": task["result"]
                    .get("statistics", {})
                    .get("total_lessons", 0),
                }

        # Thêm error nếu task thất bại
        if task["status"] == "failed":
            response["error"] = task["error"]

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/result/{task_id}", response_model=Dict[str, Any])
async def get_task_result(task_id: str) -> Dict[str, Any]:
    """
    Lấy kết quả của task đã hoàn thành (chỉ trả về result, không trả về toàn bộ task info)

    Args:
        task_id: ID của task

    Returns:
        Dict chứa kết quả task với định dạng giống /process-textbook

    Example:
        GET /api/v1/tasks/result/abc-123
    """
    try:
        task = await background_task_processor.get_task_status(task_id)

        if not task:
            raise HTTPException(
                status_code=404, detail=f"Task with ID '{task_id}' not found"
            )

        if task["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Task '{task_id}' is not completed yet. Current status: {task['status']}",
            )

        if not task.get("result"):
            raise HTTPException(
                status_code=404, detail=f"No result found for task '{task_id}'"
            )

        # Trả về kết quả với định dạng giống /process-textbook
        result = task["result"]

        return {
            "success": result.get("success", False),
            "book_id": result.get("book_id"),
            "filename": result.get("filename"),
            "book_structure": result.get("book_structure"),
            "statistics": result.get("statistics", {}),
            "processing_info": result.get("processing_info", {}),
            "message": result.get("message", "Task completed successfully"),
            "embeddings_created": result.get("embeddings_created", False),
            "embeddings_info": result.get("embeddings_info", {}),
            "task_info": {
                "task_id": task_id,
                "task_type": task["task_type"],
                "completed_at": task["completed_at"],
                "processing_time": task.get("processing_time"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task result: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/", response_model=Dict[str, Any])
async def get_all_tasks(
    limit: int = Query(100, ge=1, le=1000, description="Số lượng tasks tối đa"),
    status: str = Query(
        None, description="Lọc theo status (pending, processing, completed, failed)"
    ),
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
        result = await background_task_processor.get_all_tasks()
        tasks = result["tasks"]

        # Filter by status if provided
        if status:
            valid_statuses = ["pending", "processing", "completed", "failed"]
            if status not in valid_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Valid values: {valid_statuses}",
                )

            # Filter tasks by status
            tasks = [t for t in tasks if t["status"] == status]

        # Apply limit
        tasks = tasks[:limit]

        # Thêm thông tin chi tiết cho mỗi task
        enhanced_tasks = []
        for task in tasks:
            enhanced_task = {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "status": task["status"],
                "progress": task["progress"],
                "message": task["message"],
                "created_at": task["created_at"],
                "started_at": task["started_at"],
                "completed_at": task["completed_at"],
                "estimated_duration": task.get("estimated_duration", "Unknown"),
            }

            # Thêm quick info cho completed tasks
            if task["status"] == "completed" and task.get("result"):
                task_result = task["result"]
                enhanced_task["quick_info"] = {
                    "success": task_result.get("success", False),
                    "book_id": task_result.get("book_id"),
                    "filename": task_result.get("filename"),
                    "embeddings_created": task_result.get("embeddings_created", False),
                    "total_pages": task_result.get("statistics", {}).get(
                        "total_pages", 0
                    ),
                    "total_chapters": task_result.get("statistics", {}).get(
                        "total_chapters", 0
                    ),
                    "total_lessons": task_result.get("statistics", {}).get(
                        "total_lessons", 0
                    ),
                }

            # Thêm error info cho failed tasks
            if task["status"] == "failed":
                enhanced_task["error"] = task.get("error", "Unknown error")

            enhanced_tasks.append(enhanced_task)

        # Tính toán thống kê
        all_tasks = result["tasks"]
        total_tasks = len(all_tasks)
        processing_tasks = len([t for t in all_tasks if t["status"] == "processing"])
        completed_tasks = len([t for t in all_tasks if t["status"] == "completed"])
        failed_tasks = len([t for t in all_tasks if t["status"] == "failed"])
        pending_tasks = len([t for t in all_tasks if t["status"] == "pending"])

        return {
            "success": True,
            "data": {
                "tasks": enhanced_tasks,
                "total_tasks": total_tasks,
                "processing_tasks": processing_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "pending_tasks": pending_tasks,
            },
            "filters": {"limit": limit, "status": status},
            "summary": {
                "most_recent_task": enhanced_tasks[0] if enhanced_tasks else None,
                "success_rate": f"{(completed_tasks / total_tasks * 100):.1f}%"
                if total_tasks > 0
                else "0%",
                "active_tasks": processing_tasks + pending_tasks,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/getAllTask", response_model=Dict[str, Any])
async def get_all_task() -> Dict[str, Any]:
    """
    Lấy danh sách tất cả tasks (alias cho endpoint chính)

    Endpoint này tương đương với GET /api/v1/tasks/ nhưng với tên gọi khác
    để phù hợp với naming convention của bạn.

    Returns:
        Dict chứa danh sách tất cả tasks với thông tin chi tiết

    Example:
        GET /api/v1/tasks/getAllTask
    """
    try:
        # Gọi lại function chính với default parameters
        return await get_all_tasks(limit=100, status=None)

    except Exception as e:
        logger.error(f"Error in getAllTask: {e}")
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
                status_code=404, detail=f"Task with ID {task_id} not found"
            )

        # Try to cancel (for now, just mark as failed)
        if task["status"] not in ["pending"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task {task_id}. Only pending tasks can be cancelled. Current status: {task['status']}",
            )

        # Mark task as cancelled
        background_task_processor.mark_task_failed(task_id, "Cancelled by user")

        return {
            "success": True,
            "message": f"Task {task_id} cancelled successfully",
            "task_id": task_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_tasks(
    max_age_hours: int = Query(
        24, ge=1, le=168, description="Tuổi tối đa của tasks (giờ)"
    ),
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
        background_task_processor.task_service.cleanup_old_tasks(
            max_age_hours=max_age_hours
        )

        # Count tasks after cleanup
        after_count = len(background_task_processor.task_service.tasks)
        cleaned_count = before_count - after_count

        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} old tasks",
            "tasks_before": before_count,
            "tasks_after": after_count,
            "tasks_cleaned": cleaned_count,
            "max_age_hours": max_age_hours,
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
            "timestamp": background_task_processor.task_service.tasks
            and max(
                task["created_at"]
                for task in background_task_processor.task_service.tasks.values()
            )
            or None,
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
        "task_types": [
            "process_textbook",
            "process_textbook_auto",
            "quick_analysis",
            "process_cv",
            "create_embeddings",
            "generate_lesson_plan",
        ],
        "task_statuses": ["pending", "processing", "completed", "failed"],
        "descriptions": {
            "process_textbook": "Xử lý sách giáo khoa PDF với metadata từ người dùng",
            "process_textbook_auto": "Xử lý sách giáo khoa PDF với tự động phân tích metadata",
            "quick_analysis": "Phân tích nhanh cấu trúc sách giáo khoa và tạo embeddings",
            "process_cv": "Xử lý CV/Resume với OCR",
            "create_embeddings": "Tạo embeddings cho RAG search",
            "generate_lesson_plan": "Tạo giáo án từ nội dung sách",
        },
    }


@router.get("/debug/{task_id}", response_model=Dict[str, Any])
async def debug_task(task_id: str) -> Dict[str, Any]:
    """
    Debug task - Lấy toàn bộ thông tin task để debug

    Args:
        task_id: ID của task

    Returns:
        Dict chứa toàn bộ thông tin task
    """
    try:
        task = background_task_processor.get_task_status(task_id)

        if not task:
            raise HTTPException(
                status_code=404, detail=f"Task with ID '{task_id}' not found"
            )

        # Trả về toàn bộ thông tin task
        return {
            "success": True,
            "task": task,
            "debug_info": {
                "task_exists": True,
                "has_result": bool(task.get("result")),
                "result_keys": list(task.get("result", {}).keys())
                if task.get("result")
                else [],
                "data_keys": list(task.get("data", {}).keys())
                if task.get("data")
                else [],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error debugging task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
