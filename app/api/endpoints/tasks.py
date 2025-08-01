"""
Task Endpoints - API endpoints riêng cho quản lý tasks
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any

from app.services.background_task_processor import get_background_task_processor

logger = logging.getLogger(__name__)

router = APIRouter()


def _calculate_average_step_duration(progress_history: list) -> float:
    """Tính thời gian trung bình giữa các steps"""
    if len(progress_history) < 2:
        return 0.0

    durations = []
    for i in range(1, len(progress_history)):
        current_timestamp = progress_history[i].get("timestamp", 0)
        prev_timestamp = progress_history[i - 1].get("timestamp", 0)
        if current_timestamp and prev_timestamp:
            duration = current_timestamp - prev_timestamp
            durations.append(duration)

    return sum(durations) / len(durations) if durations else 0.0


def _calculate_total_duration(progress_history: list) -> float:
    """Tính tổng thời gian từ bắt đầu đến hiện tại"""
    if len(progress_history) < 2:
        return 0.0
    
    start_timestamp = progress_history[0].get("timestamp", 0)
    end_timestamp = progress_history[-1].get("timestamp", 0)
    
    return end_timestamp - start_timestamp if end_timestamp > start_timestamp else 0.0


def _estimate_remaining_time(task: dict) -> float:
    """Ước tính thời gian còn lại dựa trên progress velocity"""
    progress_history = task.get("progress_history", [])
    current_progress = task.get("progress", 0)
    
    if current_progress >= 100 or len(progress_history) < 2:
        return 0.0
    
    velocity = _calculate_progress_velocity(progress_history)
    if velocity <= 0:
        return 0.0
    
    remaining_progress = 100 - current_progress
    return remaining_progress / velocity


def _find_longest_step(progress_history: list) -> dict:
    """Tìm step mất thời gian lâu nhất"""
    if len(progress_history) < 2:
        return {}
    
    longest_duration = 0.0
    longest_step = {}
    
    for i in range(1, len(progress_history)):
        current_timestamp = progress_history[i].get("timestamp", 0)
        prev_timestamp = progress_history[i-1].get("timestamp", 0)
        
        if current_timestamp and prev_timestamp:
            duration = current_timestamp - prev_timestamp
            if duration > longest_duration:
                longest_duration = duration
                longest_step = {
                    "step": progress_history[i],
                    "duration": duration,
                    "progress_gained": progress_history[i].get("progress", 0) - progress_history[i-1].get("progress", 0)
                }
    
    return longest_step


def _calculate_progress_velocity(progress_history: list) -> float:
    """Tính velocity (progress/giây) trung bình"""
    if len(progress_history) < 2:
        return 0.0
    
    total_time = _calculate_total_duration(progress_history)
    if total_time <= 0:
        return 0.0
    
    total_progress = progress_history[-1].get("progress", 0) - progress_history[0].get("progress", 0)
    return total_progress / total_time


def _get_step_breakdown(progress_history: list) -> list:
    """Phân tích chi tiết từng step"""
    if len(progress_history) < 2:
        return []
    
    breakdown = []
    for i in range(1, len(progress_history)):
        current_step = progress_history[i]
        prev_step = progress_history[i-1]
        
        duration = current_step.get("timestamp", 0) - prev_step.get("timestamp", 0)
        progress_gained = current_step.get("progress", 0) - prev_step.get("progress", 0)
        
        breakdown.append({
            "step_number": i,
            "message": current_step.get("message", ""),
            "progress": current_step.get("progress", 0),
            "progress_gained": progress_gained,
            "duration": duration,
            "timestamp": current_step.get("datetime", ""),
            "velocity": progress_gained / duration if duration > 0 else 0
        })
    
    return breakdown


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
    try:
        background_processor = get_background_task_processor()
        # Sử dụng method tối ưu hơn nếu có
        if hasattr(background_processor, "get_task_status_optimized"):
            task = await background_processor.get_task_status_optimized(task_id)
        else:
            task = await background_processor.get_task_status(task_id)

        if not task:
            raise HTTPException(
                status_code=404, detail=f"Task with ID '{task_id}' not found"
            )        # Tạo response với thông tin cần thiết và progress history
        progress_history = task.get("progress_history", [])
        
        response = {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "status": task["status"],
            "current_progress": task["progress"],
            "current_message": task["message"],
            "progress_history": progress_history,
            "created_at": task["created_at"],
            "started_at": task["started_at"],
            "completed_at": task["completed_at"],
            "updated_at": task.get("updated_at"),
            "estimated_duration": task.get("estimated_duration", "Unknown"),
            # Thống kê progress
            "progress_stats": {
                "total_steps": len(progress_history),
                "completion_percentage": task["progress"],
                "started_at": progress_history[0].get("datetime") if progress_history else None,
                "last_updated": progress_history[-1].get("datetime") if progress_history else None,
                "average_step_duration": _calculate_average_step_duration(progress_history)
            }
        }

        # Thêm result nếu có (bao gồm cả partial results trong quá trình xử lý)
        if task.get("result"):
            response["result"] = task["result"]

        # Thêm thông tin nhanh cho completed tasks
        if task["status"] == "completed" and task.get("result"):
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
    Lấy result của task theo task_id

    Endpoint này chỉ trả về result của task, không bao gồm thông tin progress hay status khác.
    Dùng để lấy kết quả cuối cùng của task đã hoàn thành.
    Hỗ trợ tất cả loại task (textbook processing, smart exam, lesson plan, etc.)

    Args:
        task_id: ID của task cần lấy result

    Returns:
        Dict: Result của task hoặc error nếu task chưa hoàn thành/không tồn tại

    Example:
        GET /api/v1/tasks/result/abc-123-def

        Response (Success - Smart Exam):
        {
            "success": true,
            "task_id": "abc-123-def",
            "task_type": "generate_smart_exam",
            "result": {
                "success": true,
                "exam_id": "exam_123",
                "message": "Đề thi thông minh đã được tạo thành công",
                "statistics": {...},
                "download_link": "https://drive.google.com/...",
                "online_link": "https://docs.google.com/..."
            },
            "completed_at": "2025-07-07T18:30:00",
            "message": "Result retrieved successfully"
        }

        Response (Success - Textbook Processing):
        {
            "success": true,
            "task_id": "abc-123-def",
            "task_type": "process_textbook",
            "result": {
                "success": true,
                "book_id": "book_123",
                "filename": "textbook.pdf",
                "book_structure": {...},
                "statistics": {...}
            },
            "completed_at": "2025-07-07T18:30:00",
            "message": "Result retrieved successfully"
        }

        Response (Error):
        {
            "success": false,
            "task_id": "abc-123-def",
            "task_type": "generate_smart_exam",
            "result": {
                "success": false,
                "error": "Lỗi hệ thống: ...",
                "error_details": {...}
            },
            "completed_at": "2025-07-07T18:30:00",
            "message": "Result retrieved successfully"
        }

        Response (Not Found/Not Completed):
        {
            "success": false,
            "error": "Task not found or not completed yet",
            "task_id": "abc-123-def",
            "current_status": "in_progress",
            "progress": 45
        }
    """
    try:
        logger.info(f"Getting result for task: {task_id}")

        background_processor = get_background_task_processor()
        task = await background_processor.get_task_status(task_id)

        if not task:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": "Task not found",
                    "task_id": task_id
                }
            )

        # Kiểm tra task đã hoàn thành chưa
        task_status = task.get("status")
        if task_status != "completed":
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": f"Task not completed yet. Current status: {task_status}",
                    "task_id": task_id,
                    "task_type": task.get("task_type"),
                    "current_status": task_status,
                    "progress": task.get("progress", 0),
                    "message": task.get("message", "")
                }
            )

        # Lấy result
        result = task.get("result")
        if not result:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": "Task completed but no result found",
                    "task_id": task_id,
                    "task_type": task.get("task_type")
                }
            )

        logger.info(f"Successfully retrieved result for task: {task_id}")

        return {
            "success": True,
            "task_id": task_id,
            "task_type": task.get("task_type"),
            "result": result,
            "completed_at": task.get("completed_at"),
            "message": "Result retrieved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task result: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": f"Internal server error: {str(e)}",
                "task_id": task_id
            }
        )


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
        result = await get_background_task_processor().get_all_tasks()
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
        task = get_background_task_processor().get_task_status(task_id)
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
        get_background_task_processor().mark_task_failed(task_id, "Cancelled by user")

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
        before_count = len(get_background_task_processor().task_service.tasks)

        # Perform cleanup
        get_background_task_processor().task_service.cleanup_old_tasks(
            max_age_hours=max_age_hours
        )

        # Count tasks after cleanup
        after_count = len(get_background_task_processor().task_service.tasks)
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


async def get_task_statistics() -> Dict[str, Any]:
    """
    Lấy thống kê tổng quan về tasks

    Returns:
        Dict chứa thống kê chi tiết

    Example:
        GET /api/v1/tasks/statistics
    """
    try:
        stats = get_background_task_processor().task_service.get_task_statistics()

        return {
            "success": True,
            "statistics": stats,
            "timestamp": get_background_task_processor().task_service.tasks
            and max(
                task["created_at"]
                for task in get_background_task_processor().task_service.tasks.values()
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
            "generate_smart_exam",
            "guide_import",
        ],
        "task_statuses": ["pending", "processing", "completed", "failed"],
        "descriptions": {
            "process_textbook": "Xử lý sách giáo khoa PDF với metadata từ người dùng",
            "process_textbook_auto": "Xử lý sách giáo khoa PDF với tự động phân tích metadata",
            "quick_analysis": "Phân tích nhanh cấu trúc sách giáo khoa và tạo embeddings",
            "process_cv": "Xử lý CV/Resume với OCR",
            "create_embeddings": "Tạo embeddings cho RAG search",
            "generate_lesson_plan": "Tạo giáo án từ nội dung sách",
            "generate_smart_exam": "Tạo đề thi thông minh theo chuẩn THPT 2025",
            "guide_import": "Import hướng dẫn từ file DOCX và tạo embeddings cho RAG search",
        },
    }


async def debug_task(task_id: str) -> Dict[str, Any]:
    """
    Debug task - Lấy toàn bộ thông tin task để debug

    Args:
        task_id: ID của task

    Returns:
        Dict chứa toàn bộ thông tin task
    """
    try:
        task = get_background_task_processor().get_task_status(task_id)

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


async def get_task_progress_detailed(task_id: str) -> Dict[str, Any]:
    """
    Lấy chi tiết progress history của task với timeline và statistics

    Args:
        task_id: ID của task

    Returns:
        Dict chứa chi tiết progress history, timeline và statistics

    Example:
        GET /api/v1/tasks/progress/abc-123
    """
    try:
        task = await get_background_task_processor().get_task_status_optimized(task_id)

        if not task:
            raise HTTPException(
                status_code=404, detail=f"Task with ID '{task_id}' not found"
            )

        progress_history = task.get("progress_history", [])
        
        # Tính toán timeline
        timeline = {}
        if progress_history:
            timeline = {
                "created_at": task.get("created_at"),
                "started_at": next((step["datetime"] for step in progress_history if step["progress"] > 0), None),
                "last_update": task.get("updated_at"),
                "total_duration": _calculate_total_duration(progress_history),
                "estimated_remaining": _estimate_remaining_time(task)
            }

        # Tính toán statistics chi tiết
        statistics = {
            "total_steps": len(progress_history),
            "completion_rate": task.get("progress", 0),
            "average_step_duration": _calculate_average_step_duration(progress_history),
            "longest_step": _find_longest_step(progress_history),
            "progress_velocity": _calculate_progress_velocity(progress_history),
            "step_breakdown": _get_step_breakdown(progress_history)
        }

        return {
            "task_id": task_id,
            "status": task.get("status"),
            "current_progress": task.get("progress", 0),
            "current_message": task.get("message", ""),
            "progress_history": progress_history,
            "timeline": timeline,
            "statistics": statistics,
            "task_type": task.get("task_type"),
            "result": task.get("result") if task.get("status") == "completed" else None,
            "error": task.get("error") if task.get("status") == "failed" else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task progress details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
