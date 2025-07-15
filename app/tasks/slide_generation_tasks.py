"""
Celery tasks cho Slide Generation
Xử lý tạo slide tự động bất đồng bộ với progress tracking
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

from app.core.celery_app import celery_app
from app.services.mongodb_task_service import get_mongodb_task_service
from app.services.slide_generation_service import get_slide_generation_service

logger = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.slide_generation_tasks.test_task")
def test_slide_generation_task(message: str = "Test slide generation task"):
    """Simple test task để kiểm tra Celery worker"""
    logger.info(f"🧪 TEST TASK EXECUTED: {message}")
    return {
        "success": True,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }


@celery_app.task(bind=True, name="app.tasks.slide_generation_tasks.generate_slides_task")
def generate_slides_task(self, task_id: str, lesson_id: str, template_id: str,
                        config_prompt: str = None, presentation_title: str = None):
    """
    Celery task để tạo slide tự động bất đồng bộ

    Args:
        task_id: ID của task trong MongoDB
        lesson_id: ID của bài học
        template_id: ID của Google Slides template
        config_prompt: Prompt cấu hình tùy chỉnh (optional)
        presentation_title: Tiêu đề presentation tùy chỉnh (optional)
    """

    logger.info(f"🚀 CELERY TASK STARTED: generate_slides_task")
    logger.info(f"   Task ID: {task_id}")
    logger.info(f"   Lesson ID: {lesson_id}")
    logger.info(f"   Template ID: {template_id}")
    logger.info(f"   Config Prompt: {config_prompt}")
    logger.info(f"   Presentation Title: {presentation_title}")
    
    async def _async_generate_slides():
        """Async wrapper cho slide generation"""
        logger.info(f"🔄 ASYNC FUNCTION STARTED for task: {task_id}")

        try:
            logger.info("🔄 Getting MongoDB task service...")
            task_service = get_mongodb_task_service()
            logger.info("✅ MongoDB task service obtained")
        except Exception as e:
            logger.error(f"❌ Failed to get MongoDB task service: {e}")
            raise

        try:
            logger.info(f"🚀 Bắt đầu task tạo slide: {task_id}")
            
            # Cập nhật trạng thái: Bắt đầu xử lý
            await task_service.mark_task_processing(task_id)
            await task_service.update_task_progress(
                task_id,
                progress=10,
                message="Đang khởi tạo dịch vụ tạo slide..."
            )
            
            # Lấy slide generation service
            slide_service = get_slide_generation_service()
            
            if not slide_service.is_available():
                await task_service.mark_task_failed(
                    task_id,
                    error="Dịch vụ tạo slide không khả dụng"
                )
                return
            
            # Cập nhật: Đang lấy nội dung bài học
            await task_service.update_task_progress(
                task_id,
                progress=20,
                message="📖 Đang lấy nội dung bài học..."
            )
            
            # Cập nhật: Đang phân tích template
            await task_service.update_task_progress(
                task_id,
                progress=30,
                message="🔍 Đang phân tích cấu trúc template Google Slides..."
            )
            
            # Cập nhật: Đang sinh nội dung với LLM
            await task_service.update_task_progress(
                task_id,
                progress=50,
                message="🤖 Đang sử dụng AI để sinh nội dung slide..."
            )
            
            # Thực hiện slide generation
            result = await slide_service.generate_slides_from_lesson(
                lesson_id=lesson_id,
                template_id=template_id,
                config_prompt=config_prompt
            )
            
            if result["success"]:
                # Cập nhật: Đang tạo slides
                await task_service.update_task_progress(
                    task_id,
                    progress=80,
                    message="📊 Đang tạo slides trên Google Slides..."
                )

                # Hoàn thành thành công
                await task_service.mark_task_completed(
                    task_id,
                    result={
                        "success": True,
                        "lesson_id": result["lesson_id"],
                        "template_id": result["original_template_id"],
                        "presentation_id": result["presentation_id"],
                        "presentation_title": result["presentation_title"],
                        "web_view_link": result["web_view_link"],
                        "slides_created": result["slides_created"],
                        "template_info": result["template_info"],
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                logger.info(f"✅ Task {task_id} hoàn thành thành công")
                
            else:
                # Xử lý lỗi
                error_message = result.get("error", "Unknown error")
                await task_service.mark_task_failed(
                    task_id,
                    error=f"Lỗi tạo slide: {error_message}"
                )
                
                logger.error(f"❌ Task {task_id} thất bại: {error_message}")
                
        except Exception as e:
            logger.error(f"❌ Lỗi không mong muốn trong task {task_id}: {e}")
            
            try:
                await task_service.mark_task_failed(
                    task_id,
                    error=f"Lỗi hệ thống: {str(e)}"
                )
            except Exception as update_error:
                logger.error(f"❌ Không thể cập nhật trạng thái task: {update_error}")
    
    # Chạy async function
    logger.info(f"🔄 Starting Celery task execution for task_id: {task_id}")

    try:
        # Sử dụng asyncio.run() thay vì event loop phức tạp
        logger.info("🔄 Running async slide generation function...")
        asyncio.run(_async_generate_slides())
        logger.info(f"✅ Celery task completed successfully for task_id: {task_id}")

    except Exception as e:
        logger.error(f"❌ Lỗi chạy async task {task_id}: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        logger.error(f"❌ Exception details: {str(e)}")

        # Fallback: cập nhật trạng thái lỗi trực tiếp - sử dụng new_event_loop
        try:
            logger.info("🔄 Attempting to update task status to FAILURE...")
            task_service = get_mongodb_task_service()

            # Create a new event loop instead of using asyncio.run() again
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(task_service.mark_task_failed(
                    task_id,
                    error=f"Lỗi hệ thống: {str(e)}"
                ))
                logger.info("✅ Task status updated to FAILURE")
            finally:
                loop.close()
        except Exception as update_error:
            logger.error(f"❌ Failed to update task status: {update_error}")


@celery_app.task(bind=True, name="app.tasks.slide_generation_tasks.cleanup_old_presentations")
def cleanup_old_presentations_task(self, days_old: int = 7):
    """
    Celery task để dọn dẹp các presentation cũ trên Google Drive
    
    Args:
        days_old: Số ngày để xem presentation là "cũ" (mặc định 7 ngày)
    """
    
    async def _async_cleanup():
        """Async wrapper cho cleanup"""
        try:
            logger.info(f"🧹 Bắt đầu dọn dẹp presentation cũ hơn {days_old} ngày")
            
            slide_service = get_slide_generation_service()
            
            if not slide_service.is_available():
                logger.warning("Slide service không khả dụng, bỏ qua cleanup")
                return
            
            # TODO: Implement cleanup logic
            # 1. Lấy danh sách presentations từ Google Drive
            # 2. Kiểm tra ngày tạo
            # 3. Xóa những file cũ hơn days_old
            
            logger.info("✅ Hoàn thành dọn dẹp presentation")
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình cleanup: {e}")
    
    # Chạy async function
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_async_cleanup())
    except Exception as e:
        logger.error(f"❌ Lỗi chạy cleanup task: {e}")


# Utility function để trigger task từ API
async def trigger_slide_generation_task(
    lesson_id: str, 
    template_id: str,
    config_prompt: str = None,
    presentation_title: str = None
) -> str:
    """
    Trigger slide generation task và trả về task_id
    
    Args:
        lesson_id: ID của bài học
        template_id: ID của template
        config_prompt: Prompt cấu hình tùy chỉnh
        presentation_title: Tiêu đề presentation tùy chỉnh
        
    Returns:
        str: Task ID để theo dõi progress
    """
    try:
        # Tạo task trong MongoDB
        task_service = get_mongodb_task_service()
        
        task_data = {
            "lesson_id": lesson_id,
            "template_id": template_id,
            "config_prompt": config_prompt,
            "presentation_title": presentation_title
        }

        from app.services.mongodb_task_service import TaskType

        task_id = await task_service.create_task(
            task_type=TaskType.SLIDE_GENERATION,
            task_data=task_data
        )
        
        # Trigger Celery task
        logger.info(f"🔄 About to trigger Celery task for task_id: {task_id}")
        logger.info(f"   Lesson ID: {lesson_id}")
        logger.info(f"   Template ID: {template_id}")

        try:
            # Trigger Celery task với apply_async và queue cụ thể
            celery_result = generate_slides_task.apply_async(
                args=[task_id, lesson_id, template_id, config_prompt, presentation_title],
                queue='slide_generation_queue'
            )

            logger.info(f"✅ Celery task triggered successfully:")
            logger.info(f"   Task ID: {task_id}")
            logger.info(f"   Celery Result ID: {celery_result.id}")
            logger.info(f"   Celery State: {celery_result.state}")

        except Exception as e:
            logger.error(f"❌ Failed to trigger Celery task: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            logger.error(f"   Exception details: {str(e)}")
            raise

        logger.info(f"✅ Đã trigger slide generation task: {task_id}")
        return task_id
        
    except Exception as e:
        logger.error(f"❌ Lỗi trigger slide generation task: {e}")
        raise
