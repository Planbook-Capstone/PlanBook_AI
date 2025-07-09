"""
Background Task Processor - Xử lý các task bất đồng bộ sử dụng TaskService
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import concurrent.futures

from app.services.mongodb_task_service import mongodb_task_service, TaskType

logger = logging.getLogger(__name__)


class BackgroundTaskProcessor:
    """Service xử lý background tasks sử dụng TaskService"""

    def __init__(self):
        self.task_service = mongodb_task_service
        # ThreadPoolExecutor để chạy OCR operations không block event loop
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Task status cache để tránh query MongoDB liên tục
        self._task_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timeout = 0
        self._cache_lock = asyncio.Lock()

    async def _get_cached_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy task từ cache nếu còn hợp lệ"""
        async with self._cache_lock:
            cached = self._task_cache.get(task_id)
            if (
                cached
                and time.time() - cached.get("cached_at", 0) < self._cache_timeout
            ):
                return cached.get("data")
        return None

    async def _cache_task(self, task_id: str, task_data: Dict[str, Any]):
        """Cache task data"""
        async with self._cache_lock:
            self._task_cache[task_id] = {
                "data": task_data.copy(),
                "cached_at": time.time(),
            }

    async def get_task_status_optimized(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái task tối ưu với caching và timeout"""
        # Kiểm tra cache trước
        cached = await self._get_cached_task(task_id)
        if cached:
            return cached

        # Nếu không có cache, query MongoDB với timeout
        try:
            task = await asyncio.wait_for(
                self.task_service.get_task_status(task_id),
                timeout=2.0,  # Timeout 2 giây
            )
            if task:
                await self._cache_task(task_id, task)
            return task
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting task {task_id}")
            return None

    async def get_task_status_fast(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái task với caching để tránh chậm trễ"""
        # Kiểm tra cache trước
        cached = self._get_cached_task(task_id)
        if cached:
            return cached

        # Nếu không có cache, query MongoDB
        task = await self.task_service.get_task_status(task_id)
        if task:
            self._cache_task(task_id, task)
        return task

    async def create_task(self, task_type: str, task_data: Dict[str, Any]) -> str:
        """Tạo task mới"""
        # Convert string to TaskType enum
        task_type_enum = TaskType(task_type)
        return await self.task_service.create_task(task_type_enum, task_data)

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Lấy trạng thái task"""
        return await self.task_service.get_task_status(task_id)

    async def get_all_tasks(self) -> Dict[str, Any]:
        """Lấy tất cả tasks"""
        return await self.task_service.get_all_tasks()

    async def get_task_progress_detailed(self, task_id: str) -> Dict[str, Any]:
        """Lấy chi tiết progress với full history và analytics"""
        task = await self.task_service.get_task_status(task_id)

        if not task:
            return {"error": "Task not found"}

        progress_history = task.get("progress_history", [])

        return {
            "task_id": task_id,
            "status": task.get("status"),
            "current_progress": task.get("progress", 0),
            "current_message": task.get("message", ""),
            "progress_history": progress_history,
            "timeline": {
                "created_at": task.get("created_at"),
                "started_at": next(
                    (
                        step["timestamp"]
                        for step in progress_history
                        if step["progress"] > 0
                    ),
                    None,
                ),
                "last_update": task.get("updated_at"),
                "duration": time.time() - task.get("created_at", time.time())
                if isinstance(task.get("created_at"), (int, float))
                else 0,
            },
            "statistics": {
                "total_steps": len(progress_history),
                "completion_rate": task.get("progress", 0),
            },
        }

    async def update_task_progress(
        self, task_id: str, progress: int, message: str = None
    ):
        """Cập nhật tiến độ task"""
        await self.task_service.update_task_progress(task_id, progress, message)

    async def mark_task_processing(self, task_id: str):
        """Đánh dấu task đang xử lý"""
        await self.task_service.mark_task_processing(task_id)

    async def mark_task_completed(self, task_id: str, result: Dict[str, Any]):
        """Đánh dấu task hoàn thành"""
        await self.task_service.mark_task_completed(task_id, result)

    async def mark_task_failed(self, task_id: str, error: str):
        """Đánh dấu task thất bại"""
        await self.task_service.mark_task_failed(task_id, error)

    async def process_pdf_task(self, task_id: str):
        """Xử lý PDF task bất đồng bộ"""

        task = await self.task_service.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            await self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            file_content = task["data"]["file_content"]
            filename = task["data"]["filename"]
            metadata = task["data"]["metadata"]
            create_embeddings = task["data"].get("create_embeddings", True)

            await self.update_task_progress(task_id, 10, "Starting PDF processing...")

            # Import services
            from app.services.enhanced_textbook_service import enhanced_textbook_service
            from app.services.qdrant_service import qdrant_service

            # Bước 1: OCR và phân tích cấu trúc
            await self.update_task_progress(task_id, 20, "Extracting text with OCR...")

            enhanced_result = (
                await enhanced_textbook_service.process_textbook_to_structure(
                    pdf_content=file_content, filename=filename, book_metadata=metadata
                )
            )

            if not enhanced_result.get("success"):
                raise Exception(
                    f"PDF processing failed: {enhanced_result.get('error')}"
                )

            await self.update_task_progress(
                task_id, 60, "PDF structure analysis completed"
            )

            # Bước 2: Tạo embeddings nếu được yêu cầu
            embeddings_result = None
            if create_embeddings:
                await self.update_task_progress(task_id, 70, "Creating embeddings...")

                book_structure_dict = enhanced_result["book"]
                if isinstance(book_structure_dict, str):
                    book_structure_dict = json.loads(book_structure_dict)

                embeddings_result = await qdrant_service.process_textbook(
                    book_id=metadata.get("id"), book_structure=book_structure_dict
                )

                if embeddings_result.get("success"):
                    await self.update_task_progress(
                        task_id, 90, "Embeddings created successfully"
                    )
                else:
                    logger.warning(
                        f"Embeddings creation failed: {embeddings_result.get('error')}"
                    )

            # Tạo kết quả cuối cùng
            result = {
                "success": True,
                "book_id": metadata.get("id"),
                "filename": filename,
                "book_structure": enhanced_result["book"],
                "statistics": {
                    "total_pages": enhanced_result.get("total_pages", 0),
                    "total_chapters": enhanced_result.get("total_chapters", 0),
                    "total_lessons": enhanced_result.get("total_lessons", 0),
                },
                "processing_info": {
                    "ocr_applied": True,
                    "llm_analysis": True,
                    "processing_method": "enhanced_ocr_async",
                },
                "embeddings_created": embeddings_result.get("success", False)
                if embeddings_result
                else False,
                "embeddings_info": {
                    "collection_name": embeddings_result.get("collection_name")
                    if embeddings_result
                    else None,
                    "vector_count": embeddings_result.get("total_chunks", 0)
                    if embeddings_result
                    else 0,
                    "vector_dimension": embeddings_result.get("vector_dimension")
                    if embeddings_result
                    else None,
                },
            }

            # ✅ Hoàn tất task
            await self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing PDF task {task_id}: {e}")
            await self.mark_task_failed(task_id, str(e))

    async def process_cv_task(self, task_id: str):
        """Xử lý CV task bất đồng bộ"""

        task = await self.task_service.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            await self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            file_content = task["data"]["file_content"]
            filename = task["data"]["filename"]

            await self.update_task_progress(task_id, 20, "Processing CV with OCR...")

            # Import CV parser service
            from app.services.cv_parser_service import cv_parser_service

            # Xử lý CV - cần extract text từ PDF trước
            from app.services.simple_ocr_service import simple_ocr_service

            # Extract text từ PDF
            (
                extracted_text,
                ocr_metadata,
            ) = await simple_ocr_service.extract_text_from_pdf(file_content, filename)

            # Parse CV từ text
            cv_result = await cv_parser_service.parse_cv_to_structured_data(
                cv_text=extracted_text
            )

            if not cv_result.get("success"):
                raise Exception(f"CV processing failed: {cv_result.get('error')}")

            await self.update_task_progress(task_id, 80, "CV processing completed")

            # Tạo kết quả
            result = {
                "success": True,
                "filename": filename,
                "cv_data": cv_result.get("cv_data", {}),
                "processing_info": {
                    "ocr_applied": True,
                    "processing_method": "cv_ocr_async",
                },
            }

            await self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing CV task {task_id}: {e}")
            await self.mark_task_failed(task_id, str(e))

    async def process_embeddings_task(self, task_id: str):
        """Xử lý embeddings task bất đồng bộ"""

        task = await self.task_service.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            await self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            book_id = task["data"]["book_id"]
            book_structure = task["data"]["book_structure"]

            await self.update_task_progress(task_id, 20, "Creating embeddings...")

            # Import Qdrant service
            from app.services.qdrant_service import qdrant_service

            # Tạo embeddings
            embeddings_result = await qdrant_service.process_textbook(
                book_id=book_id, book_structure=book_structure
            )

            if not embeddings_result.get("success"):
                raise Exception(
                    f"Embeddings creation failed: {embeddings_result.get('error')}"
                )

            await self.update_task_progress(
                task_id, 90, "Embeddings created successfully"
            )

            # Tạo kết quả
            result = {
                "success": True,
                "book_id": book_id,
                "embeddings_info": {
                    "collection_name": embeddings_result.get("collection_name"),
                    "vector_count": embeddings_result.get("total_chunks", 0),
                    "vector_dimension": embeddings_result.get("vector_dimension"),
                },
            }

            await self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing embeddings task {task_id}: {e}")
            await self.mark_task_failed(task_id, str(e))

    async def create_quick_analysis_task(
        self,
        pdf_content: bytes,
        filename: str,
        create_embeddings: bool = True,
        lesson_id: Optional[str] = None,
    ) -> str:
        """Tạo task phân tích nhanh sách giáo khoa - sử dụng Celery"""

        task_data = {
            "file_content": pdf_content,
            "filename": filename,
            "create_embeddings": create_embeddings,
            "lesson_id": lesson_id,
        }

        # Sử dụng Celery thay vì asyncio.create_task
        from app.services.celery_task_service import celery_task_service

        task_id = await celery_task_service.create_and_dispatch_task(
            task_type="quick_analysis", task_data=task_data
        )

        return task_id

    async def create_guide_import_task(
        self,
        docx_content: bytes,
        filename: str,
        create_embeddings: bool = True,
    ) -> str:
        """Tạo task import hướng dẫn từ file DOCX - sử dụng Celery"""

        task_data = {
            "file_content": docx_content,
            "filename": filename,
            "create_embeddings": create_embeddings,
        }

        # Sử dụng Celery thay vì asyncio.create_task
        from app.services.celery_task_service import celery_task_service

        task_id = await celery_task_service.create_and_dispatch_task(
            task_type="guide_import", task_data=task_data
        )

        return task_id

    async def create_lesson_plan_content_task(
        self,
        lesson_plan_json: Dict[str, Any],
        lesson_id: Optional[str] = None,
    ) -> str:
        """Tạo task sinh nội dung giáo án - sử dụng Celery"""

        task_data = {
            "lesson_plan_json": lesson_plan_json,
            "lesson_id": lesson_id,
        }

        # Sử dụng Celery thay vì asyncio.create_task
        from app.services.celery_task_service import celery_task_service

        task_id = await celery_task_service.create_and_dispatch_task(
            task_type="lesson_plan_content_generation", task_data=task_data
        )

        return task_id

    async def create_smart_exam_task(
        self,
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Tạo task tạo đề thi thông minh - sử dụng Celery"""

        try:
            task_data = {
                "request_data": request_data,
            }

            # Sử dụng Celery thay vì asyncio.create_task
            from app.services.celery_task_service import celery_task_service

            task_id = await celery_task_service.create_and_dispatch_task(
                task_type="smart_exam_generation", task_data=task_data
            )

            return {
                "success": True,
                "task_id": task_id
            }

        except Exception as e:
            logger.error(f"Lỗi tạo smart exam task: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # DEPRECATED: Phương thức này đã được thay thế bằng Celery task
    # Sử dụng create_quick_analysis_task() thay thế
    async def process_quick_analysis_task(self, task_id: str):
        """
        DEPRECATED: Method này đã được thay thế bằng Celery task
        Sử dụng create_quick_analysis_task() để tạo task và Celery worker sẽ xử lý
        """
        logger.warning(
            f"process_quick_analysis_task() is deprecated. Task {task_id} should be processed by Celery worker."
        )

        # Redirect to Celery task status check
        task_status = await self.task_service.get_task_status(task_id)
        if task_status:
            logger.info(f"Task {task_id} status: {task_status.get('status')}")
        else:
            logger.error(f"Task {task_id} not found in MongoDB")

    async def process_pdf_auto_task(self, task_id: str):
        """Xử lý task PDF với tự động phân tích metadata"""

        task = await self.task_service.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            await self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            file_content = task["data"]["file_content"]
            filename = task["data"]["filename"]
            create_embeddings = task["data"].get("create_embeddings", True)

            await self.update_task_progress(
                task_id, 10, "Starting PDF processing with auto metadata detection..."
            )

            # Import services
            from app.services.integrated_textbook_service import (
                integrated_textbook_service,
            )
            from app.services.qdrant_service import qdrant_service

            # Bước 1: Tự động phân tích metadata và cấu trúc
            await self.update_task_progress(
                task_id, 20, "Analyzing PDF content and extracting metadata..."
            )

            integrated_result = await integrated_textbook_service.process_pdf_complete(
                pdf_content=file_content, filename=filename
            )

            if not integrated_result.get("success"):
                raise Exception(
                    f"PDF analysis failed: {integrated_result.get('error')}"
                )

            # Lấy metadata và structure đã được phân tích tự động
            extracted_metadata = integrated_result["extracted_metadata"]
            book_structure = integrated_result["formatted_structure"]

            await self.update_task_progress(
                task_id, 60, "PDF structure analysis completed"
            )

            # Bước 2: Tạo embeddings nếu được yêu cầu
            embeddings_result = None
            if create_embeddings:
                await self.update_task_progress(task_id, 70, "Creating embeddings...")

                embeddings_result = await qdrant_service.process_textbook(
                    book_id=extracted_metadata.get("id", "unknown"),
                    book_structure=book_structure,
                )

                if embeddings_result.get("success"):
                    await self.update_task_progress(
                        task_id, 90, "Embeddings created successfully"
                    )
                else:
                    logger.warning(
                        f"Embeddings creation failed: {embeddings_result.get('error')}"
                    )

            # Tạo kết quả cuối cùng
            result = {
                "success": True,
                "book_id": extracted_metadata.get("id"),
                "filename": filename,
                "auto_detected_metadata": extracted_metadata,
                "book_structure": book_structure,
                "statistics": {
                    "total_pages": integrated_result.get("processing_info", {}).get(
                        "total_pages", 0
                    ),
                    "total_chapters": len(book_structure.get("chapters", [])),
                    "total_lessons": sum(
                        len(ch.get("lessons", []))
                        for ch in book_structure.get("chapters", [])
                    ),
                },
                "processing_info": {
                    "ocr_applied": True,
                    "llm_analysis": True,
                    "auto_metadata_detection": True,
                    "processing_method": "integrated_auto_async",
                },
                "embeddings_created": embeddings_result.get("success", False)
                if embeddings_result
                else False,
                "embeddings_info": {
                    "collection_name": embeddings_result.get("collection_name")
                    if embeddings_result
                    else None,
                    "vector_count": embeddings_result.get("total_chunks", 0)
                    if embeddings_result
                    else 0,
                    "vector_dimension": embeddings_result.get("vector_dimension")
                    if embeddings_result
                    else None,
                }
                if embeddings_result
                else None,
            }

            await self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing auto PDF task {task_id}: {e}")
            await self.mark_task_failed(task_id, str(e))

    async def process_guide_import_task(self, task_id: str):
        """Xử lý task import hướng dẫn từ file DOCX"""
        task = await self.task_service.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            await self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            file_content = task["data"]["file_content"]
            filename = task["data"]["filename"]
            create_embeddings = task["data"].get("create_embeddings", True)

            await self.update_task_progress(task_id, 10, "Starting DOCX guide import...")

            # Import services
            from app.services.exam_import_service import exam_import_service
            from app.services.qdrant_service import qdrant_service

            # Bước 1: Trích xuất text từ DOCX
            await self.update_task_progress(task_id, 20, "Extracting text from DOCX...")

            extracted_text = exam_import_service._extract_text_from_docx_bytes(file_content)

            if not extracted_text or len(extracted_text.strip()) < 50:
                raise Exception("Không thể trích xuất nội dung từ file DOCX hoặc nội dung quá ngắn")

            await self.update_task_progress(task_id, 40, "Text extraction completed")

            # Bước 2: Tạo guide ID và metadata
            import uuid
            guide_id = f"guide_{uuid.uuid4().hex[:8]}"

            # Bước 3: Tạo embeddings nếu được yêu cầu
            embeddings_result = None
            if create_embeddings:
                await self.update_task_progress(task_id, 60, "Creating embeddings for guide content...")

                # Sử dụng process_textbook với text_content cho guide
                embeddings_result = await qdrant_service.process_textbook(
                    book_id=guide_id,
                    text_content=extracted_text,
                    lesson_id=f"guide_{filename}",
                    book_title=f"Guide: {filename}"
                )

                if embeddings_result.get("success"):
                    await self.update_task_progress(
                        task_id, 80, "Embeddings created successfully"
                    )
                else:
                    logger.warning(
                        f"Embeddings creation failed: {embeddings_result.get('error')}"
                    )

            # Tạo kết quả cuối cùng
            result = {
                "success": True,
                "guide_id": guide_id,
                "filename": filename,
                "content_length": len(extracted_text),
                "content_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                "processing_info": {
                    "file_type": "docx",
                    "processing_method": "guide_import",
                    "text_extraction": True,
                },
                "embeddings_created": embeddings_result.get("success", False) if embeddings_result else False,
                "embeddings_info": {
                    "collection_name": embeddings_result.get("collection_name") if embeddings_result else None,
                    "vector_count": embeddings_result.get("total_chunks", 0) if embeddings_result else 0,
                    "vector_dimension": embeddings_result.get("vector_dimension") if embeddings_result else None,
                },
                "message": "Guide imported successfully and ready for RAG search"
            }

            await self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing guide import task {task_id}: {e}")
            await self.mark_task_failed(task_id, str(e))

    async def get_task_with_polling(
        self, task_id: str, timeout: int = 300, poll_interval: int = 2
    ) -> Dict[str, Any]:
        """
        Lấy task với polling cho đến khi hoàn thành hoặc timeout

        Args:
            task_id: ID của task
            timeout: Thời gian timeout (giây) - mặc định 5 phút
            poll_interval: Khoảng cách giữa các lần poll (giây)

        Returns:
            Task data khi hoàn thành

        Raises:
            TimeoutError: Khi task không hoàn thành trong thời gian cho phép
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            task = await self.task_service.get_task_status(task_id)

            if task and task.get("status") in ["completed", "failed"]:
                return task

            # Log progress nếu có
            if task and task.get("progress"):
                logger.info(
                    f"Task {task_id} progress: {task.get('progress')}% - {task.get('message', '')}"
                )

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} không hoàn thành trong {timeout} giây")

    async def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """Lấy progress của task một cách nhanh chóng"""
        task = await self.task_service.get_task_status(task_id)

        if not task:
            return {"error": "Task not found"}

        return {
            "task_id": task_id,
            "status": task.get("status"),
            "progress": task.get("progress", 0),
            "message": task.get("message", ""),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "estimated_completion": self._estimate_completion_time(task),
        }

    def _estimate_completion_time(self, task: Dict[str, Any]) -> str:
        """Ước tính thời gian hoàn thành dựa trên progress"""
        progress = task.get("progress", 0)
        created_at = task.get("created_at")

        if progress > 0 and created_at:
            elapsed = time.time() - created_at
            estimated_total = (elapsed / progress) * 100
            remaining = estimated_total - elapsed

            if remaining > 0:
                return f"~{int(remaining)} giây"

        return "Đang tính toán..."


# Singleton instance
background_task_processor = BackgroundTaskProcessor()


def get_background_task_processor() -> BackgroundTaskProcessor:
    """Lấy singleton instance của BackgroundTaskProcessor"""
    return background_task_processor
