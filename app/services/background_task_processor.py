"""
Background Task Processor - Xử lý các task bất đồng bộ sử dụng TaskService
"""

import asyncio
import logging
import json
from typing import Dict, Any

from app.services.mongodb_task_service import mongodb_task_service, TaskType

logger = logging.getLogger(__name__)


class BackgroundTaskProcessor:
    """Service xử lý background tasks sử dụng TaskService"""

    def __init__(self):
        self.task_service = mongodb_task_service

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

            await self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing PDF task {task_id}: {e}")
            await self.mark_task_failed(task_id, str(e))

    async def process_cv_task(self, task_id: str):
        """Xử lý CV task bất đồng bộ"""

        task = self.task_service.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            file_content = task["data"]["file_content"]
            filename = task["data"]["filename"]

            self.update_task_progress(task_id, 20, "Processing CV with OCR...")

            # Import CV service
            from app.services.cv_service import cv_service

            # Xử lý CV
            cv_result = await cv_service.process_cv(
                pdf_content=file_content, filename=filename
            )

            if not cv_result.get("success"):
                raise Exception(f"CV processing failed: {cv_result.get('error')}")

            self.update_task_progress(task_id, 80, "CV processing completed")

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

            self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing CV task {task_id}: {e}")
            self.mark_task_failed(task_id, str(e))

    async def process_embeddings_task(self, task_id: str):
        """Xử lý embeddings task bất đồng bộ"""

        task = self.task_service.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            book_id = task["data"]["book_id"]
            book_structure = task["data"]["book_structure"]

            self.update_task_progress(task_id, 20, "Creating embeddings...")

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

            self.update_task_progress(task_id, 90, "Embeddings created successfully")

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

            self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing embeddings task {task_id}: {e}")
            self.mark_task_failed(task_id, str(e))

    async def create_quick_analysis_task(
        self, pdf_content: bytes, filename: str, create_embeddings: bool = True
    ) -> str:
        """Tạo task phân tích nhanh sách giáo khoa"""

        task_data = {
            "file_content": pdf_content,
            "filename": filename,
            "create_embeddings": create_embeddings,
        }

        task_id = await self.create_task("quick_analysis", task_data)

        # Bắt đầu xử lý task bất đồng bộ
        asyncio.create_task(self.process_quick_analysis_task(task_id))

        return task_id

    async def process_quick_analysis_task(self, task_id: str):
        """Xử lý task phân tích nhanh sách giáo khoa"""

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
                task_id, 10, "Starting quick textbook analysis..."
            )

            # Import services
            from app.services.enhanced_textbook_service import enhanced_textbook_service
            from app.services.qdrant_service import qdrant_service
            import uuid

            # Bước 1: OCR và phân tích cấu trúc
            await self.update_task_progress(task_id, 20, "Extracting pages with OCR...")

            pages_data = await enhanced_textbook_service._extract_pages_with_ocr(
                file_content
            )

            await self.update_task_progress(task_id, 40, "Analyzing book structure...")

            # Tạo metadata tự động
            book_metadata = {
                "id": str(uuid.uuid4())[:8],  # Tạo ID ngắn
                "title": filename.replace(".pdf", ""),
                "subject": "Chưa xác định",
                "grade": "Chưa xác định",
                "author": "Chưa xác định",
                "language": "vi",
            }

            book_structure = (
                await enhanced_textbook_service._analyze_book_structure_enhanced(
                    pages_data, book_metadata
                )
            )

            if not book_structure:
                raise Exception("Failed to analyze textbook structure")

            await self.update_task_progress(
                task_id, 60, "Book structure analysis completed"
            )

            # Bước 2: Tạo embeddings nếu được yêu cầu
            embeddings_result = None
            if create_embeddings:
                await self.update_task_progress(task_id, 70, "Creating embeddings...")

                embeddings_result = await qdrant_service.process_textbook(
                    book_id=book_metadata.get("id"), book_structure=book_structure
                )

                if embeddings_result.get("success"):
                    await self.update_task_progress(
                        task_id, 90, "Embeddings created successfully"
                    )
                else:
                    logger.warning(
                        f"Embeddings creation failed: {embeddings_result.get('error')}"
                    )

            # Tạo kết quả với định dạng giống /process-textbook
            result = {
                "success": True,
                "book_id": book_metadata.get("id"),
                "filename": filename,
                "book_structure": book_structure,
                "statistics": {
                    "total_pages": len(pages_data),
                    "total_chapters": len(book_structure.get("chapters", [])),
                    "total_lessons": sum(
                        len(chapter.get("lessons", []))
                        for chapter in book_structure.get("chapters", [])
                    ),
                },
                "processing_info": {
                    "ocr_applied": True,
                    "llm_analysis": True,
                    "processing_method": "quick_analysis_async",
                },
                "message": "Quick textbook analysis completed successfully with searchable embeddings in Qdrant",
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
            logger.error(f"Error processing quick analysis task {task_id}: {e}")
            await self.mark_task_failed(task_id, str(e))

    async def process_pdf_auto_task(self, task_id: str):
        """Xử lý task PDF với tự động phân tích metadata"""

        task = self.task_service.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            file_content = task["data"]["file_content"]
            filename = task["data"]["filename"]
            create_embeddings = task["data"].get("create_embeddings", True)

            self.update_task_progress(
                task_id, 10, "Starting PDF processing with auto metadata detection..."
            )

            # Import services
            from app.services.integrated_textbook_service import (
                integrated_textbook_service,
            )
            from app.services.qdrant_service import qdrant_service

            # Bước 1: Tự động phân tích metadata và cấu trúc
            self.update_task_progress(
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

            self.update_task_progress(task_id, 60, "PDF structure analysis completed")

            # Bước 2: Tạo embeddings nếu được yêu cầu
            embeddings_result = None
            if create_embeddings:
                self.update_task_progress(task_id, 70, "Creating embeddings...")

                embeddings_result = await qdrant_service.process_textbook(
                    book_id=extracted_metadata.get("id"), book_structure=book_structure
                )

                if embeddings_result.get("success"):
                    self.update_task_progress(
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

            self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing auto PDF task {task_id}: {e}")
            self.mark_task_failed(task_id, str(e))


# Singleton instance
background_task_processor = BackgroundTaskProcessor()
