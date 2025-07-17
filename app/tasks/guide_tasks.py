"""
Celery tasks cho xử lý guide import từ file DOCX với LLM formatting và smart chunking
"""

import asyncio
import logging
from typing import Dict, Any
from celery import Celery
from app.core.celery_app import celery_app
from app.services.mongodb_task_service import get_mongodb_task_service

logger = logging.getLogger(__name__)


def run_async_task(coro):
    """Helper để chạy async function trong Celery task"""
    try:
        # Always create a new event loop for each task to avoid "Event loop is closed" error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(coro)
        finally:
            # Always close the loop after use
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Wait for all tasks to be cancelled
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                loop.close()
            except Exception as cleanup_error:
                logger.warning(f"Error during loop cleanup: {cleanup_error}")

    except Exception as e:
        logger.error(f"Error in run_async_task: {e}")
        raise e


async def process_guide_import_task(task_id: str) -> Dict[str, Any]:
    """
    Xử lý task import hướng dẫn từ file DOCX với LLM formatting và smart chunking

    Luồng xử lý:
    1. Trích xuất text từ DOCX
    2. LLM formatting để cấu trúc lại nội dung
    3. Upload DOCX file lên Supabase Storage
    4. Smart chunking và tạo embeddings

    Lưu ý: Báo lỗi thay vì fallback khi có lỗi xảy ra
    """
    task_service = get_mongodb_task_service()
    task = await task_service.get_task_status(task_id)
    if not task:
        logger.error(f"Task {task_id} not found")
        return {"success": False, "error": "Task not found"}

    try:
        await task_service.mark_task_processing(task_id)

        # Lấy dữ liệu task
        file_content = task["data"]["file_content"]
        filename = task["data"]["filename"]
        book_id = task["data"]["book_id"]
        create_embeddings = task["data"].get("create_embeddings", True)

        await task_service.update_task_progress(task_id, 10, "Starting DOCX guide import...")

        # Import services
        from app.services.exam_import_service import get_exam_import_service
        from app.services.llm_service import get_llm_service
        from app.services.qdrant_service import get_qdrant_service

        exam_import_service = get_exam_import_service()
        llm_service = get_llm_service()
        qdrant_service = get_qdrant_service()

        # Bước 1: Trích xuất text từ DOCX
        await task_service.update_task_progress(task_id, 15, "Extracting text from DOCX...")

        extracted_text = exam_import_service._extract_text_from_docx_bytes(file_content)

        if not extracted_text or len(extracted_text.strip()) < 50:
            raise Exception("Không thể trích xuất nội dung từ file DOCX hoặc nội dung quá ngắn")

        await task_service.update_task_progress(task_id, 25, "Text extraction completed")

        # Bước 2: LLM formatting để cấu trúc lại nội dung
        await task_service.update_task_progress(task_id, 35, "Formatting content with LLM...")

        # Sử dụng format_document_text với document_type="guide" để tối ưu cho guide content
        format_result = await llm_service.format_document_text(
            raw_text=extracted_text,
            document_type="guide"
        )

        if not format_result.get("success"):
            # Báo lỗi thay vì fallback
            error_msg = f"LLM formatting failed: {format_result.get('error', 'Unknown error')}"
            logger.error(error_msg)
            raise Exception(error_msg)

        formatted_text = format_result["formatted_text"]
        await task_service.update_task_progress(task_id, 50, "Content formatting completed")

        # Bước 3: Upload DOCX file lên Supabase Storage
        await task_service.update_task_progress(task_id, 50, "Uploading DOCX to Supabase Storage...")

        file_url = None
        uploaded_at = None
        try:
            from app.services.supabase_storage_service import get_supabase_storage_service

            supabase_service = get_supabase_storage_service()
            if supabase_service.is_available():
                upload_result = await supabase_service.upload_document_file(
                    file_content=file_content,
                    book_id=book_id,
                    lesson_id=f"guide_{filename.replace('.docx', '')}",
                    original_filename=filename,
                    file_type="docx"
                )

                if upload_result.get("success"):
                    file_url = upload_result.get("file_url")
                    uploaded_at = upload_result.get("uploaded_at")
                    logger.info(f"✅ DOCX uploaded to Supabase: {file_url}")
                    logger.info(f"✅ Upload time: {uploaded_at}")
                    logger.info(f"🔍 Full upload result: {upload_result}")
                else:
                    logger.error(f"❌ Failed to upload DOCX to Supabase: {upload_result.get('error')}")
                    logger.error(f"🔍 Full upload result: {upload_result}")
            else:
                logger.warning("Supabase service not available, skipping DOCX upload")
        except Exception as e:
            logger.warning(f"Error uploading DOCX to Supabase: {e}")

        # Bước 4: Tạo embeddings với smart chunking nếu được yêu cầu
        embeddings_result = None
        if create_embeddings:
            await task_service.update_task_progress(task_id, 70, "Creating embeddings with smart chunking...")

            # Log metadata trước khi tạo embeddings
            logger.info(f"🔍 Creating embeddings with metadata:")
            logger.info(f"   - book_id: {book_id}")
            logger.info(f"   - lesson_id: guide_{filename}")
            logger.info(f"   - content_type: guide")
            logger.info(f"   - file_url: {file_url}")
            logger.info(f"   - uploaded_at: {uploaded_at}")

            # Sử dụng process_textbook với formatted content và metadata từ Supabase
            embeddings_result = await qdrant_service.process_textbook(
                book_id=book_id,
                content=formatted_text,  # Sử dụng formatted text thay vì raw text
                lesson_id=f"guide_{filename}",
                content_type="guide",
                file_url=file_url,  # Truyền URL của file DOCX từ Supabase
                uploaded_at=uploaded_at  # Truyền thời gian upload
            )

            if not embeddings_result.get("success"):
                # Báo lỗi thay vì fallback
                error_msg = f"Embeddings creation failed: {embeddings_result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                raise Exception(error_msg)

            await task_service.update_task_progress(task_id, 85, "Embeddings created successfully")

        # Tạo kết quả cuối cùng
        result = {
            "success": True,
            "book_id": book_id,
            "filename": filename,
            "content_length": len(formatted_text),  # Sử dụng formatted text length
            "content_preview": formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text,
            "processing_info": {
                "file_type": "docx",
                "processing_method": "guide_import_with_llm_formatting",
                "text_extraction": True,
                "llm_formatting": True,
                "smart_chunking": True,
                "original_length": len(extracted_text),
                "formatted_length": len(formatted_text),
            },
            "file_storage": {
                "uploaded_to_supabase": file_url is not None,
                "file_url": file_url,
                "uploaded_at": uploaded_at,
            },
            "embeddings_created": embeddings_result.get("success", False) if embeddings_result else False,
            "embeddings_info": {
                "collection_name": embeddings_result.get("collection_name") if embeddings_result else None,
                "vector_count": embeddings_result.get("total_chunks", 0) if embeddings_result else 0,
                "vector_dimension": embeddings_result.get("vector_dimension") if embeddings_result else None,
            },
            "message": "Guide imported successfully with LLM formatting, smart chunking, and uploaded to Supabase storage, ready for RAG search"
        }

        await task_service.update_task_progress(task_id, 100, "Guide import completed successfully")
        await task_service.mark_task_completed(task_id, result)
        return result

    except Exception as e:
        logger.error(f"Error processing guide import task {task_id}: {e}")
        await task_service.mark_task_failed(task_id, str(e))
        return {"success": False, "error": str(e)}


@celery_app.task(bind=True, name="app.tasks.guide_tasks.process_guide_import")
def process_guide_import(self, task_id: str):
    """
    Celery task để xử lý guide import từ file DOCX với LLM formatting và smart chunking

    Args:
        task_id: ID của task trong MongoDB
    """
    logger.info(f"Starting Celery task for guide import: {task_id}")

    try:
        # Chạy task processing trực tiếp trong guide_tasks
        result = run_async_task(process_guide_import_task(task_id))

        logger.info(f"Completed Celery task for guide import: {task_id}")
        return {"status": "completed", "task_id": task_id, "result": result}

    except Exception as e:
        logger.error(f"Error in Celery guide import task {task_id}: {e}")

        # Mark task as failed in MongoDB
        try:
            run_async_task(
                get_mongodb_task_service().mark_task_failed(task_id, str(e))
            )
        except Exception as mark_error:
            logger.error(f"Failed to mark task as failed: {mark_error}")

        # Re-raise để Celery biết task failed
        raise e
