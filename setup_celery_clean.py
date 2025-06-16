"""
Script để setup Celery một cách clean và chuẩn chỉnh cho PlanBookAI
Tạo tất cả các file cần thiết và cập nhật background_task_processor
"""

import os
import shutil

def create_celery_tasks():
    """Tạo các file task cho Celery"""
    
    # Tạo thư mục tasks nếu chưa có
    tasks_dir = "app/tasks"
    if not os.path.exists(tasks_dir):
        os.makedirs(tasks_dir)
    
    # 1. Tạo app/tasks/pdf_tasks.py
    pdf_tasks_content = '''"""
PDF Processing Tasks for Celery Workers
Các task xử lý PDF chạy trong background workers
"""

import asyncio
import logging
from typing import Dict, Any
from celery import current_task

from app.core.celery_app import celery_app, task_with_retry
from app.services.mongodb_task_service import mongodb_task_service

logger = logging.getLogger(__name__)

def run_async_task(coro):
    """Helper để chạy async function trong Celery task"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@task_with_retry()
def process_pdf_quick_analysis(self, task_id: str) -> Dict[str, Any]:
    """Celery task xử lý phân tích nhanh PDF"""
    logger.info(f"Starting PDF quick analysis task: {task_id}")
    
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'message': 'Starting PDF quick analysis...'}
        )
        
        result = run_async_task(_process_quick_analysis_async(task_id))
        logger.info(f"PDF quick analysis completed: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in PDF quick analysis task {task_id}: {e}")
        run_async_task(mongodb_task_service.mark_task_failed(task_id, str(e)))
        raise

async def _process_quick_analysis_async(task_id: str) -> Dict[str, Any]:
    """Async implementation của quick analysis"""
    
    task = await mongodb_task_service.get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")
    
    try:
        await mongodb_task_service.mark_task_processing(task_id)
        
        file_content = task["data"]["file_content"]
        filename = task["data"]["filename"]
        create_embeddings = task["data"].get("create_embeddings", True)
        
        await mongodb_task_service.update_task_progress(
            task_id, 10, "Starting quick textbook analysis..."
        )
        
        from app.services.enhanced_textbook_service import enhanced_textbook_service
        from app.services.qdrant_service import qdrant_service
        import uuid
        
        await mongodb_task_service.update_task_progress(
            task_id, 20, "Extracting pages with OCR..."
        )
        
        pages_data = await enhanced_textbook_service._extract_pages_with_ocr(file_content)
        
        if not pages_data.get("success"):
            raise Exception(f"OCR extraction failed: {pages_data.get('error')}")
        
        await mongodb_task_service.update_task_progress(
            task_id, 40, "Analyzing document structure..."
        )
        
        structure_result = await enhanced_textbook_service._analyze_structure_with_llm(
            pages_data["pages"]
        )
        
        if not structure_result.get("success"):
            raise Exception(f"Structure analysis failed: {structure_result.get('error')}")
        
        await mongodb_task_service.update_task_progress(
            task_id, 60, "Processing content and creating embeddings..."
        )
        
        embeddings_info = {"created": False, "count": 0}
        
        if create_embeddings:
            book_id = str(uuid.uuid4())
            total_embeddings = 0
            
            for chapter in structure_result.get("book_structure", {}).get("chapters", []):
                for lesson in chapter.get("lessons", []):
                    if lesson.get("content", {}).get("text"):
                        embedding_result = await qdrant_service.create_embeddings_for_lesson(
                            book_id=book_id,
                            chapter_id=chapter.get("chapter_id"),
                            lesson_id=lesson.get("lesson_id"),
                            lesson_content=lesson["content"]
                        )
                        
                        if embedding_result.get("success"):
                            total_embeddings += embedding_result.get("embeddings_count", 0)
            
            embeddings_info = {
                "created": True,
                "count": total_embeddings,
                "book_id": book_id
            }
        
        await mongodb_task_service.update_task_progress(task_id, 90, "Finalizing results...")
        
        result = {
            "success": True,
            "filename": filename,
            "book_structure": structure_result.get("book_structure", {}),
            "statistics": structure_result.get("statistics", {}),
            "processing_info": {
                "method": "quick_analysis_celery",
                "ocr_applied": True,
                "structure_analyzed": True,
                "embeddings_created": embeddings_info["created"],
            },
            "embeddings_info": embeddings_info,
        }
        
        await mongodb_task_service.mark_task_completed(task_id, result)
        return result
        
    except Exception as e:
        logger.error(f"Error in quick analysis async processing {task_id}: {e}")
        await mongodb_task_service.mark_task_failed(task_id, str(e))
        raise

@task_with_retry()
def process_pdf_textbook(self, task_id: str) -> Dict[str, Any]:
    """Celery task xử lý sách giáo khoa PDF với metadata"""
    logger.info(f"Starting PDF textbook processing task: {task_id}")
    
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'message': 'Starting PDF textbook processing...'}
        )
        
        result = run_async_task(_process_textbook_async(task_id))
        logger.info(f"PDF textbook processing completed: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in PDF textbook processing task {task_id}: {e}")
        run_async_task(mongodb_task_service.mark_task_failed(task_id, str(e)))
        raise

async def _process_textbook_async(task_id: str) -> Dict[str, Any]:
    """Async implementation của textbook processing"""
    
    task = await mongodb_task_service.get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")
    
    try:
        await mongodb_task_service.mark_task_processing(task_id)
        
        file_content = task["data"]["file_content"]
        filename = task["data"]["filename"]
        metadata = task["data"]["metadata"]
        create_embeddings = task["data"].get("create_embeddings", True)
        
        await mongodb_task_service.update_task_progress(task_id, 10, "Starting PDF processing...")
        
        from app.services.enhanced_textbook_service import enhanced_textbook_service
        from app.services.qdrant_service import qdrant_service
        
        await mongodb_task_service.update_task_progress(task_id, 20, "Extracting text with OCR...")
        
        enhanced_result = await enhanced_textbook_service.process_textbook_to_structure(
            pdf_content=file_content, filename=filename, book_metadata=metadata
        )
        
        if not enhanced_result.get("success"):
            raise Exception(f"Textbook processing failed: {enhanced_result.get('error')}")
        
        await mongodb_task_service.update_task_progress(task_id, 70, "Creating embeddings...")
        
        embeddings_info = {"created": False, "count": 0}
        
        if create_embeddings and enhanced_result.get("book_structure"):
            embedding_result = await qdrant_service.create_embeddings_for_textbook(
                enhanced_result["book_structure"]
            )
            
            if embedding_result.get("success"):
                embeddings_info = {
                    "created": True,
                    "count": embedding_result.get("embeddings_count", 0),
                    "book_id": embedding_result.get("book_id")
                }
        
        await mongodb_task_service.update_task_progress(task_id, 90, "Finalizing results...")
        
        result = {
            **enhanced_result,
            "embeddings_info": embeddings_info,
            "processing_info": {
                **enhanced_result.get("processing_info", {}),
                "method": "textbook_celery",
                "embeddings_created": embeddings_info["created"],
            }
        }
        
        await mongodb_task_service.mark_task_completed(task_id, result)
        return result
        
    except Exception as e:
        logger.error(f"Error in textbook async processing {task_id}: {e}")
        await mongodb_task_service.mark_task_failed(task_id, str(e))
        raise

@task_with_retry()
def process_pdf_textbook_auto(self, task_id: str) -> Dict[str, Any]:
    """Celery task xử lý sách giáo khoa PDF với auto metadata detection"""
    logger.info(f"Starting PDF textbook auto processing task: {task_id}")
    
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'message': 'Starting PDF auto processing...'}
        )
        
        result = run_async_task(_process_textbook_auto_async(task_id))
        logger.info(f"PDF textbook auto processing completed: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in PDF textbook auto processing task {task_id}: {e}")
        run_async_task(mongodb_task_service.mark_task_failed(task_id, str(e)))
        raise

async def _process_textbook_auto_async(task_id: str) -> Dict[str, Any]:
    """Async implementation của textbook auto processing"""
    
    task = await mongodb_task_service.get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")
    
    try:
        await mongodb_task_service.mark_task_processing(task_id)
        
        file_content = task["data"]["file_content"]
        filename = task["data"]["filename"]
        create_embeddings = task["data"].get("create_embeddings", True)
        
        await mongodb_task_service.update_task_progress(
            task_id, 10, "Starting PDF processing with auto metadata detection..."
        )
        
        from app.services.integrated_textbook_service import integrated_textbook_service
        from app.services.qdrant_service import qdrant_service
        
        await mongodb_task_service.update_task_progress(
            task_id, 20, "Analyzing PDF content and extracting metadata..."
        )
        
        integrated_result = await integrated_textbook_service.process_pdf_complete(
            pdf_content=file_content, filename=filename
        )
        
        if not integrated_result.get("success"):
            raise Exception(f"Auto processing failed: {integrated_result.get('error')}")
        
        await mongodb_task_service.update_task_progress(task_id, 70, "Creating embeddings...")
        
        embeddings_info = {"created": False, "count": 0}
        
        if create_embeddings and integrated_result.get("book_structure"):
            embedding_result = await qdrant_service.create_embeddings_for_textbook(
                integrated_result["book_structure"]
            )
            
            if embedding_result.get("success"):
                embeddings_info = {
                    "created": True,
                    "count": embedding_result.get("embeddings_count", 0),
                    "book_id": embedding_result.get("book_id")
                }
        
        await mongodb_task_service.update_task_progress(task_id, 90, "Finalizing results...")
        
        result = {
            **integrated_result,
            "embeddings_info": embeddings_info,
            "processing_info": {
                **integrated_result.get("processing_info", {}),
                "method": "textbook_auto_celery",
                "embeddings_created": embeddings_info["created"],
            }
        }
        
        await mongodb_task_service.mark_task_completed(task_id, result)
        return result
        
    except Exception as e:
        logger.error(f"Error in textbook auto async processing {task_id}: {e}")
        await mongodb_task_service.mark_task_failed(task_id, str(e))
        raise
'''
    
    # Ghi file pdf_tasks.py
    with open(f"{tasks_dir}/pdf_tasks.py", "w", encoding="utf-8") as f:
        f.write(pdf_tasks_content)
    
    print("✅ Created app/tasks/pdf_tasks.py")

if __name__ == "__main__":
    print("🚀 Setting up Celery for PlanBookAI...")
    create_celery_tasks()
    print("🎉 Celery setup completed!")
