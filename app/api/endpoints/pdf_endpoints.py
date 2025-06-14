"""
PDF Endpoints - Endpoint đơn giản để xử lý PDF với OCR và LLM formatting
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import Dict, Any

from app.services.llm_service import llm_service
from app.services.textbook_parser_service import textbook_parser_service
from app.services.enhanced_textbook_service import enhanced_textbook_service
from app.services.background_task_processor import background_task_processor

logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/process-textbook-async", response_model=Dict[str, Any])
async def process_textbook_async(
    file: UploadFile = File(...),
    create_embeddings: bool = Form(True),
) -> Dict[str, Any]:
    """
    Xử lý sách giáo khoa bất đồng bộ với tự động phân tích metadata

    Upload PDF và nhận task_id ngay lập tức, không cần đợi xử lý hoàn thành.
    Hệ thống sẽ tự động phân tích và trích xuất metadata từ nội dung PDF.

    Args:
        file: File PDF sách giáo khoa
        create_embeddings: Có tạo embeddings cho RAG search không

    Returns:
        Dict chứa task_id để theo dõi tiến độ

    Example:
        1. Upload: POST /process-textbook-async
        2. Nhận: {"task_id": "abc-123", "status": "pending"}
        3. Theo dõi: GET /task-status/abc-123
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(
            f"Received async PDF upload: {file.filename} ({len(file_content)} bytes)"
        )

        # Tạo background task với auto metadata detection
        task_data = {
            "file_content": file_content,
            "filename": file.filename,
            "auto_detect_metadata": True,  # Flag để tự động phân tích metadata
            "create_embeddings": create_embeddings,
        }

        task_id = background_task_processor.create_task(
            task_type="process_textbook_auto", task_data=task_data
        )

        # Bắt đầu xử lý bất đồng bộ trong background
        import threading

        def run_background_task():
            """Chạy task trong thread riêng biệt"""
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    background_task_processor.process_pdf_auto_task(task_id)
                )
            finally:
                loop.close()

        # Chạy trong thread riêng để không block response
        thread = threading.Thread(target=run_background_task, daemon=True)
        thread.start()

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": f"PDF processing started with auto metadata detection. Use /task-status/{task_id} to check progress.",
            "filename": file.filename,
            "estimated_time": "2-5 minutes",
            "check_status_url": f"/api/v1/tasks/status/{task_id}",
            "auto_metadata": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting async PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/process-textbook", response_model=Dict[str, Any])
async def process_textbook(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    create_embeddings: bool = Form(True),  # Thêm tham số mới
) -> Dict[str, Any]:
    """
    Xử lý sách giáo khoa thành cấu trúc dữ liệu cho giáo án

    Args:
        file: PDF file của sách giáo khoa
        metadata: JSON string chứa metadata của sách
        create_embeddings: Có tạo embeddings cho RAG không

    Returns:
        Dict containing processing results
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Parse metadata
        import json

        try:
            book_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON format")

        # Validate required metadata fields
        required_fields = ["id", "title"]
        for field in required_fields:
            if field not in book_metadata:
                raise HTTPException(
                    status_code=400, detail=f"Missing required metadata field: {field}"
                )

        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Processing textbook: {file.filename} ({len(file_content)} bytes)")

        # Process textbook with enhanced service for better structure
        enhanced_result = await enhanced_textbook_service.process_textbook_to_structure(
            pdf_content=file_content,
            filename=file.filename,
            book_metadata=book_metadata,
        )

        if not enhanced_result.get("success", False):
            # Fallback to old service if enhanced fails
            logger.warning("Enhanced processing failed, falling back to old service")
            result = await textbook_parser_service.process_textbook(
                pdf_content=file_content,
                filename=file.filename,
                book_metadata=book_metadata,
            )

            return {
                "success": result.get("success", False),
                "book_id": result.get("book_id"),
                "book_path": result.get("book_path"),
                "lessons_processed": result.get("lessons_processed", 0),
                "total_pages": result.get("total_pages", 0),
                "processing_error": result.get("error"),
                "message": result.get(
                    "message", "Textbook processing completed (fallback mode)"
                ),
            }

        # Return enhanced structure with OCR results
        result = {
            "success": True,
            "book_id": book_metadata.get("id"),
            "filename": file.filename,
            "book_structure": enhanced_result["book"],
            "statistics": {
                "total_pages": enhanced_result.get("total_pages", 0),
                "total_chapters": enhanced_result.get("total_chapters", 0),
                "total_lessons": enhanced_result.get("total_lessons", 0),
            },
            "processing_info": {
                "ocr_applied": True,
                "llm_analysis": llm_service.is_available(),
                "processing_method": "enhanced_ocr",
            },
            "message": "Textbook processed successfully with OCR structure analysis",
        }

        # Thêm phần tạo embeddings nếu được yêu cầu
        if create_embeddings:
            try:
                from app.services.qdrant_service import qdrant_service

                # SỬA LỖI: Sử dụng result["book_structure"] thay vì enhanced_result["book"]
                book_structure_dict = result["book_structure"]

                logger.info(
                    f"Creating embeddings for book_id: {book_metadata.get('id')}"
                )
                logger.debug(f"Book structure type: {type(book_structure_dict)}")

                # Đảm bảo book_structure_dict là dictionary
                if isinstance(book_structure_dict, str):
                    import json

                    book_structure_dict = json.loads(book_structure_dict)

                # Tạo embeddings và lưu vào Qdrant
                logger.info("Calling qdrant_service.process_textbook...")
                embedding_result = await qdrant_service.process_textbook(
                    book_id=book_metadata.get("id"),
                    book_structure=book_structure_dict,  # Gửi đi dictionary đã được parse
                )

                logger.info(f"Embedding result: {embedding_result}")

                # Thêm thông tin về embeddings vào kết quả
                result["embeddings_created"] = embedding_result.get("success", False)
                result["embeddings_info"] = {
                    "collection_name": embedding_result.get("collection_name"),
                    "vector_count": embedding_result.get("total_chunks", 0),
                    "vector_dimension": embedding_result.get("vector_dimension"),
                }

                if embedding_result.get("success", False):
                    result["message"] += " with searchable embeddings in Qdrant"
                else:
                    result["message"] += (
                        f" (embeddings failed: {embedding_result.get('error', 'unknown error')})"
                    )

            except Exception as e:
                logger.error(f"Embeddings creation failed with exception: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                result["embeddings_created"] = False
                result["embeddings_error"] = str(e)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing textbook: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/quick-textbook-analysis", response_model=Dict[str, Any])
async def quick_textbook_analysis(
    file: UploadFile = File(...),
    create_embeddings: bool = Form(True),
) -> Dict[str, Any]:
    """
    Phân tích nhanh cấu trúc sách giáo khoa với xử lý bất đồng bộ

    Upload PDF và nhận task_id ngay lập tức. Hệ thống sẽ:
    1. Phân tích cấu trúc sách (chapters, lessons)
    2. Tự động trích xuất metadata
    3. Tạo embeddings và lưu vào Qdrant (nếu được yêu cầu)
    4. Trả về kết quả với định dạng giống /process-textbook

    Args:
        file: PDF file của sách giáo khoa
        create_embeddings: Có tạo embeddings cho RAG search không

    Returns:
        Dict chứa task_id để theo dõi tiến độ
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(
            f"Starting quick analysis task for: {file.filename} ({len(file_content)} bytes)"
        )

        # Tạo task bất đồng bộ
        task_id = await background_task_processor.create_quick_analysis_task(
            pdf_content=file_content,
            filename=file.filename,
            create_embeddings=create_embeddings,
        )

        return {
            "success": True,
            "task_id": task_id,
            "filename": file.filename,
            "status": "processing",
            "message": "Quick textbook analysis task created successfully. Use /api/v1/tasks/{task_id}/status to check progress.",
            "endpoints": {
                "check_status": f"/api/v1/tasks/{task_id}/status",
                "get_result": f"/api/v1/tasks/{task_id}/result",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating quick analysis task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbooks", response_model=Dict[str, Any])
async def list_all_textbooks() -> Dict[str, Any]:
    """
    Lấy danh sách tất cả sách giáo khoa đã được xử lý

    Returns:
        Dict chứa danh sách tất cả textbooks với ID và metadata
    """
    try:
        from app.services.qdrant_service import qdrant_service

        # Lấy danh sách từ Qdrant collections
        textbooks = []

        if qdrant_service.qdrant_client:
            collections = qdrant_service.qdrant_client.get_collections().collections

            for collection in collections:
                if collection.name.startswith("textbook_"):
                    book_id = collection.name.replace("textbook_", "")

                    # Lấy metadata từ collection
                    try:
                        # Lấy một point bất kỳ để lấy metadata
                        search_result = qdrant_service.qdrant_client.scroll(
                            collection_name=collection.name, limit=1, with_payload=True
                        )

                        if search_result[0]:  # Có points
                            point = search_result[0][0]
                            payload = point.payload or {}

                            textbooks.append(
                                {
                                    "book_id": book_id,
                                    "title": payload.get("book_title", "Unknown"),
                                    "subject": payload.get("subject", "Unknown"),
                                    "grade": payload.get("grade", "Unknown"),
                                    "collection_name": collection.name,
                                    "vector_count": getattr(
                                        collection, "vectors_count", 0
                                    ),
                                    "status": "available",
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Error getting metadata for {book_id}: {e}")
                        textbooks.append(
                            {
                                "book_id": book_id,
                                "title": "Unknown",
                                "collection_name": collection.name,
                                "vector_count": getattr(collection, "vectors_count", 0),
                                "status": "available",
                                "error": str(e),
                            }
                        )

        return {
            "success": True,
            "total_textbooks": len(textbooks),
            "textbooks": textbooks,
            "message": f"Found {len(textbooks)} textbooks",
        }

    except Exception as e:
        logger.error(f"Error listing textbooks: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbooks-full", response_model=Dict[str, Any])
async def list_all_textbooks_full_format() -> Dict[str, Any]:
    """
    Lấy danh sách tất cả sách giáo khoa với format đầy đủ như /process-textbook

    Returns:
        Dict chứa danh sách tất cả textbooks với format đầy đủ
    """
    try:
        from app.services.qdrant_service import qdrant_service
        from qdrant_client import models as qdrant_models

        # Lấy danh sách từ Qdrant collections
        textbooks = []

        if qdrant_service.qdrant_client:
            collections = qdrant_service.qdrant_client.get_collections().collections

            for collection in collections:
                if collection.name.startswith("textbook_"):
                    book_id = collection.name.replace("textbook_", "")

                    # Tìm metadata point để lấy original_book_structure
                    try:
                        search_result = qdrant_service.qdrant_client.scroll(
                            collection_name=collection.name,
                            scroll_filter=qdrant_models.Filter(
                                must=[
                                    qdrant_models.FieldCondition(
                                        key="type",
                                        match=qdrant_models.MatchValue(
                                            value="metadata"
                                        ),
                                    )
                                ]
                            ),
                            limit=1,
                            with_payload=True,
                        )

                        if search_result[0]:  # Có metadata point
                            metadata_point = search_result[0][0]
                            payload = metadata_point.payload or {}

                            # Lấy original_book_structure nếu có
                            original_structure = payload.get("original_book_structure")

                            if original_structure:
                                # Trả về format đầy đủ như /process-textbook
                                textbook_data = {
                                    "success": True,
                                    "book_id": book_id,
                                    "filename": f"{book_id}.pdf",  # Không có filename gốc
                                    "book_structure": original_structure,
                                    "statistics": {
                                        "total_pages": payload.get(
                                            "book_total_pages", 0
                                        ),
                                        "total_chapters": len(
                                            original_structure.get("chapters", [])
                                        ),
                                        "total_lessons": sum(
                                            len(ch.get("lessons", []))
                                            for ch in original_structure.get(
                                                "chapters", []
                                            )
                                        ),
                                    },
                                    "processing_info": {
                                        "ocr_applied": True,
                                        "llm_analysis": True,
                                        "processing_method": "retrieved_from_qdrant",
                                        "processed_at": payload.get("processed_at"),
                                    },
                                    "embeddings_created": True,
                                    "embeddings_info": {
                                        "collection_name": collection.name,
                                        "vector_count": getattr(
                                            collection, "vectors_count", 0
                                        ),
                                        "vector_dimension": 384,  # Default dimension
                                    },
                                    "message": "Textbook retrieved successfully from Qdrant vector database",
                                }
                                textbooks.append(textbook_data)
                            else:
                                # Fallback format nếu không có original_structure
                                logger.warning(
                                    f"No original_book_structure found for {book_id}"
                                )

                    except Exception as e:
                        logger.warning(
                            f"Error getting full metadata for {book_id}: {e}"
                        )

        return {
            "success": True,
            "total_textbooks": len(textbooks),
            "textbooks": textbooks,
            "message": f"Found {len(textbooks)} textbooks with full format",
        }

    except Exception as e:
        logger.error(f"Error listing textbooks with full format: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbook/{book_id}/structure", response_model=Dict[str, Any])
async def get_textbook_structure(book_id: str) -> Dict[str, Any]:
    """
    Lấy cấu trúc của sách giáo khoa đã xử lý

    Args:
        book_id: ID của sách

    Returns:
        Dict containing book structure
    """
    try:
        # Thử lấy từ file system trước (legacy data)
        structure = await textbook_parser_service.get_book_structure(book_id)

        if structure:
            return {"success": True, "book_id": book_id, "structure": structure}

        # Nếu không tìm thấy trong file system, thử lấy từ Qdrant
        from app.services.qdrant_service import qdrant_service

        # Kiểm tra collection có tồn tại không
        collection_name = f"textbook_{book_id}"
        try:
            if not qdrant_service.qdrant_client:
                raise Exception("Qdrant client not available")

            collections = qdrant_service.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name in existing_names:
                # Lấy tất cả points từ collection để tạo structure
                all_points = qdrant_service.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=True,
                )

                if all_points[0]:  # Có data
                    metadata = None
                    chapters = {}

                    # Phân loại points
                    for point in all_points[0]:
                        payload = point.payload or {}
                        point_type = payload.get("type", "")

                        if point_type == "metadata":
                            metadata = payload
                        elif point_type == "title":
                            chapter_id = payload.get("chapter_id")
                            chapter_title = payload.get("chapter_title")
                            lesson_id = payload.get("lesson_id")
                            lesson_title = payload.get("lesson_title")

                            if chapter_id and chapter_id not in chapters:
                                chapters[chapter_id] = {
                                    "chapter_id": chapter_id,
                                    "chapter_title": chapter_title,
                                    "lessons": [],
                                }

                            if chapter_id and lesson_id:
                                chapters[chapter_id]["lessons"].append(
                                    {
                                        "lesson_id": lesson_id,
                                        "lesson_title": lesson_title,
                                    }
                                )

                    # Thêm lesson IDs để có thể sử dụng API endpoints
                    for chapter in chapters.values():
                        for lesson in chapter.get("lessons", []):
                            lesson["api_endpoints"] = {
                                "get_content": f"/api/v1/pdf/textbook/{book_id}/lesson/{lesson.get('lesson_id')}",
                                "search": f"/api/v1/pdf/textbook/{book_id}/search?query=<your_query>",
                            }

                    structure = {
                        "metadata": {
                            "book_id": book_id,
                            "total_chunks": metadata.get("total_chunks", 0)
                            if metadata
                            else 0,
                            "processed_at": metadata.get("processed_at")
                            if metadata
                            else None,
                            "model": metadata.get("model") if metadata else None,
                            "source": "qdrant",
                        },
                        "chapters": list(chapters.values()),
                        "api_usage": {
                            "book_structure": f"/api/v1/pdf/textbook/{book_id}/structure",
                            "search_book": f"/api/v1/pdf/textbook/{book_id}/search?query=<your_query>",
                            "global_search": "/api/v1/pdf/search?query=<your_query>",
                        },
                    }

                    return {"success": True, "book_id": book_id, "structure": structure}

        except Exception as e:
            logger.warning(f"Error checking Qdrant for book {book_id}: {e}")

        # Không tìm thấy ở đâu cả
        raise HTTPException(
            status_code=404,
            detail=f"Textbook with ID '{book_id}' not found in file system or Qdrant",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting textbook structure: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbook/{book_id}/lesson/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson_content(book_id: str, lesson_id: str) -> Dict[str, Any]:
    """
    Lấy nội dung chi tiết của một bài học từ Qdrant hoặc file system

    Args:
        book_id: ID của sách
        lesson_id: ID của bài học

    Returns:
        Dict containing lesson content
    """
    try:
        # Thử lấy từ file system trước (legacy data)
        lesson_content = await textbook_parser_service.get_lesson_content(
            book_id, lesson_id
        )

        if lesson_content:
            return {
                "success": True,
                "book_id": book_id,
                "lesson_id": lesson_id,
                "content": lesson_content,
                "source": "file_system",
            }

        # Nếu không tìm thấy trong file system, thử lấy từ Qdrant
        from app.services.qdrant_service import qdrant_service
        from qdrant_client import models as qdrant_models

        collection_name = f"textbook_{book_id}"

        try:
            if not qdrant_service.qdrant_client:
                raise Exception("Qdrant client not available")

            # Tìm tất cả chunks của lesson này
            search_result = qdrant_service.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="lesson_id",
                            match=qdrant_models.MatchValue(value=lesson_id),
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
            )

            if not search_result[0]:  # Không có points
                raise HTTPException(
                    status_code=404,
                    detail=f"Lesson '{lesson_id}' not found in textbook '{book_id}'",
                )

            # Tổng hợp nội dung từ các chunks
            lesson_chunks = []
            lesson_info = {}

            for point in search_result[0]:
                payload = point.payload or {}

                # Lấy thông tin lesson từ chunk đầu tiên
                if not lesson_info:
                    lesson_info = {
                        "lesson_id": lesson_id,
                        "lesson_title": payload.get("lesson_title", "Unknown"),
                        "chapter_id": payload.get("chapter_id", "Unknown"),
                        "chapter_title": payload.get("chapter_title", "Unknown"),
                        "book_title": payload.get("book_title", "Unknown"),
                        "subject": payload.get("subject", "Unknown"),
                        "grade": payload.get("grade", "Unknown"),
                        # Lấy thông tin pages và images từ payload
                        "lesson_start_page": payload.get("lesson_start_page"),
                        "lesson_end_page": payload.get("lesson_end_page"),
                        "lesson_images": payload.get("lesson_images", []),
                        "lesson_pages": payload.get("lesson_pages", []),
                        "lesson_total_pages": payload.get("lesson_total_pages", 0),
                        "lesson_has_images": payload.get("lesson_has_images", False),
                    }

                lesson_chunks.append(
                    {
                        "type": payload.get("type", "content"),
                        "text": payload.get("text", ""),
                        "chunk_id": str(point.id),
                    }
                )

            # Sắp xếp chunks theo type (title trước, content sau)
            lesson_chunks.sort(key=lambda x: 0 if x["type"] == "title" else 1)

            # Tạo content text từ các chunks
            content_text_parts = []
            for chunk in lesson_chunks:
                if chunk.get("type") == "title":
                    content_text_parts.append(f"\n=== {chunk.get('text', '')} ===\n")
                else:
                    content_text_parts.append(chunk.get("text", ""))

            content_text = "\n\n".join(content_text_parts).strip()

            # Tạo cấu trúc giống /process-textbook
            book_structure = {
                "book_info": {
                    "title": lesson_info.get("book_title", "Unknown"),
                    "subject": lesson_info.get("subject", "Unknown"),
                    "grade": lesson_info.get("grade", "Unknown"),
                    "book_id": book_id,
                    "total_pages": "Unknown",  # Không có thông tin pages từ Qdrant
                },
                "chapters": [
                    {
                        "chapter_id": lesson_info.get("chapter_id"),
                        "chapter_title": lesson_info.get("chapter_title"),
                        "lessons": [
                            {
                                "lesson_id": lesson_id,
                                "lesson_title": lesson_info.get("lesson_title"),
                                "start_page": lesson_info.get("lesson_start_page"),
                                "end_page": lesson_info.get("lesson_end_page"),
                                "content": {
                                    "text": content_text,
                                    "images": lesson_info.get("lesson_images", []),
                                    "pages": lesson_info.get("lesson_pages", []),
                                    "total_pages": lesson_info.get(
                                        "lesson_total_pages", 0
                                    ),
                                    "has_images": lesson_info.get(
                                        "lesson_has_images", False
                                    ),
                                    "chunks_info": {
                                        "total_chunks": len(lesson_chunks),
                                        "chunk_types": list(
                                            set(
                                                chunk.get("type", "content")
                                                for chunk in lesson_chunks
                                            )
                                        ),
                                    },
                                },
                            }
                        ],
                    }
                ],
            }

            return {
                "success": True,
                "book_id": book_id,
                "lesson_id": lesson_id,
                "book_structure": book_structure,
                "source": "qdrant",
                "statistics": {
                    "total_chapters": 1,
                    "total_lessons": 1,
                    "total_chunks": len(lesson_chunks),
                },
                "navigation": {
                    "book_structure": f"/api/v1/pdf/textbook/{book_id}/structure",
                    "search_in_book": f"/api/v1/pdf/textbook/{book_id}/search?query=<your_query>",
                    "global_search": "/api/v1/pdf/search?query=<your_query>",
                },
                "message": f"Lesson '{lesson_id}' retrieved successfully from vector database",
            }

        except Exception as e:
            logger.warning(f"Error getting lesson from Qdrant: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Lesson '{lesson_id}' not found in textbook '{book_id}'",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lesson content: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbook/{book_id}/search", response_model=Dict[str, Any])
async def search_textbook(
    book_id: str, query: str, limit: int = Query(5, ge=1, le=20)
) -> Dict[str, Any]:
    """
    Tìm kiếm trong sách giáo khoa bằng RAG với Qdrant

    Args:
        book_id: ID của sách
        query: Câu truy vấn tìm kiếm
        limit: Số lượng kết quả tối đa

    Returns:
        Dict chứa kết quả tìm kiếm
    """
    try:
        from app.services.qdrant_service import qdrant_service

        # Tìm kiếm trực tiếp với Qdrant (không cần kiểm tra file system)
        search_result = await qdrant_service.search_textbook(
            book_id=book_id, query=query, limit=limit
        )

        if not search_result.get("success", False):
            # Nếu search failed, có thể do collection không tồn tại
            error_msg = search_result.get("error", "Unknown error")
            if "Collection not found" in error_msg:
                raise HTTPException(
                    status_code=404,
                    detail=f"Textbook with ID '{book_id}' not found. Please process the textbook first.",
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Search failed: {error_msg}",
                )

        return {
            "success": True,
            "book_id": book_id,
            "query": query,
            "results": search_result.get("results", []),
            "message": f"Found {len(search_result.get('results', []))} results",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching textbook: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/search", response_model=Dict[str, Any])
async def search_all_textbooks(
    query: str, limit: int = Query(10, ge=1, le=50)
) -> Dict[str, Any]:
    """
    Tìm kiếm trong TẤT CẢ sách giáo khoa (Global Search)

    Endpoint này cho phép người dùng tìm kiếm mà không cần biết book_id cụ thể.
    Hệ thống sẽ tìm kiếm trong tất cả sách đã được xử lý.

    Args:
        query: Câu truy vấn tìm kiếm (từ khóa, câu hỏi, chủ đề)
        limit: Số lượng kết quả tối đa (1-50, mặc định 10)

    Returns:
        Dict chứa kết quả tìm kiếm từ tất cả sách

    Examples:
        - /api/v1/pdf/search?query=hóa học là gì
        - /api/v1/pdf/search?query=định nghĩa nguyên tử&limit=5
        - /api/v1/pdf/search?query=bài tập về liên kết hóa học
    """
    try:
        from app.services.qdrant_service import qdrant_service

        logger.info(f"Global search query: '{query}' with limit: {limit}")

        # Tìm kiếm trong tất cả textbooks
        search_result = await qdrant_service.search_all_textbooks(
            query=query, limit=limit
        )

        if not search_result.get("success", False):
            error_msg = search_result.get("error", "Unknown error")
            if "No textbooks found" in error_msg:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "total_collections_searched": 0,
                    "message": "No textbooks have been processed yet. Please upload and process textbooks first.",
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Search failed: {error_msg}",
                )

        return {
            "success": True,
            "query": query,
            "results": search_result.get("results", []),
            "total_collections_searched": search_result.get(
                "total_collections_searched", 0
            ),
            "collections_searched": search_result.get("collections_searched", []),
            "message": search_result.get("message", "Search completed"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in global search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        from app.services.simple_ocr_service import simple_ocr_service

        # Check OCR service availability
        supported_langs = simple_ocr_service.get_supported_languages()

        # Check LLM service availability
        llm_available = llm_service.is_available()

        return {
            "status": "healthy",
            "services": {
                "pdf_ocr": "available",
                "llm_analysis": "available" if llm_available else "unavailable",
                "textbook_processing": "available",
                "async_processing": "available",
                "vector_search": "available",
            },
            "supported_languages": supported_langs,
            "llm_status": "Gemini API configured"
            if llm_available
            else "Gemini API not configured",
            "available_endpoints": [
                "/process-textbook-async",
                "/process-textbook",
                "/quick-textbook-analysis",
                "/textbooks",  # NEW: List all textbooks
                "/textbook/{book_id}/structure",
                "/textbook/{book_id}/lesson/{lesson_id}",
                "/textbook/{book_id}/search",
                "/search",
                "/health",
            ],
            "usage_flow": {
                "1": "Upload PDF: POST /process-textbook-async",
                "2": "List textbooks: GET /textbooks",
                "3": "Get structure: GET /textbook/{book_id}/structure",
                "4": "Get lesson: GET /textbook/{book_id}/lesson/{lesson_id}",
                "5": "Search: GET /search?query=your_query",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
