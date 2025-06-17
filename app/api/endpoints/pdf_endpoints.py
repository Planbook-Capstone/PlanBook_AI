"""
PDF Endpoints - Endpoint đơn giản để xử lý PDF với OCR và LLM formatting
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import Dict, Any, Optional

from app.services.llm_service import llm_service
from app.services.enhanced_textbook_service import enhanced_textbook_service
from app.services.textbook_parser_service import textbook_parser_service
from app.services.background_task_processor import background_task_processor

logger = logging.getLogger(__name__)


router = APIRouter()


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
                )  # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Processing textbook: {file.filename} ({len(file_content)} bytes)")

        # Process textbook with enhanced service
        enhanced_result = await enhanced_textbook_service.process_textbook_to_structure(
            pdf_content=file_content,
            filename=file.filename,
            book_metadata=book_metadata,
        )

        if not enhanced_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Textbook processing failed: {enhanced_result.get('error', 'Unknown error')}",
            )

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
    lesson_id: Optional[str] = Form(None),
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
        lesson_id: ID bài học tùy chọn để liên kết với lesson cụ thể

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
        )  # Tạo task bất đồng bộ
        task_id = await background_task_processor.create_quick_analysis_task(
            pdf_content=file_content,
            filename=file.filename,
            create_embeddings=create_embeddings,
            lesson_id=lesson_id,
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


@router.get("/getAllTextBook", response_model=Dict[str, Any])
async def get_all_textbook() -> Dict[str, Any]:
    """
    Lấy danh sách tất cả sách giáo khoa với format đầy đủ như /process-textbook

    Endpoint này trả về tất cả textbooks đã được xử lý với format chuẩn:
    - book_structure với book_info, chapters, lessons
    - content đầy đủ với text, images, pages
    - statistics và processing_info
    - embeddings_info

    Returns:
        Dict chứa danh sách tất cả textbooks với format đầy đủ

    Example:
        GET /api/v1/pdf/getAllTextBook
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
                                    "filename": f"{book_id}.pdf",
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
                                        "vector_dimension": 384,
                                    },
                                    "message": "Textbook retrieved successfully from Qdrant vector database",
                                }
                                textbooks.append(textbook_data)
                            else:
                                # Fallback: Tạo structure từ chunks nếu không có original_structure
                                logger.info(
                                    f"Creating structure from chunks for {book_id}"
                                )

                                # Lấy tất cả chunks để tái tạo structure
                                all_chunks = qdrant_service.qdrant_client.scroll(
                                    collection_name=collection.name,
                                    limit=1000,
                                    with_payload=True,
                                )

                                if all_chunks[0]:
                                    chapters = {}
                                    book_info = {
                                        "title": payload.get("book_title", "Unknown"),
                                        "subject": payload.get(
                                            "book_subject", "Unknown"
                                        ),
                                        "grade": payload.get("book_grade", "Unknown"),
                                        "total_pages": payload.get(
                                            "book_total_pages", 0
                                        ),
                                    }

                                    # Tái tạo structure từ chunks
                                    for point in all_chunks[0]:
                                        chunk_payload = point.payload or {}
                                        if chunk_payload.get("type") in [
                                            "title",
                                            "content",
                                        ]:
                                            chapter_id = chunk_payload.get("chapter_id")
                                            lesson_id = chunk_payload.get("lesson_id")

                                            if (
                                                chapter_id
                                                and chapter_id not in chapters
                                            ):
                                                chapters[chapter_id] = {
                                                    "chapter_id": chapter_id,
                                                    "chapter_title": chunk_payload.get(
                                                        "chapter_title", "Unknown"
                                                    ),
                                                    "start_page": chunk_payload.get(
                                                        "chapter_start_page"
                                                    ),
                                                    "end_page": chunk_payload.get(
                                                        "chapter_end_page"
                                                    ),
                                                    "lessons": [],
                                                }

                                            if lesson_id:
                                                # Kiểm tra lesson đã tồn tại chưa
                                                existing_lesson = next(
                                                    (
                                                        l
                                                        for l in chapters[chapter_id][
                                                            "lessons"
                                                        ]
                                                        if l["lesson_id"] == lesson_id
                                                    ),
                                                    None,
                                                )

                                                if not existing_lesson:
                                                    chapters[chapter_id][
                                                        "lessons"
                                                    ].append(
                                                        {
                                                            "lesson_id": lesson_id,
                                                            "lesson_title": chunk_payload.get(
                                                                "lesson_title",
                                                                "Unknown",
                                                            ),
                                                            "start_page": chunk_payload.get(
                                                                "lesson_start_page"
                                                            ),
                                                            "end_page": chunk_payload.get(
                                                                "lesson_end_page"
                                                            ),
                                                            "content": {
                                                                "text": "Content available via lesson endpoint",
                                                                "images": chunk_payload.get(
                                                                    "lesson_images", []
                                                                ),
                                                                "pages": chunk_payload.get(
                                                                    "lesson_pages", []
                                                                ),
                                                                "total_pages": chunk_payload.get(
                                                                    "lesson_total_pages",
                                                                    0,
                                                                ),
                                                                "has_images": chunk_payload.get(
                                                                    "lesson_has_images",
                                                                    False,
                                                                ),
                                                            },
                                                        }
                                                    )

                                    # Tạo book_structure
                                    book_structure = {
                                        "book_info": book_info,
                                        "chapters": list(chapters.values()),
                                    }

                                    textbook_data = {
                                        "success": True,
                                        "book_id": book_id,
                                        "filename": f"{book_id}.pdf",
                                        "book_structure": book_structure,
                                        "statistics": {
                                            "total_pages": book_info.get(
                                                "total_pages", 0
                                            ),
                                            "total_chapters": len(chapters),
                                            "total_lessons": sum(
                                                len(ch["lessons"])
                                                for ch in chapters.values()
                                            ),
                                        },
                                        "processing_info": {
                                            "ocr_applied": True,
                                            "llm_analysis": True,
                                            "processing_method": "reconstructed_from_chunks",
                                            "processed_at": payload.get("processed_at"),
                                        },
                                        "embeddings_created": True,
                                        "embeddings_info": {
                                            "collection_name": collection.name,
                                            "vector_count": getattr(
                                                collection, "vectors_count", 0
                                            ),
                                            "vector_dimension": 384,
                                        },
                                        "message": "Textbook structure reconstructed from vector chunks",
                                    }
                                    textbooks.append(textbook_data)

                    except Exception as e:
                        logger.warning(f"Error processing textbook {book_id}: {e}")
                        # Thêm basic info nếu có lỗi
                        textbooks.append(
                            {
                                "success": False,
                                "book_id": book_id,
                                "error": str(e),
                                "collection_name": collection.name,
                                "status": "error",
                            }
                        )

        return {
            "success": True,
            "total_textbooks": len(textbooks),
            "textbooks": textbooks,
            "message": f"Retrieved {len(textbooks)} textbooks successfully",
        }

    except Exception as e:
        logger.error(f"Error in getAllTextBook: {e}")
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
    try:  # Thử lấy từ file system trước (legacy data)
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


@router.get("/lesson/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson_content_by_id(lesson_id: str) -> Dict[str, Any]:
    """
    Lấy nội dung chi tiết của một bài học chỉ bằng lesson_id

    API này tìm kiếm lesson trong tất cả textbooks và trả về nội dung đầy đủ
    với format giống như /process-textbook.

    Args:
        lesson_id: ID của bài học cần lấy

    Returns:
        Dict chứa nội dung đầy đủ của bài học với format chuẩn

    Example:
        GET /api/v1/pdf/lesson/lesson_01_01
    """
    try:
        from app.services.qdrant_service import qdrant_service
        from qdrant_client import models as qdrant_models

        if not qdrant_service.qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant service not available")

        # Tìm kiếm lesson trong tất cả collections
        collections = qdrant_service.qdrant_client.get_collections().collections
        lesson_found = False
        lesson_data = None
        book_id = None

        for collection in collections:
            if collection.name.startswith("textbook_"):
                try:
                    # Tìm lesson trong collection này
                    search_result = qdrant_service.qdrant_client.scroll(
                        collection_name=collection.name,
                        scroll_filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="lesson_id",
                                    match=qdrant_models.MatchValue(value=lesson_id),
                                )
                            ]
                        ),
                        limit=100,  # Lấy nhiều chunks của lesson
                        with_payload=True,
                    )

                    if search_result[0]:  # Tìm thấy lesson
                        lesson_found = True
                        book_id = collection.name.replace("textbook_", "")

                        # Tổng hợp nội dung từ các chunks
                        lesson_chunks = []
                        lesson_info = {}

                        for point in search_result[0]:
                            payload = point.payload or {}

                            # Lấy thông tin lesson từ chunk đầu tiên
                            if not lesson_info:
                                lesson_info = {
                                    "lesson_id": lesson_id,
                                    "lesson_title": payload.get(
                                        "lesson_title", "Unknown"
                                    ),
                                    "chapter_id": payload.get("chapter_id", "Unknown"),
                                    "chapter_title": payload.get(
                                        "chapter_title", "Unknown"
                                    ),
                                    "book_title": payload.get("book_title", "Unknown"),
                                    "subject": payload.get("subject", "Unknown"),
                                    "grade": payload.get("grade", "Unknown"),
                                    # Lấy thông tin pages và images từ payload
                                    "lesson_start_page": payload.get(
                                        "lesson_start_page"
                                    ),
                                    "lesson_end_page": payload.get("lesson_end_page"),
                                    "lesson_images": payload.get("lesson_images", []),
                                    "lesson_pages": payload.get("lesson_pages", []),
                                    "lesson_total_pages": payload.get(
                                        "lesson_total_pages", 0
                                    ),
                                    "lesson_has_images": payload.get(
                                        "lesson_has_images", False
                                    ),
                                }

                            # Chỉ lấy content chunks, bỏ qua title chunks
                            if payload.get("type") == "content":
                                lesson_chunks.append(
                                    {
                                        "text": payload.get("text", ""),
                                        "chunk_index": payload.get("chunk_index", 0),
                                        "type": payload.get("type", "content"),
                                    }
                                )

                        # Sắp xếp chunks theo thứ tự
                        lesson_chunks.sort(key=lambda x: x.get("chunk_index", 0))

                        # Ghép nội dung text
                        content_text = "\n\n".join(
                            [chunk["text"] for chunk in lesson_chunks]
                        )

                        # Tạo response với format đầy đủ như /process-textbook
                        lesson_data = {
                            "success": True,
                            "book_id": book_id,
                            "lesson_id": lesson_id,
                            "book_structure": {
                                "book_info": {
                                    "title": lesson_info.get("book_title", "Unknown"),
                                    "subject": lesson_info.get("subject", "Unknown"),
                                    "grade": lesson_info.get("grade", "Unknown"),
                                    "total_pages": lesson_info.get(
                                        "lesson_total_pages", 0
                                    ),
                                },
                                "chapters": [
                                    {
                                        "chapter_id": lesson_info.get("chapter_id"),
                                        "chapter_title": lesson_info.get(
                                            "chapter_title"
                                        ),
                                        "start_page": lesson_info.get(
                                            "lesson_start_page"
                                        ),
                                        "end_page": lesson_info.get("lesson_end_page"),
                                        "lessons": [
                                            {
                                                "lesson_id": lesson_id,
                                                "lesson_title": lesson_info.get(
                                                    "lesson_title"
                                                ),
                                                "start_page": lesson_info.get(
                                                    "lesson_start_page"
                                                ),
                                                "end_page": lesson_info.get(
                                                    "lesson_end_page"
                                                ),
                                                "content": {
                                                    "text": content_text,
                                                    "images": lesson_info.get(
                                                        "lesson_images", []
                                                    ),
                                                    "pages": lesson_info.get(
                                                        "lesson_pages", []
                                                    ),
                                                    "total_pages": lesson_info.get(
                                                        "lesson_total_pages", 0
                                                    ),
                                                    "has_images": lesson_info.get(
                                                        "lesson_has_images", False
                                                    ),
                                                    "chunks_info": {
                                                        "total_chunks": len(
                                                            lesson_chunks
                                                        ),
                                                        "chunk_types": list(
                                                            set(
                                                                chunk.get(
                                                                    "type", "content"
                                                                )
                                                                for chunk in lesson_chunks
                                                            )
                                                        ),
                                                    },
                                                },
                                            }
                                        ],
                                    }
                                ],
                            },
                            "statistics": {
                                "total_pages": lesson_info.get("lesson_total_pages", 0),
                                "total_chapters": 1,
                                "total_lessons": 1,
                            },
                            "processing_info": {
                                "ocr_applied": True,
                                "llm_analysis": True,
                                "processing_method": "retrieved_from_qdrant_by_lesson_id",
                            },
                            "embeddings_created": True,
                            "embeddings_info": {
                                "collection_name": collection.name,
                                "vector_count": len(search_result[0]),
                                "vector_dimension": 384,
                            },
                            "message": f"Lesson content retrieved successfully from {collection.name}",
                        }
                        break  # Đã tìm thấy, thoát khỏi loop                except Exception as e:
                except Exception as e:
                    logger.warning(f"Error searching lesson in {collection.name}: {e}")
                    continue

        if not lesson_found:
            raise HTTPException(
                status_code=404,
                detail=f"Lesson with ID '{lesson_id}' not found in any textbook",
            )

        if lesson_data is None:
            raise HTTPException(
                status_code=500,
                detail="Lesson data is None - internal error",
            )

        return lesson_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lesson content for {lesson_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbook/{book_id}/lesson/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson_content(book_id: str, lesson_id: str) -> Dict[str, Any]:
    """
    [DEPRECATED] Lấy nội dung chi tiết của một bài học từ Qdrant hoặc file system

    ⚠️ API này đã deprecated. Sử dụng /lesson/{lesson_id} thay thế.

    API này redirect đến endpoint mới /lesson/{lesson_id} vì không cần book_id nữa.
    Endpoint mới sẽ tự động tìm lesson trong tất cả textbooks.

    Args:
        book_id: ID của sách (không cần thiết nữa)
        lesson_id: ID của bài học

    Returns:
        Dict chứa nội dung chi tiết của bài học

    Example:
        GET /api/v1/pdf/textbook/book_001/lesson/lesson_01_01
        -> Redirect to: GET /api/v1/pdf/lesson/lesson_01_01
    """
    try:
        # Redirect đến endpoint mới
        logger.info(
            f"Redirecting /textbook/{book_id}/lesson/{lesson_id} to /lesson/{lesson_id}"
        )
        return await get_lesson_content_by_id(lesson_id)

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

        logger.info(
            f"Global search query: '{query}' with limit: {limit}"
        )  # Implement global search logic here since the method was removed
        # Get all collections starting with "textbook_"
        try:
            if not qdrant_service.qdrant_client:
                raise HTTPException(
                    status_code=500, detail="Qdrant service not available"
                )

            collections = qdrant_service.qdrant_client.get_collections().collections
            textbook_collections = [
                c.name for c in collections if c.name.startswith("textbook_")
            ]

            if not textbook_collections:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "total_collections_searched": 0,
                    "message": "No textbooks have been processed yet. Please upload and process textbooks first.",
                }

            # Search each textbook collection
            all_results = []
            for collection_name in textbook_collections:
                book_id = collection_name.replace("textbook_", "")
                book_search_result = await qdrant_service.search_textbook(
                    book_id=book_id, query=query, limit=limit
                )

                if book_search_result.get("success") and book_search_result.get(
                    "results"
                ):
                    for result in book_search_result["results"]:
                        result["book_id"] = book_id
                        result["collection_name"] = collection_name
                        all_results.append(result)

            # Sort by score and limit results
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            top_results = all_results[:limit]

            search_result = {
                "success": True,
                "query": query,
                "results": top_results,
                "total_collections_searched": len(textbook_collections),
                "message": f"Found {len(top_results)} results from {len(textbook_collections)} textbooks",
            }

        except Exception as e:
            logger.error(f"Error in global search implementation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}",
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


@router.get("/search-books", response_model=Dict[str, Any])
async def search_books_metadata(
    query: str = Query(..., description="Search query for books"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
) -> Dict[str, Any]:
    """
    Tìm kiếm sách và trả về metadata đơn giản

    Endpoint này tìm kiếm trong tất cả textbooks và trả về thông tin metadata
    cơ bản của sách thay vì nội dung chi tiết.

    Args:
        query: Từ khóa tìm kiếm (title, author, subject, etc.)
        limit: Số lượng kết quả tối đa (1-50, mặc định 10)

    Returns:
        Dict chứa danh sách sách với metadata đơn giản

    Example:
        GET /api/v1/pdf/search-books?query=hóa học&limit=5
    """
    try:
        from app.services.qdrant_service import qdrant_service
        from qdrant_client import models as qdrant_models

        logger.info(f"Searching books with query: '{query}' limit: {limit}")

        if not qdrant_service.qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant service not available")

        # Lấy danh sách tất cả collections
        collections = qdrant_service.qdrant_client.get_collections().collections
        books_found = []

        for collection in collections:
            if collection.name.startswith("textbook_"):
                book_id = collection.name.replace("textbook_", "")

                try:
                    # Tìm metadata point để lấy thông tin sách
                    search_result = qdrant_service.qdrant_client.scroll(
                        collection_name=collection.name,
                        scroll_filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="type",
                                    match=qdrant_models.MatchValue(value="metadata"),
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
                        book_info = {}

                        if original_structure:
                            book_info = original_structure.get("book_info", {})
                        else:
                            # Fallback từ payload
                            book_info = {
                                "title": payload.get("book_title", "Unknown"),
                                "subject": payload.get("book_subject", "Unknown"),
                                "grade": payload.get("book_grade", "Unknown"),
                                "total_pages": payload.get("book_total_pages", 0),
                            }

                        # Tạo metadata theo format yêu cầu
                        book_metadata = {
                            "id": book_id,
                            "title": book_info.get("title", "Unknown"),
                            "author": book_info.get("author", "Bộ Giáo dục và Đào tạo"),
                            "publisher": book_info.get(
                                "publisher", "Nhà xuất bản Giáo dục Việt Nam"
                            ),
                            "published_year": book_info.get("published_year", 2024),
                            "isbn": book_info.get(
                                "isbn",
                                f"978-604-{book_id[:3]}-{book_id[3:6]}-{book_id[6:]}",
                            ),
                            "language": book_info.get("language", "vi"),
                            "categories": [
                                book_info.get("subject", "Giáo dục"),
                                f"Lớp {book_info.get('grade', 'Chưa xác định')}",
                            ],
                            "description": f"Sách giáo khoa {book_info.get('subject', 'Unknown')} - {book_info.get('title', 'Unknown')}",
                            "pages": book_info.get("total_pages", 0),
                            "format": "PDF",
                        }

                        # Kiểm tra xem sách có match với query không
                        searchable_text = " ".join(
                            [
                                book_metadata["title"].lower(),
                                book_metadata["author"].lower(),
                                book_metadata["publisher"].lower(),
                                " ".join(book_metadata["categories"]).lower(),
                                book_metadata["description"].lower(),
                            ]
                        )

                        if query.lower() in searchable_text:
                            books_found.append(book_metadata)

                except Exception as e:
                    logger.warning(
                        f"Error processing collection {collection.name}: {e}"
                    )
                    continue

        # Sắp xếp theo relevance (title match trước)
        def relevance_score(book):
            score = 0
            query_lower = query.lower()
            if query_lower in book["title"].lower():
                score += 10
            if query_lower in book["description"].lower():
                score += 5
            if query_lower in " ".join(book["categories"]).lower():
                score += 3
            if query_lower in book["author"].lower():
                score += 2
            return score

        books_found.sort(key=relevance_score, reverse=True)

        # Áp dụng limit
        books_found = books_found[:limit]

        return {
            "success": True,
            "query": query,
            "total_results": len(books_found),
            "books": books_found,
            "message": f"Found {len(books_found)} books matching '{query}'",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search-books: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/search-textbooks-simple", response_model=Dict[str, Any])
async def search_textbooks_simple(
    query: str = Query(..., description="Search query for textbooks"),
    limit: int = Query(10, ge=1, le=20, description="Maximum number of results"),
) -> Dict[str, Any]:
    """
    Tìm kiếm sách giáo khoa đơn giản và trả về format đầy đủ như /process-textbook

    Endpoint này sử dụng getAllTextBook rồi filter theo query để đảm bảo
    trả về đúng format với content đầy đủ.

    Args:
        query: Từ khóa tìm kiếm (title, subject, content, etc.)
        limit: Số lượng kết quả tối đa (1-20, mặc định 10)

    Returns:
        Dict chứa danh sách textbooks với format đầy đủ như /process-textbook

    Example:
        GET /api/v1/pdf/search-textbooks-simple?query=hóa học&limit=5
    """
    try:
        logger.info(f"Simple search textbooks with query: '{query}' limit: {limit}")

        # Lấy tất cả textbooks từ getAllTextBook
        all_textbooks_response = await get_all_textbook()

        if not all_textbooks_response.get("success"):
            raise HTTPException(status_code=500, detail="Failed to get textbooks")

        all_textbooks = all_textbooks_response.get("textbooks", [])
        matched_textbooks = []

        for textbook in all_textbooks:
            # Tạo searchable text từ textbook data
            book_structure = textbook.get("book_structure", {})
            book_info = book_structure.get("book_info", {})
            chapters = book_structure.get("chapters", [])

            searchable_texts = [
                str(textbook.get("book_id", "")).lower(),
                str(book_info.get("title", "")).lower(),
                str(book_info.get("subject", "")).lower(),
                str(book_info.get("grade", "")).lower(),
            ]

            # Thêm text từ chapters và lessons
            for chapter in chapters:
                searchable_texts.append(str(chapter.get("chapter_title", "")).lower())
                for lesson in chapter.get("lessons", []):
                    searchable_texts.append(str(lesson.get("lesson_title", "")).lower())
                    content = lesson.get("content", {})
                    if isinstance(content, dict):
                        lesson_text = str(content.get("text", ""))
                        # Lấy một phần text để search (không quá dài)
                        searchable_texts.append(lesson_text[:500].lower())

            searchable_text = " ".join(searchable_texts)

            # Kiểm tra match
            if query.lower() in searchable_text:
                # Tạo response với format đầy đủ như /process-textbook
                textbook_data = {
                    "success": True,
                    "book_id": textbook.get("book_id"),
                    "filename": f"{textbook.get('book_id')}.pdf",
                    "book_structure": book_structure,  # Giữ nguyên structure từ getAllTextBook
                    "statistics": {
                        "total_pages": book_info.get("total_pages", 0),
                        "total_chapters": len(chapters),
                        "total_lessons": sum(
                            len(ch.get("lessons", [])) for ch in chapters
                        ),
                    },
                    "processing_info": {
                        "ocr_applied": True,
                        "llm_analysis": True,
                        "processing_method": "simple_search_from_getAllTextBook",
                        "processed_at": textbook.get("processed_at"),
                    },
                    "embeddings_created": True,
                    "embeddings_info": {
                        "collection_name": f"textbook_{textbook.get('book_id')}",
                        "vector_count": 0,  # Will be updated if needed
                        "vector_dimension": 384,
                    },
                    "message": f"Textbook found matching '{query}' with full content structure",
                }
                matched_textbooks.append(textbook_data)

        # Sắp xếp theo relevance (title match trước)
        def relevance_score(textbook):
            score = 0
            query_lower = query.lower()
            book_info = textbook.get("book_structure", {}).get("book_info", {})

            if query_lower in str(textbook.get("book_id", "")).lower():
                score += 15
            if query_lower in str(book_info.get("title", "")).lower():
                score += 10
            if query_lower in str(book_info.get("subject", "")).lower():
                score += 8
            if query_lower in str(book_info.get("grade", "")).lower():
                score += 5

            return score

        matched_textbooks.sort(key=relevance_score, reverse=True)

        # Áp dụng limit
        matched_textbooks = matched_textbooks[:limit]

        return {
            "success": True,
            "query": query,
            "total_results": len(matched_textbooks),
            "textbooks": matched_textbooks,
            "message": f"Found {len(matched_textbooks)} textbooks matching '{query}' with full structure",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search-textbooks-simple: {e}")
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
                "/getAllTextBook",  # Enhanced textbook list
                "/textbook/{book_id}/structure",
                "/lesson/{lesson_id}",  # Get lesson by ID only
                "/textbook/{book_id}/lesson/{lesson_id}",  # DEPRECATED: Redirects to /lesson/{lesson_id}
                "/textbook/{book_id}/search",
                "/search",  # Content search
                "/search-books",  # Book metadata search
                "/search-textbooks-simple",  # Full textbook structure search (Simple)
                "/health",
            ],
            "usage_flow": {
                "1": "Upload PDF: POST /quick-textbook-analysis",
                "2": "List textbooks: GET /getAllTextBook",
                "3": "Search textbooks: GET /search-textbooks-simple?query=your_query",  # RECOMMENDED: Full structure
                "4": "Search books (metadata): GET /search-books?query=your_query",  # Metadata only
                "5": "Get structure: GET /textbook/{book_id}/structure",
                "6": "Get lesson: GET /lesson/{lesson_id}",  # No book_id needed
                "7": "Search content: GET /search?query=your_query",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
