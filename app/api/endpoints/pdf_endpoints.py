"""
PDF Endpoints - Endpoint đơn giản để xử lý PDF với OCR và LLM formatting
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import Dict, Any, Optional, Optional

from app.services.llm_service import llm_service
from app.services.enhanced_textbook_service import enhanced_textbook_service
from app.services.background_task_processor import background_task_processor

logger = logging.getLogger(__name__)


router = APIRouter()


# @router.post("/process-textbook", response_model=Dict[str, Any])
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


@router.post("/import", response_model=Dict[str, Any])
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


# @router.get("/search-books", response_model=Dict[str, Any])
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
            else "Gemini API not configured",            "available_endpoints": [
                "/process-textbook-async",
                "/process-textbook",
                "/quick-textbook-analysis",
                "/getAllTextBook",  # Enhanced textbook list
                "/textbook/{lesson_id}",  # NEW: Get textbook by lesson ID
                "/textbook/{book_id}/structure",
                "/lesson/{lesson_id}",  # Get lesson by ID only
                "/textbook/{book_id}/lesson/{lesson_id}",  # DEPRECATED: Redirects to /lesson/{lesson_id}
                "/textbook/{book_id}/search",
                "/search",  # Content search
                "/search-books",  # Book metadata search
                "/search-textbooks-simple",  # Full textbook structure search (Simple)
                "/textbook",  # DELETE: Flexible textbook deletion
                "/health",
            ],            "usage_flow": {
                "1": "Upload PDF: POST /quick-textbook-analysis",
                "2": "List textbooks: GET /getAllTextBook",
                "3": "Get textbook by lesson: GET /textbook/{lesson_id}",  # NEW
                "4": "Search textbooks: GET /search-textbooks-simple?query=your_query",  # RECOMMENDED: Full structure
                "5": "Search books (metadata): GET /search-books?query=your_query",  # Metadata only
                "6": "Get structure: GET /textbook/{book_id}/structure",
                "7": "Get lesson: GET /lesson/{lesson_id}",  # No book_id needed
                "8": "Search content: GET /search?query=your_query",
                "9": "Delete textbook: DELETE /textbook?textbook_id=ID or DELETE /textbook?lesson_id=ID",  # NEW
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@router.get("/textbook/{lesson_id}", response_model=Dict[str, Any])
async def get_textbook_by_lesson_id(lesson_id: str) -> Dict[str, Any]:
    """
    Lấy nội dung lesson theo lesson_id

    Args:
        lesson_id: ID của lesson cần lấy

    Returns:
        Dict chứa lesson_content và book_id
    """
    from app.services.textbook_retrieval_service import textbook_retrieval_service

    result = await textbook_retrieval_service.get_lesson_content(lesson_id)

    return {
        "book_id": result["book_id"],
        "lesson_content": result["lesson_content"]
    }



@router.delete("/textbook", response_model=Dict[str, Any])
async def delete_textbook_flexible(
    textbook_id: Optional[str] = Query(None, description="Textbook ID to delete"),
    lesson_id: Optional[str] = Query(None, description="Lesson ID to find and delete textbook")
) -> Dict[str, Any]:
    """
    Xóa textbook linh hoạt - nhận vào textbook_id HOẶC lesson_id

    Endpoint này cho phép xóa textbook bằng một trong hai cách:
    1. Cung cấp textbook_id để xóa trực tiếp
    2. Cung cấp lesson_id để tìm và xóa textbook chứa lesson đó

    Args:
        textbook_id: (Optional) ID của textbook cần xóa
        lesson_id: (Optional) ID của lesson để tìm textbook cần xóa

    Returns:
        Dict chứa kết quả xóa

    Examples:
        DELETE /api/v1/pdf/textbook?textbook_id=abc123
        DELETE /api/v1/pdf/textbook?lesson_id=lesson_01_01
    """
    try:
        from app.services.qdrant_service import qdrant_service

        # Validation: phải có ít nhất một trong hai parameters
        if not textbook_id and not lesson_id:
            raise HTTPException(
                status_code=400,
                detail="Either 'textbook_id' or 'lesson_id' parameter is required"
            )

        # Validation: không được cung cấp cả hai parameters
        if textbook_id and lesson_id:
            raise HTTPException(
                status_code=400,
                detail="Cannot provide both 'textbook_id' and 'lesson_id'. Choose one."
            )

        if not qdrant_service.qdrant_client:
            raise HTTPException(
                status_code=503, 
                detail="Qdrant service is not available"
            )

        # Xóa bằng textbook_id
        if textbook_id:
            logger.info(f"Deleting textbook by ID: {textbook_id}")
            result = await qdrant_service.delete_textbook_by_id(textbook_id)
            operation = "delete_by_textbook_id"
            identifier = textbook_id

        # Xóa bằng lesson_id
        else:  # lesson_id
            logger.info(f"Deleting textbook by lesson_id: {lesson_id}")
            result = await qdrant_service.delete_textbook_by_lesson_id(lesson_id)
            operation = "delete_by_lesson_id"
            identifier = lesson_id

        if not result.get("success"):
            if "not found" in result.get("error", "").lower():
                raise HTTPException(
                    status_code=404,
                    detail=result.get("error")
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to delete textbook")
                )

        logger.info(f"Successfully deleted textbook: {identifier}")
        return {
            "success": True,
            "message": result.get("message"),
            "deleted_textbook": {
                "book_id": result.get("book_id"),
                "lesson_id": result.get("lesson_id") if lesson_id else None,
                "collection_name": result.get("collection_name"),
                "deleted_vectors": result.get("deleted_vectors", 0)
            },
            "operation": operation,
            "identifier_used": {
                "textbook_id": textbook_id,
                "lesson_id": lesson_id
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in flexible textbook deletion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

