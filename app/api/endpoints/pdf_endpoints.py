"""
PDF Endpoints - Endpoint đơn giản để xử lý PDF với OCR và LLM formatting
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import Dict, Any, Optional

from app.services.llm_service import get_llm_service
from app.services.semantic_analysis_service import get_semantic_analysis_service
from app.services.background_task_processor import get_background_task_processor

logger = logging.getLogger(__name__)


router = APIRouter()



@router.post("/import", response_model=Dict[str, Any])
async def quick_textbook_analysis(
    file: UploadFile = File(...),
    create_embeddings: bool = Form(True),
    lesson_id: Optional[str] = Form(None),
    isImportGuide: bool = Form(False),
) -> Dict[str, Any]:
    """
    Phân tích nhanh cấu trúc sách giáo khoa hoặc import hướng dẫn với xử lý bất đồng bộ

    Upload PDF/DOCX và nhận task_id ngay lập tức. Hệ thống sẽ:

    Với PDF (sách giáo khoa):
    1. Phân tích cấu trúc sách (chapters, lessons)
    2. Tự động trích xuất metadata
    3. Tạo embeddings và lưu vào Qdrant (nếu được yêu cầu)
    4. Trả về kết quả với định dạng giống /process-textbook

    Với DOCX (hướng dẫn):
    1. Trích xuất nội dung text từ DOCX
    2. Tạo embeddings cho nội dung hướng dẫn
    3. Lưu vào Qdrant collection riêng cho guides
    4. Hỗ trợ tìm kiếm và RAG cho hướng dẫn

    Args:
        file: PDF file của sách giáo khoa hoặc DOCX file của hướng dẫn
        create_embeddings: Có tạo embeddings cho RAG search không
        lesson_id: ID bài học tùy chọn để liên kết với lesson cụ thể
        isImportGuide: True nếu import file DOCX làm hướng dẫn, False cho PDF sách giáo khoa

    Returns:
        Dict chứa task_id để theo dõi tiến độ
    """
    try:
        # Validate file type based on import mode
        if isImportGuide:
            # Validate DOCX file for guide import
            if not file.filename or not file.filename.lower().endswith(".docx"):
                raise HTTPException(status_code=400, detail="Guide import only supports DOCX files")
        else:
            # Validate PDF file for textbook import
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Textbook import only supports PDF files")

        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(
            f"Starting {'guide import' if isImportGuide else 'textbook analysis'} task for: {file.filename} ({len(file_content)} bytes)"
        )

        if isImportGuide:
            # Create guide import task
            task_id = await get_background_task_processor().create_guide_import_task(
                docx_content=file_content,
                filename=file.filename,
                create_embeddings=create_embeddings,
            )

            return {
                "success": True,
                "task_id": task_id,
                "filename": file.filename,
                "status": "processing",
                "import_type": "guide",
                "message": "Guide import task created successfully. Use /api/v1/tasks/{task_id}/status to check progress.",
                "endpoints": {
                    "check_status": f"/api/v1/tasks/{task_id}/status",
                    "get_result": f"/api/v1/tasks/{task_id}/result",
                },
            }
        else:
            # Create textbook analysis task
            task_id = await get_background_task_processor().create_quick_analysis_task(
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
                "import_type": "textbook",
                "message": "Quick textbook analysis task created successfully. Use /api/v1/tasks/{task_id}/status to check progress.",
                "endpoints": {
                    "check_status": f"/api/v1/tasks/{task_id}/status",
                    "get_result": f"/api/v1/tasks/{task_id}/result",
                },
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating {'guide import' if isImportGuide else 'textbook analysis'} task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbooks", response_model=Dict[str, Any])
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
        from app.services.qdrant_service import get_qdrant_service
        qdrant_service = get_qdrant_service()
        qdrant_service._ensure_service_initialized()
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
        from app.services.qdrant_service import get_qdrant_service
        qdrant_service = get_qdrant_service()
        qdrant_service._ensure_service_initialized()

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
        llm_service = get_llm_service()
        llm_available = llm_service.is_available()

        # Check Qdrant and embedding model
        from app.services.qdrant_service import get_qdrant_service
        qdrant_service = get_qdrant_service()
        embedding_available = qdrant_service.embedding_model is not None
        qdrant_available = qdrant_service.qdrant_client is not None

        embedding_model_name = "Unknown"
        vector_dimension = "Unknown"
        if embedding_available:
            from app.core.config import settings
            embedding_model_name = settings.EMBEDDING_MODEL
            vector_dimension = qdrant_service.vector_size

        return {
            "status": "healthy",
            "services": {
                "pdf_ocr": "available",
                "llm_analysis": "available" if llm_available else "unavailable",
                "textbook_processing": "available",
                "async_processing": "available",
                "vector_search": "available" if (embedding_available and qdrant_available) else "unavailable",
                "semantic_analysis": "available" if llm_available else "fallback_mode",
            },
            "supported_languages": supported_langs,
            "llm_status": "Gemini API configured" if llm_available else "Gemini API not configured",
            "embedding_model": embedding_model_name,
            "vector_dimension": vector_dimension,
            "qdrant_status": "connected" if qdrant_available else "disconnected",
            "available_endpoints": [
                "/import",  # Quick textbook analysis & Guide import
                "/textbooks",  # Get all textbooks
                "/textbook/{lesson_id}",  # Get textbook by lesson ID
                "/search",  # Global content search
                "/search-semantic",  # Semantic search with filters
                "/rag-query",  # RAG endpoint with LLM
                "/test-semantic-analysis",  # Test semantic analysis
                "/textbook",  # DELETE: Flexible textbook deletion
                "/health",
            ],
            "usage_flow": {
                "1a": "Upload PDF (Textbook): POST /import (with isImportGuide=false)",
                "1b": "Upload DOCX (Guide): POST /import (with isImportGuide=true)",
                "2": "List textbooks: GET /textbooks",
                "3": "Get textbook by lesson: GET /textbook/{lesson_id}",
                "4": "Global search: GET /search?query=your_query",
                "5": "Semantic search: GET /search-semantic?query=your_query&semantic_tags=definition,example",
                "6": "RAG Query (Recommended): POST /rag-query?query=your_question&lesson_id=lesson123",  # Best for Q&A
                "7": "Delete textbook: DELETE /textbook?textbook_id=your_id OR DELETE /textbook?lesson_id=your_lesson_id"
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@router.get("/rag-query", response_model=Dict[str, Any])
async def rag_query(
    query: str = Query(..., description="Câu hỏi của người dùng"),
    book_id: Optional[str] = Query(None, description="ID sách cụ thể (tùy chọn)"),
    lesson_id: Optional[str] = Query(None, description="ID bài học cụ thể (tùy chọn)"),
    limit: int = Query(5, description="Số lượng kết quả tìm kiếm tối đa"),
    semantic_tags: Optional[str] = Query(None, description="Lọc theo semantic tags"),
    temperature: float = Query(0.3, description="Temperature cho LLM response"),
    max_tokens: int = Query(2000, description="Số token tối đa cho response")
) -> Dict[str, Any]:
    """
    RAG endpoint kết hợp semantic search và LLM để trả lời câu hỏi người dùng

    Workflow:
    1. Nhận câu hỏi từ người dùng
    2. Sử dụng semantic search để tìm nội dung liên quan (có thể filter theo book_id hoặc lesson_id)
    3. Gửi context + câu hỏi cho LLM để tạo câu trả lời
    4. Làm sạch text và trả về câu trả lời kèm sources

    Examples:
        POST /api/v1/pdf/rag-query?query=Nguyên tử là gì?
        POST /api/v1/pdf/rag-query?query=Nguyên tử là gì?&lesson_id=lesson123
        POST /api/v1/pdf/rag-query?query=Nguyên tử là gì?&book_id=book456&semantic_tags=definition
    """
    try:
        # Sử dụng RAG service để xử lý toàn bộ workflow
        from app.services.rag_service import rag_service

        result = await rag_service.process_rag_query(
            query=query,
            book_id=book_id,
            lesson_id=lesson_id,
            limit=limit,
            semantic_tags=semantic_tags,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.get("/search-semantic", response_model=Dict[str, Any])
async def search_with_semantic_filters(
    query: str = Query(..., description="Câu truy vấn tìm kiếm"),
    book_id: Optional[str] = Query(None, description="ID sách (nếu không có sẽ tìm trong tất cả sách)"),
    semantic_tags: Optional[str] = Query(None, description="Các semantic tags cần filter, phân cách bằng dấu phẩy (VD: definition,example)"),
    difficulty: Optional[str] = Query(None, description="Mức độ khó: basic, intermediate, advanced"),
    has_examples: Optional[bool] = Query(None, description="Lọc nội dung có ví dụ"),
    has_formulas: Optional[bool] = Query(None, description="Lọc nội dung có công thức"),
    min_confidence: Optional[float] = Query(0.0, ge=0.0, le=1.0, description="Confidence tối thiểu cho semantic tags"),
    limit: int = Query(10, ge=1, le=50, description="Số lượng kết quả tối đa")
) -> Dict[str, Any]:
    """
    Tìm kiếm với semantic filters nâng cao

    Args:
        query: Câu truy vấn tìm kiếm
        book_id: ID sách cụ thể (optional)
        semantic_tags: Danh sách semantic tags để filter
        difficulty: Mức độ khó
        has_examples: Có ví dụ hay không
        has_formulas: Có công thức hay không
        min_confidence: Confidence tối thiểu
        limit: Số lượng kết quả

    Returns:
        Dict chứa kết quả tìm kiếm với semantic metadata

    Examples:
        - /api/v1/pdf/search-semantic?query=nguyên tử&semantic_tags=definition,theory&difficulty=basic
        - /api/v1/pdf/search-semantic?query=bài tập&has_examples=true&min_confidence=0.7
    """
    try:
        # Sử dụng RAG service để xử lý semantic search
        from app.services.rag_service import rag_service

        result = await rag_service.search_with_semantic_filters(
            query=query,
            book_id=book_id,
            semantic_tags=semantic_tags,
            difficulty=difficulty,
            has_examples=has_examples,
            has_formulas=has_formulas,
            min_confidence=min_confidence,
            limit=limit
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(e)}"
        )


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
