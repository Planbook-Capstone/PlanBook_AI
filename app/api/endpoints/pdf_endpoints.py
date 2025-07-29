"""
PDF Endpoints - Endpoint đơn giản để xử lý PDF với OCR và LLM formatting
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import Dict, Any, Optional

from app.services.llm_service import get_llm_service
from app.services.background_task_processor import get_background_task_processor
from app.services.qdrant_service import get_qdrant_service

logger = logging.getLogger(__name__)


router = APIRouter()



@router.post("/import", response_model=Dict[str, Any])
async def quick_textbook_analysis(
    file: UploadFile = File(...),
    book_id: Optional[str] = Form(None, description="ID của textbook (bắt buộc cho PDF, tùy chọn cho DOCX guide)"),
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
        book_id: ID của textbook (bắt buộc cho PDF, tùy chọn cho DOCX guide)
        create_embeddings: Có tạo embeddings cho RAG search không
        lesson_id: ID bài học tùy chọn để liên kết với lesson cụ thể
        isImportGuide: True nếu import file DOCX làm hướng dẫn, False cho PDF sách giáo khoa

    Returns:
        Dict chứa task_id để theo dõi tiến độ

    Examples:
        # Import PDF textbook với book_id bắt buộc
        curl -X POST "http://localhost:8000/api/v1/pdf/import" \
             -F "file=@textbook.pdf" \
             -F "book_id=hoa12" \
             -F "create_embeddings=true" \
             -F "isImportGuide=false"

        # Import DOCX guide với book_id tùy chọn
        curl -X POST "http://localhost:8000/api/v1/pdf/import" \
             -F "file=@guide.docx" \
             -F "book_id=guide_hoa12" \
             -F "create_embeddings=true" \
             -F "isImportGuide=true"
    """
    try:
        # Validate file type and book_id based on import mode
        if isImportGuide:
            # Validate DOCX file for guide import
            if not file.filename or not file.filename.lower().endswith(".docx"):
                raise HTTPException(status_code=400, detail="Guide import only supports DOCX files")
            # book_id is optional for guides, will use filename if not provided
            if not book_id:
                book_id = file.filename.replace(".docx", "").replace(" ", "_").lower()
        else:
            # Validate PDF file for textbook import
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Textbook import only supports PDF files")
            # book_id is required for textbooks
            if not book_id:
                raise HTTPException(
                    status_code=400,
                    detail="book_id is required for textbook import. Please provide a unique book_id (e.g., 'hoa12', 'toan10')"
                )

        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(
            f"Starting {'guide import' if isImportGuide else 'textbook analysis'} task for: {file.filename} "
            f"with book_id: {book_id} ({len(file_content)} bytes)"
        )

        if isImportGuide:
            # Create guide import task
            task_id = await get_background_task_processor().create_guide_import_task(
                docx_content=file_content,
                filename=file.filename,
                book_id=book_id,
                create_embeddings=create_embeddings,
            )

            return {
                "success": True,
                "task_id": task_id,
                "book_id": book_id,
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
                book_id=book_id,
                create_embeddings=create_embeddings,
                lesson_id=lesson_id,
            )

            return {
                "success": True,
                "task_id": task_id,
                "book_id": book_id,
                "filename": file.filename,
                "status": "processing",
                "import_type": "textbook",
                "lesson_id": lesson_id,
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


@router.get("/lessons", response_model=Dict[str, Any])
async def get_all_lessons(book_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Lấy tất cả bài học textbook từ Qdrant

    Endpoint này trả về danh sách bài học textbook đã được import vào hệ thống với các thông tin:
    - bookId: ID của sách
    - lessonId: ID của bài học
    - fileUrl: URL của file PDF trên Supabase Storage
    - uploaded_at: Thời gian upload file lên Supabase (ưu tiên)
    - processed_at: Thời gian xử lý (fallback)
    - content_type: Loại nội dung (textbook)
    - total_chunks: Số lượng chunks được tạo

    Args:
        book_id: Optional - Filter theo book ID cụ thể

    Returns:
        Dict chứa danh sách bài học textbook và thông tin tổng quan

    Examples:
        curl -X GET "http://localhost:8000/api/v1/pdf/lessons"
        curl -X GET "http://localhost:8000/api/v1/pdf/lessons?book_id=hoa12"
    """
    try:
        from app.services.qdrant_service import get_qdrant_service

        qdrant_service = get_qdrant_service()

        logger.info(f"Getting textbook lessons from Qdrant" + (f" for book_id={book_id}" if book_id else "..."))
        result = await qdrant_service.get_lessons_by_type(content_type="textbook", book_id=book_id)

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve lessons: {result.get('error', 'Unknown error')}"
            )

        logger.info(f"Successfully retrieved {result.get('total_lessons', 0)} textbook lessons" + (f" for book_id={book_id}" if book_id else ""))

        return {
            "success": True,
            "data": {
                "lessons": result.get("lessons", []),
                "total_lessons": result.get("total_lessons", 0),
                "collections_processed": result.get("collections_processed", 0),
                "content_type": "textbook",
                "book_id": book_id
            },
            "message": f"Retrieved {result.get('total_lessons', 0)} textbook lessons successfully" + (f" for book_id={book_id}" if book_id else "")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_all_lessons: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/guides", response_model=Dict[str, Any])
async def get_all_guides(book_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Lấy tất cả guide từ Qdrant

    Endpoint này trả về danh sách guide đã được import vào hệ thống với các thông tin:
    - bookId: ID của sách
    - lessonId: ID của bài học
    - fileUrl: URL của file DOCX trên Supabase Storage
    - uploaded_at: Thời gian upload file lên Supabase (ưu tiên)
    - processed_at: Thời gian xử lý (fallback)
    - content_type: Loại nội dung (guide)
    - total_chunks: Số lượng chunks được tạo

    Args:
        book_id: Optional - Filter theo book ID cụ thể

    Returns:
        Dict chứa danh sách guide và thông tin tổng quan

    Examples:
        curl -X GET "http://localhost:8000/api/v1/pdf/guides"
        curl -X GET "http://localhost:8000/api/v1/pdf/guides?book_id=guide_hoa12"
    """
    try:
        from app.services.qdrant_service import get_qdrant_service

        qdrant_service = get_qdrant_service()

        logger.info(f"Getting guide lessons from Qdrant" + (f" for book_id={book_id}" if book_id else "..."))
        result = await qdrant_service.get_lessons_by_type(content_type="guide", book_id=book_id)

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve guides: {result.get('error', 'Unknown error')}"
            )

        logger.info(f"Successfully retrieved {result.get('total_lessons', 0)} guides" + (f" for book_id={book_id}" if book_id else ""))

        return {
            "success": True,
            "data": {
                "guides": result.get("lessons", []),  # Đổi tên từ lessons thành guides
                "total_guides": result.get("total_lessons", 0),
                "collections_processed": result.get("collections_processed", 0),
                "content_type": "guide",
                "book_id": book_id
            },
            "message": f"Retrieved {result.get('total_lessons', 0)} guides successfully" + (f" for book_id={book_id}" if book_id else "")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_all_guides: {e}")
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
        qdrant_service = get_qdrant_service()

        # Đảm bảo service được khởi tạo
        qdrant_service._ensure_service_initialized()

        result = await qdrant_service.get_all_textbooks()

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get textbooks: {result.get('error', 'Unknown error')}"
            )

        return result

    except Exception as e:
        logger.error(f"Error in getAllTextBook: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




@router.get("/search", response_model=Dict[str, Any])
async def search_all_textbooks(
    query: str,
    limit: int = Query(10, ge=1, le=50),
    book_id: Optional[str] = Query(None, description="Filter by specific book ID"),
    lesson_id: Optional[str] = Query(None, description="Filter by specific lesson ID")
) -> Dict[str, Any]:
    """
    Tìm kiếm trong TẤT CẢ sách giáo khoa (Global Search) với metadata filtering

    Endpoint này cho phép người dùng tìm kiếm với các filter metadata:
    - Tìm kiếm toàn bộ (không có filter)
    - Tìm kiếm trong sách cụ thể (book_id)
    - Tìm kiếm trong bài học cụ thể (lesson_id)

    Args:
        query: Câu truy vấn tìm kiếm (từ khóa, câu hỏi, chủ đề)
        limit: Số lượng kết quả tối đa (1-50, mặc định 10)
        book_id: ID sách cụ thể để filter (tùy chọn)
        lesson_id: ID bài học cụ thể để filter (tùy chọn)

    Returns:
        Dict chứa kết quả tìm kiếm với metadata đầy đủ bao gồm book_id, lesson_id

    Examples:
        - /api/v1/pdf/search?query=hóa học là gì
        - /api/v1/pdf/search?query=nguyên tử&book_id=hoa12
        - /api/v1/pdf/search?query=định nghĩa&lesson_id=hoa12_bai1&limit=5
    """
    try:
        qdrant_service = get_qdrant_service()

        logger.info(f"Global search query: '{query}' with limit: {limit}, book_id: {book_id}, lesson_id: {lesson_id}")

        # Tạo semantic filters từ parameters
        semantic_filters = {}
        if book_id:
            semantic_filters["book_id"] = book_id
        if lesson_id:
            semantic_filters["lesson_id"] = lesson_id

        # Sử dụng global_search từ unified collection với filters
        search_result = await qdrant_service.global_search(
            query=query,
            limit=limit,
            book_id=semantic_filters.get("book_id") if semantic_filters else None,
            lesson_id=semantic_filters.get("lesson_id") if semantic_filters else None
        )

        if not search_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {search_result.get('error', 'Unknown error')}",
            )

        return search_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in global search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




@router.get("/textbook/{book_id}/info", response_model=Dict[str, Any])
async def get_textbook_info(book_id: str) -> Dict[str, Any]:
    """
    Lấy thông tin chi tiết về textbook theo book_id với metadata đầy đủ

    Args:
        book_id: ID của textbook cần lấy thông tin

    Returns:
        Dict chứa thông tin chi tiết về textbook bao gồm:
        - Metadata cơ bản (title, chapters, lessons)
        - Danh sách lessons với lesson_id
        - Thống kê về content

    Examples:
        GET /api/v1/pdf/textbook/hoa12/info
    """
    try:
        qdrant_service = get_qdrant_service()
        result = await qdrant_service.get_textbook_info_by_book_id(book_id)

        if not result.get("success"):
            if "not found" in result.get("error", "").lower():
                raise HTTPException(
                    status_code=404,
                    detail=result.get("error")
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to get textbook info")
                )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting textbook info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbook/{book_id}/lessons", response_model=Dict[str, Any])
async def get_textbook_lessons(book_id: str) -> Dict[str, Any]:
    """
    Lấy danh sách tất cả lessons trong textbook theo book_id

    Args:
        book_id: ID của textbook

    Returns:
        Dict chứa danh sách lessons với lesson_id và metadata

    Examples:
        GET /api/v1/pdf/textbook/hoa12/lessons
    """
    try:
        qdrant_service = get_qdrant_service()
        result = await qdrant_service.get_textbook_lessons_by_book_id(book_id)

        if not result.get("success"):
            if "not found" in result.get("error", "").lower():
                raise HTTPException(
                    status_code=404,
                    detail=result.get("error")
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to get textbook lessons")
                )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting textbook lessons: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/lesson/{lesson_id}/check", response_model=Dict[str, Any])
async def check_lesson_id_exists(lesson_id: str) -> Dict[str, Any]:
    """
    Kiểm tra lesson_id đã tồn tại chưa

    Args:
        lesson_id: ID của lesson cần kiểm tra

    Returns:
        Dict chứa thông tin về lesson_id existence

    Examples:
        GET /api/v1/pdf/lesson/hoa12_bai1/check

        Response nếu tồn tại:
        {
            "success": true,
            "exists": true,
            "lesson_id": "hoa12_bai1",
            "existing_book_id": "hoa12",
            "message": "Lesson ID 'hoa12_bai1' already exists in book 'hoa12'"
        }

        Response nếu chưa tồn tại:
        {
            "success": true,
            "exists": false,
            "lesson_id": "hoa12_bai1",
            "message": "Lesson ID 'hoa12_bai1' is available"
        }
    """
    try:
        qdrant_service = get_qdrant_service()
        result = await qdrant_service.check_lesson_id_exists(lesson_id)

        return result

    except Exception as e:
        logger.error(f"Error checking lesson_id {lesson_id}: {e}")
        return {
            "success": False,
            "exists": False,
            "error": f"Failed to check lesson_id: {str(e)}",
            "lesson_id": lesson_id
        }

@router.get("/lesson/{lesson_id}/info", response_model=Dict[str, Any])
async def get_lesson_info(lesson_id: str) -> Dict[str, Any]:
    """
    Lấy thông tin chi tiết về lesson theo lesson_id

    Args:
        lesson_id: ID của lesson

    Returns:
        Dict chứa thông tin chi tiết về lesson và metadata

    Examples:
        GET /api/v1/pdf/lesson/hoa12_bai1/info
    """
    try:
        qdrant_service = get_qdrant_service()
        result = await qdrant_service.get_lesson_info_by_lesson_id(lesson_id)

        if not result.get("success"):
            if "not found" in result.get("error", "").lower():
                raise HTTPException(
                    status_code=404,
                    detail=result.get("error")
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to get lesson info")
                )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lesson info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        from app.services.simple_ocr_service import get_simple_ocr_service

        # Check OCR service availability
        simple_ocr_service = get_simple_ocr_service()
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
                "/textbook/{book_id}/info",  # Get textbook info with metadata
                "/textbook/{book_id}/lessons",  # Get lessons list by book_id
                "/lesson/{lesson_id}/info",  # Get lesson info with metadata
                "/lesson/{lesson_id}/check",  # Check lesson_id existence
                "/search",  # Global content search with book_id/lesson_id filters
                "/update-lesson-id",  # PUT: Update lesson_id in book
                "/update-book-id",  # PUT: Update book_id in Qdrant
                "/textbook",  # DELETE: Flexible textbook deletion
                "/health",
            ],
            "usage_flow": {
                "1a": "Upload PDF (Textbook): POST /import (with isImportGuide=false)",
                "1b": "Upload DOCX (Guide): POST /import (with isImportGuide=true)",
                "2": "Check lesson_id: GET /lesson/{lesson_id}/check",
                "3": "List textbooks: GET /textbooks",
                "4": "Get textbook info: GET /textbook/{book_id}/info",
                "5": "Get lessons list: GET /textbook/{book_id}/lessons",
                "6": "Get lesson info: GET /lesson/{lesson_id}/info",
                "7": "Global search: GET /search?query=your_query&book_id=book123&lesson_id=lesson456",
                "8": "Update lesson_id: PUT /update-lesson-id?book_id=xxx&old_lesson_id=yyy&new_lesson_id=zzz",
                "9": "Update book_id: PUT /update-book-id?old_book_id=xxx&new_book_id=yyy",
                "10": "Delete textbook: DELETE /textbook?textbook_id=your_id OR DELETE /textbook?lesson_id=your_lesson_id"
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}




@router.put("/update-lesson-id", response_model=Dict[str, Any])
async def update_lesson_id(
    book_id: str = Query(..., description="ID của book chứa lesson cần update"),
    old_lesson_id: str = Query(..., description="lessonID cũ cần thay đổi"),
    new_lesson_id: str = Query(..., description="lessonID mới")
) -> Dict[str, Any]:
    """
    Update tất cả lessonID cũ thành lessonID mới trong một bookID

    Endpoint này sẽ tìm tất cả points trong collection của book_id có lesson_id cũ
    và update thành lesson_id mới.

    Args:
        book_id: ID của book chứa lesson cần update
        old_lesson_id: lessonID cũ cần thay đổi
        new_lesson_id: lessonID mới

    Returns:
        Dict chứa kết quả update với số lượng points đã được update

    Examples:
        PUT /api/v1/pdf/update-lesson-id?book_id=hoa12&old_lesson_id=hoa12_bai1&new_lesson_id=hoa12_lesson1
    """
    try:
        qdrant_service = get_qdrant_service()

        logger.info(f"Updating lesson_id from '{old_lesson_id}' to '{new_lesson_id}' in book '{book_id}'")

        result = await qdrant_service.update_lesson_id_in_book(
            book_id=book_id,
            old_lesson_id=old_lesson_id,
            new_lesson_id=new_lesson_id
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to update lesson_id")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating lesson_id: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/update-book-id", response_model=Dict[str, Any])
async def update_book_id(
    old_book_id: str = Query(..., description="bookID cũ cần thay đổi"),
    new_book_id: str = Query(..., description="bookID mới")
) -> Dict[str, Any]:
    """
    Update bookID cũ thành bookID mới trong Qdrant

    Endpoint này sẽ:
    1. Tìm tất cả collections có old_book_id (textbook_xxx, guide_xxx)
    2. Tạo collections mới với new_book_id
    3. Copy tất cả points với book_id được update
    4. Xóa collections cũ

    Args:
        old_book_id: bookID cũ cần thay đổi
        new_book_id: bookID mới

    Returns:
        Dict chứa kết quả update với thông tin các collections đã được xử lý

    Examples:
        PUT /api/v1/pdf/update-book-id?old_book_id=hoa12&new_book_id=chemistry12
    """
    try:
        qdrant_service = get_qdrant_service()

        logger.info(f"Updating book_id from '{old_book_id}' to '{new_book_id}'")

        result = await qdrant_service.update_book_id_in_qdrant(
            old_book_id=old_book_id,
            new_book_id=new_book_id
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to update book_id")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating book_id: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("", response_model=Dict[str, Any])
async def delete_textbook_content(
    book_id: str = Query(..., description="Book ID - required for both operations"),
    lesson_id: Optional[str] = Query(None, description="Lesson ID to delete specific lesson")
) -> Dict[str, Any]:
    """
    Xóa nội dung textbook/guide - Xóa cả trên Qdrant và Supabase

    - book_id only: Xóa toàn bộ collection của book và tất cả files liên quan trên Supabase
    - book_id + lesson_id: Xóa lesson cụ thể trong collection của book và file tương ứng trên Supabase

    Args:
        book_id: ID của book (bắt buộc)
        lesson_id: ID của lesson cần xóa (optional - nếu có thì xóa lesson, không có thì xóa book)

    Returns:
        Dict chứa kết quả xóa từ cả Qdrant và Supabase

    Examples:
        DELETE /api/v1/pdf?book_id=hoa12                           # Xóa toàn bộ sách Hóa 12 và files
        DELETE /api/v1/pdf?book_id=hoa12&lesson_id=hoa12_bai1      # Xóa bài 1 trong sách Hóa 12 và file
        DELETE /api/v1/pdf?book_id=guide_hoa12                     # Xóa toàn bộ guide Hóa 12 và files
    """
    try:
        qdrant_service = get_qdrant_service()

        # Bước 1: Lấy danh sách file URLs cần xóa từ Supabase trước khi xóa khỏi Qdrant
        logger.info(f"Getting file URLs for deletion: book_id={book_id}, lesson_id={lesson_id}")
        file_urls = await qdrant_service.get_file_urls_for_deletion(book_id, lesson_id)

        # Bước 2: Xóa files từ Supabase
        supabase_results = []
        if file_urls:
            try:
                from app.services.supabase_storage_service import get_supabase_storage_service

                supabase_service = get_supabase_storage_service()
                if supabase_service.is_available():
                    logger.info(f"Deleting {len(file_urls)} files from Supabase")
                    for file_url in file_urls:
                        delete_result = await supabase_service.delete_file_by_url(file_url)
                        supabase_results.append({
                            "file_url": file_url,
                            "success": delete_result.get("success", False),
                            "error": delete_result.get("error") if not delete_result.get("success") else None
                        })
                        if delete_result.get("success"):
                            logger.info(f"✅ Deleted from Supabase: {file_url}")
                        else:
                            logger.warning(f"⚠️ Failed to delete from Supabase: {file_url} - {delete_result.get('error')}")
                else:
                    logger.warning("Supabase service not available, skipping file deletion")
            except Exception as e:
                logger.warning(f"Error deleting files from Supabase: {e}")

        # Bước 3: Xóa từ Qdrant
        # Nếu có lesson_id: xóa lesson cụ thể trong book
        if lesson_id:
            logger.info(f"Deleting lesson '{lesson_id}' from book '{book_id}' in Qdrant")
            result = await qdrant_service.delete_lesson_in_book_clean(book_id, lesson_id)

            return {
                "success": True,
                "operation": "delete_lesson",
                "book_id": book_id,
                "lesson_id": lesson_id,
                "message": f"Lesson '{lesson_id}' deleted successfully from book '{book_id}'",
                "qdrant_details": result,
                "supabase_results": supabase_results,
                "files_deleted": len([r for r in supabase_results if r["success"]])
            }

        # Nếu không có lesson_id: xóa toàn bộ book
        else:
            logger.info(f"Deleting entire book: {book_id} from Qdrant")
            result = await qdrant_service.delete_book_clean(book_id)

            return {
                "success": True,
                "operation": "delete_book",
                "book_id": book_id,
                "message": f"Book '{book_id}' deleted successfully",
                "qdrant_details": result,
                "supabase_results": supabase_results,
                "files_deleted": len([r for r in supabase_results if r["success"]])
            }

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.warning(f"Resource not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Service error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting textbook content: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
