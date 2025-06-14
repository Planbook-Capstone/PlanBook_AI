"""
PDF Endpoints - Endpoint đơn giản để xử lý PDF với OCR và LLM formatting
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel

from app.services.pdf_service import simple_pdf_service
from app.services.llm_service import llm_service
from app.services.cv_parser_service import cv_parser_service
from app.services.textbook_parser_service import textbook_parser_service
from app.services.enhanced_textbook_service import enhanced_textbook_service

logger = logging.getLogger(__name__)


class FormatTextRequest(BaseModel):
    text: str
    document_type: Optional[str] = "general"


class TextbookMetadata(BaseModel):
    id: str
    title: str
    author: Optional[str] = None
    language: str = "vi"
    grade: Optional[str] = None
    subject: Optional[str] = None


router = APIRouter()


@router.post("/extract-text", response_model=Dict[str, Any])
async def extract_text_from_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract text from PDF using OCR

    Args:
        file: PDF file to process

    Returns:
        Dict containing extracted text and metadata
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Processing PDF file: {file.filename} ({len(file_content)} bytes)")

        # Process PDF
        result = await simple_pdf_service.extract_text_from_pdf(
            file_content=file_content, filename=file.filename
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process PDF: {result.get('error', 'Unknown error')}",
            )

        return {
            "success": True,
            "filename": result["filename"],
            "extracted_text": result["extracted_text"],
            "metadata": result["metadata"],
            "message": "PDF processed successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/extract-and-format", response_model=Dict[str, Any])
async def extract_and_format_pdf(
    file: UploadFile = File(...), document_type: str = "general"
) -> Dict[str, Any]:
    """
    Extract text from PDF and format it using LLM

    Args:
        file: PDF file to process
        document_type: Type of document (cv, report, letter, general)

    Returns:
        Dict containing extracted and formatted text
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
            f"Processing and formatting PDF: {file.filename} ({len(file_content)} bytes)"
        )

        # Extract text from PDF
        extract_result = await simple_pdf_service.extract_text_from_pdf(
            file_content=file_content, filename=file.filename
        )

        # Luôn lấy text, ngay cả khi extraction thất bại
        extracted_text = extract_result.get("extracted_text", "")
        extraction_success = extract_result.get("success", False)

        # Format text using LLM - ngay cả khi extracted_text rỗng hoặc có lỗi
        if extracted_text.strip():
            format_result = await llm_service.format_document_text(
                raw_text=extracted_text, document_type=document_type
            )
        else:
            # Nếu không có text, tạo thông báo để LLM format
            fallback_text = f"[PDF_PROCESSING_INFO] File: {file.filename}\nStatus: Could not extract text from PDF. This may be a scanned document or image-based PDF.\nSuggestion: Please try using OCR tools or convert to text format."
            format_result = await llm_service.format_document_text(
                raw_text=fallback_text, document_type="info"
            )

        extraction_success = extract_result.get("success", False)

        return {
            "success": True,  # Luôn trả về success
            "filename": extract_result["filename"],
            "extraction_metadata": extract_result["metadata"],
            "extraction_success": extraction_success,
            "extracted_text": extracted_text,
            "formatted_text": format_result.get("formatted_text", extracted_text),
            "formatting_success": format_result.get("success", False),
            "formatting_error": format_result.get("error"),
            "document_type": document_type,
            "message": "PDF processed and formatted successfully"
            if extraction_success
            else "PDF processed with limited text extraction, but formatting applied",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing and formatting PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/format-text", response_model=Dict[str, Any])
async def format_text_only(request: FormatTextRequest) -> Dict[str, Any]:
    """
    Format text using LLM (without PDF extraction)

    Args:
        request: Text and document type to format

    Returns:
        Dict containing formatted text
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        logger.info(f"Formatting text of length: {len(request.text)} characters")

        # Format text using LLM
        format_result = await llm_service.format_document_text(
            raw_text=request.text, document_type=request.document_type or "general"
        )

        return {
            "success": format_result.get("success", False),
            "original_text": request.text,
            "formatted_text": format_result.get("formatted_text", request.text),
            "formatting_error": format_result.get("error"),
            "document_type": request.document_type,
            "original_length": len(request.text),
            "formatted_length": len(format_result.get("formatted_text", "")),
            "message": "Text formatted successfully"
            if format_result.get("success")
            else "Text formatting failed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error formatting text: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/extract-and-parse", response_model=Dict[str, Any])
async def extract_and_parse_cv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract text from PDF và parse thành structured CV data

    Args:
        file: PDF file to process

    Returns:
        Dict containing structured CV data with fields
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
            f"Processing and parsing CV: {file.filename} ({len(file_content)} bytes)"
        )

        # Extract text from PDF
        extract_result = await simple_pdf_service.extract_text_from_pdf(
            file_content=file_content, filename=file.filename
        )

        extracted_text = extract_result.get("extracted_text", "")
        extraction_success = extract_result.get("success", False)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400, detail="Could not extract text from PDF"
            )

        # Parse CV to structured data
        parse_result = await cv_parser_service.parse_cv_to_structured_data(
            extracted_text
        )

        return {
            "success": True,
            "filename": extract_result["filename"],
            "extraction_metadata": extract_result["metadata"],
            "extraction_success": extraction_success,
            "parsing_success": parse_result.get("success", False),
            "parsing_method": parse_result.get("parsing_method", "unknown"),
            "cv_data": parse_result.get("cv_data"),
            "parsing_error": parse_result.get("error"),
            "message": "CV processed and parsed successfully"
            if parse_result.get("success")
            else "CV processed but parsing failed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing and parsing CV: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/parse-text", response_model=Dict[str, Any])
async def parse_cv_text(request: FormatTextRequest) -> Dict[str, Any]:
    """
    Parse CV text thành structured data (without PDF extraction)

    Args:
        request: CV text to parse

    Returns:
        Dict containing structured CV data
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        logger.info(f"Parsing CV text of length: {len(request.text)} characters")

        # Parse CV to structured data
        parse_result = await cv_parser_service.parse_cv_to_structured_data(request.text)

        return {
            "success": parse_result.get("success", False),
            "original_text": request.text,
            "parsing_method": parse_result.get("parsing_method", "unknown"),
            "cv_data": parse_result.get("cv_data"),
            "parsing_error": parse_result.get("error"),
            "message": "CV text parsed successfully"
            if parse_result.get("success")
            else "CV text parsing failed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing CV text: {e}")
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


@router.post("/process-textbook-enhanced", response_model=Dict[str, Any])
async def process_textbook_enhanced(
    file: UploadFile = File(...),
    title: str = Form(...),
    subject: str = Form("Chưa xác định"),
    grade: str = Form("Chưa xác định"),
    author: str = Form(None),
    language: str = Form("vi"),
) -> Dict[str, Any]:
    """
    Xử lý sách giáo khoa với OCR cải tiến và trả về cấu trúc: Sách → Chương → Bài → Nội dung

    Args:
        file: PDF file sách giáo khoa
        title: Tiêu đề sách
        subject: Môn học (Toán, Lý, Hóa, ...)
        grade: Lớp (10, 11, 12, ...)
        author: Tác giả
        language: Ngôn ngữ (vi, en)

    Returns:
        Dict chứa cấu trúc sách hoàn chỉnh
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
            f"Processing enhanced textbook: {file.filename} ({len(file_content)} bytes)"
        )

        # Prepare book metadata
        book_metadata = {
            "title": title,
            "subject": subject,
            "grade": grade,
            "author": author,
            "language": language,
        }

        # Process textbook with enhanced service
        result = await enhanced_textbook_service.process_textbook_to_structure(
            pdf_content=file_content,
            filename=file.filename,
            book_metadata=book_metadata,
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process textbook: {result.get('error', 'Unknown error')}",
            )

        return {
            "success": True,
            "filename": file.filename,
            "book": result["book"],
            "statistics": {
                "total_pages": result.get("total_pages", 0),
                "total_chapters": result.get("total_chapters", 0),
                "total_lessons": result.get("total_lessons", 0),
            },
            "message": "Textbook processed successfully with enhanced OCR",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing enhanced textbook: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/quick-textbook-analysis", response_model=Dict[str, Any])
async def quick_textbook_analysis(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Phân tích nhanh cấu trúc sách giáo khoa (chỉ trả về outline, không xử lý nội dung chi tiết)

    Args:
        file: PDF file sách giáo khoa

    Returns:
        Dict chứa outline cấu trúc sách
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Quick analysis for: {file.filename} ({len(file_content)} bytes)")

        # Extract pages for analysis only
        pages_data = await enhanced_textbook_service._extract_pages_with_ocr(
            file_content
        )

        # Analyze structure only
        book_structure = (
            await enhanced_textbook_service._analyze_book_structure_enhanced(
                pages_data, {"title": file.filename.replace(".pdf", "")}
            )
        )

        # Return outline only
        outline = {
            "book_info": book_structure.get("book_info", {}),
            "chapters_outline": [],
        }

        for chapter in book_structure.get("chapters", []):
            chapter_outline = {
                "chapter_id": chapter["chapter_id"],
                "chapter_title": chapter["chapter_title"],
                "page_range": f"{chapter['start_page']}-{chapter['end_page']}",
                "lessons_count": len(chapter.get("lessons", [])),
                "lessons_outline": [
                    {
                        "lesson_id": lesson["lesson_id"],
                        "lesson_title": lesson["lesson_title"],
                        "page_range": f"{lesson['start_page']}-{lesson['end_page']}",
                    }
                    for lesson in chapter.get("lessons", [])
                ],
            }
            outline["chapters_outline"].append(chapter_outline)

        return {
            "success": True,
            "filename": file.filename,
            "outline": outline,
            "statistics": {
                "total_pages": len(pages_data),
                "total_chapters": len(outline["chapters_outline"]),
                "total_lessons": sum(
                    ch["lessons_count"] for ch in outline["chapters_outline"]
                ),
            },
            "message": "Quick textbook analysis completed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick textbook analysis: {e}")
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
        structure = await textbook_parser_service.get_book_structure(book_id)

        if not structure:
            raise HTTPException(
                status_code=404, detail=f"Textbook with ID '{book_id}' not found"
            )

        return {"success": True, "book_id": book_id, "structure": structure}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting textbook structure: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/textbook/{book_id}/lesson/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson_content(book_id: str, lesson_id: str) -> Dict[str, Any]:
    """
    Lấy nội dung chi tiết của một bài học

    Args:
        book_id: ID của sách
        lesson_id: ID của bài học

    Returns:
        Dict containing lesson content
    """
    try:
        lesson_content = await textbook_parser_service.get_lesson_content(
            book_id, lesson_id
        )

        if not lesson_content:
            raise HTTPException(
                status_code=404,
                detail=f"Lesson '{lesson_id}' not found in textbook '{book_id}'",
            )

        return {
            "success": True,
            "book_id": book_id,
            "lesson_id": lesson_id,
            "content": lesson_content,
        }

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

        # Kiểm tra xem sách có tồn tại không
        structure = await textbook_parser_service.get_book_structure(book_id)
        if not structure:
            raise HTTPException(
                status_code=404, detail=f"Textbook with ID '{book_id}' not found"
            )

        # Tìm kiếm với Qdrant
        search_result = await qdrant_service.search_textbook(
            book_id=book_id, query=query, limit=limit
        )

        if not search_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {search_result.get('error', 'Unknown error')}",
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
                "llm_formatting": "available" if llm_available else "unavailable",
                "cv_parsing": "available" if llm_available else "basic_only",
                "textbook_processing": "available",
            },
            "supported_languages": supported_langs,
            "llm_status": "Gemini API configured"
            if llm_available
            else "Gemini API not configured",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
