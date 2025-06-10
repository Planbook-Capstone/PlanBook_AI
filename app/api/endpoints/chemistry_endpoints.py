from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from fastapi.responses import StreamingResponse
import os
import logging
from typing import Dict, Any
from bson import ObjectId
from app.database.models import (
    PDFUploadRequest, PDFUploadResponse,
    LessonPlanGenerationRequest, LessonPlanGenerationResponse
)
from app.services.pdf_service import pdf_service
from app.services.file_storage_service import file_storage
from app.agents.lesson_plan_agent import ChemistryLessonAgent
from app.database.connection import get_database, LESSON_PLANS_COLLECTION

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize Chemistry Lesson Agent (singleton pattern)
_chemistry_agent = None

def get_chemistry_agent():
    global _chemistry_agent
    if _chemistry_agent is None:
        _chemistry_agent = ChemistryLessonAgent()
    return _chemistry_agent

@router.post("/upload-textbook", response_model=PDFUploadResponse)
async def upload_chemistry_textbook(
    file: UploadFile = File(...),
    title: str = Form("Sách Giáo Khoa Hóa Học"),
    grade: str = Form("12"),
    publisher: str = Form("NXB Giáo dục Việt Nam"),
    academic_year: str = Form("2024-2025")
):
    """
    Upload và xử lý file PDF sách giáo khoa Hóa học với GridFS storage
    """
    try:
        # Validate file type (relaxed for testing)
        allowed_extensions = ['.pdf', '.txt']  # Allow .txt for testing
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF hoặc TXT (testing)")
        
        # Validate grade
        if grade not in ["10", "11", "12"]:
            raise HTTPException(status_code=400, detail="Lớp phải là 10, 11 hoặc 12")
        
        logger.info(f"Uploading textbook: {title} - Grade {grade}")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File rỗng")
        
        # Process với GridFS storage
        result = await pdf_service.process_textbook_pdf(
            file_content=file_content,
            filename=file.filename,
            book_title=title,
            grade=grade,
            publisher=publisher,
            academic_year=academic_year
        )
        
        logger.info(f"Successfully processed textbook: {result['textbook_id']}")
        
        return PDFUploadResponse(
            textbook_id=result["textbook_id"],
            message=f"Sách giáo khoa '{title}' đã được upload và lưu vào GridFS thành công",
            processing_status="completed",
            pdf_file_id=result.get("pdf_file_id"),
            total_chapters=result.get("total_chapters", 0),
            total_pages=result.get("total_pages", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload textbook: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi xử lý file: {str(e)}"
        )

@router.post("/generate-lesson-plan", response_model=LessonPlanGenerationResponse)
async def generate_chemistry_lesson_plan(request: LessonPlanGenerationRequest):
    """
    Tạo giáo án Hóa học sử dụng RAG và Gemini AI
    
    - **topic**: Chủ đề bài học
    - **grade**: Lớp (10, 11, 12)
    - **duration**: Thời lượng bài học (phút)
    - **objectives**: Mục tiêu cụ thể (optional)
    - **teaching_method**: Phương pháp dạy học
    - **include_experiments**: Có bao gồm thí nghiệm không
    """
    try:
        # Validate grade
        if request.grade not in ["10", "11", "12"]:
            raise HTTPException(status_code=400, detail="Lớp phải là 10, 11 hoặc 12")
        
        # Validate duration
        if request.duration < 15 or request.duration > 90:
            raise HTTPException(status_code=400, detail="Thời lượng phải từ 15-90 phút")
        
        logger.info(f"Generating lesson plan: {request.topic} - Grade {request.grade}")

        # Generate lesson plan using Chemistry Agent (now async)
        chemistry_agent = get_chemistry_agent()
        result = await chemistry_agent.process(
            topic=request.topic,
            grade=request.grade,
            duration=request.duration,
            objectives=request.objectives,
            teaching_method=request.teaching_method,
            include_experiments=request.include_experiments
        )
        
        # Extract result data
        agent_result = result.get("result", {})
        lesson_plan_id = agent_result.get("lesson_plan_id")
        docx_file_id = agent_result.get("docx_file_id")
        generation_time = agent_result.get("generation_time", 0)
        
        if not lesson_plan_id or not docx_file_id:
            raise ValueError("Không thể tạo giáo án hoặc file DOCX")
        
        # Create download URL
        download_url = f"/api/v1/chemistry/download-lesson-plan/{lesson_plan_id}"
        
        logger.info(f"Successfully generated lesson plan: {lesson_plan_id}")
        
        return LessonPlanGenerationResponse(
            lesson_plan_id=lesson_plan_id,
            docx_download_url=download_url,
            docx_file_id=docx_file_id,
            message=f"Giáo án '{request.topic}' đã được tạo và lưu vào GridFS thành công",
            generation_time=generation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate lesson plan: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi tạo giáo án: {str(e)}"
        )

@router.get("/download-lesson-plan/{lesson_plan_id}")
async def download_lesson_plan(lesson_plan_id: str):
    """
    Download file DOCX giáo án từ GridFS
    """
    try:
        # Get lesson plan from database
        db = await get_database()
        collection = db[LESSON_PLANS_COLLECTION]
        
        lesson_plan = await collection.find_one({"_id": ObjectId(lesson_plan_id)})
        
        if not lesson_plan:
            raise HTTPException(status_code=404, detail="Không tìm thấy giáo án")
        
        docx_file_id = lesson_plan.get("docx_file_id")
        
        if not docx_file_id:
            raise HTTPException(status_code=404, detail="Không tìm thấy file DOCX")
        
        # Stream file từ GridFS
        return await file_storage.get_file_stream_response(ObjectId(docx_file_id))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download lesson plan: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi download file: {str(e)}"
        )

@router.get("/lesson-plans")
async def list_lesson_plans(
    grade: str = None,
    limit: int = 10,
    skip: int = 0
):
    """
    Lấy danh sách giáo án đã tạo
    
    - **grade**: Lọc theo lớp (optional)
    - **limit**: Số lượng kết quả
    - **skip**: Bỏ qua số lượng kết quả
    """
    try:
        from app.database.connection import get_database_sync, LESSON_PLANS_COLLECTION
        
        db = get_database_sync()
        collection = db[LESSON_PLANS_COLLECTION]
        
        # Build query filter
        query_filter = {}
        if grade:
            query_filter["grade"] = grade
        
        # Get lesson plans
        cursor = collection.find(query_filter).sort("created_date", -1).skip(skip).limit(limit)
        lesson_plans = list(cursor)
        
        # Convert ObjectId to string
        for plan in lesson_plans:
            plan["_id"] = str(plan["_id"])
            if "source_lessons" in plan:
                plan["source_lessons"] = [str(id) for id in plan["source_lessons"]]
        
        # Get total count
        total_count = collection.count_documents(query_filter)
        
        return {
            "lesson_plans": lesson_plans,
            "total_count": total_count,
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        logger.error(f"Failed to list lesson plans: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi lấy danh sách giáo án: {str(e)}"
        )

@router.get("/textbooks")
async def list_textbooks():
    """
    Lấy danh sách sách giáo khoa đã upload
    """
    try:
        from app.database.connection import get_database_sync, CHEMISTRY_TEXTBOOK_COLLECTION
        
        db = get_database_sync()
        collection = db[CHEMISTRY_TEXTBOOK_COLLECTION]
        
        textbooks = list(collection.find().sort("upload_date", -1))
        
        # Convert ObjectId to string
        for book in textbooks:
            book["_id"] = str(book["_id"])
        
        return {
            "textbooks": textbooks,
            "total_count": len(textbooks)
        }
        
    except Exception as e:
        logger.error(f"Failed to list textbooks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi lấy danh sách sách: {str(e)}"
        )

@router.get("/agent-status")
async def get_agent_status():
    """
    Lấy trạng thái của Chemistry Lesson Agent
    """
    try:
        chemistry_agent = get_chemistry_agent()
        agent_info = chemistry_agent.get_agent_info()

        return {
            "agent_info": agent_info,
            "status": "active",
            "supported_grades": chemistry_agent.get_supported_grades()
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi lấy trạng thái agent: {str(e)}"
        )

@router.get("/files/{file_type}")
async def list_files_by_type(
    file_type: str,
    grade: str = None,
    limit: int = 50
):
    """
    List files theo type từ GridFS
    
    - **file_type**: Loại file (pdf, docx)
    - **grade**: Lọc theo lớp (optional)
    - **limit**: Số lượng kết quả
    """
    try:
        filters = {}
        if grade:
            filters["grade"] = grade
        
        files = await file_storage.list_files_by_type(
            file_type=file_type,
            limit=limit,
            **filters
        )
        
        return {
            "success": True,
            "files": files,
            "total": len(files)
        }
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Xóa file từ GridFS
    """
    try:
        success = await file_storage.delete_file(ObjectId(file_id))
        
        if success:
            return {"success": True, "message": "File đã được xóa thành công"}
        else:
            raise HTTPException(status_code=404, detail="Không tìm thấy file")
            
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage-stats")
async def get_storage_stats():
    """
    Thống kê storage sử dụng
    """
    try:
        # Get file counts by type
        pdf_files = await file_storage.list_files_by_type("pdf", limit=1000)
        docx_files = await file_storage.list_files_by_type("docx", limit=1000)
        
        total_size_pdf = sum(f["file_size"] for f in pdf_files)
        total_size_docx = sum(f["file_size"] for f in docx_files)
        
        return {
            "success": True,
            "stats": {
                "pdf_files": {
                    "count": len(pdf_files),
                    "total_size_mb": round(total_size_pdf / (1024 * 1024), 2)
                },
                "docx_files": {
                    "count": len(docx_files),
                    "total_size_mb": round(total_size_docx / (1024 * 1024), 2)
                },
                "total_files": len(pdf_files) + len(docx_files),
                "total_size_mb": round((total_size_pdf + total_size_docx) / (1024 * 1024), 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
