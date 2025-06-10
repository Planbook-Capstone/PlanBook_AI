from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import BeforeValidator
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime
from bson import ObjectId

def validate_object_id(v: Any) -> ObjectId:
    """Validate ObjectId"""
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str):
        if ObjectId.is_valid(v):
            return ObjectId(v)
    raise ValueError("Invalid ObjectId")

# Custom ObjectId type for Pydantic v2
PyObjectId = Annotated[ObjectId, BeforeValidator(validate_object_id)]

# Base config for all models
base_config = ConfigDict(
    populate_by_name=True,
    arbitrary_types_allowed=True,
    json_encoders={ObjectId: str}
)

class ChemistryTextbook(BaseModel):
    """Model cho sách giáo khoa Hóa học"""
    model_config = base_config

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    title: str = Field(..., description="Tên sách giáo khoa")
    grade: str = Field(..., description="Lớp (10, 11, 12)")
    publisher: str = Field(..., description="Nhà xuất bản")
    year: int = Field(..., description="Năm xuất bản")
    file_path: str = Field(..., description="Đường dẫn file PDF")
    file_size: int = Field(..., description="Kích thước file (bytes)")
    total_pages: int = Field(..., description="Tổng số trang")
    upload_date: datetime = Field(default_factory=datetime.now)
    processed: bool = Field(default=False, description="Đã xử lý chưa")

class ChemistryChapter(BaseModel):
    """Model cho chương trong sách Hóa học"""
    model_config = base_config

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    textbook_id: PyObjectId = Field(..., description="ID sách giáo khoa")
    chapter_number: int = Field(..., description="Số thứ tự chương")
    title: str = Field(..., description="Tên chương")
    description: Optional[str] = Field(None, description="Mô tả chương")
    start_page: int = Field(..., description="Trang bắt đầu")
    end_page: int = Field(..., description="Trang kết thúc")
    content: str = Field(..., description="Nội dung text của chương")
    created_date: datetime = Field(default_factory=datetime.now)

class ChemistryLesson(BaseModel):
    """Model cho bài học trong chương"""
    model_config = base_config

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    chapter_id: PyObjectId = Field(..., description="ID chương")
    textbook_id: PyObjectId = Field(..., description="ID sách giáo khoa")
    lesson_number: int = Field(..., description="Số thứ tự bài")
    title: str = Field(..., description="Tên bài học")
    objectives: List[str] = Field(default=[], description="Mục tiêu bài học")
    content: str = Field(..., description="Nội dung bài học")
    key_concepts: List[str] = Field(default=[], description="Khái niệm chính")
    formulas: List[str] = Field(default=[], description="Công thức hóa học")
    experiments: List[Dict[str, Any]] = Field(default=[], description="Thí nghiệm")
    exercises: List[Dict[str, Any]] = Field(default=[], description="Bài tập")
    start_page: int = Field(..., description="Trang bắt đầu")
    end_page: int = Field(..., description="Trang kết thúc")
    created_date: datetime = Field(default_factory=datetime.now)

class ChemistryEmbedding(BaseModel):
    """Model cho vector embeddings của nội dung Hóa học"""
    model_config = base_config

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    content_id: PyObjectId = Field(..., description="ID của content (lesson/chapter)")
    content_type: str = Field(..., description="Loại content: lesson, chapter, textbook")
    text_chunk: str = Field(..., description="Đoạn text được embedding")
    chunk_index: int = Field(..., description="Thứ tự chunk trong content")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: Dict[str, Any] = Field(default={}, description="Metadata bổ sung")
    created_date: datetime = Field(default_factory=datetime.now)

class GeneratedLessonPlan(BaseModel):
    """Model cho giáo án đã tạo"""
    model_config = base_config

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    title: str = Field(..., description="Tên giáo án")
    subject: str = Field(default="Hóa học", description="Môn học")
    grade: str = Field(..., description="Lớp")
    topic: str = Field(..., description="Chủ đề")
    duration: int = Field(..., description="Thời lượng (phút)")

    # Nội dung giáo án
    objectives: List[str] = Field(default=[], description="Mục tiêu")
    materials: List[str] = Field(default=[], description="Dụng cụ, thiết bị")
    activities: List[Dict[str, Any]] = Field(default=[], description="Các hoạt động")
    assessment: Dict[str, Any] = Field(default={}, description="Đánh giá")
    homework: List[str] = Field(default=[], description="Bài tập về nhà")

    # Metadata
    source_lessons: List[PyObjectId] = Field(default=[], description="Bài học nguồn")
    generated_by: str = Field(default="ChemistryLessonAgent", description="Agent tạo")
    prompt_used: str = Field(default="", description="Prompt đã sử dụng")
    docx_file_path: Optional[str] = Field(None, description="Đường dẫn file DOCX")

    created_date: datetime = Field(default_factory=datetime.now)
    updated_date: datetime = Field(default_factory=datetime.now)

# Request/Response schemas cho API
class PDFUploadRequest(BaseModel):
    """Request cho upload PDF"""
    title: str = Field(..., description="Tên sách")
    grade: str = Field(..., description="Lớp (10, 11, 12)")
    publisher: str = Field(default="NXB Giáo dục Việt Nam", description="Nhà xuất bản")
    year: int = Field(default=2023, description="Năm xuất bản")

class PDFUploadResponse(BaseModel):
    """Response cho upload PDF"""
    textbook_id: str = Field(..., description="ID sách đã upload")
    message: str = Field(..., description="Thông báo")
    processing_status: str = Field(..., description="Trạng thái xử lý")
    pdf_file_id: Optional[str] = Field(None, description="ID file PDF trong GridFS")
    total_chapters: int = Field(default=0, description="Tổng số chương")
    total_pages: int = Field(default=0, description="Tổng số trang")

class LessonPlanGenerationRequest(BaseModel):
    """Request cho tạo giáo án"""
    topic: str = Field(..., description="Chủ đề bài học")
    grade: str = Field(..., description="Lớp (10, 11, 12)")
    duration: int = Field(default=45, description="Thời lượng (phút)")
    objectives: Optional[List[str]] = Field(None, description="Mục tiêu cụ thể")
    teaching_method: str = Field(default="Tương tác", description="Phương pháp dạy")
    include_experiments: bool = Field(default=True, description="Có thí nghiệm không")

class LessonPlanGenerationResponse(BaseModel):
    """Response cho tạo giáo án"""
    lesson_plan_id: str = Field(..., description="ID giáo án")
    docx_download_url: str = Field(..., description="URL download file DOCX")
    docx_file_id: Optional[str] = Field(None, description="ID file DOCX trong GridFS")
    message: str = Field(..., description="Thông báo")
    generation_time: float = Field(..., description="Thời gian tạo (giây)")

class FileMetadata(BaseModel):
    """Base metadata cho files trong GridFS"""
    model_config = base_config
    
    file_type: str = Field(..., description="Loại file (pdf, docx, pptx)")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="Kích thước file (bytes)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(default="system", description="Người tạo")

class TextbookFileMetadata(FileMetadata):
    """Metadata cho PDF sách giáo khoa"""
    book_title: str = Field(..., description="Tên sách")
    grade: str = Field(..., description="Lớp học")
    academic_year: str = Field(..., description="Năm học")
    publisher: Optional[str] = Field(None, description="Nhà xuất bản")

class LessonPlanFileMetadata(FileMetadata):
    """Metadata cho DOCX giáo án"""
    lesson_plan_id: str = Field(..., description="ID giáo án")
    lesson_title: str = Field(..., description="Tên bài học")
    grade: str = Field(..., description="Lớp học")
    topic: str = Field(..., description="Chủ đề")
    duration: int = Field(default=45, description="Thời lượng (phút)")

class StoredFileInfo(BaseModel):
    """Response model cho stored file"""
    model_config = base_config
    
    file_id: str = Field(..., description="GridFS file ID")
    filename: str = Field(..., description="Tên file")
    metadata: Dict[str, Any] = Field(..., description="File metadata")
    upload_date: datetime = Field(..., description="Ngày upload")
    file_size: int = Field(..., description="Kích thước file")
