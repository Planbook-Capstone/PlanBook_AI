"""
Models cho online document responses
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class OnlineDocumentLinks(BaseModel):
    """Model cho các link của document online"""
    view: str = Field(..., description="Link để xem document")
    edit: Optional[str] = Field(None, description="Link để chỉnh sửa document (nếu có)")
    preview: Optional[str] = Field(None, description="Link preview document")
    download: Optional[str] = Field(None, description="Link download document")


class OnlineDocumentResponse(BaseModel):
    """Model cho response khi tạo document online thành công"""
    success: bool = Field(..., description="Trạng thái thành công")
    message: str = Field(..., description="Thông báo")
    
    # Document info
    file_id: Optional[str] = Field(None, description="ID của file trên cloud storage")
    filename: str = Field(..., description="Tên file")
    mime_type: Optional[str] = Field(None, description="MIME type của file")
    
    # Links
    links: OnlineDocumentLinks = Field(..., description="Các link truy cập document")
    primary_link: str = Field(..., description="Link chính để truy cập document")
    
    # Metadata
    created_at: str = Field(..., description="Thời gian tạo")
    storage_provider: str = Field(..., description="Nhà cung cấp lưu trữ (Google Drive, OneDrive, etc.)")
    
    # Additional info
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Thông tin bổ sung")


class ExamOnlineResponse(OnlineDocumentResponse):
    """Model cho response khi tạo đề thi online"""
    exam_id: str = Field(..., description="ID của bài kiểm tra")
    lesson_id: str = Field(..., description="ID bài học")
    mon_hoc: str = Field(..., description="Môn học")
    lop: int = Field(..., description="Lớp")
    total_questions: int = Field(..., description="Tổng số câu hỏi")
    search_quality: Optional[float] = Field(None, description="Chất lượng tìm kiếm nội dung")


class LessonPlanOnlineResponse(OnlineDocumentResponse):
    """Model cho response khi tạo giáo án online"""
    lesson_id: str = Field(..., description="ID bài học")
    framework_id: str = Field(..., description="ID khung giáo án")
    framework_name: str = Field(..., description="Tên khung giáo án")


class OnlineDocumentError(BaseModel):
    """Model cho lỗi khi tạo document online"""
    success: bool = Field(False, description="Trạng thái thất bại")
    error: str = Field(..., description="Mô tả lỗi")
    error_code: Optional[str] = Field(None, description="Mã lỗi")
    fallback_available: bool = Field(False, description="Có thể fallback về download file không")
    fallback_message: Optional[str] = Field(None, description="Thông báo fallback")
