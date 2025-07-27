"""
Pydantic models cho Slide Generation API
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class SlideGenerationRequest(BaseModel):
    """Request model cho slide generation API"""
    
    lesson_id: str = Field(
        ..., 
        description="ID của bài học cần tạo slide",
        example="lesson_123"
    )
    
    template_id: str = Field(
        ..., 
        description="ID của Google Slides template (từ URL: https://docs.google.com/presentation/d/{template_id}/edit)",
        example="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
    )
    
    config_prompt: Optional[str] = Field(
        None,
        description="Prompt cấu hình tùy chỉnh cho LLM (optional)",
        example="Tạo slide với phong cách sinh động, phù hợp với học sinh lớp 10"
    )
    
    presentation_title: Optional[str] = Field(
        None,
        description="Tiêu đề tùy chỉnh cho presentation (optional)",
        example="Bài học Toán 10 - Hàm số bậc nhất"
    )

    @validator('template_id')
    def validate_template_id(cls, v):
        """Validate template_id format"""
        if not v or len(v.strip()) == 0:
            raise ValueError('template_id cannot be empty')
        
        # Nếu là URL đầy đủ, extract ID
        if 'docs.google.com/presentation/d/' in v:
            try:
                # Extract ID từ URL
                parts = v.split('/d/')
                if len(parts) > 1:
                    id_part = parts[1].split('/')[0]
                    return id_part
            except:
                pass
        
        return v.strip()

    @validator('lesson_id')
    def validate_lesson_id(cls, v):
        """Validate lesson_id format"""
        if not v or len(v.strip()) == 0:
            raise ValueError('lesson_id cannot be empty')
        return v.strip()


class TemplateInfo(BaseModel):
    """Thông tin về template đã sử dụng"""
    
    title: str = Field(description="Tiêu đề của template")
    layouts_count: int = Field(description="Số lượng layout trong template")


class SlideGenerationResponse(BaseModel):
    """Response model cho slide generation API"""
    
    success: bool = Field(description="Trạng thái thành công")
    
    lesson_id: Optional[str] = Field(
        None, 
        description="ID của bài học đã xử lý"
    )
    
    template_id: Optional[str] = Field(
        None, 
        description="ID của template đã sử dụng"
    )
    
    presentation_id: Optional[str] = Field(
        None, 
        description="ID của Google Slides presentation đã tạo"
    )
    
    presentation_title: Optional[str] = Field(
        None, 
        description="Tiêu đề của presentation đã tạo"
    )
    
    web_view_link: Optional[str] = Field(
        None, 
        description="Link để xem/chỉnh sửa presentation trên Google Slides"
    )
    
    slides_created: Optional[int] = Field(
        None, 
        description="Số lượng slide đã tạo"
    )
    
    template_info: Optional[TemplateInfo] = Field(
        None, 
        description="Thông tin về template đã sử dụng"
    )
    
    error: Optional[str] = Field(
        None, 
        description="Thông báo lỗi nếu có"
    )
    
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Thời gian tạo"
    )


class SlideGenerationTaskRequest(BaseModel):
    """Request model cho Celery task slide generation"""

    lesson_id: str = Field(
        ...,
        description="ID của bài học cần tạo slide"
    )

    template_id: str = Field(
        ...,
        description="ID của Google Slides template"
    )

    config_prompt: Optional[str] = Field(
        None,
        description="Prompt cấu hình tùy chỉnh cho LLM"
    )

    presentation_title: Optional[str] = Field(
        None,
        description="Tiêu đề tùy chỉnh cho presentation"
    )


# Models cho JSON Template Processing
class SlideElement(BaseModel):
    """Model cho element trong slide"""

    id: str = Field(..., description="ID của element")
    type: str = Field(..., description="Loại element (text, image, etc.)")
    x: int = Field(..., description="Vị trí X")
    y: int = Field(..., description="Vị trí Y")
    width: int = Field(..., description="Chiều rộng")
    height: int = Field(..., description="Chiều cao")
    text: str = Field(..., description="Nội dung text")
    style: Dict[str, Any] = Field(..., description="Style của element")


class SlideTemplate(BaseModel):
    """Model cho slide template"""

    id: str = Field(..., description="ID của slide")
    title: str = Field(..., description="Tiêu đề slide")
    elements: List[SlideElement] = Field(..., description="Danh sách elements")
    isVisible: bool = Field(True, description="Slide có hiển thị không")
    background: str = Field("#ffffff", description="Màu nền slide")


class JsonTemplateRequest(BaseModel):
    """Request model cho JSON template processing"""

    lesson_id: str = Field(
        ...,
        description="ID của bài học cần tạo slide"
    )

    slides: List[Dict[str, Any]] = Field(
        ...,
        description="Danh sách slides đã được phân tích sẵn với description"
    )

    config_prompt: Optional[str] = Field(
        None,
        description="Prompt cấu hình tùy chỉnh cho LLM"
    )


class JsonTemplateResponse(BaseModel):
    """Response model cho JSON template processing"""

    success: bool = Field(..., description="Trạng thái thành công")
    lesson_id: Optional[str] = Field(None, description="ID bài học")
    processed_template: Optional[Dict[str, Any]] = Field(None, description="Template đã xử lý")
    slides_created: Optional[int] = Field(None, description="Số slide đã tạo")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có")
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Thời gian tạo"
    )


class SlideGenerationTaskResponse(BaseModel):
    """Response model cho Celery task slide generation"""
    
    task_id: str = Field(description="ID của Celery task")
    status: str = Field(description="Trạng thái task (PENDING, PROGRESS, SUCCESS, FAILURE)")
    message: str = Field(description="Thông báo trạng thái")


class SlideGenerationProgress(BaseModel):
    """Model cho progress tracking của slide generation"""
    
    step: str = Field(description="Bước hiện tại")
    progress: int = Field(description="Phần trăm hoàn thành (0-100)")
    message: str = Field(description="Thông báo chi tiết")
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Thông tin chi tiết bổ sung"
    )


class SlideGenerationError(BaseModel):
    """Model cho error response"""
    
    success: bool = Field(default=False, description="Luôn là False cho error")
    error_code: str = Field(description="Mã lỗi")
    error_message: str = Field(description="Thông báo lỗi")
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Chi tiết lỗi"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Thời gian xảy ra lỗi"
    )


# Models cho API lấy thông tin Google Slides
class SlideInfoRequest(BaseModel):
    """Request model cho API lấy thông tin Google Slides"""

    presentation_id: str = Field(
        ...,
        description="ID của Google Slides presentation cần lấy thông tin",
        example="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
    )

    @validator('presentation_id')
    def validate_presentation_id(cls, v):
        """Validate presentation_id format"""
        if not v or len(v.strip()) == 0:
            raise ValueError('presentation_id cannot be empty')

        # Nếu là URL đầy đủ, extract ID
        if 'docs.google.com/presentation/d/' in v:
            try:
                parts = v.split('/d/')
                if len(parts) > 1:
                    id_part = parts[1].split('/')[0]
                    return id_part
            except:
                pass

        return v.strip()


class SlideElementInfo(BaseModel):
    """Thông tin chi tiết về một element trong slide"""

    objectId: str = Field(description="ID của element")
    type: str = Field(description="Loại element (shape, image, table, etc.)")
    text: Optional[str] = Field(None, description="Nội dung text (nếu có)")
    position: Optional[Dict[str, Any]] = Field(None, description="Vị trí của element")
    size: Optional[Dict[str, Any]] = Field(None, description="Kích thước của element")
    transform: Optional[Dict[str, Any]] = Field(None, description="Thông tin transform (translate, scale, shear)")
    properties: Optional[Dict[str, Any]] = Field(None, description="Các thuộc tính và style của element")


class SlideInfo(BaseModel):
    """Thông tin chi tiết về một slide"""

    slide_id: str = Field(description="ID của slide")
    slide_index: int = Field(description="Vị trí của slide (0-based)")
    layout: Optional[str] = Field(None, description="Layout của slide")
    elements: List[SlideElementInfo] = Field(default_factory=list, description="Danh sách các elements trong slide")
    properties: Optional[Dict[str, Any]] = Field(None, description="Các thuộc tính khác của slide")


class SlideInfoResponse(BaseModel):
    """Response model cho API lấy thông tin Google Slides"""

    success: bool = Field(description="Trạng thái thành công")
    presentation_id: Optional[str] = Field(None, description="ID của presentation")
    title: Optional[str] = Field(None, description="Tiêu đề của presentation")
    slide_count: Optional[int] = Field(None, description="Số lượng slides")
    slides: Optional[List[SlideInfo]] = Field(None, description="Thông tin chi tiết về các slides")
    web_view_link: Optional[str] = Field(None, description="Link để xem/chỉnh sửa presentation")
    created_time: Optional[str] = Field(None, description="Thời gian tạo presentation")
    last_modified_time: Optional[str] = Field(None, description="Thời gian chỉnh sửa cuối cùng")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có")


# Error codes constants
class SlideGenerationErrorCodes:
    """Constants cho error codes"""

    LESSON_NOT_FOUND = "LESSON_NOT_FOUND"
    TEMPLATE_NOT_ACCESSIBLE = "TEMPLATE_NOT_ACCESSIBLE"
    TEMPLATE_ANALYSIS_FAILED = "TEMPLATE_ANALYSIS_FAILED"
    LLM_GENERATION_FAILED = "LLM_GENERATION_FAILED"
    SLIDES_CREATION_FAILED = "SLIDES_CREATION_FAILED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INVALID_REQUEST = "INVALID_REQUEST"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
