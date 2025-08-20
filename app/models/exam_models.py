"""
Pydantic models cho chức năng tạo bài kiểm tra từ ma trận đề thi
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Literal

class CauHoiModel(BaseModel):
    """Model cho một câu hỏi được tạo ra"""
    stt: int = Field(..., description="Số thứ tự câu hỏi")
    loai_cau: Literal["TN", "DT", "DS", "TL"] = Field(..., description="Loại câu hỏi")
    muc_do: str = Field(..., description="Mức độ nhận thức")
    noi_dung_cau_hoi: str = Field(..., description="Nội dung câu hỏi")
    dap_an: Optional[Dict[str, Any]] = Field(None, description="Đáp án cho câu hỏi")
    giai_thich: Optional[str] = Field(None, description="Giải thích đáp án")
    bai_hoc: str = Field(..., description="Bài học liên quan")
    noi_dung_kien_thuc: str = Field(..., description="Nội dung kiến thức liên quan")


class ExamResponse(BaseModel):
    """Model cho response khi tạo bài kiểm tra thành công"""
    exam_id: str = Field(..., description="ID của bài kiểm tra được tạo")
    ten_truong: str = Field(..., description="Tên trường học")
    mon_hoc: str = Field(..., description="Môn học")
    lop: int = Field(..., description="Lớp")
    tong_so_cau: int = Field(..., description="Tổng số câu hỏi")
    cau_hoi: List[CauHoiModel] = Field(..., description="Danh sách câu hỏi")
    thong_ke: Dict[str, Any] = Field(..., description="Thống kê về đề thi")
    created_at: str = Field(..., description="Thời gian tạo")
    docx_file_path: Optional[str] = Field(None, description="Đường dẫn file DOCX")


class ExamStatistics(BaseModel):
    """Model cho thống kê đề thi"""
    tong_so_cau: int
    phan_bo_theo_loai: Dict[str, int]  # {"TN": 5, "DT": 3, "DS": 2, "TL": 0}
    phan_bo_theo_muc_do: Dict[str, int]  # {"Nhận biết": 4, "Thông hiểu": 4, "Vận dụng": 2}
    phan_bo_theo_bai: Dict[str, int]  # {"Bài 1": 5, "Bài 2": 5}


class SearchContentRequest(BaseModel):
    """Model cho request tìm kiếm nội dung bài học"""
    lesson_id: str = Field(..., description="ID bài học")
    search_terms: List[str] = Field(..., description="Các từ khóa tìm kiếm")
    limit: int = Field(10, ge=1, le=50, description="Số lượng kết quả tối đa")


class LessonContentResponse(BaseModel):
    """Model cho response nội dung bài học"""
    lesson_id: str
    lesson_title: str
    chapter_title: str
    content_chunks: List[Dict[str, Any]]
    total_chunks: int
    search_quality: float  # Điểm chất lượng search (0-1)


class ExamGenerationError(BaseModel):
    """Model cho lỗi trong quá trình tạo đề thi"""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
