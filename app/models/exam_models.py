"""
Pydantic models cho chức năng tạo bài kiểm tra từ ma trận đề thi
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


class MucDoModel(BaseModel):
    """Model cho mức độ nhận thức trong ma trận đề thi"""
    loai: Literal["Nhận biết", "Thông hiểu", "Vận dụng", "Vận dụng cao"] = Field(
        ..., description="Mức độ nhận thức theo Bloom's taxonomy"
    )
    so_cau: int = Field(..., ge=1, description="Số câu hỏi cho mức độ này")
    loai_cau: List[Literal["TN", "DT", "DS", "TL"]] = Field(
        ..., description="Các loại câu hỏi: TN=Trắc nghiệm, DT=Điền từ, DS=Đúng/Sai, TL=Tự luận"
    )

    @validator('loai_cau')
    def validate_loai_cau(cls, v):
        if not v:
            raise ValueError("Phải có ít nhất một loại câu hỏi")
        valid_types = ["TN", "DT", "DS", "TL"]
        for loai in v:
            if loai not in valid_types:
                raise ValueError(f"Loại câu hỏi không hợp lệ: {loai}. Chỉ chấp nhận: {valid_types}")
        return v


class NoiDungModel(BaseModel):
    """Model cho nội dung cụ thể trong bài học"""
    ten_noi_dung: str = Field(..., description="Tên nội dung cụ thể")
    yeu_cau_can_dat: str = Field(..., description="Yêu cầu cần đạt cho nội dung này")
    muc_do: List[MucDoModel] = Field(..., description="Các mức độ nhận thức cho nội dung này")

    @validator('muc_do')
    def validate_muc_do(cls, v):
        if not v:
            raise ValueError("Phải có ít nhất một mức độ nhận thức")
        return v


class CauHinhDeModel(BaseModel):
    """Model cho cấu hình đề thi theo từng bài"""
    bai: str = Field(..., description="Tên bài học")
    so_cau: int = Field(..., ge=1, description="Tổng số câu hỏi cho bài này")
    noi_dung: List[NoiDungModel] = Field(..., description="Các nội dung cụ thể trong bài")

    @validator('noi_dung')
    def validate_noi_dung(cls, v):
        if not v:
            raise ValueError("Phải có ít nhất một nội dung")
        return v

    @validator('so_cau')
    def validate_so_cau_consistency(cls, v, values):
        """Kiểm tra tổng số câu có khớp với tổng số câu trong các nội dung không"""
        if 'noi_dung' in values:
            total_cau = sum(
                sum(muc_do.so_cau for muc_do in nd.muc_do) 
                for nd in values['noi_dung']
            )
            if total_cau != v:
                raise ValueError(f"Tổng số câu ({v}) không khớp với tổng số câu trong nội dung ({total_cau})")
        return v


class ExamMatrixRequest(BaseModel):
    """Model cho request tạo bài kiểm tra từ ma trận đề thi"""
    lesson_id: str = Field(..., description="ID của bài học cần tạo đề thi")
    mon_hoc: str = Field(..., description="Tên môn học")
    lop: int = Field(..., ge=1, le=12, description="Lớp học (1-12)")
    tong_so_cau: int = Field(..., ge=1, description="Tổng số câu hỏi trong đề thi")
    cau_hinh_de: List[CauHinhDeModel] = Field(..., description="Cấu hình đề thi theo từng bài")

    @validator('cau_hinh_de')
    def validate_cau_hinh_de(cls, v):
        if not v:
            raise ValueError("Phải có ít nhất một cấu hình đề")
        return v

    @validator('tong_so_cau')
    def validate_tong_so_cau_consistency(cls, v, values):
        """Kiểm tra tổng số câu có khớp với tổng số câu trong cấu hình đề không"""
        if 'cau_hinh_de' in values:
            total_cau = sum(cau_hinh.so_cau for cau_hinh in values['cau_hinh_de'])
            if total_cau != v:
                raise ValueError(f"Tổng số câu ({v}) không khớp với tổng số câu trong cấu hình đề ({total_cau})")
        return v


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
    lesson_id: str = Field(..., description="ID bài học")
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
