"""
Models cho hệ thống sinh đề thi thông minh theo chuẩn THPT 2025
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime


class ObjectivesModel(BaseModel):
    """Model cho mức độ nhận thức trong từng phần"""
    Biết: int = Field(0, ge=0, description="Số câu hỏi mức độ Biết")
    Hiểu: int = Field(0, ge=0, description="Số câu hỏi mức độ Hiểu") 
    Vận_dụng: int = Field(0, ge=0, description="Số câu hỏi mức độ Vận dụng")

    class Config:
        # Cho phép sử dụng alias để map từ "Vận dụng" sang "Vận_dụng"
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "Biết": 2,
                "Hiểu": 1,
                "Vận_dụng": 0
            }
        }


class PartModel(BaseModel):
    """Model cho từng phần trong đề thi"""
    part: Literal[1, 2, 3] = Field(..., description="Số phần (1, 2, hoặc 3)")
    objectives: ObjectivesModel = Field(..., description="Phân bổ mức độ nhận thức cho phần này")

    @validator('objectives')
    def validate_objectives_not_empty(cls, v):
        # Cho phép phần có 0 câu hỏi (user có thể chỉ muốn tạo một số phần)
        # Validation tổng thể sẽ được kiểm tra ở LessonMatrixModel
        return v


class LessonMatrixModel(BaseModel):
    """Model cho ma trận của một bài học"""
    lessonId: str = Field(..., description="ID của bài học")
    totalQuestions: int = Field(..., ge=1, description="Tổng số câu hỏi cho bài học này")
    parts: List[PartModel] = Field(..., description="Phân bổ câu hỏi theo từng phần")

    @validator('parts')
    def validate_parts(cls, v):
        if not v:
            raise ValueError("Phải có ít nhất một phần")

        # Kiểm tra không có phần trùng lặp
        part_numbers = [part.part for part in v]
        if len(set(part_numbers)) != len(part_numbers):
            raise ValueError("Không được có phần trùng lặp")

        # Kiểm tra ít nhất một phần phải có câu hỏi
        total_questions = 0
        for part in v:
            part_total = part.objectives.Biết + part.objectives.Hiểu + part.objectives.Vận_dụng
            total_questions += part_total

        if total_questions == 0:
            raise ValueError("Mỗi bài học phải có tối thiểu 1 câu hỏi")

        return v

    @validator('totalQuestions')
    def validate_total_questions_match_parts(cls, v, values):
        if 'parts' in values:
            total_from_parts = 0
            for part in values['parts']:
                part_total = part.objectives.Biết + part.objectives.Hiểu + part.objectives.Vận_dụng
                total_from_parts += part_total
            
            if total_from_parts != v:
                raise ValueError(f"Tổng số câu hỏi ({v}) không khớp với tổng từ các phần ({total_from_parts})")
        return v


class SmartExamRequest(BaseModel):
    """Model cho request tạo đề thi thông minh theo chuẩn THPT 2025"""
    school: str = Field(..., description="Tên trường học")
    grade: int = Field(..., ge=1, le=12, description="Khối lớp (1-12)")
    subject: str = Field(..., description="Tên môn học")
    examTitle: str = Field(..., description="Tiêu đề đề thi (VD: Kiểm tra giữa kỳ 1)")
    duration: int = Field(..., ge=15, le=180, description="Thời gian làm bài (phút)")
    examCode: Optional[str] = Field(None, description="Mã đề thi (4 số, VD: 0335). Nếu không truyền sẽ tự động random")
    outputFormat: Literal["docx"] = Field("docx", description="Định dạng file xuất ra")
    outputLink: Literal["online"] = Field("online", description="Loại link trả về")
    bookID: Optional[str] = Field(None, description="ID của sách giáo khoa (optional). Nếu có thì chỉ tìm lessons trong collection textbook_{bookID}")
    matrix: List[LessonMatrixModel] = Field(..., description="Ma trận đề thi theo bài học")

    @validator('matrix')
    def validate_matrix_not_empty(cls, v):
        if not v:
            raise ValueError("Ma trận đề thi không được rỗng")
        return v

    @validator('examCode')
    def validate_exam_code(cls, v):
        if v is not None:
            # Kiểm tra mã đề phải là 4 số
            if not v.isdigit() or len(v) != 4:
                raise ValueError("Mã đề phải là 4 số (VD: 0335)")
        return v

    @validator('subject')
    def validate_subject(cls, v):
        valid_subjects = [
            "Toán", "Ngữ văn", "Tiếng Anh", "Vật lý", "Hóa học", "Sinh học",
            "Lịch sử", "Địa lý", "Giáo dục công dân", "Tin học", "Công nghệ"
        ]
        if v not in valid_subjects:
            # Cho phép các môn học khác nhưng log warning
            pass
        return v

    class Config:
        schema_extra = {
            "example": {
                "school": "Trường THPT ABC",
                "grade": 12,
                "subject": "Hóa học",
                "examTitle": "Kiểm tra giữa kỳ 1",
                "duration": 45,
                "examCode": "0335",
                "outputFormat": "docx",
                "outputLink": "online",
                "bookID": "hoa12",
                "matrix": [
                    {
                        "lessonId": "hoa12_bai1",
                        "totalQuestions": 7,
                        "parts": [
                            {
                                "part": 1,
                                "objectives": {
                                    "Biết": 2,
                                    "Hiểu": 0,
                                    "Vận_dụng": 0
                                }
                            },
                            {
                                "part": 2,
                                "objectives": {
                                    "Biết": 0,
                                    "Hiểu": 3,
                                    "Vận_dụng": 0
                                }
                            },
                            {
                                "part": 3,
                                "objectives": {
                                    "Biết": 0,
                                    "Hiểu": 1,
                                    "Vận_dụng": 1
                                }
                            }
                        ]
                    }
                ]
            }
        }


class SmartExamResponse(BaseModel):
    """Model cho response của API tạo đề thi thông minh"""
    success: bool = Field(..., description="Trạng thái thành công")
    exam_id: Optional[str] = Field(None, description="ID của đề thi được tạo")
    message: str = Field(..., description="Thông báo kết quả")
    online_links: Optional[Dict[str, str]] = Field(None, description="Links online để truy cập đề thi")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Thống kê về đề thi")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "exam_id": "exam_20250628_143022",
                "message": "Đề thi đã được tạo thành công",
                "online_links": {
                    "view_link": "https://docs.google.com/document/d/...",
                    "edit_link": "https://docs.google.com/document/d/.../edit"
                },
                "statistics": {
                    "total_questions": 28,
                    "part_1_questions": 18,
                    "part_2_questions": 4,
                    "part_3_questions": 6,
                    "lessons_used": 2
                }
            }
        }


class SmartExamError(BaseModel):
    """Model cho lỗi trong quá trình tạo đề thi thông minh"""
    success: bool = Field(False, description="Trạng thái thất bại")
    message: str = Field("", description="Thông báo lỗi")
    error: str = Field(..., description="Mô tả lỗi")
    error_code: Optional[str] = Field(None, description="Mã lỗi")
    details: Optional[Dict[str, Any]] = Field(None, description="Chi tiết lỗi")

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Không tìm thấy nội dung bài học",
                "error_code": "LESSON_NOT_FOUND",
                "details": {
                    "missing_lessons": ["hoa12_bai1", "hoa12_bai2"],
                    "available_lessons": ["hoa12_bai3", "hoa12_bai4"]
                }
            }
        }


class ExamStatistics(BaseModel):
    """Model cho thống kê đề thi"""
    total_questions: int = Field(..., description="Tổng số câu hỏi")
    part_1_questions: int = Field(..., description="Số câu Phần I")
    part_2_questions: int = Field(..., description="Số câu Phần II") 
    part_3_questions: int = Field(..., description="Số câu Phần III")
    lessons_used: int = Field(..., description="Số bài học được sử dụng")
    difficulty_distribution: Dict[str, int] = Field(..., description="Phân bổ theo mức độ")
    generation_time: float = Field(..., description="Thời gian tạo đề (giây)")
    created_at: str = Field(..., description="Thời gian tạo")

    class Config:
        schema_extra = {
            "example": {
                "total_questions": 28,
                "part_1_questions": 18,
                "part_2_questions": 4,
                "part_3_questions": 6,
                "lessons_used": 2,
                "difficulty_distribution": {
                    "Biết": 12,
                    "Hiểu": 10,
                    "Vận_dụng": 6
                },
                "generation_time": 45.2,
                "created_at": "2025-06-28T14:30:22"
            }
        }
