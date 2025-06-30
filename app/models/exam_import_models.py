"""
Models cho chức năng import đề thi từ file DOCX
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class ExamImportRequest(BaseModel):
    """Request model cho import đề thi từ DOCX"""
    file_content: str = Field(..., description="Nội dung file DOCX đã được extract")
    filename: Optional[str] = Field(None, description="Tên file gốc")
    additional_instructions: Optional[str] = Field(None, description="Hướng dẫn bổ sung cho LLM")


class QuestionOption(BaseModel):
    """Model cho đáp án trắc nghiệm"""
    A: str
    B: str
    C: str
    D: str


class TrueFalseStatement(BaseModel):
    """Model cho câu đúng/sai"""
    text: str
    answer: bool


class TrueFalseStatements(BaseModel):
    """Model cho tập hợp câu đúng/sai"""
    a: TrueFalseStatement
    b: TrueFalseStatement
    c: TrueFalseStatement
    d: TrueFalseStatement


class MultipleChoiceQuestion(BaseModel):
    """Model cho câu hỏi trắc nghiệm nhiều lựa chọn"""
    id: int
    question: str
    options: QuestionOption
    answer: str


class TrueFalseQuestion(BaseModel):
    """Model cho câu hỏi đúng/sai"""
    id: int
    question: str
    statements: TrueFalseStatements


class ShortAnswerQuestion(BaseModel):
    """Model cho câu hỏi trả lời ngắn"""
    id: int
    question: str
    answer: str


class ExamPart(BaseModel):
    """Model cho một phần của đề thi"""
    part: str = Field(..., description="Tên phần (Phần I, Phần II, Phần III)")
    title: str = Field(..., description="Tiêu đề phần")
    description: str = Field(..., description="Mô tả phần")
    questions: List[Union[MultipleChoiceQuestion, TrueFalseQuestion, ShortAnswerQuestion]]


class ImportedExamData(BaseModel):
    """Model cho dữ liệu đề thi đã import"""
    subject: str = Field(..., description="Môn học")
    grade: int = Field(..., description="Lớp")
    duration_minutes: int = Field(..., description="Thời gian làm bài (phút)")
    school: str = Field(..., description="Tên trường")
    exam_code: Optional[str] = Field(None, description="Mã đề thi")
    atomic_masses: Optional[str] = Field(None, description="Bảng nguyên tử khối (cho môn Hóa)")
    parts: List[ExamPart]


class ExamImportResponse(BaseModel):
    """Response model cho import đề thi"""
    success: bool
    message: str
    data: Optional[ImportedExamData] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    imported_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExamImportError(BaseModel):
    """Error model cho import đề thi"""
    success: bool = False
    message: str
    error: str
    error_code: str
    details: Dict[str, Any] = Field(default_factory=dict)
    imported_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExamImportStatistics(BaseModel):
    """Statistics cho đề thi đã import"""
    total_questions: int
    part_1_questions: int = Field(0, description="Số câu trắc nghiệm nhiều lựa chọn")
    part_2_questions: int = Field(0, description="Số câu đúng/sai")
    part_3_questions: int = Field(0, description="Số câu trả lời ngắn")
    has_atomic_masses: bool = Field(False, description="Có bảng nguyên tử khối không")
    processing_quality: float = Field(0.0, description="Chất lượng xử lý (0-1)")
