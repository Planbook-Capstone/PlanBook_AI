from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import json
import os

from app.services.lesson_plan_framework_service import get_lesson_plan_framework_service
from app.services.llm_service import LLMService
from app.services.docx_export_service import docx_export_service
from app.services.docx_upload_service import docx_upload_service
from app.models.online_document_models import OnlineDocumentResponse


logger = logging.getLogger(__name__)
router = APIRouter()


# Helper functions
async def _create_lesson_plan_prompt(framework: Dict[str, Any], user_config: List[Dict[str, Any]], lesson_data: Dict[str, Any]) -> str:
    """
    Tạo prompt chi tiết để gửi cho LLM

    Args:
        framework: Khung giáo án mẫu
        user_config: Cấu hình từ người dùng
        lesson_data: Thông tin bài học

    Returns:
        str: Prompt đầy đủ cho LLM
    """

    # Chuyển đổi user_config thành text dễ đọc
    user_config_text = _format_user_config(user_config)

    # Lấy thông tin quan trọng từ lesson_data
    lesson_info = _extract_lesson_info(lesson_data)

    # Tạo prompt chuyên nghiệp như giáo viên thực tế
    framework_structure = framework.get('structure', {})
    framework_name = framework.get('name', '')

    prompt = f"""
Bạn là một giáo viên Việt Nam giàu kinh nghiệm, đã soạn hàng nghìn giáo án thực tế. Hãy tạo một giáo án CHI TIẾT, THỰC TIỄN và CHUYÊN NGHIỆP như những giáo án mà các giáo viên giỏi thực sự sử dụng trong lớp học.

## KHUNG GIÁO ÁN: {framework_name}
{json.dumps(framework_structure, ensure_ascii=False, indent=2)}

## THÔNG TIN BÀI HỌC:
{lesson_info}

## THÔNG TIN CỤ THỂ TỪ GIÁO VIÊN:
{user_config_text}

## QUY TẮC XỬ LÝ THÔNG TIN NGHIÊM NGẶT:

### 1. THÔNG TIN CÁ NHÂN - KHÔNG ĐƯỢC THÊM TEXT GIẢI THÍCH:
- **CÓ THÔNG TIN**: Viết trực tiếp giá trị, KHÔNG thêm ghi chú
  * VÍ DỤ: "Trường: FPT University" (KHÔNG viết "Trường: FPT University (Nếu có thông tin khác...)")
  * VÍ DỤ: "Giáo viên: Hong Thinh Thinh" (KHÔNG thêm bất kỳ ghi chú nào)

- **KHÔNG CÓ THÔNG TIN**: Viết "..." hoặc bỏ trống
  * VÍ DỤ: "Tổ: ..." hoặc "Tổ: "
  * TUYỆT ĐỐI KHÔNG viết "(Nếu có thông tin khác từ người dùng...)"

### 2. DANH SÁCH THÔNG TIN CÁ NHÂN:
- Trường, Tổ, Họ và tên giáo viên, Tên bài dạy, Môn học/Hoạt động giáo dục, Lớp, Thời gian thực hiện

### 3. NỘI DUNG GIÁO ÁN:
- Có thể sáng tạo dựa trên thông tin đã cho
- Sử dụng thông tin từ "Về kiến thức", "Về năng lực", "Về phẩm chất"

### 4. VÍ DỤ ĐÚNG VÀ SAI:

**✅ ĐÚNG:**
```
I. Thông tin chung
• Trường: FPT University
• Tổ: ...
• Giáo viên: Hong Thinh Thinh
• Lớp: ...
```

**❌ SAI:**
```
I. Thông tin chung
• Trường: FPT University (Nếu có thông tin khác từ người dùng, hãy thay thế thông tin này)
• Tổ: Không có thông tin
• Giáo viên: Hong Thinh Thinh
• Lớp: 10 (Nếu có thông tin khác từ người dùng, hãy thay thế thông tin này)
```

### 5. CẤM TUYỆT ĐỐI:
- KHÔNG viết "(Nếu có thông tin khác từ người dùng, hãy thay thế thông tin này)"
- KHÔNG viết "Không có thông tin" - thay bằng "..."
- KHÔNG viết bất kỳ ghi chú giải thích nào trong thông tin cá nhân
- KHÔNG thêm hướng dẫn cho người đọc
- KHÔNG TỰ TẠO thông tin khi thấy "[Để trống - không có thông tin từ người dùng]"
- KHÔNG ĐOÁN hoặc SÁNG TẠO thông tin cá nhân (tên trường, tổ, lớp, môn học...)

### 6. XỬ LÝ "[Để trống - không có thông tin từ người dùng]":
- Khi thấy text này → Viết "..." hoặc bỏ trống hoàn toàn
- TUYỆT ĐỐI KHÔNG tự tạo thông tin như "Tổ Khoa học Tự nhiên", "Lớp 10", "Hóa học"
- Chỉ sử dụng thông tin có sẵn từ người dùng

## TIÊU CHUẨN GIÁO ÁN CHUYÊN NGHIỆP:

### 1. MỤC TIÊU PHẢI CỤ THỂ, ĐO LƯỜNG ĐƯỢC:
- Không viết chung chung như "học sinh hiểu được"
- Viết cụ thể: "Sau bài học, học sinh làm đúng ít nhất 8/10 bài tập về..."
- Mục tiêu phải liên kết trực tiếp với nội dung bài học

### 2. HOẠT ĐỘNG DẠY HỌC PHẢI THỰC TẾ:
- Ghi rõ thời gian từng hoạt động (VD: 5 phút, 15 phút)
- Mô tả CỤ THỂ những gì giáo viên nói, làm
- Mô tả CỤ THỂ những gì học sinh làm
- Có câu hỏi cụ thể, bài tập cụ thể
- Có cách xử lý khi học sinh không hiểu

### 3. NỘI DUNG PHẢI CHÍNH XÁC:
- Dựa chính xác vào nội dung bài học đã cung cấp
- Có ví dụ minh họa cụ thể
- Có bài tập thực hành với đáp án
- Liên hệ với thực tế cuộc sống

### 4. ĐÁNH GIÁ PHẢI CỤ THỂ:
- Nêu rõ tiêu chí đánh giá
- Có rubric đánh giá (nếu cần)
- Cách thức kiểm tra hiểu bài

### 5. CHUẨN BỊ PHẢI CHI TIẾT:
- Liệt kê cụ thể từng vật dụng cần thiết
- Chuẩn bị trước những gì
- Dự phòng khi thiết bị hỏng

### 6. NGÔN NGỮ GIÁO VIÊN THỰC TẾ:
- Viết câu nói cụ thể của giáo viên: "Các em hãy mở SGK trang 45..."
- Ghi rõ cách gọi học sinh: "Gọi em A lên bảng", "Em B nhận xét"
- Mô tả hành động: "Giáo viên đi quanh lớp kiểm tra"
- Dự kiến phản ứng học sinh và cách xử lý

### 7. BÀI TẬP VÀ VÍ DỤ CỤ THỂ:
- Đưa ra số liệu cụ thể trong ví dụ
- Có đáp án chi tiết
- Phân loại bài tập theo mức độ (dễ → khó)
- Chuẩn bị bài tập dự phòng cho học sinh giỏi

## YÊU CẦU VIẾT GIÁO ÁN:

1. **VIẾT NHU GIÁO VIÊN THỰC TẾ**: Dùng ngôn ngữ giáo viên, không học thuật quá
2. **CHI TIẾT CỤ THỂ**: Giáo viên khác đọc là có thể dạy ngay được
3. **THỜI GIAN CHÍNH XÁC**: Chia thời gian hợp lý cho từng hoạt động
4. **CÂU HỎI CỤ THỂ**: Viết sẵn những câu hỏi giáo viên sẽ hỏi
5. **XỬ LÝ TÌNH HUỐNG**: Dự kiến khó khăn và cách giải quyết
6. **LIÊN HỆ THỰC TẾ**: Đưa ra ví dụ từ cuộc sống hàng ngày
7. **PHƯƠNG PHÁP ĐA DẠNG**: Kết hợp nhiều phương pháp dạy học

## VÍ DỤ CÁCH VIẾT CHI TIẾT:

**Thay vì viết:** "Giáo viên giới thiệu bài mới"
**Hãy viết:** "Giáo viên nói: 'Hôm nay chúng ta sẽ học về... Trước tiên, các em hãy quan sát hình ảnh này (chỉ vào bảng phụ) và cho cô biết các em thấy gì?' Gọi 3-4 em trả lời, ghi lại ý kiến trên bảng."

**Thay vì viết:** "Học sinh làm bài tập"
**Hãy viết:** "Học sinh làm bài tập 1 trang 45 SGK trong 8 phút. Giáo viên đi kiểm tra, nhắc nhở em nào chưa làm. Gọi em A lên bảng trình bày, các em khác nhận xét. Nếu sai, gọi em B sửa lại."

## LƯU Ý QUAN TRỌNG:
- Viết hoàn toàn bằng tiếng Việt
- Không thêm ghi chú bằng tiếng Anh
- Kết thúc giáo án bằng phần "Ghi chú" và "Đánh giá"
- Đảm bảo tính thực tiễn cao, có thể áp dụng ngay trong lớp học

BẮT ĐẦU VIẾT GIÁO ÁN CHUYÊN NGHIỆP:
"""

    return prompt


def _format_user_config(user_config: List[Dict[str, Any]]) -> str:
    """Chuyển đổi user_config thành text dễ đọc với xử lý cấu trúc phức tạp"""
    if not user_config:
        return "Không có cấu hình cụ thể từ người dùng. Hãy tạo giáo án với thông tin cơ bản."

    formatted_text = ""

    def format_field(field: Dict[str, Any], indent_level: int = 0) -> str:
        """Hàm đệ quy để format field ở bất kỳ level nào"""
        indent = "  " * indent_level
        field_text = ""

        field_name = field.get('field_name', '')
        label = field.get('label', '')
        default_value = field.get('default_value', '')
        data_type = field.get('data_type', '')

        # Danh sách các field thông tin cá nhân cần để trống nếu không có
        personal_info_fields = [
            'school_name', 'group', 'teacher_name', 'topic_title',
            'subject_activity', 'grade', 'duration'
        ]

        # Hiển thị thông tin field
        if label:
            field_text += f"{indent}- {label}: "

            # Xử lý thông tin dựa trên label để xác định loại field
            is_personal_info = label in [
                'Trường', 'Tổ', 'Họ và tên giáo viên', 'Tên bài dạy',
                'Môn học/Hoạt động giáo dục', 'Lớp', 'Thời gian thực hiện'
            ]

            # Lấy giá trị thực tế
            actual_value = ""
            if default_value and str(default_value).strip():
                # Có default_value
                actual_value = str(default_value).strip()
            elif is_personal_info and field_name and field_name not in personal_info_fields:
                # Với thông tin cá nhân, field_name có thể chứa giá trị thực tế
                # Chỉ sử dụng field_name làm giá trị nếu nó KHÔNG phải tên field chuẩn
                standard_field_names = ['school_name', 'group', 'teacher_name', 'topic_title', 'subject_activity', 'grade', 'duration']
                if field_name not in standard_field_names:
                    actual_value = field_name

            if actual_value:
                field_text += f"{actual_value}\n"
            else:
                # Xử lý khác nhau cho thông tin cá nhân và nội dung giáo án
                if is_personal_info:
                    field_text += "[Để trống - không có thông tin từ người dùng]\n"
                else:
                    field_text += "[Sử dụng thông tin mặc định phù hợp]\n"

        # Xử lý nested fields
        nested_fields = field.get('fields', [])
        if nested_fields:
            for nested_field in nested_fields:
                field_text += format_field(nested_field, indent_level + 1)

        return field_text

    for group in user_config:
        group_name = group.get('group_name', '')
        formatted_text += f"\n### {group_name}\n"

        fields = group.get('fields', [])
        if not fields:
            formatted_text += "- Không có thông tin cụ thể\n"
            continue

        for field in fields:
            formatted_text += format_field(field)

    return formatted_text


def _extract_lesson_info(lesson_data: Dict[str, Any]) -> str:
    """Trích xuất thông tin quan trọng từ lesson_data"""

    if lesson_data.get('success') and lesson_data.get('book_structure'):
        # Dữ liệu từ Qdrant
        book_structure = lesson_data.get('book_structure', {})
        book_info = book_structure.get('book_info', {})
        chapters = book_structure.get('chapters', [])

        # Tìm lesson trong chapters
        lesson_content = ""
        chapter_title = ""
        lesson_title = ""

        for chapter in chapters:
            lessons = chapter.get('lessons', [])
            for lesson in lessons:
                if lesson.get('lesson_id') == lesson_data.get('lesson_id'):
                    lesson_title = lesson.get('lesson_title', '')
                    lesson_content = lesson.get('lesson_content', '')
                    chapter_title = chapter.get('chapter_title', '')
                    break

        lesson_info = f"""
ID bài học: {lesson_data.get('lesson_id', '')}
Tên bài học: {lesson_title}
Môn học: {book_info.get('subject', '')}
Lớp: {book_info.get('grade', '')}

Thông tin chương:
- Tên chương: {chapter_title}

Nội dung bài học chi tiết:
{lesson_content[:3000] if lesson_content else 'Không có nội dung chi tiết'}...

Tổng số trang sách: {book_info.get('total_pages', 0)}
"""
    else:
        # Dữ liệu giả hoặc cấu trúc khác
        lesson_info = f"""
ID bài học: {lesson_data.get('lesson_id', '')}
Tên bài học: {lesson_data.get('lesson_title', 'Không có tiêu đề')}
Nội dung bài học:
{lesson_data.get('lesson_content', 'Không có nội dung')[:2000]}...

Thông tin chương:
- Tên chương: {lesson_data.get('chapter_info', {}).get('chapter_title', '')}
- Vị trí: Chương {lesson_data.get('position_in_book', {}).get('chapter_number', '')}, Bài {lesson_data.get('position_in_book', {}).get('lesson_number', '')}

Số trang: {', '.join(map(str, lesson_data.get('lesson_pages', [])))}
"""

    return lesson_info


# Pydantic models for request/response
class LessonPlanRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    duration: int  # minutes
    learning_objectives: List[str]
    materials_needed: List[str]
    student_level: str
    special_requirements: Optional[str] = None
    framework_id: Optional[str] = None  # ID của khung giáo án đã upload (sử dụng _id của MongoDB)


class LessonPlanGenerateRequest(BaseModel):
    framework_id: str  # ID của khung giáo án mẫu
    user_config: List[Dict[str, Any]]  # JSON cấu hình từ người dùng
    lesson_id: Any # ID của bài học để lấy thông tin


class LessonPlanResponse(BaseModel):
    lesson_plan_id: Any
    content: dict
    framework_used: str
    created_at: str


@router.post("/lesson-plan-framework")
async def upload_lesson_plan_framework(
    framework_name: str = Form(...), framework_file: UploadFile = File(...)
):
    """
    Upload khung giáo án template
    """
    try:
        # Validate file type
        if not framework_file.filename or not framework_file.filename.endswith(
            (".pdf", ".docx", ".doc", ".txt")
        ):
            raise HTTPException(
                status_code=400, detail="Chỉ hỗ trợ file PDF, Word hoặc Text"
            )

        # Process framework file using service
        framework_service = get_lesson_plan_framework_service()
        result = await framework_service.process_framework_file(
            framework_name, framework_file
        )

        return {
            "message": "Khung giáo án đã được upload thành công",
            "framework_name": result["name"],
            "framework_id": result["id"],  # Sử dụng id từ MongoDB _id
            "filename": result["filename"],
            "structure": result["structure"],
            "created_at": result["created_at"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi upload khung giáo án: {str(e)}"
        )


@router.post("/upload-docx-to-online", response_model=Dict[str, Any])
async def upload_docx_to_online_document(
    file: UploadFile = File(..., description="File DOCX cần chuyển thành online document"),
    convert_to_google_docs: bool = Form(True, description="Có convert thành Google Docs không (default: True)")
):
    """
    Upload file DOCX offline và chuyển thành file DOCX online

    Endpoint này nhận file DOCX từ client và:
    1. Validate file type và size
    2. Lưu file tạm thời
    3. Upload lên Google Drive
    4. Tạo các link truy cập online
    5. Xóa file tạm và trả về thông tin online document

    Args:
        file: File DOCX được upload từ client
        convert_to_google_docs: Có convert thành Google Docs để edit online không

    Returns:
        OnlineDocumentResponse: Thông tin online document với các link truy cập
    """
    try:
        logger.info(f"Processing DOCX upload: {file.filename}")

        # Xử lý upload file thông qua service
        result = await docx_upload_service.process_docx_upload_to_online(
            uploaded_file=file,
            convert_to_google_docs=convert_to_google_docs
        )

        # Kiểm tra kết quả
        if not result.get("success", False):
            error_code = result.get("error_code", "UNKNOWN_ERROR")
            error_message = result.get("error", "Lỗi không xác định")

            # Map error codes to HTTP status codes
            status_code_map = {
                "INVALID_FILE_TYPE": 400,
                "FILE_TOO_LARGE": 413,
                "TEMP_FILE_ERROR": 500,
                "UPLOAD_FAILED": 503,
                "PROCESSING_ERROR": 500,
                "RESPONSE_ERROR": 500
            }

            status_code = status_code_map.get(error_code, 500)

            raise HTTPException(
                status_code=status_code,
                detail=f"[{error_code}] {error_message}"
            )

        logger.info(f"Successfully processed DOCX upload: {file.filename}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_docx_to_online_document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi không mong muốn: {str(e)}"
        )




# @router.post("/lesson-plan-generate-old", response_model=LessonPlanResponse)
async def generate_lesson_plan_old(request: LessonPlanRequest):
    """
    [DEPRECATED] Tạo giáo án dựa trên dữ liệu người dùng và khung giáo án (phiên bản cũ)

    ⚠️ API này đã deprecated. Sử dụng /lesson-plan-generate với LessonPlanGenerateRequest thay thế.
    """
    try:
        # Validate required fields
        if not request.subject or not request.topic:
            raise HTTPException(status_code=400, detail="Môn học và chủ đề là bắt buộc")

        # Mock response for backward compatibility
        generated_plan = {
            "title": f"Giáo án {request.subject} - {request.topic}",
            "grade": request.grade,
            "duration": request.duration,
            "objectives": request.learning_objectives,
            "materials": request.materials_needed,
            "activities": [
                {
                    "phase": "Khởi động",
                    "duration": 10,
                    "content": "Hoạt động khởi động...",
                },
                {
                    "phase": "Phát triển",
                    "duration": request.duration - 20,
                    "content": "Nội dung chính...",
                },
                {"phase": "Củng cố", "duration": 10, "content": "Hoạt động củng cố..."},
            ],
        }

        return LessonPlanResponse(
            lesson_plan_id="plan_123",
            content=generated_plan,
            framework_used=request.framework_id or "default",
            created_at="2025-06-18T10:00:00Z",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo giáo án: {str(e)}")


@router.get("/frameworks")
async def get_available_frameworks():
    """
    Lấy danh sách các khung giáo án có sẵn
    """
    try:
        logger.info("Getting all frameworks...")
        framework_service = get_lesson_plan_framework_service()
        frameworks = await framework_service.get_all_frameworks()
        logger.info(f"Found {len(frameworks)} frameworks")
        return {"frameworks": frameworks}
    except Exception as e:
        logger.error(f"Error getting frameworks: {e}")
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi lấy danh sách frameworks: {str(e)}"
        )


@router.get("/frameworks/{framework_id}")
async def get_framework_by_id(framework_id: str):
    """
    Lấy thông tin chi tiết của một framework
    """
    try:
        framework_service = get_lesson_plan_framework_service()
        framework = await framework_service.get_framework_by_id(
            framework_id
        )
        if not framework:
            raise HTTPException(status_code=404, detail="Không tìm thấy framework")
        return framework
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy framework: {str(e)}")


@router.delete("/frameworks/{framework_id}")
async def delete_framework(framework_id: str):
    """
    Xóa một framework
    """
    try:
        framework_service = get_lesson_plan_framework_service()
        success = await framework_service.delete_framework(framework_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy framework")
        return {"message": "Framework đã được xóa thành công"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa framework: {str(e)}")


# @router.post("/frameworks/seed")
async def seed_sample_frameworks():
    """
    Tạo dữ liệu mẫu cho frameworks (dev only)
    """
    try:
        framework_service = get_lesson_plan_framework_service()
        await framework_service._ensure_initialized()

        # Sample frameworks data (không cần framework_id)
        sample_frameworks = [
            {
                "name": "Khung giáo án 5E",
                "filename": "5E_framework.txt",
                "original_text": "Khung giáo án 5E bao gồm 5 giai đoạn: Engage, Explore, Explain, Elaborate, Evaluate",
                "structure": {
                    "phases": [
                        {"name": "Engage", "description": "Tạo hứng thú, kích thích học sinh"},
                        {"name": "Explore", "description": "Khám phá, tìm hiểu"},
                        {"name": "Explain", "description": "Giải thích, trình bày"},
                        {"name": "Elaborate", "description": "Mở rộng, vận dụng"},
                        {"name": "Evaluate", "description": "Đánh giá, kiểm tra"}
                    ]
                },
                "created_at": datetime.now(datetime.timezone.utc),
                "updated_at": datetime.now(datetime.timezone.utc),
                "status": "active"
            },
            {
                "name": "Khung giáo án truyền thống",
                "filename": "traditional_framework.txt",
                "original_text": "Khung giáo án truyền thống bao gồm: Kiểm tra bài cũ, Bài mới, Củng cố, Dặn dò",
                "structure": {
                    "phases": [
                        {"name": "Kiểm tra bài cũ", "description": "Ôn tập kiến thức đã học"},
                        {"name": "Bài mới", "description": "Trình bày nội dung bài học"},
                        {"name": "Củng cố", "description": "Tóm tắt, khắc sâu kiến thức"},
                        {"name": "Dặn dò", "description": "Giao bài tập về nhà"}
                    ]
                },
                "created_at": datetime.now(datetime.timezone.utc),
                "updated_at": datetime.now(datetime.timezone.utc),
                "status": "active"
            }
        ]
          # Insert sample data
        if framework_service.frameworks_collection is not None:
            for framework in sample_frameworks:
                # Check if already exists by name
                existing = await framework_service.frameworks_collection.find_one(
                    {"name": framework["name"], "status": "active"}
                )
                if not existing:
                    await framework_service.frameworks_collection.insert_one(framework)
                    
        return {
            "message": "Sample frameworks seeded successfully",
            "count": len(sample_frameworks)
        }
        
    except Exception as e:
        logger.error(f"Error seeding frameworks: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi seed frameworks: {str(e)}")


@router.post("/lesson-plan-export-docx")
async def export_lesson_plan_to_docx(lesson_plan_data: dict):
    """
    Xuất giáo án ra file DOCX với format đẹp

    Args:
        lesson_plan_data: Dữ liệu giáo án từ API generate

    Returns:
        FileResponse: File DOCX để download
    """
    try:
        # Tạo file DOCX
        filepath = docx_export_service.create_lesson_plan_docx(lesson_plan_data)

        # Kiểm tra file có tồn tại không
        if not os.path.exists(filepath):
            raise HTTPException(status_code=500, detail="Không thể tạo file DOCX")

        # Tạo tên file download
        filename = os.path.basename(filepath)

        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Error exporting to DOCX: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xuất file DOCX: {str(e)}")


# @router.get("/lesson-plan-export-docx/{lesson_plan_id}")
async def export_lesson_plan_by_id_to_docx(lesson_plan_id: str):
    """
    Xuất giáo án ra file DOCX theo ID (nếu có lưu trữ)

    Args:
        lesson_plan_id: ID của giáo án

    Returns:
        FileResponse: File DOCX để download
    """
    try:
        # TODO: Implement logic to retrieve lesson plan by ID from database
        # For now, return error message
        raise HTTPException(
            status_code=501,
            detail="Chức năng này chưa được implement. Vui lòng sử dụng POST endpoint với dữ liệu giáo án."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting lesson plan by ID: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xuất file DOCX: {str(e)}")


# Pydantic models for lesson plan content generation
class LessonPlanContentRequest(BaseModel):
    lesson_plan_json: Dict[str, Any]
    lesson_id: Optional[Any] = None
    user_id: Optional[str] = None
    book_id: Optional[str] = Field(None, description="ID của sách giáo khoa (optional). Nếu có thì chỉ tìm lesson content trong collection textbook_{book_id}")
    tool_log_id:Optional[Any] = None

class LessonPlanContentResponse(BaseModel):
    success: bool
    lesson_plan: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None


@router.post("/generate-lesson-plan-content")
async def generate_lesson_plan_content(request: LessonPlanContentRequest):
    """
    Sinh nội dung chi tiết cho giáo án từ cấu trúc JSON với Celery task

    Args:
        request: Request chứa JSON giáo án, lesson_id (optional), và book_id (optional)

    Returns:
        Dict chứa task_id để theo dõi tiến độ

    Example:
        POST /api/v1/lesson-plan/generate-lesson-plan-content
        {
            "lesson_plan_json": {...},
            "lesson_id": "hoa12_bai1",
            "book_id": "hoa12",
            "user_id": "user123"
        }
    """
    try:
        logger.info("Starting lesson plan content generation task...")

        # Validate đầu vào
        if not request.lesson_plan_json:
            raise HTTPException(
                status_code=400,
                detail="lesson_plan_json is required"
            )

        # Import background_task_processor
        from app.services.background_task_processor import get_background_task_processor

        # Tạo task bất đồng bộ
        background_task_processor = get_background_task_processor()
        task_id = await background_task_processor.create_lesson_plan_content_task(
            lesson_plan_json=request.lesson_plan_json,
            lesson_id=request.lesson_id,
            user_id=request.user_id,
            book_id=request.book_id
        )

        return {
            "success": True,
            "task_id": task_id,
            "status": "processing",
            "message": "Lesson plan content generation task created successfully. Use /api/v1/tasks/{task_id}/status to check progress.",
            "endpoints": {
                "check_status": f"/api/v1/tasks/{task_id}/status",
                "get_result": f"/api/v1/tasks/{task_id}/result",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating lesson plan content generation task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
