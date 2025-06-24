from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import json
import os

from app.services.lesson_plan_framework_service import lesson_plan_framework_service
from app.services.llm_service import LLMService
from app.services.docx_export_service import docx_export_service
from app.api.endpoints.pdf_endpoints import get_lesson_content_by_id

logger = logging.getLogger(__name__)
router = APIRouter()


# Helper functions
async def _create_lesson_plan_sections_prompts(framework: Dict[str, Any], user_config: List[Dict[str, Any]], lesson_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tạo danh sách các prompt cho từng phần của giáo án để gọi LLM nhiều lần

    Args:
        framework: Khung giáo án mẫu
        user_config: Cấu hình từ người dùng
        lesson_data: Thông tin bài học

    Returns:
        List[Dict]: Danh sách các prompt cho từng phần
    """

    # Chuyển đổi user_config thành text dễ đọc
    user_config_text = _format_user_config(user_config)

    # Trích xuất thông tin cụ thể từ user_config
    extracted_info = _extract_user_config_info(user_config)

    # Lấy thông tin quan trọng từ lesson_data
    lesson_info = _extract_lesson_info(lesson_data)

    # Lấy cấu trúc framework
    framework_structure = framework.get('structure', {})
    framework_name = framework.get('name', '')
    sections = framework_structure.get('sections', [])

    # Tạo prompt chung cho tất cả các phần
    common_context = f"""
Bạn là một giáo viên Việt Nam giàu kinh nghiệm, đã soạn hàng nghìn giáo án thực tế.

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

### 2. CẤM TUYỆT ĐỐI:
- KHÔNG viết "(Nếu có thông tin khác từ người dùng, hãy thay thế thông tin này)"
- KHÔNG viết "Không có thông tin" - thay bằng "..."
- KHÔNG viết bất kỳ ghi chú giải thích nào trong thông tin cá nhân
- KHÔNG thêm hướng dẫn cho người đọc
- KHÔNG TỰ TẠO thông tin khi thấy "[Để trống - không có thông tin từ người dùng]"
- KHÔNG ĐOÁN hoặc SÁNG TẠO thông tin cá nhân (tên trường, tổ, lớp, môn học...)

### 3. XỬ LÝ "[Để trống - không có thông tin từ người dùng]":
- Khi thấy text này → Viết "..." hoặc bỏ trống hoàn toàn
- TUYỆT ĐỐI KHÔNG tự tạo thông tin như "Tổ Khoa học Tự nhiên", "Lớp 10", "Hóa học"
- Chỉ sử dụng thông tin có sẵn từ người dùng

## LƯU Ý QUAN TRỌNG:
- Viết hoàn toàn bằng tiếng Việt
- Không thêm ghi chú bằng tiếng Anh
- Đảm bảo tính thực tiễn cao, có thể áp dụng ngay trong lớp học
- Viết như giáo viên chuyên nghiệp với kinh nghiệm lâu năm
"""

    # Tạo danh sách prompts cho từng phần
    section_prompts = []

    # LUÔN SỬ DỤNG SECTIONS CHI TIẾT với cấu trúc thư mục đúng
    sections = [
        {
            "name": "Thông tin chung",
            "description": "Thông tin cơ bản về bài học",
            "required": True,
            "section_type": "info"
        },
        {
            "name": "I. Mục tiêu",
            "description": "Mục tiêu bài học",
            "required": True,
            "section_type": "objectives"
        },
        {
            "name": "II. Thiết bị dạy học và học liệu",
            "description": "Tài liệu và thiết bị cần thiết",
            "required": True,
            "section_type": "materials"
        },
        {
            "name": "III. Tiến trình dạy học",
            "description": "Các hoạt động dạy học",
            "required": True,
            "section_type": "teaching_process",
            "subsections": [
                {
                    "name": "1. Hoạt động khởi động",
                    "description": "Hoạt động mở đầu bài học",
                    "section_type": "warmup"
                },
                {
                    "name": "2. Hình thành kiến thức",
                    "description": "Hoạt động hình thành kiến thức mới",
                    "section_type": "knowledge_formation"
                },
                {
                    "name": "3. Luyện tập",
                    "description": "Bài tập củng cố kiến thức",
                    "section_type": "practice"
                },
                {
                    "name": "4. Vận dụng và mở rộng",
                    "description": "Ứng dụng thực tế và mở rộng",
                    "section_type": "application"
                }
            ]
        },
        {
            "name": "Phiếu học tập chi tiết",
            "description": "Các phiếu học tập và đáp án",
            "required": True,
            "section_type": "worksheets"
        }
    ]

    for i, section in enumerate(sections):
        section_name = section.get('name', f'Phần {i+1}')
        section_description = section.get('description', '')
        section_type = section.get('section_type', 'general')
        subsections = section.get('subsections', [])

        # Nếu section có subsections, tạo prompt cho từng subsection
        if subsections:
            # Tạo prompt cho section chính (như III. TIẾN TRÌNH DẠY HỌC)
            main_prompt = _create_main_section_prompt(section_name, section_description, common_context, i+1, len(sections))
            section_prompts.append({
                "section_name": section_name,
                "section_index": i,
                "prompt": main_prompt,
                "required": section.get('required', True)
            })

            # Tạo prompt cho từng subsection
            for j, subsection in enumerate(subsections):
                sub_name = subsection.get('name', f'{section_name} - Phần {j+1}')
                sub_description = subsection.get('description', '')
                sub_type = subsection.get('section_type', 'general')

                sub_prompt = _create_specialized_prompt(sub_name, sub_description, [], common_context, lesson_data, extracted_info, f"{i+1}.{j+1}", f"{len(sections)} sections", sub_type)

                section_prompts.append({
                    "section_name": sub_name,
                    "section_index": f"{i}.{j}",
                    "prompt": sub_prompt,
                    "required": True
                })
        else:
            # Tạo prompt chuyên biệt dựa trên loại section
            section_prompt = _create_specialized_prompt(section_name, section_description, subsections, common_context, lesson_data, extracted_info, i+1, len(sections), section_type)

            section_prompts.append({
                "section_name": section_name,
                "section_index": i,
                "prompt": section_prompt,
                "required": section.get('required', True)
            })

    return section_prompts


def _extract_user_config_info(user_config: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Trích xuất thông tin cụ thể từ user_config
    """
    info = {
        "school_name": "",
        "teacher_name": "",
        "subject": "",
        "grade": "",
        "lesson_name": "",
        "time_duration": ""
    }

    try:
        for group in user_config:
            fields = group.get('fields', [])
            for field in fields:
                field_name = field.get('field_name', '')
                label = field.get('label', '').lower()

                # Mapping dựa trên label
                if 'trường' in label:
                    info['school_name'] = field_name
                elif 'giáo viên' in label or 'họ và tên' in label:
                    info['teacher_name'] = field_name
                elif 'môn học' in label:
                    info['subject'] = field_name
                elif 'lớp' in label:
                    info['grade'] = field_name
                elif 'tên bài' in label or 'bài dạy' in label:
                    info['lesson_name'] = field_name
                elif 'thời gian' in label:
                    info['time_duration'] = field_name

    except Exception as e:
        logger.error(f"Error extracting user config info: {e}")

    return info



def _create_main_section_prompt(section_name: str, section_description: str, common_context: str, section_index, total_sections) -> str:
    """Tạo prompt cho section chính có subsections"""
    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### YÊU CẦU:
- Chỉ tạo TIÊU ĐỀ CHÍNH cho section này
- Không tạo nội dung chi tiết (sẽ được tạo ở các subsections)
- Sử dụng cấu trúc thư mục đúng

### ĐỊNH DẠNG ĐẦU RA:
{section_name.upper()}

{section_description}

BẮT ĐẦU VIẾT TIÊU ĐỀ CHÍNH:"""


def _create_specialized_prompt(section_name: str, section_description: str, subsections: List, common_context: str, lesson_data: Dict[str, Any], user_info: Dict[str, str], section_index, total_sections, section_type: str = "general") -> str:
    """
    Tạo prompt chuyên biệt cho từng loại section
    """

    # Lấy thông tin bài học để tùy chỉnh prompt
    lesson_content = ""
    subject = user_info.get('subject', '')
    grade = user_info.get('grade', '')
    lesson_name = user_info.get('lesson_name', '')

    # Nếu không có từ user_config, lấy từ lesson_data
    if not subject or not grade:
        if lesson_data.get('success') and lesson_data.get('book_structure'):
            book_info = lesson_data.get('book_structure', {}).get('book_info', {})
            if not subject:
                subject = book_info.get('subject', '')
            if not grade:
                grade = book_info.get('grade', '')

    # Lấy nội dung bài học
    if lesson_data.get('success') and lesson_data.get('book_structure'):
        chapters = lesson_data.get('book_structure', {}).get('chapters', [])
        for chapter in chapters:
            lessons = chapter.get('lessons', [])
            for lesson in lessons:
                if lesson.get('lesson_id') == lesson_data.get('lesson_id'):
                    lesson_content = lesson.get('lesson_content', '')
                    if not lesson_name:
                        lesson_name = lesson.get('lesson_title', '')
                    break

    # Tạo prompt chuyên biệt dựa trên section_type hoặc tên section
    if section_type == "objectives" or "mục tiêu" in section_name.lower():
        return _create_objectives_prompt(section_name, common_context, lesson_content, subject, grade, lesson_name, section_index, total_sections)
    elif section_type == "materials" or "thiết bị" in section_name.lower() or "học liệu" in section_name.lower():
        return _create_materials_prompt(section_name, common_context, lesson_content, subject, grade, lesson_name, section_index, total_sections)
    elif section_type == "warmup" or "khởi động" in section_name.lower():
        return _create_warmup_prompt(section_name, common_context, lesson_content, subject, grade, lesson_name, section_index, total_sections)
    elif section_type == "knowledge_formation" or "hình thành kiến thức" in section_name.lower() or "kiến thức" in section_name.lower():
        return _create_knowledge_prompt(section_name, common_context, lesson_content, subject, grade, lesson_name, section_index, total_sections, subsections)
    elif section_type == "practice" or "luyện tập" in section_name.lower():
        return _create_practice_prompt(section_name, common_context, lesson_content, subject, grade, lesson_name, section_index, total_sections)
    elif section_type == "worksheets" or "phiếu học tập" in section_name.lower():
        return _create_worksheet_prompt(section_name, common_context, lesson_content, subject, grade, lesson_name, section_index, total_sections)
    elif section_type == "application" or "vận dụng" in section_name.lower():
        return _create_application_prompt(section_name, common_context, lesson_content, subject, grade, lesson_name, section_index, total_sections)
    else:
        return _create_general_prompt(section_name, section_description, subsections, common_context, section_index, total_sections)


def _create_objectives_prompt(section_name: str, common_context: str, lesson_content: str, subject: str, grade: str, lesson_name: str, section_index, total_sections) -> str:
    """Prompt chuyên biệt cho phần mục tiêu"""
    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### THÔNG TIN BÀI HỌC CỤ THỂ:
- Tên bài: {lesson_name}
- Môn học: {subject}
- Lớp: {grade}
- Nội dung: {lesson_content[:600]}...

### YÊU CẦU ĐẶC BIỆT CHO MỤC TIÊU:
- PHẢI dựa trên nội dung bài học cụ thể ở trên
- PHẢI sử dụng động từ hành động có thể đo lường được
- PHẢI phù hợp với môn {subject} lớp {grade}
- PHẢI chia rõ: kiến thức, kỹ năng, thái độ
- PHẢI cụ thể, không chung chung

### ĐỘNG TỪ HÀNH ĐỘNG CHO TỪNG MỨC ĐỘ:
**Kiến thức:** nêu được, nhận biết được, liệt kê được, định nghĩa được, phân biệt được
**Kỹ năng:** tính được, giải được, vẽ được, phân tích được, so sánh được, vận dụng được
**Thái độ:** có thái độ, rèn luyện, hình thành, phát triển

### VÍ DỤ MỤC TIÊU TỐT:
✅ "Học sinh nêu được định nghĩa đạo hàm tại một điểm"
✅ "Học sinh tính được đạo hàm của hàm số y = x² bằng định nghĩa"
✅ "Học sinh có thái độ tích cực trong việc khám phá ý nghĩa hình học của đạo hàm"

### ĐỊNH DẠNG ĐẦU RA:
## I. MỤC TIÊU

### 1. Về kiến thức
- [Mục tiêu kiến thức cụ thể 1]
- [Mục tiêu kiến thức cụ thể 2]
- [Mục tiêu kiến thức cụ thể 3]

### 2. Về kỹ năng
- [Mục tiêu kỹ năng cụ thể 1]
- [Mục tiêu kỹ năng cụ thể 2]

### 3. Về thái độ
- [Mục tiêu thái độ cụ thể 1]
- [Mục tiêu thái độ cụ thể 2]

BẮT ĐẦU VIẾT MỤC TIÊU CỤ THỂ:"""


def _create_knowledge_prompt(section_name: str, common_context: str, lesson_content: str, subject: str, grade: str, lesson_name: str, section_index, total_sections, subsections: List) -> str:
    """Prompt chuyên biệt cho phần hình thành kiến thức - CỰC KỲ CHI TIẾT"""
    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### YÊU CẦU ĐẶC BIỆT - PHẢI CỰC KỲ CHI TIẾT:
- PHẢI dựa trên nội dung bài học: {lesson_content[:800]}...
- PHẢI chia thành các hoạt động con nhỏ (2.1, 2.2, 2.3...)
- MỖI HOẠT ĐỘNG CON PHẢI có cấu trúc: a. Mục tiêu, b. Nội dung, c. Sản phẩm, d. Tổ chức thực hiện
- PHẢI có câu hỏi cụ thể với số thứ tự (Câu 1, Câu 2...)
- PHẢI có "DỰ KIẾN CÂU TRẢ LỜI" chi tiết cho từng câu hỏi
- PHẢI có bảng "HOẠT ĐỘNG CỦA GIÁO VIÊN" và "HOẠT ĐỘNG CỦA HỌC SINH"
- PHẢI có "Kiến thức trọng tâm" sau mỗi hoạt động con
- PHẢI có phiếu học tập cụ thể với nội dung thật

### MẪU CẤU TRÚC BẮT BUỘC:
## {section_name.upper()}

**2.1 Hoạt động tìm hiểu về: [Tên khái niệm đầu tiên]**
**a. Mục tiêu**
- [Mục tiêu cụ thể cho hoạt động này]
- [Mục tiêu về kỹ năng]

**b. Nội dung**
Học sinh trả lời các câu hỏi sau:

**CÂU HỎI**
Câu 1: [Câu hỏi cụ thể với nội dung thật]
Câu 2: [Câu hỏi cụ thể tiếp theo]
[Có thể có hình ảnh: "Quan sát hình X.Y SGK trang Z"]

**c. Sản phẩm**
Câu trả lời của học sinh

**DỰ KIẾN CÂU TRẢ LỜI**
Câu 1: [Đáp án chi tiết cho câu 1]
Câu 2: [Đáp án chi tiết cho câu 2]

**d. Tổ chức thực hiện**
| HOẠT ĐỘNG CỦA GIÁO VIÊN | HOẠT ĐỘNG CỦA HỌC SINH |
|-------------------------|------------------------|
| **Bước 1: Chuyển giao nhiệm vụ học tập**<br>[Mô tả cụ thể GV làm gì] | **Nhận nhiệm vụ**<br>[HS làm gì] |
| **Bước 2: Thực hiện nhiệm vụ**<br>[GV theo dõi, hỗ trợ] | [HS thực hiện cụ thể] |
| **Bước 3: Báo cáo kết quả và thảo luận**<br>[GV gọi HS trả lời] | [HS trả lời, thảo luận] |
| **Bước 4: Kết luận và nhận định**<br>[GV nhận xét, chốt kiến thức] | [HS nhận xét, ghi chép] |

**Kiến thức trọng tâm:**
- [Kiến thức cốt lõi của hoạt động này]

**2.2 Hoạt động tìm hiểu về: [Tên khái niệm thứ hai]**
[Lặp lại cấu trúc tương tự]

### VÍ DỤ CÁCH VIẾT ĐÚNG:
✅ "Câu 1: Quan sát hình 2.1 SGK trang 45, nguyên tử gồm những phần nào?"
✅ "DỰ KIẾN CÂU TRẢ LỜI: Nguyên tử gồm hạt nhân ở trung tâm và các electron chuyển động xung quanh"
✅ "Yêu cầu học sinh quan sát hình 2.1 SGK trang 45 và trả lời câu hỏi"

### TUYỆT ĐỐI KHÔNG ĐƯỢC:
❌ Viết "[Câu hỏi về cấu tạo nguyên tử]"
❌ Viết "[Đáp án dự kiến]"
❌ Bỏ trống phần "DỰ KIẾN CÂU TRẢ LỜI"
❌ Không có bảng tổ chức thực hiện

BẮT ĐẦU VIẾT HOẠT ĐỘNG HÌNH THÀNH KIẾN THỨC CỰC KỲ CHI TIẾT:"""


def _create_practice_prompt(section_name: str, common_context: str, lesson_content: str, subject: str, grade: str, lesson_name: str, section_index, total_sections) -> str:
    """Prompt chuyên biệt cho phần luyện tập"""
    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### THÔNG TIN BÀI HỌC:
- Tên bài: {lesson_name}
- Môn học: {subject}
- Lớp: {grade}

### YÊU CẦU ĐẶC BIỆT CHO LUYỆN TẬP:
- PHẢI có bài tập cụ thể với số liệu thật
- PHẢI có đáp án chi tiết với cách giải từng bước
- PHẢI phân loại: bài tập cơ bản, nâng cao
- PHẢI có trò chơi học tập hoặc hoạt động tương tác
- PHẢI phù hợp với nội dung: {lesson_content[:400]}...

### CẤU TRÚC BẮT BUỘC:
1. **Bài tập cơ bản (3-4 bài):** Áp dụng trực tiếp công thức
2. **Bài tập nâng cao (1-2 bài):** Yêu cầu phân tích, tổng hợp
3. **Trò chơi học tập:** Quiz, đúng-sai, ghép cặp
4. **Đáp án chi tiết:** Lời giải từng bước cho tất cả bài

### ĐỊNH DẠNG ĐẦU RA:
## {section_name.upper()}

**Bài tập cơ bản:**
1. [Bài tập 1 với số liệu cụ thể]
2. [Bài tập 2 với số liệu cụ thể]
3. [Bài tập 3 với số liệu cụ thể]

**Bài tập nâng cao:**
1. [Bài tập phân tích, tổng hợp]
2. [Bài tập ứng dụng]

**Trò chơi học tập:**
[Mô tả hoạt động tương tác cụ thể]

**ĐÁP ÁN CHI TIẾT:**
[Lời giải từng bước cho tất cả bài tập]

BẮT ĐẦU VIẾT BÀI TẬP CỤ THỂ:"""


def _create_worksheet_prompt(section_name: str, common_context: str, lesson_content: str, subject: str, grade: str, lesson_name: str, section_index, total_sections) -> str:
    """Prompt chuyên biệt cho phần phiếu học tập - CỰC KỲ CHI TIẾT"""
    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### YÊU CẦU ĐẶC BIỆT - PHẢI CỰC KỲ CHI TIẾT:
- PHẢI tạo NHIỀU phiếu học tập cụ thể (PHT số 1, 2, 3...)
- MỖI PHIẾU PHẢI có nội dung thật, không phải template
- PHẢI có "TRẢ LỜI PHIẾU HỌC TẬP SỐ X" chi tiết cho từng phiếu
- PHẢI dựa trên nội dung: {lesson_content[:600]}...
- PHẢI phù hợp với trình độ {grade}
- PHẢI có thể in ra sử dụng ngay trong lớp học

### MẪU CẤU TRÚC BẮT BUỘC:
## {section_name.upper()}

**PHIẾU HỌC TẬP SỐ 1: [TÊN CHỦ ĐỀ CỤ THỂ]**

[Nội dung câu hỏi cụ thể, có thể có bảng, sơ đồ]

1. [Câu hỏi cụ thể với nội dung thật]
2. [Câu hỏi tiếp theo]
3. [Bài tập tính toán nếu có]

**TRẢ LỜI PHIẾU HỌC TẬP SỐ 1**

1. [Đáp án chi tiết cho câu 1]
2. [Đáp án chi tiết cho câu 2]
3. [Đáp án chi tiết cho câu 3]

**PHIẾU HỌC TẬP SỐ 2: [CHỦ ĐỀ KHÁC]**

[Nội dung phiếu thứ 2]

**TRẢ LỜI PHIẾU HỌC TẬP SỐ 2**

[Đáp án cho phiếu thứ 2]

**PHIẾU HỌC TẬP SỐ 3: [CHỦ ĐỀ THỨ 3]**

[Nội dung phiếu thứ 3]

**TRẢ LỜI PHIẾU HỌC TẬP SỐ 3**

[Đáp án cho phiếu thứ 3]

### VÍ DỤ CÁCH VIẾT ĐÚNG:
✅ "PHIẾU HỌC TẬP SỐ 1: SỰ TÌM RA ELECTRON"
✅ "1. Màn huỳnh quang (màn phosphorus) sẽ phát sáng khi..."
✅ "TRẢ LỜI PHIẾU HỌC TẬP SỐ 1: 1. Màn huỳnh quang sẽ phát sáng, cho phép xác định vị trí..."

### TUYỆT ĐỐI KHÔNG ĐƯỢC:
❌ Viết "[Câu hỏi về định nghĩa]"
❌ Viết "[Đáp án cho tất cả câu hỏi]"
❌ Tạo phiếu học tập chung chung
❌ Không có đáp án chi tiết

BẮT ĐẦU VIẾT CÁC PHIẾU HỌC TẬP CỤ THỂ:"""


def _create_materials_prompt(section_name: str, common_context: str, lesson_content: str, subject: str, grade: str, lesson_name: str, section_index, total_sections) -> str:
    """Prompt chuyên biệt cho phần thiết bị dạy học"""
    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### YÊU CẦU ĐẶC BIỆT CHO THIẾT BỊ DẠY HỌC:
- PHẢI liệt kê cụ thể từng thiết bị, tài liệu cần thiết
- PHẢI phân chia rõ: cho giáo viên và cho học sinh
- PHẢI phù hợp với môn {subject} và nội dung bài học
- CÓ THỂ đề xuất video YouTube, website cụ thể
- PHẢI thực tế, có thể chuẩn bị được

### ĐỊNH DẠNG ĐẦU RA:
## II. THIẾT BỊ DẠY HỌC VÀ HỌC LIỆU

### 1. Cho giáo viên
- [Thiết bị cụ thể 1]
- [Thiết bị cụ thể 2]
- [Tài liệu tham khảo cụ thể]
- [Video/website hỗ trợ]

### 2. Cho học sinh
- [Sách giáo khoa và tài liệu]
- [Dụng cụ học tập cần thiết]
- [Kiến thức tiên quyết]

BẮT ĐẦU VIẾT THIẾT BỊ CỤ THỂ:"""


def _create_warmup_prompt(section_name: str, common_context: str, lesson_content: str, subject: str, grade: str, lesson_name: str, section_index, total_sections) -> str:
    """Prompt chuyên biệt cho hoạt động khởi động - CỰC KỲ CHI TIẾT"""
    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### YÊU CẦU ĐẶC BIỆT - PHẢI CỰC KỲ CHI TIẾT:
- PHẢI có cấu trúc: a. Mục tiêu, b. Nội dung, c. Sản phẩm, d. Tổ chức thực hiện
- PHẢI có câu hỏi cụ thể với số thứ tự (Câu 1, Câu 2...)
- PHẢI có "DỰ KIẾN TRẢ LỜI CÂU HỎI KHỞI ĐỘNG" chi tiết
- PHẢI có bảng "HOẠT ĐỘNG CỦA GIÁO VIÊN" và "HOẠT ĐỘNG CỦA HỌC SINH"
- PHẢI có hình ảnh cụ thể (có thể tham khảo SGK hoặc đề xuất hình ảnh)
- PHẢI liên quan đến nội dung: {lesson_content[:400]}...
- Thời gian: 5-7 phút

### MẪU CẤU TRÚC BẮT BUỘC:
## {section_name.upper()}

**a. Mục tiêu**
- [Mục tiêu cụ thể về tạo hứng thú]
- [Mục tiêu về kích thích tò mò]

**b. Nội dung**
Học sinh trả lời các câu hỏi sau:

**CÂU HỎI KHỞI ĐỘNG**

[Mô tả hình ảnh cụ thể, ví dụ:]
Hình a. [Mô tả hình ảnh 1]     Hình b. [Mô tả hình ảnh 2]

Câu 1: [Câu hỏi cụ thể liên quan hình ảnh/tình huống]
Câu 2: [Câu hỏi tiếp theo dẫn dắt]
Câu 3: [Câu hỏi kết nối với bài học]
Câu 4: [Câu hỏi mở rộng tư duy]
Câu 5: [Hoàn thành bảng K-W-L nếu phù hợp]

**c. Sản phẩm**
Câu trả lời của học sinh.

**DỰ KIẾN TRẢ LỜI CÂU HỎI KHỞI ĐỘNG**

Câu 1: [Đáp án chi tiết cho câu 1]
Câu 2: [Đáp án chi tiết cho câu 2]
Câu 3: [Đáp án chi tiết cho câu 3]
Câu 4: [Đáp án chi tiết cho câu 4]
Câu 5: [Đáp án cho câu 5 nếu có]

**d. Tổ chức thực hiện**
| HOẠT ĐỘNG CỦA GIÁO VIÊN | HOẠT ĐỘNG CỦA HỌC SINH |
|-------------------------|------------------------|
| **Bước 1: Chuyển giao nhiệm vụ học tập**<br>Yêu cầu học sinh hoạt động cá nhân trả lời câu hỏi.<br>[Mô tả cụ thể GV trình bày hình ảnh, tình huống] | **Nhận nhiệm vụ**<br>[HS quan sát, lắng nghe] |
| **Bước 2: Thực hiện nhiệm vụ**<br>Theo dõi và hỗ trợ cho HS. | Suy nghĩ và trả lời câu hỏi |
| **Bước 3: Báo cáo kết quả và thảo luận**<br>Gọi HS trả lời câu hỏi | Trả lời câu hỏi |
| **Bước 4: Kết luận và nhận định**<br>Nhận xét và dẫn dắt vào bài | [HS lắng nghe, chuẩn bị cho bài mới] |

### VÍ DỤ CÁCH VIẾT ĐÚNG:
✅ "Câu 1: Để nhìn rõ các cầu thủ trong một trận bóng đá ngoài sân vận động thì người xem có thể dùng thiết bị gì?"
✅ "DỰ KIẾN TRẢ LỜI: Để nhìn rõ các cầu thủ thì người xem có thể dùng ống nhòm."
✅ "Hình a. Cổ động viên trên sân Mỹ Đình     Hình b. Quan sát vi khuẩn bằng kính hiển vi"

BẮT ĐẦU VIẾT HOẠT ĐỘNG KHỞI ĐỘNG CỰC KỲ CHI TIẾT:"""


def _create_application_prompt(section_name: str, common_context: str, lesson_content: str, subject: str, grade: str, lesson_name: str, section_index, total_sections) -> str:
    """Prompt chuyên biệt cho phần vận dụng"""
    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### YÊU CẦU ĐẶC BIỆT CHO VẬN DỤNG:
- PHẢI có tình huống thực tế cụ thể
- PHẢI có câu hỏi mở rộng, tư duy phản biện
- PHẢI liên hệ với đời sống, khoa học, công nghệ
- PHẢI dựa trên kiến thức đã học: {lesson_content[:400]}...
- Thời gian: 3-5 phút

### ĐỊNH DẠNG ĐẦU RA:
## {section_name.upper()}

**Mục tiêu:** [Liên hệ kiến thức với thực tế]

**Nội dung:**
- Tình huống thực tế: [Tình huống cụ thể trong đời sống]
- Câu hỏi vận dụng: [2-3 câu hỏi mở rộng]
- Ứng dụng thực tiễn: [Ứng dụng trong khoa học, công nghệ]

**Sản phẩm:**
- [Sản phẩm cụ thể học sinh tạo ra]
- [Câu trả lời cho tình huống thực tế]

**Tổ chức thực hiện:** [Cách thức tổ chức cụ thể]

BẮT ĐẦU VIẾT HOẠT ĐỘNG VẬN DỤNG:"""


def _create_general_prompt(section_name: str, section_description: str, subsections: List, common_context: str, section_index, total_sections) -> str:
    """Prompt chung cho các section khác"""
    subsections_text = ', '.join([str(item) if isinstance(item, str) else item.get('name', str(item)) for item in subsections]) if subsections else 'Không có thành phần con cụ thể'

    return f"""{common_context}

## NHIỆM VỤ CỤ THỂ:
Bạn đang tạo phần "{section_name}" của giáo án. Đây là phần {section_index}/{total_sections} của toàn bộ giáo án.

### MÔ TẢ PHẦN NÀY:
{section_description}

### CÁC THÀNH PHẦN CON:
{subsections_text}

### YÊU CẦU:
- Tạo nội dung CỤ THỂ, CHI TIẾT cho phần này
- KHÔNG sử dụng template trong ngoặc vuông []
- Phù hợp với cấu trúc giáo án chuyên nghiệp

### ĐỊNH DẠNG ĐẦU RA:
## {section_name.upper()}

[Nội dung chi tiết của phần này]

BẮT ĐẦU VIẾT PHẦN "{section_name.upper()}":"""


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

### 3. VÍ DỤ ĐÚNG VÀ SAI:

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

### 4. CẤM TUYỆT ĐỐI:
- KHÔNG viết "(Nếu có thông tin khác từ người dùng, hãy thay thế thông tin này)"
- KHÔNG viết "Không có thông tin" - thay bằng "..."
- KHÔNG viết bất kỳ ghi chú giải thích nào trong thông tin cá nhân
- KHÔNG thêm hướng dẫn cho người đọc
- KHÔNG TỰ TẠO thông tin khi thấy "[Để trống - không có thông tin từ người dùng]"
- KHÔNG ĐOÁN hoặc SÁNG TẠO thông tin cá nhân (tên trường, tổ, lớp, môn học...)

### 5. XỬ LÝ "[Để trống - không có thông tin từ người dùng]":
- Khi thấy text này → Viết "..." hoặc bỏ trống hoàn toàn
- TUYỆT ĐỐI KHÔNG tự tạo thông tin như "Tổ Khoa học Tự nhiên", "Lớp 10", "Hóa học"
- Chỉ sử dụng thông tin có sẵn từ người dùng

## TIÊU CHUẨN GIÁO ÁN CHUYÊN NGHIỆP:

**YÊU CẦU QUAN TRỌNG NHẤT:**
- KHÔNG viết template hoặc hướng dẫn trong ngoặc vuông []
- PHẢI tạo nội dung CỤ THỂ, CHI TIẾT cho từng phần dựa trên nội dung bài học thực tế
- PHẢI có các phiếu học tập hoàn chỉnh với nội dung thực tế
- PHẢI có câu hỏi, bài tập, đáp án cụ thể
- PHẢI có hoạt động chi tiết cho từng bước
- PHẢI tham khảo và trích dẫn nội dung từ sách giáo khoa
- PHẢI có hình ảnh, sơ đồ, bảng biểu cụ thể từ SGK

**CÁCH VIẾT ĐÚNG:**
✅ "Câu hỏi 1: Quan sát hình 2.1 SGK trang 45, em hãy cho biết nguyên tử gồm những phần nào?"
❌ "[Câu hỏi về cấu tạo nguyên tử]"

✅ "Phiếu học tập 1: Hoàn thành bảng 2.1 SGK trang 46 về tính chất các hạt cơ bản"
❌ "[Mô tả nội dung phiếu học tập số 1]"

✅ "Sử dụng hình 3.2 SGK để giải thích quá trình quang hợp"
❌ "[Sử dụng hình ảnh minh họa]"

**TEMPLATE GIÁO ÁN LINH HOẠT CHO TẤT CẢ MÔN HỌC (LỚP 10-12):**

## I. MỤC TIÊU BÀI HỌC

**Về năng lực chung:**
- Tự chủ và tự học: Học sinh chủ động tìm hiểu nội dung bài học qua SGK, tài liệu tham khảo, thảo luận nhóm để xây dựng kiến thức một cách có hệ thống
- Giao tiếp, hợp tác: Sử dụng ngôn ngữ khoa học chính xác của môn học, thảo luận nhóm hiệu quả với vai trò cụ thể (thư ký, báo cáo viên, thành viên)
- Giải quyết vấn đề và sáng tạo: Phân tích, so sánh, tổng hợp thông tin để giải quyết các vấn đề cụ thể trong bài học

**Năng lực chuyên môn:**
- Nhận thức: Nêu được các khái niệm, định luật, công thức, quy tắc cụ thể trong bài học
- Tìm hiểu thế giới: Giải thích được các hiện tượng, thí nghiệm, ví dụ thực tế có trong bài học
- Vận dụng kiến thức: Áp dụng kiến thức để giải quyết bài tập, tình huống thực tế liên quan đến nội dung bài học

**Về phẩm chất:**
- Có thái độ tích cực, hứng thú tìm hiểu kiến thức mới
- Rèn luyện tính cẩn thận, chính xác trong học tập và làm bài
- Có ý thức học tập suốt đời, yêu thích môn học

## II. THIẾT BỊ DẠY HỌC VÀ HỌC LIỆU

**Cho giáo viên:**
- Sách giáo khoa và sách giáo viên của môn học
- Máy chiếu/TV, laptop để trình chiếu hình ảnh, video minh họa
- Bảng phụ hoặc poster minh họa các khái niệm chính trong bài
- Video/animation liên quan đến nội dung bài học (nếu có)
- Hình ảnh, sơ đồ, bảng biểu từ SGK và các nguồn tài liệu bổ trợ
- Thiết bị thí nghiệm (nếu bài học có thí nghiệm)
- Phiếu học tập, bài tập bổ sung

**Cho học sinh:**
- Sách giáo khoa của môn học (theo chương trình hiện hành)
- Vở ghi chép và dụng cụ học tập cơ bản
- Máy tính cầm tay (nếu cần thiết cho môn Toán, Lý, Hóa)
- Kiến thức tiên quyết: Các bài học trước đó có liên quan
- Tài liệu tham khảo bổ sung (nếu có)

## III. TIẾN TRÌNH DẠY HỌC

### Hoạt động 1: Khởi động (5-7 phút)
**Mục tiêu:** Tạo hứng thú học tập, kích thích tò mò về nội dung bài học mới
**Nội dung:**
- Đặt câu hỏi mở đầu liên quan đến kinh nghiệm sống của học sinh
- Sử dụng tình huống thực tế, hiện tượng thú vị để dẫn dắt vào bài
- Ôn tập kiến thức cũ có liên quan (nếu cần)
- Giới thiệu mục tiêu và nội dung chính của bài học
**Sản phẩm:** Học sinh có hứng thú, sẵn sàng tiếp nhận kiến thức mới
**Tổ chức thực hiện:**
- GV: Đặt câu hỏi, tạo tình huống, ghi các ý kiến lên bảng
- HS: Suy nghĩ, thảo luận, phát biểu ý kiến

### Hoạt động 2: Hình thành kiến thức mới (25-30 phút)

**YÊU CẦU ĐẶC BIỆT CHO PHẦN NÀY:**
- PHẢI chia thành các hoạt động con cụ thể (2.1, 2.2, 2.3...)
- PHẢI có nội dung kiến thức chi tiết từ SGK
- PHẢI trích dẫn cụ thể hình ảnh, bảng biểu, công thức từ SGK
- PHẢI có câu hỏi dẫn dắt cụ thể cho từng phần kiến thức
- PHẢI có phiếu học tập với nội dung thực tế
- PHẢI có đáp án chi tiết cho các câu hỏi và bài tập

**2.1 [Tên khái niệm/kiến thức đầu tiên] (X phút)**
**Mục tiêu:** Học sinh nắm được khái niệm/kiến thức cụ thể này
**Nội dung:**
- Trình bày kiến thức chi tiết từ SGK (định nghĩa, công thức, quy tắc...)
- Sử dụng hình ảnh/sơ đồ cụ thể từ SGK (ví dụ: "Quan sát hình X.Y SGK trang Z")
- Đưa ra ví dụ minh họa cụ thể
- Câu hỏi dẫn dắt từ dễ đến khó
- Phiếu học tập với bài tập cụ thể
**Sản phẩm:** Học sinh hiểu và vận dụng được kiến thức này
**Tổ chức:** [Cá nhân/nhóm/cả lớp] trong [X] phút

**2.2 [Tên khái niệm/kiến thức thứ hai] (Y phút)**
**Mục tiêu:** Học sinh nắm được mối liên hệ giữa các kiến thức
**Nội dung:**
- Liên hệ với kiến thức đã học ở phần 2.1
- Trình bày nội dung mới với ví dụ cụ thể
- Sử dụng bảng so sánh, phân loại (nếu có trong SGK)
- Thực hiện thí nghiệm/quan sát (nếu có)
- Câu hỏi kiểm tra hiểu bài
**Sản phẩm:** Bảng tổng hợp, sơ đồ, kết quả thí nghiệm
**Tổ chức:** [Phương pháp cụ thể] trong [Y] phút

**[Tiếp tục với 2.3, 2.4... tùy theo nội dung bài học]**

### Hoạt động 3: Luyện tập (8-12 phút)
**Mục tiêu:** Củng cố và vận dụng kiến thức đã học
**Nội dung:**
- Bài tập cơ bản: Áp dụng trực tiếp công thức, định nghĩa
- Bài tập nâng cao: Yêu cầu phân tích, so sánh, tổng hợp
- Trò chơi học tập: Quiz, đúng-sai, ghép cặp...
- Chữa bài tập với lời giải chi tiết
**Sản phẩm:**
- Lời giải chi tiết cho từng bài tập
- Học sinh nắm vững kiến thức và biết cách vận dụng
**Tổ chức:** Cá nhân → nhóm → cả lớp

### Hoạt động 4: Vận dụng (3-5 phút)
**Mục tiêu:** Liên hệ kiến thức với thực tế, mở rộng tư duy
**Nội dung:**
- Tình huống thực tế liên quan đến bài học
- Câu hỏi mở rộng, tư duy phản biện
- Ứng dụng kiến thức vào đời sống, khoa học, công nghệ
- Dự đoán, giải thích hiện tượng
**Sản phẩm:**
- Sơ đồ tư duy tóm tắt bài học
- Câu trả lời cho tình huống thực tế
**Tổ chức:** Thảo luận nhóm, trình bày kết quả

### Hoạt động 5: Củng cố và dặn dò (2-3 phút)
**Mục tiêu:** Tóm tắt kiến thức, hướng dẫn học tập tiếp theo
**Nội dung:**
- Tóm tắt những kiến thức trọng tâm của bài
- Nhấn mạnh các công thức, quy tắc quan trọng
- Giao nhiệm vụ về nhà: bài tập SGK, chuẩn bị bài mới
- Hướng dẫn phương pháp học tập hiệu quả
**Sản phẩm:** Học sinh nắm được kiến thức cốt lõi và nhiệm vụ về nhà
**Tổ chức:** Giáo viên tóm tắt, học sinh ghi chép

## IV. PHỤ LỤC - PHIẾU HỌC TẬP VÀ BÀI TẬP

**YÊU CẦU QUAN TRỌNG:**
- PHẢI tạo phiếu học tập cụ thể dựa trên nội dung bài học thực tế
- PHẢI có đáp án chi tiết cho tất cả câu hỏi và bài tập
- PHẢI phù hợp với trình độ học sinh (lớp 10/11/12)
- PHẢI liên quan trực tiếp đến kiến thức trong SGK

**PHIẾU HỌC TẬP 1: [TÊN CHỦ ĐỀ CHÍNH CỦA BÀI]**
Họ tên: ........................ Lớp: ......... Ngày: .........

**Phần A: Kiến thức cơ bản**
1. [Câu hỏi về định nghĩa, khái niệm cơ bản]
2. [Câu hỏi điền vào chỗ trống dựa trên SGK]
3. [Câu hỏi trắc nghiệm về nội dung chính]

**Phần B: Vận dụng**
1. [Bài tập áp dụng công thức, quy tắc]
2. [Bài tập phân tích, so sánh]
3. [Bài tập tình huống thực tế]

**PHIẾU HỌC TẬP 2: [CHỦ ĐỀ PHỤ HOẶC NÂNG CAO]**
**Dành cho hoạt động nhóm**

1. [Bài tập yêu cầu thảo luận nhóm]
2. [Dự án nhỏ, điều tra, khảo sát]
3. [Bài tập sáng tạo, mở rộng]

**ĐÁP ÁN CHI TIẾT:**

**Phiếu học tập 1:**
1. [Đáp án câu 1 với giải thích]
2. [Đáp án câu 2 với cách làm]
3. [Đáp án câu 3 với lý do chọn]

**Phiếu học tập 2:**
1. [Hướng dẫn giải và đáp án mẫu]
2. [Tiêu chí đánh giá dự án]
3. [Gợi ý và định hướng làm bài]

**STYLE VIẾT:** Như giáo viên chuyên nghiệp, ngôn ngữ khoa học chính xác, phù hợp học sinh lớp 10.

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

    # Kiểm tra cấu trúc dữ liệu từ get_lesson_content_by_id
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

Số trang: {', '.join(map(str, lesson_data.get('lesson_pages', []))) if lesson_data.get('lesson_pages') else 'Không xác định'}
"""

    return lesson_info


async def _generate_lesson_plan_by_sections(framework: Dict[str, Any], user_config: List[Dict[str, Any]], lesson_data: Dict[str, Any], llm_service) -> str:
    """
    Tạo giáo án bằng cách chia nhỏ thành nhiều phần và gọi LLM nhiều lần

    Args:
        framework: Khung giáo án mẫu
        user_config: Cấu hình từ người dùng
        lesson_data: Thông tin bài học
        llm_service: Service LLM để gọi API

    Returns:
        str: Nội dung giáo án hoàn chỉnh được ghép từ các phần
    """
    try:
        # 1. Tạo danh sách prompts cho từng phần
        section_prompts = await _create_lesson_plan_sections_prompts(framework, user_config, lesson_data)

        # 2. Tạo phần tiêu đề và thông tin chung trước
        lesson_info = _extract_lesson_info(lesson_data)
        framework_name = framework.get('name', '')

        # Lấy thông tin bài học để tạo tiêu đề
        lesson_title = ""
        subject = ""
        grade = ""

        if lesson_data.get('success') and lesson_data.get('book_structure'):
            book_info = lesson_data.get('book_structure', {}).get('book_info', {})
            chapters = lesson_data.get('book_structure', {}).get('chapters', [])

            for chapter in chapters:
                lessons = chapter.get('lessons', [])
                for lesson in lessons:
                    if lesson.get('lesson_id') == lesson_data.get('lesson_id'):
                        lesson_title = lesson.get('lesson_title', '')
                        break

            subject = book_info.get('subject', '')
            grade = book_info.get('grade', '')

        # 3. Tạo header cho giáo án
        header = f"""# GIÁO ÁN {subject.upper() if subject else 'MÔN HỌC'}

**Tên bài:** {lesson_title}
**Khung giáo án:** {framework_name}
**Lớp:** {grade}
**Ngày tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M')}

---

"""

        # 4. Gọi LLM cho từng phần
        generated_sections = []

        logger.info(f"Generating lesson plan with {len(section_prompts)} sections")

        for section_info in section_prompts:
            section_name = section_info['section_name']
            section_prompt = section_info['prompt']
            section_index = section_info['section_index']

            logger.info(f"Generating section {section_index}/{len(section_prompts)}: {section_name}")

            try:
                # Gọi LLM cho phần này
                response = llm_service.model.generate_content(section_prompt)
                section_content = response.text

                # Thêm vào danh sách
                generated_sections.append({
                    'name': section_name,
                    'content': section_content,
                    'index': section_index
                })

                logger.info(f"Successfully generated section: {section_name}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error generating section {section_name}: {e}")

                # Nếu lỗi quota, raise exception để trả về lỗi cho user
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.error(f"Quota exceeded for section {section_name}")
                    raise HTTPException(
                        status_code=429,
                        detail=f"Đã hết quota API. Vui lòng thử lại sau hoặc nâng cấp gói dịch vụ. Chi tiết: {error_msg}"
                    )
                else:
                    # Lỗi khác, cũng raise exception
                    raise HTTPException(
                        status_code=500,
                        detail=f"Lỗi khi tạo nội dung cho phần '{section_name}': {error_msg}"
                    )

        # 5. Ghép tất cả các phần lại thành giáo án hoàn chỉnh
        full_lesson_plan = header

        # Sắp xếp theo thứ tự index (xử lý cả string và int)
        def sort_key(x):
            index = x['index']
            if isinstance(index, str):
                # Nếu là string như "3.0", chuyển thành tuple (3, 0) để sort
                parts = index.split('.')
                return tuple(int(p) for p in parts)
            else:
                # Nếu là int, chuyển thành tuple (index,)
                return (index,)

        generated_sections.sort(key=sort_key)

        for section in generated_sections:
            full_lesson_plan += section['content'] + "\n\n"

        # 6. Thêm phần kết thúc
        full_lesson_plan += """---

## GHI CHÚ
- Giáo án được tạo tự động bởi hệ thống AI
- Giáo viên có thể điều chỉnh nội dung cho phù hợp với tình hình thực tế lớp học
- Thời gian các hoạt động có thể linh hoạt tùy theo khả năng tiếp thu của học sinh

## ĐÁNH GIÁ
- [ ] Đã chuẩn bị đầy đủ tài liệu, thiết bị dạy học
- [ ] Đã xem xét và điều chỉnh nội dung phù hợp với học sinh
- [ ] Đã chuẩn bị các câu hỏi dự phòng và bài tập bổ sung
- [ ] Đã lên kế hoạch xử lý các tình huống có thể xảy ra trong giờ học

---
*Giáo án được tạo bởi PlanBook AI - Hệ thống hỗ trợ soạn giáo án thông minh*
"""

        logger.info(f"Successfully generated complete lesson plan with {len(generated_sections)} sections")
        return full_lesson_plan

    except Exception as e:
        logger.error(f"Error in _generate_lesson_plan_by_sections: {e}")
        raise


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
    lesson_id: str  # ID của bài học để lấy thông tin


class LessonPlanResponse(BaseModel):
    lesson_plan_id: str
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
        result = await lesson_plan_framework_service.process_framework_file(
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


@router.post("/lesson-plan-generate")
async def generate_lesson_plan(request: LessonPlanGenerateRequest):
    """
    Tạo giáo án chi tiết dựa trên framework, cấu hình người dùng và thông tin bài học
    Trả về file DOCX để download

    Args:
        request: Chứa framework_id, user_config và lesson_id

    Returns:
        FileResponse: File DOCX chứa giáo án đã được format đẹp
    """
    try:
        # 1. Lấy framework template
        framework = await lesson_plan_framework_service.get_framework_by_id(request.framework_id)
        if not framework:
            raise HTTPException(status_code=404, detail="Không tìm thấy framework")

        # 2. Lấy thông tin bài học
        lesson_data = None
        try:
            logger.info(f"Attempting to get lesson content for: {request.lesson_id}")
            lesson_data = await get_lesson_content_by_id(request.lesson_id)
            logger.info(f"Successfully retrieved lesson data: {lesson_data.get('success', False)}")
        except Exception as e:
            logger.warning(f"Failed to get real lesson data: {e}")

        if not lesson_data or not lesson_data.get('success'):
            # Tạo lesson giả với thông tin cơ bản để test
            logger.info("Creating mock lesson data for testing")
            lesson_data = {
                "success": True,
                "lesson_id": request.lesson_id,
                "book_structure": {
                    "book_info": {
                        "title": "Sách giáo khoa mẫu",
                        "subject": "Toán học",
                        "grade": "Lớp 10",
                        "total_pages": 200
                    },
                    "chapters": [
                        {
                            "chapter_title": "Chương 1: Khái niệm cơ bản",
                            "lessons": [
                                {
                                    "lesson_id": request.lesson_id,
                                    "lesson_title": f"Bài học {request.lesson_id}",
                                    "lesson_content": f"Đây là nội dung chi tiết của bài học {request.lesson_id}. Bài học này giới thiệu các khái niệm cơ bản và phương pháp giải quyết vấn đề. Học sinh sẽ được học về lý thuyết, thực hành và ứng dụng vào thực tế. Nội dung bao gồm: định nghĩa, tính chất, ví dụ minh họa và bài tập thực hành."
                                }
                            ]
                        }
                    ]
                }
            }

        # 3. Khởi tạo LLM service
        llm_service = LLMService()
        if not llm_service.is_available():
            raise HTTPException(status_code=503, detail="LLM service không khả dụng")

        # 4. Tạo giáo án bằng cách chia nhỏ thành nhiều phần
        logger.info("Generating lesson plan using multi-section approach")
        generated_content = await _generate_lesson_plan_by_sections(
            framework, request.user_config, lesson_data, llm_service
        )

        # 6. Tạo dữ liệu giáo án để xuất DOCX
        lesson_plan_id = f"plan_{request.lesson_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        lesson_plan_data = {
            "lesson_plan_id": lesson_plan_id,
            "content": {
                "generated_plan": generated_content,
                "framework_used": framework["name"],
                "lesson_info": {
                    "lesson_id": request.lesson_id,
                    "lesson_title": lesson_data.get("lesson_title", ""),
                },
                "user_config": request.user_config
            },
            "framework_used": request.framework_id,
            "created_at": datetime.now().isoformat(),
        }

        # 7. Tạo file DOCX
        filepath = docx_export_service.create_lesson_plan_docx(lesson_plan_data)

        # 8. Kiểm tra file có tồn tại không
        if not os.path.exists(filepath):
            raise HTTPException(status_code=500, detail="Không thể tạo file DOCX")

        # 9. Tạo tên file download
        filename = os.path.basename(filepath)

        # 10. Trả về file DOCX
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating lesson plan: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo giáo án: {str(e)}")


@router.post("/lesson-plan-generate-json", response_model=LessonPlanResponse)
async def generate_lesson_plan_json(request: LessonPlanGenerateRequest):
    """
    Tạo giáo án chi tiết và trả về JSON (cho backward compatibility)

    Args:
        request: Chứa framework_id, user_config và lesson_id

    Returns:
        LessonPlanResponse: Giáo án đã được tạo bởi AI (JSON format)
    """
    try:
        # 1. Lấy framework template
        framework = await lesson_plan_framework_service.get_framework_by_id(request.framework_id)
        if not framework:
            raise HTTPException(status_code=404, detail="Không tìm thấy framework")

        # 2. Lấy thông tin bài học
        lesson_data = None
        try:
            logger.info(f"Attempting to get lesson content for: {request.lesson_id}")
            lesson_data = await get_lesson_content_by_id(request.lesson_id)
            logger.info(f"Successfully retrieved lesson data: {lesson_data.get('success', False)}")
        except Exception as e:
            logger.warning(f"Failed to get real lesson data: {e}")

        if not lesson_data or not lesson_data.get('success'):
            # Tạo lesson giả với thông tin cơ bản để test
            logger.info("Creating mock lesson data for testing")
            lesson_data = {
                "success": True,
                "lesson_id": request.lesson_id,
                "book_structure": {
                    "book_info": {
                        "title": "Sách giáo khoa mẫu",
                        "subject": "Toán học",
                        "grade": "Lớp 10",
                        "total_pages": 200
                    },
                    "chapters": [
                        {
                            "chapter_title": "Chương 1: Khái niệm cơ bản",
                            "lessons": [
                                {
                                    "lesson_id": request.lesson_id,
                                    "lesson_title": f"Bài học {request.lesson_id}",
                                    "lesson_content": f"Đây là nội dung chi tiết của bài học {request.lesson_id}. Bài học này giới thiệu các khái niệm cơ bản và phương pháp giải quyết vấn đề. Học sinh sẽ được học về lý thuyết, thực hành và ứng dụng vào thực tế. Nội dung bao gồm: định nghĩa, tính chất, ví dụ minh họa và bài tập thực hành."
                                }
                            ]
                        }
                    ]
                }
            }

        # 3. Khởi tạo LLM service
        llm_service = LLMService()
        if not llm_service.is_available():
            raise HTTPException(status_code=503, detail="LLM service không khả dụng")

        # 4. Tạo giáo án bằng cách chia nhỏ thành nhiều phần
        logger.info("Generating lesson plan JSON using multi-section approach")
        generated_content = await _generate_lesson_plan_by_sections(
            framework, request.user_config, lesson_data, llm_service
        )

        # 6. Tạo response JSON
        lesson_plan_id = f"plan_{request.lesson_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return LessonPlanResponse(
            lesson_plan_id=lesson_plan_id,
            content={
                "generated_plan": generated_content,
                "framework_used": framework["name"],
                "lesson_info": {
                    "lesson_id": request.lesson_id,
                    "lesson_title": lesson_data.get("lesson_title", ""),
                },
                "user_config": request.user_config
            },
            framework_used=request.framework_id,
            created_at=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating lesson plan JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo giáo án: {str(e)}")


@router.post("/lesson-plan-generate-old", response_model=LessonPlanResponse)
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
        frameworks = await lesson_plan_framework_service.get_all_frameworks()
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
        framework = await lesson_plan_framework_service.get_framework_by_id(
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
        success = await lesson_plan_framework_service.delete_framework(framework_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy framework")
        return {"message": "Framework đã được xóa thành công"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa framework: {str(e)}")


@router.post("/frameworks/seed")
async def seed_sample_frameworks():
    """
    Tạo dữ liệu mẫu cho frameworks (dev only)
    """
    try:
        await lesson_plan_framework_service._ensure_initialized()
        
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
        if lesson_plan_framework_service.frameworks_collection is not None:
            for framework in sample_frameworks:
                # Check if already exists by name
                existing = await lesson_plan_framework_service.frameworks_collection.find_one(
                    {"name": framework["name"], "status": "active"}
                )
                if not existing:
                    await lesson_plan_framework_service.frameworks_collection.insert_one(framework)
                    
        return {
            "message": "Sample frameworks seeded successfully",
            "count": len(sample_frameworks)
        }
        
    except Exception as e:
        logger.error(f"Error seeding frameworks: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi seed frameworks: {str(e)}")


@router.post("/lesson-plan-generate-sections-test")
async def generate_lesson_plan_sections_test(request: LessonPlanGenerateRequest):
    """
    Test endpoint để kiểm tra việc tạo giáo án theo từng phần
    Trả về JSON với thông tin chi tiết về từng phần được tạo

    Args:
        request: Chứa framework_id, user_config và lesson_id

    Returns:
        Dict: Thông tin chi tiết về quá trình tạo từng phần
    """
    try:
        # 1. Lấy framework template
        framework = await lesson_plan_framework_service.get_framework_by_id(request.framework_id)
        if not framework:
            raise HTTPException(status_code=404, detail="Không tìm thấy framework")

        # 2. Lấy thông tin bài học (mock data để test)
        lesson_data = {
            "success": True,
            "lesson_id": request.lesson_id,
            "book_structure": {
                "book_info": {
                    "title": "Sách giáo khoa mẫu",
                    "subject": "Hóa học",
                    "grade": "Lớp 10",
                    "total_pages": 200
                },
                "chapters": [
                    {
                        "chapter_title": "Chương 2: Thành phần của nguyên tử",
                        "lessons": [
                            {
                                "lesson_id": request.lesson_id,
                                "lesson_title": "THÀNH PHẦN CỦA NGUYÊN TỬ",
                                "lesson_content": "Nguyên tử được cấu tạo từ các hạt cơ bản: proton, neutron và electron. Proton và neutron tạo thành hạt nhân nguyên tử, electron chuyển động xung quanh hạt nhân. Số proton quyết định tính chất hóa học của nguyên tố."
                            }
                        ]
                    }
                ]
            }
        }

        # 3. Tạo danh sách prompts cho từng phần
        section_prompts = await _create_lesson_plan_sections_prompts(framework, request.user_config, lesson_data)

        # 4. Trả về thông tin test
        return {
            "message": "Test tạo giáo án theo từng phần",
            "framework_name": framework.get('name', ''),
            "lesson_id": request.lesson_id,
            "total_sections": len(section_prompts),
            "sections": [
                {
                    "index": section['section_index'],
                    "name": section['section_name'],
                    "required": section['required'],
                    "prompt_length": len(section['prompt'])
                }
                for section in section_prompts
            ],
            "sample_prompt": section_prompts[0]['prompt'][:500] + "..." if section_prompts else "No prompts generated"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sections test: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi test: {str(e)}")


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


@router.get("/lesson-plan-export-docx/{lesson_plan_id}")
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
