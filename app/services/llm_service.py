"""
LLM Service để cấu trúc lại text bằng Gemini API
"""
import logging
from typing import Dict, Any, Optional
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service sử dụng Gemini API để cấu trúc lại text
    """
    
    def __init__(self):
        self.model = None
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini API"""
        try:
            if not settings.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not set. LLM features will be disabled.")
                return
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            # Use the latest available model
            self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
            logger.info("Gemini API initialized with gemini-1.5-flash-latest")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            self.model = None
    
    async def format_cv_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Cấu trúc lại text CV thành format đẹp
        
        Args:
            raw_text: Text thô từ PDF
            
        Returns:
            Dict chứa text đã được cấu trúc lại
        """
        try:
            if not self.model:
                return {
                    "success": False,
                    "error": "LLM service not available. Please set GEMINI_API_KEY.",
                    "formatted_text": raw_text
                }
            
            prompt = f"""
Bạn là một chuyên gia HR và CV formatting chuyên nghiệp. Hãy cấu trúc lại CV sau thành format CHUẨN QUỐC TẾ, đẹp mắt và chuyên nghiệp với LAYOUT ĐẸP.

YÊU CẦU FORMAT CHUYÊN NGHIỆP:

1. **HEADER LAYOUT ĐẸP:**
   ```
   # TÊN ĐẦY ĐỦ
   ## Chức danh chuyên nghiệp

   📧 Email | 📱 Phone | 🔗 LinkedIn | 💻 GitHub
   📍 Location
   ```

2. **CẤU TRÚC CHUẨN:**
   - Professional Summary (2-3 câu ngắn gọn, highlight value)
   - Core Skills (dạng grid 3-4 cột, không quá dài 1 hàng)
   - Professional Experience (format đẹp với icons)
   - Education (compact, highlight achievements)
   - Key Projects (3-4 dự án top, format card-style)
   - Technical Skills (phân loại rõ ràng theo nhóm)

3. **FORMATTING RULES ĐẸP:**
   - Sử dụng icons/emojis cho sections: 💼 🎓 🚀 ⚡
   - Skills dạng grid: `Skill1` `Skill2` `Skill3` (3-4 per line)
   - Dates format: MM/YYYY - MM/YYYY
   - Bullet points với action verbs mạnh
   - Spacing đều đặn, không quá dài 1 hàng
   - Contact info trên nhiều dòng cho đẹp

4. **LAYOUT OPTIMIZATION:**
   - Header: Tên to, chức danh nhỏ hơn, contact info xuống dòng
   - Skills: Chia thành 3-4 skills per line
   - Experience: Company | Position | Duration (xuống dòng)
   - Projects: Title trên, description và tech stack xuống dòng
   - Không để text quá dài 1 hàng

5. **PROFESSIONAL CONTENT:**
   - Quantify achievements với số liệu
   - Action verbs: Developed, Implemented, Led, Achieved
   - Focus vào impact và results
   - Technical skills phân nhóm: Frontend | Backend | Database | Tools

Text CV cần format:
{raw_text}

Hãy trả về CV với LAYOUT ĐẸP, không dài 1 hàng, format chuyên nghiệp:
"""
            
            response = self.model.generate_content(prompt)
            formatted_text = response.text
            
            return {
                "success": True,
                "formatted_text": formatted_text,
                "original_length": len(raw_text),
                "formatted_length": len(formatted_text)
            }
            
        except Exception as e:
            logger.error(f"Error formatting CV text: {e}")
            return {
                "success": False,
                "error": str(e),
                "formatted_text": raw_text
            }
    
    async def format_document_text(self, raw_text: str, document_type: str = "general") -> Dict[str, Any]:
        """
        Cấu trúc lại text tài liệu thành format đẹp
        
        Args:
            raw_text: Text thô từ PDF
            document_type: Loại tài liệu (cv, report, letter, etc.)
            
        Returns:
            Dict chứa text đã được cấu trúc lại
        """
        try:
            if not self.model:
                return {
                    "success": False,
                    "error": "LLM service not available. Please set GEMINI_API_KEY.",
                    "formatted_text": raw_text
                }
            
            # Tùy chỉnh prompt theo loại tài liệu
            if document_type.lower() == "cv":
                return await self.format_cv_text(raw_text)

            # Nếu là general nhưng có vẻ như CV, format như CV
            if document_type.lower() == "general" and self._is_cv_content(raw_text):
                return await self.format_cv_text(raw_text)

            # Xử lý trường hợp đặc biệt khi text extraction thất bại
            if raw_text.startswith("[PDF_EXTRACTION_FAILED]") or raw_text.startswith("[PDF_PROCESSING_INFO]"):
                prompt = f"""
Bạn là một chuyên gia hỗ trợ xử lý tài liệu. Người dùng đã gặp vấn đề khi xử lý file PDF.

Thông tin lỗi:
{raw_text}

Hãy tạo một thông báo hữu ích và chuyên nghiệp bằng tiếng Việt để:
1. Giải thích vấn đề đã xảy ra
2. Đưa ra các giải pháp khả thi
3. Hướng dẫn cách khắc phục
4. Sử dụng markdown formatting để dễ đọc

Trả về thông báo hỗ trợ:
"""
            else:
                prompt = f"""
Bạn là một chuyên gia về formatting và editing tài liệu. Hãy cấu trúc lại tài liệu sau thành format đẹp, dễ đọc và chuyên nghiệp.

Yêu cầu:
1. Sắp xếp thông tin theo thứ tự logic
2. Định dạng rõ ràng với tiêu đề và cấu trúc phân cấp
3. Sửa lỗi chính tả và ngữ pháp
4. Giữ nguyên tất cả thông tin quan trọng
5. Sử dụng markdown formatting
6. Loại bỏ các ký tự lạ và format lỗi

Loại tài liệu: {document_type}

Text cần format:
{raw_text}

Hãy trả về tài liệu đã được format đẹp:
"""
            
            response = self.model.generate_content(prompt)
            formatted_text = response.text
            
            return {
                "success": True,
                "formatted_text": formatted_text,
                "original_length": len(raw_text),
                "formatted_length": len(formatted_text),
                "document_type": document_type
            }
            
        except Exception as e:
            logger.error(f"Error formatting document text: {e}")
            return {
                "success": False,
                "error": str(e),
                "formatted_text": raw_text
            }
    
    def _is_cv_content(self, text: str) -> bool:
        """Check if text content looks like a CV/Resume"""
        cv_keywords = [
            'curriculum vitae', 'cv', 'resume', 'experience', 'education', 'skills',
            'objective', 'work experience', 'employment', 'projects', 'achievements',
            'qualifications', 'developer', 'engineer', 'manager', 'analyst',
            'university', 'college', 'degree', 'gpa', 'graduation',
            'github', 'linkedin', 'portfolio', 'email', 'phone',
            'kinh nghiem', 'hoc van', 'ky nang', 'du an', 'thanh tich',
            'bang cap', 'tot nghiep', 'cong ty', 'lam viec'
        ]

        text_lower = text.lower()
        keyword_count = sum(1 for keyword in cv_keywords if keyword in text_lower)

        # If more than 3 CV-related keywords found, likely a CV
        return keyword_count >= 3

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.model is not None

# Global instance
llm_service = LLMService()
