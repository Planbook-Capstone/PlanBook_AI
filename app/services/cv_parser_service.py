"""
CV Parser Service - Extract và structure dữ liệu CV thành fields
"""
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

class PersonalInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None
    position: Optional[str] = None

class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    major: Optional[str] = None
    gpa: Optional[str] = None
    duration: Optional[str] = None
    achievements: List[str] = []

class WorkExperience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    description: List[str] = []

class Project(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    tech_stack: List[str] = []
    github_links: List[str] = []
    demo_link: Optional[str] = None

class TechnicalSkills(BaseModel):
    programming_languages: List[str] = []
    frameworks: List[str] = []
    databases: List[str] = []
    tools: List[str] = []

class CVData(BaseModel):
    personal_info: PersonalInfo
    professional_summary: Optional[str] = None
    education: List[Education] = []
    work_experience: List[WorkExperience] = []
    projects: List[Project] = []
    technical_skills: TechnicalSkills
    hobbies: List[str] = []
    soft_skills: List[str] = []

class CVParserService:
    """Service để parse CV text thành structured data"""
    
    def __init__(self):
        pass
    
    async def parse_cv_to_structured_data(self, cv_text: str) -> Dict[str, Any]:
        """
        Parse CV text thành structured data với các fields cụ thể
        
        Args:
            cv_text: Raw text từ CV
            
        Returns:
            Dict chứa structured CV data
        """
        try:
            if not llm_service.is_available():
                logger.warning("LLM service not available, using basic parsing")
                return await self._basic_parse(cv_text)
            
            # Sử dụng LLM để extract structured data
            structured_data = await self._llm_parse(cv_text)
            
            return {
                "success": True,
                "cv_data": structured_data,
                "parsing_method": "llm_enhanced"
            }
            
        except Exception as e:
            logger.error(f"Error parsing CV: {e}")
            return {
                "success": False,
                "error": str(e),
                "cv_data": None
            }
    
    async def _llm_parse(self, cv_text: str) -> Dict[str, Any]:
        """Sử dụng LLM để extract structured data"""
        
        prompt = f"""
Bạn là một chuyên gia phân tích CV. Hãy extract thông tin từ CV sau thành JSON structure chính xác.

YÊU CẦU OUTPUT JSON:
{{
  "personal_info": {{
    "full_name": "Tên đầy đủ",
    "email": "email@example.com",
    "phone": "số điện thoại",
    "linkedin": "LinkedIn URL",
    "github": "GitHub URL", 
    "location": "địa chỉ",
    "position": "chức danh hiện tại"
  }},
  "professional_summary": "Tóm tắt chuyên nghiệp 2-3 câu",
  "education": [
    {{
      "institution": "Tên trường",
      "degree": "Bằng cấp",
      "major": "Chuyên ngành",
      "gpa": "Điểm GPA",
      "duration": "Thời gian học",
      "achievements": ["Thành tích 1", "Thành tích 2"]
    }}
  ],
  "work_experience": [
    {{
      "company": "Tên công ty",
      "position": "Vị trí",
      "duration": "Thời gian làm việc",
      "location": "Địa điểm",
      "description": ["Mô tả công việc 1", "Mô tả công việc 2"]
    }}
  ],
  "projects": [
    {{
      "name": "Tên dự án",
      "role": "Vai trò",
      "description": "Mô tả dự án",
      "tech_stack": ["Tech1", "Tech2", "Tech3"],
      "github_links": ["GitHub URL"],
      "demo_link": "Demo URL nếu có"
    }}
  ],
  "technical_skills": {{
    "programming_languages": ["Java", "Python", "JavaScript"],
    "frameworks": ["ReactJS", "Spring Boot", "NextJS"],
    "databases": ["SQL Server", "Firebase"],
    "tools": ["Git", "Docker", "VS Code"]
  }},
  "hobbies": ["Sở thích 1", "Sở thích 2"],
  "soft_skills": ["Kỹ năng mềm 1", "Kỹ năng mềm 2"]
}}

RULES:
1. Chỉ trả về JSON hợp lệ, không có text thêm
2. Nếu không tìm thấy thông tin, để null hoặc array rỗng
3. Extract chính xác URLs, emails, phone numbers
4. Phân loại technical skills đúng category
5. Tách rõ ràng projects và work experience

CV Text:
{cv_text}

JSON Output:
"""
        
        try:
            response = llm_service.model.generate_content(prompt)
            json_text = response.text.strip()
            
            # Clean JSON text
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            if json_text.startswith('```'):
                json_text = json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            
            # Parse JSON
            import json
            structured_data = json.loads(json_text)
            
            return structured_data
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return await self._basic_parse(cv_text)
    
    async def _basic_parse(self, cv_text: str) -> Dict[str, Any]:
        """Basic parsing without LLM"""
        
        # Extract basic info using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'[\+]?[1-9]?[0-9]{7,15}'
        
        emails = re.findall(email_pattern, cv_text)
        phones = re.findall(phone_pattern, cv_text)
        
        return {
            "personal_info": {
                "full_name": self._extract_name(cv_text),
                "email": emails[0] if emails else None,
                "phone": phones[0] if phones else None,
                "linkedin": self._extract_linkedin(cv_text),
                "github": self._extract_github(cv_text),
                "location": None,
                "position": self._extract_position(cv_text)
            },
            "professional_summary": None,
            "education": [],
            "work_experience": [],
            "projects": [],
            "technical_skills": {
                "programming_languages": [],
                "frameworks": [],
                "databases": [],
                "tools": []
            },
            "hobbies": [],
            "soft_skills": []
        }
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name from text"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 5 and len(line) < 50 and not '@' in line and not 'http' in line:
                # Likely a name
                return line.title()
        return None
    
    def _extract_linkedin(self, text: str) -> Optional[str]:
        """Extract LinkedIn URL"""
        linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+'
        matches = re.findall(linkedin_pattern, text)
        return matches[0] if matches else None
    
    def _extract_github(self, text: str) -> Optional[str]:
        """Extract GitHub URL"""
        github_pattern = r'https?://(?:www\.)?github\.com/[A-Za-z0-9_-]+'
        matches = re.findall(github_pattern, text)
        return matches[0] if matches else None
    
    def _extract_position(self, text: str) -> Optional[str]:
        """Extract position/title"""
        positions = ['developer', 'engineer', 'programmer', 'analyst', 'manager']
        text_lower = text.lower()
        for position in positions:
            if position in text_lower:
                return position.title()
        return None

# Global instance
cv_parser_service = CVParserService()
