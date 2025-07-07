"""
Semantic Analysis Service - Sử dụng LLM để phân tích semantic với multi-label và confidence
"""
import logging
import json
from typing import Dict, List, Any, Optional
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


class SemanticAnalysisService:
    """
    Service phân tích semantic content sử dụng LLM
    Trả về multi-label với confidence scores
    """
    
    def __init__(self):
        self.semantic_categories = [
            "definition",      # Định nghĩa, khái niệm
            "example",         # Ví dụ, minh họa
            "formula",         # Công thức, phương trình
            "exercise",        # Bài tập, thực hành
            "theory",          # Lý thuyết, giải thích
            "summary",         # Tóm tắt, kết luận
            "procedure",       # Quy trình, các bước
            "comparison",      # So sánh, đối chiếu
            "application",     # Ứng dụng thực tế
            "header"           # Tiêu đề, đề mục
        ]
        
        self.difficulty_levels = ["basic", "intermediate", "advanced"]
        
    async def analyze_content_semantic(self, content: str) -> Dict[str, Any]:
        """
        Phân tích semantic của content sử dụng LLM
        
        Args:
            content: Nội dung cần phân tích
            
        Returns:
            Dict chứa semantic_tags với confidence scores và metadata
        """
        try:
            if not content or not content.strip():
                return self._get_default_result()
            
            # Kiểm tra LLM service
            if not llm_service.is_available():
                logger.warning("LLM service not available, using fallback analysis")
                return self._fallback_analysis(content)
            
            # Tạo prompt cho LLM
            prompt = self._create_analysis_prompt(content)
            
            # Gọi LLM
            result = await llm_service.generate_content(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000
            )
            
            if result["success"]:
                return self._parse_llm_response(result["text"], content)
            else:
                logger.warning(f"LLM analysis failed: {result['error']}")
                return self._fallback_analysis(content)
                
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return self._fallback_analysis(content)
    
    def _create_analysis_prompt(self, content: str) -> str:
        """Tạo prompt cho LLM phân tích semantic"""
        
        categories_str = ", ".join(self.semantic_categories)
        
        prompt = f"""
Phân tích nội dung giáo dục sau và xác định các loại semantic phù hợp.

NỘI DUNG CẦN PHÂN TÍCH:
{content[:2000]}  # Giới hạn độ dài để tránh vượt quá token limit

CÁC LOẠI SEMANTIC CÓ THỂ:
{categories_str}

YÊU CẦU:
1. Xác định TẤT CẢ các loại semantic phù hợp (có thể nhiều loại)
2. Đánh giá độ tin cậy (confidence) từ 0.0 đến 1.0 cho mỗi loại
3. Xác định độ khó: basic, intermediate, advanced
4. Trích xuất các khái niệm chính (tối đa 5 khái niệm)
5. Xác định có chứa: examples, definitions, formulas (true/false)

ĐỊNH DẠNG TRẢI VỀ JSON:
{{
    "semantic_tags": [
        {{"type": "definition", "confidence": 0.9}},
        {{"type": "example", "confidence": 0.7}}
    ],
    "difficulty": "intermediate",
    "key_concepts": ["khái niệm 1", "khái niệm 2"],
    "contains_examples": true,
    "contains_definitions": true,
    "contains_formulas": false
}}

Chỉ trả về JSON, không có text khác.
"""
        return prompt
    
    def _parse_llm_response(self, response: str, original_content: str) -> Dict[str, Any]:
        """Parse response từ LLM thành structured data"""
        try:
            # Tìm JSON trong response
            response = response.strip()
            
            # Tìm JSON block
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            # Validate và clean data
            return self._validate_and_clean_result(parsed, original_content)
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_analysis(original_content)
    
    def _validate_and_clean_result(self, parsed: Dict, content: str) -> Dict[str, Any]:
        """Validate và clean kết quả từ LLM"""
        
        # Validate semantic_tags
        semantic_tags = []
        if "semantic_tags" in parsed and isinstance(parsed["semantic_tags"], list):
            for tag in parsed["semantic_tags"]:
                if isinstance(tag, dict) and "type" in tag and "confidence" in tag:
                    tag_type = tag["type"]
                    confidence = float(tag["confidence"])
                    
                    # Validate type và confidence
                    if tag_type in self.semantic_categories and 0.0 <= confidence <= 1.0:
                        semantic_tags.append({
                            "type": tag_type,
                            "confidence": round(confidence, 2)
                        })
        
        # Nếu không có tags hợp lệ, thêm default
        if not semantic_tags:
            semantic_tags = [{"type": "theory", "confidence": 0.5}]
        
        # Validate difficulty
        difficulty = parsed.get("difficulty", "basic")
        if difficulty not in self.difficulty_levels:
            difficulty = "basic"
        
        # Validate key_concepts
        key_concepts = parsed.get("key_concepts", [])
        if not isinstance(key_concepts, list):
            key_concepts = []
        key_concepts = [str(concept) for concept in key_concepts[:5]]  # Limit to 5
        
        # Validate boolean flags
        contains_examples = bool(parsed.get("contains_examples", False))
        contains_definitions = bool(parsed.get("contains_definitions", False))
        contains_formulas = bool(parsed.get("contains_formulas", False))
        
        return {
            "semantic_tags": semantic_tags,
            "difficulty": difficulty,
            "key_concepts": key_concepts,
            "contains_examples": contains_examples,
            "contains_definitions": contains_definitions,
            "contains_formulas": contains_formulas,
            "analysis_method": "llm"
        }
    
    def _fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback analysis khi LLM không khả dụng"""
        
        content_lower = content.lower()
        semantic_tags = []
        
        # Simple keyword-based detection với confidence thấp
        if any(kw in content_lower for kw in ['định nghĩa', 'khái niệm', 'definition']):
            semantic_tags.append({"type": "definition", "confidence": 0.6})
        
        if any(kw in content_lower for kw in ['ví dụ', 'example', 'minh họa']):
            semantic_tags.append({"type": "example", "confidence": 0.6})
        
        if any(kw in content_lower for kw in ['công thức', 'formula', 'phương trình']):
            semantic_tags.append({"type": "formula", "confidence": 0.6})
        
        if any(kw in content_lower for kw in ['bài tập', 'exercise', 'thực hành']):
            semantic_tags.append({"type": "exercise", "confidence": 0.6})
        
        # Default nếu không detect được gì
        if not semantic_tags:
            semantic_tags = [{"type": "theory", "confidence": 0.4}]
        
        return {
            "semantic_tags": semantic_tags,
            "difficulty": "basic",
            "key_concepts": [],
            "contains_examples": "ví dụ" in content_lower or "example" in content_lower,
            "contains_definitions": "định nghĩa" in content_lower or "definition" in content_lower,
            "contains_formulas": "công thức" in content_lower or "formula" in content_lower,
            "analysis_method": "fallback"
        }
    
    def _get_default_result(self) -> Dict[str, Any]:
        """Kết quả mặc định cho content rỗng"""
        return {
            "semantic_tags": [{"type": "theory", "confidence": 0.1}],
            "difficulty": "basic",
            "key_concepts": [],
            "contains_examples": False,
            "contains_definitions": False,
            "contains_formulas": False,
            "analysis_method": "default"
        }


# Lazy loading global instance để tránh khởi tạo ngay khi import
_semantic_analysis_service_instance = None

def get_semantic_analysis_service() -> SemanticAnalysisService:
    """
    Lấy singleton instance của SemanticAnalysisService
    Lazy initialization

    Returns:
        SemanticAnalysisService: Service instance
    """
    global _semantic_analysis_service_instance
    if _semantic_analysis_service_instance is None:
        _semantic_analysis_service_instance = SemanticAnalysisService()
    return _semantic_analysis_service_instance

# Backward compatibility - deprecated, sử dụng get_semantic_analysis_service() thay thế
# Lazy loading để tránh khởi tạo ngay khi import
def _get_semantic_analysis_service_lazy():
    """Lazy loading cho backward compatibility"""
    return get_semantic_analysis_service()

# Tạo proxy object để lazy loading
class _SemanticAnalysisServiceProxy:
    def __getattr__(self, name):
        return getattr(_get_semantic_analysis_service_lazy(), name)

    def __call__(self, *args, **kwargs):
        return _get_semantic_analysis_service_lazy()(*args, **kwargs)

semantic_analysis_service = _SemanticAnalysisServiceProxy()
