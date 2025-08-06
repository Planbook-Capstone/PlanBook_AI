"""
Fast Semantic Service - Rule-based semantic analysis thay thế LLM
"""

import re
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class FastSemanticService:
    """Rule-based semantic analysis service - nhanh và không cần LLM"""
    
    def __init__(self):
        # Patterns cho semantic tags
        self.semantic_patterns = {
            "definition": [
                r'(là|được gọi là|được định nghĩa|định nghĩa|khái niệm|có nghĩa là)',
                r'(được hiểu là|có thể hiểu|được coi là)',
                r'(chính là|tức là|nghĩa là)',
            ],
            "example": [
                r'(ví dụ|thí dụ|chẳng hạn|như|cụ thể|minh họa)',
                r'(hình \d+|bảng \d+|sơ đồ \d+)',
                r'(trường hợp|tình huống|khi)',
            ],
            "formula": [
                r'[A-Z][a-z]?\s*[+\-=→]\s*[A-Z]',  # Chemical reactions
                r'\w+\s*=\s*\w+',                   # Math equations
                r'[A-Z]\([A-Z]+\)\d*',              # Chemical compounds
                r'(công thức|phương trình|biểu thức)',
            ],
            "theory": [
                r'(lý thuyết|nguyên lý|định luật|quy luật)',
                r'(giả thuyết|học thuyết|quan điểm)',
                r'(theo|dựa trên|căn cứ vào)',
            ],
            "process": [
                r'(quá trình|quy trình|các bước|thứ tự)',
                r'(đầu tiên|tiếp theo|cuối cùng|sau đó)',
                r'(giai đoạn|bước \d+|thực hiện)',
            ],
            "classification": [
                r'(phân loại|chia thành|gồm có|bao gồm)',
                r'(loại|nhóm|dạng|kiểu)',
                r'(được chia|có thể chia)',
            ]
        }
        
        # Patterns cho key concepts
        self.concept_patterns = [
            r'[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ][a-zàáạảãâầấậẩẫăằắặẳẵ]+(?:\s+[a-zàáạảãâầấậẩẫăằắặẳẵ]+){0,2}',  # Thuật ngữ
            r'[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*',  # Chemical formulas
            r'\b[A-Z]{2,}\b',  # Abbreviations
        ]
        
        # Difficulty indicators theo chuẩn THPT 2025 (Hóa học)
        self.difficulty_indicators = {
            "basic": [
                # Nhận biết - 40% đề thi
                r'(khái niệm|định nghĩa|tên gọi|ký hiệu)',
                r'(công thức phân tử|công thức cấu tạo)',
                r'(tính chất|đặc điểm|màu sắc|trạng thái)',
                r'(thuộc nhóm|loại hợp chất|phân loại)',
                r'(số hiệu nguyên tử|khối lượng nguyên tử)',
            ],
            "intermediate": [
                # Thông hiểu - 35-40% đề thi
                r'(giải thích|tại sao|nguyên nhân|do đâu)',
                r'(so sánh|phân biệt|khác nhau|giống nhau)',
                r'(liên quan|ảnh hưởng|tác động|phụ thuộc)',
                r'(dự đoán|nhận xét|kết luận|suy ra)',
                r'(điều kiện|yếu tố ảnh hưởng|cơ chế)',
            ],
            "advanced": [
                # Vận dụng - 20-25% đề thi
                r'(tính toán|xác định|tìm|khối lượng|thể tích)',
                r'(nồng độ|số mol|hiệu suất|độ tan|ph)',
                r'(phân tích|đánh giá|nhận định|bình luận)',
                r'(thiết kế|thí nghiệm|phương pháp|quy trình)',
                r'(ứng dụng|sử dụng|trong thực tế|sản xuất)',
            ]
        }

    def analyze_chunk_semantic(self, text: str) -> Dict[str, Any]:
        """
        Phân tích semantic cho chunk text
        
        Args:
            text: Nội dung chunk
            
        Returns:
            Dict chứa semantic info
        """
        if not text or not text.strip():
            return self._empty_semantic_info()
        
        text_lower = text.lower()
        
        # Phát hiện semantic tags
        semantic_tags = self._detect_semantic_tags(text_lower)
        
        # Trích xuất key concepts
        key_concepts = self._extract_key_concepts(text)
        
        # Phát hiện content features
        features = self._detect_content_features(text_lower)
        
        # Ước tính difficulty
        difficulty = self._estimate_difficulty(text_lower)
        
        return {
            "semantic_tags": semantic_tags,
            "key_concepts": key_concepts[:10],  # Giới hạn 10 concepts
            "contains_examples": features["has_examples"],
            "contains_definitions": features["has_definitions"],
            "contains_formulas": features["has_formulas"],
            "estimated_difficulty": difficulty,
            "analysis_method": "fast_semantic",
            "word_count": len(text.split()),
            "char_count": len(text),
        }

    def _detect_semantic_tags(self, text_lower: str) -> List[Dict[str, Any]]:
        """Phát hiện semantic tags từ text"""
        tags = []
        
        for tag_type, patterns in self.semantic_patterns.items():
            confidence = 0.0
            matched_patterns = 0
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matched_patterns += 1
                    confidence += 0.3  # Mỗi pattern match tăng 0.3
            
            if matched_patterns > 0:
                # Normalize confidence
                confidence = min(confidence, 1.0)
                tags.append({
                    "type": tag_type,
                    "confidence": round(confidence, 2),
                    "matched_patterns": matched_patterns
                })
        
        # Sort by confidence
        tags.sort(key=lambda x: x["confidence"], reverse=True)
        return tags[:5]  # Lấy top 5 tags

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Trích xuất key concepts từ text"""
        concepts = set()
        
        for pattern in self.concept_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, str) and len(match) > 2:
                    concepts.add(match.strip())
        
        # Filter out common words
        filtered_concepts = []
        common_words = {"Trong", "Với", "Khi", "Nếu", "Để", "Có", "Được", "Này", "Đó"}
        
        for concept in concepts:
            if concept not in common_words and len(concept) > 2:
                filtered_concepts.append(concept)
        
        return sorted(list(filtered_concepts))

    def _detect_content_features(self, text_lower: str) -> Dict[str, bool]:
        """Phát hiện các features của content"""
        return {
            "has_examples": bool(re.search(r'(ví dụ|thí dụ|chẳng hạn|minh họa)', text_lower)),
            "has_definitions": bool(re.search(r'(là|được gọi là|định nghĩa|có nghĩa là)', text_lower)),
            "has_formulas": bool(re.search(r'([A-Z][a-z]?\d*|=|→|↔)', text_lower)),
            "has_process": bool(re.search(r'(bước|giai đoạn|quá trình|thứ tự)', text_lower)),
            "has_classification": bool(re.search(r'(phân loại|chia thành|gồm có)', text_lower)),
        }

    def _estimate_difficulty(self, text_lower: str) -> str:
        """
        Ước tính mức độ khó của content theo chuẩn THPT 2025
        basic = Nhận biết, intermediate = Thông hiểu, advanced = Vận dụng
        """
        scores = {"basic": 0, "intermediate": 0, "advanced": 0}

        # Đếm điểm dựa trên từ khóa
        for level, patterns in self.difficulty_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[level] += 1

        # Phân tích bổ sung cho hóa học
        # Có số liệu, đơn vị đo lường -> vận dụng
        if re.search(r'\d+[.,]\d+|\d+\s*(g|ml|l|mol|m|%|°c|atm|bar)', text_lower):
            scores["advanced"] += 2

        # Có phương trình hóa học -> thông hiểu hoặc vận dụng
        if re.search(r'[A-Z][a-z]?\s*\+|→|↔|=', text_lower):
            scores["intermediate"] += 1

        # Có từ "tính", "tìm", "xác định" -> vận dụng
        if re.search(r'tính|tìm|xác định|cho biết', text_lower):
            scores["advanced"] += 1

        # Có công thức hóa học phức tạp -> intermediate/advanced
        chemical_formula_pattern = r'[A-Z][a-z]?\d*(\([A-Z][a-z]?\d*\))?\d*'
        if re.search(chemical_formula_pattern, text_lower):
            scores["intermediate"] += 1

        # Heuristics dựa trên độ dài
        word_count = len(text_lower.split())
        if word_count < 30:
            scores["basic"] += 1
        elif word_count > 150:
            scores["advanced"] += 1
        else:
            scores["intermediate"] += 1

        # Return level với score cao nhất
        max_level = max(scores.items(), key=lambda x: x[1])
        return max_level[0]

    def _empty_semantic_info(self) -> Dict[str, Any]:
        """Trả về semantic info rỗng"""
        return {
            "semantic_tags": [],
            "key_concepts": [],
            "contains_examples": False,
            "contains_definitions": False,
            "contains_formulas": False,
            "estimated_difficulty": "basic",
            "analysis_method": "empty",
            "word_count": 0,
            "char_count": 0,
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Phân tích batch nhiều texts cùng lúc"""
        results = []
        for text in texts:
            result = self.analyze_chunk_semantic(text)
            results.append(result)
        return results


# Singleton instance
_fast_semantic_service = None

def get_fast_semantic_service() -> FastSemanticService:
    """Get singleton instance của FastSemanticService"""
    global _fast_semantic_service
    if _fast_semantic_service is None:
        _fast_semantic_service = FastSemanticService()
    return _fast_semantic_service
