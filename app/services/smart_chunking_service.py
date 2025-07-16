"""
Smart Chunking Service - Rule-Based + Token Constraint chunking cho sách giáo khoa Việt Nam
Áp dụng pipeline mới: sent_tokenize → semantic grouping → overlap smart
"""

import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Thông tin về một chunk với semantic completeness"""
    text: str
    chunk_type: str  # "definition", "example", "table", "exercise", "content"
    semantic_tag: str  # "theory", "practice", "reference"
    parent_title: str
    concepts: List[str]
    token_count: int
    overlap_context: str = ""  # Context từ chunk trước

    @property
    def is_semantic_complete(self) -> bool:
        """Kiểm tra chunk có đầy đủ ngữ nghĩa không"""
        return self.chunk_type in ["definition", "example", "table", "exercise"] or self.token_count >= 100


class SmartChunkingService:
    """Service chunking thông minh với Rule-Based + Token Constraint pipeline"""

    def __init__(self):
        # Patterns nhận diện "ngắt nghĩa" tự nhiên
        self.semantic_break_patterns = {
            'definition_start': [
                r'^(Định nghĩa|ĐỊNH NGHĨA)[:\.]',
                r'(là|được gọi là|được định nghĩa|khái niệm về)',
                r'^[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ][^\.]*\s+(là|được gọi là)'
            ],
            'example_start': [
                r'^(Bài|BÀI)\s+\d+[\.\:]',
                r'^(Ví dụ|VÍ DỤ)\s*\d*[\.\:]',
                r'^(Hãy cho biết|Tính|Xác định)',
                r'^\d+\.\s*[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]'
            ],
            'table_start': [
                r'^(Bảng|BẢNG)\s+\d+',
                r'^\|.*\|.*\|',  # Markdown table
                r'^(Đơn vị|Khối lượng|Tên|STT)',
                r'.*\|.*\|.*'  # Table row pattern
            ],
            'section_start': [
                r'^([IVXLC]+|\d+)\.\s*[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]',
                r'^(Chương|CHƯƠNG)\s+([IVXLC]+|\d+)',
                r'^[-+*]\s*[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]'
            ]
        }

        # Token limits
        self.max_tokens = 400
        self.min_tokens = 50
        self.overlap_sentences = 2

    def chunk_textbook_content(self, text: str, max_tokens: int = 400) -> List[ChunkInfo]:
        """
        Pipeline chunking mới: Rule-Based + Token Constraint

        Args:
            text: Nội dung cần chia
            max_tokens: Số tokens tối đa cho một chunk

        Returns:
            List các ChunkInfo với semantic completeness
        """
        if not text or not text.strip():
            return []

        # Bước 1: sent_tokenize để chia văn bản thành câu
        sentences = self._tokenize_sentences(text)

        # Bước 2: Gom các câu thành blocks theo logic semantic
        semantic_blocks = self._group_sentences_by_semantics(sentences)

        # Bước 3: Tạo chunks với token constraint
        chunks = self._create_chunks_with_constraints(semantic_blocks, max_tokens)

        # Bước 4: Thêm overlap smart
        final_chunks = self._add_smart_overlap(chunks)

        return final_chunks

    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Bước 1: Chia văn bản thành câu với NLTK
        Xử lý tốt dấu chấm trong công thức: 1.602 x 10^-19 C.
        """
        # Preprocess để bảo vệ công thức khỏi bị tách sai
        protected_text = self._protect_formulas(text)

        # Tokenize sentences
        sentences = nltk.sent_tokenize(protected_text, language='english')  # Vietnamese not available

        # Restore formulas
        sentences = [self._restore_formulas(sent) for sent in sentences]

        # Clean up
        sentences = [sent.strip() for sent in sentences if sent.strip()]

        return sentences

    def _protect_formulas(self, text: str) -> str:
        """Bảo vệ công thức khỏi bị tách sai bởi sent_tokenize"""
        # Protect scientific notation: 1.602 x 10^-19
        text = re.sub(r'(\d+[.,]\d+)\s*x\s*10\^?(-?\d+)', r'\1×10POWER\2', text)

        # Protect chemical formulas with dots
        text = re.sub(r'([A-Z][a-z]?\d*)\.\s*([A-Z])', r'\1FORMULA\2', text)

        # Protect numbered items: 1. 2. 3. (but not at end of sentence)
        text = re.sub(r'^([IVXLC]+|\d+)\.\s+([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ])', r'\1ITEM \2', text, flags=re.MULTILINE)

        # Protect "Bài 1." patterns
        text = re.sub(r'(Bài|BÀI)\s+(\d+)\.\s+', r'\1 \2ITEM ', text, flags=re.IGNORECASE)

        return text

    def _restore_formulas(self, text: str) -> str:
        """Khôi phục công thức đã được bảo vệ"""
        # Restore scientific notation
        text = re.sub(r'(\d+[.,]\d+)×10POWER(-?\d+)', r'\1 x 10^\2', text)

        # Restore chemical formulas
        text = text.replace('FORMULA', '. ')

        # Restore numbered items
        text = text.replace('ITEM ', '. ')

        return text

    def _group_sentences_by_semantics(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Bước 2: Gom các câu thành blocks theo logic semantic
        Gom liên tục cho đến khi gặp semantic break
        """
        if not sentences:
            return []

        blocks = []
        current_block = {
            'sentences': [],
            'chunk_type': 'content',
            'semantic_tag': 'theory',
            'parent_title': ''
        }

        for sentence in sentences:
            # Kiểm tra có phải semantic break không
            break_info = self._detect_semantic_break(sentence)

            if break_info and current_block['sentences']:
                # Kết thúc block hiện tại
                blocks.append(current_block)

                # Bắt đầu block mới
                current_block = {
                    'sentences': [sentence],
                    'chunk_type': break_info['chunk_type'],
                    'semantic_tag': break_info['semantic_tag'],
                    'parent_title': break_info.get('title', '')
                }
            else:
                # Tiếp tục gộp vào block hiện tại
                current_block['sentences'].append(sentence)

        # Thêm block cuối
        if current_block['sentences']:
            blocks.append(current_block)

        return blocks

    def _detect_semantic_break(self, sentence: str) -> Optional[Dict[str, str]]:
        """Phát hiện điểm ngắt nghĩa tự nhiên"""
        sentence_clean = sentence.strip()

        # Kiểm tra definition start
        for pattern in self.semantic_break_patterns['definition_start']:
            if re.search(pattern, sentence_clean, re.IGNORECASE):
                return {
                    'chunk_type': 'definition',
                    'semantic_tag': 'theory',
                    'title': sentence_clean[:50] + '...' if len(sentence_clean) > 50 else sentence_clean
                }

        # Kiểm tra example/exercise start
        for pattern in self.semantic_break_patterns['example_start']:
            if re.search(pattern, sentence_clean, re.IGNORECASE):
                return {
                    'chunk_type': 'example',
                    'semantic_tag': 'practice',
                    'title': sentence_clean[:50] + '...' if len(sentence_clean) > 50 else sentence_clean
                }

        # Kiểm tra table start
        for pattern in self.semantic_break_patterns['table_start']:
            if re.search(pattern, sentence_clean, re.IGNORECASE):
                return {
                    'chunk_type': 'table',
                    'semantic_tag': 'reference',
                    'title': sentence_clean[:50] + '...' if len(sentence_clean) > 50 else sentence_clean
                }

        # Kiểm tra section start
        for pattern in self.semantic_break_patterns['section_start']:
            if re.search(pattern, sentence_clean, re.IGNORECASE):
                return {
                    'chunk_type': 'content',
                    'semantic_tag': 'theory',
                    'title': sentence_clean[:50] + '...' if len(sentence_clean) > 50 else sentence_clean
                }

        return None

    def _create_chunks_with_constraints(self, blocks: List[Dict[str, Any]], max_tokens: int) -> List[ChunkInfo]:
        """
        Bước 3: Tạo chunks với token constraint
        Đảm bảo mỗi chunk chứa trọn 1 đơn vị ngữ nghĩa
        """
        chunks = []

        for block in blocks:
            sentences = block['sentences']
            chunk_type = block['chunk_type']
            semantic_tag = block['semantic_tag']
            parent_title = block['parent_title']

            # Nếu block nhỏ, tạo 1 chunk
            block_text = ' '.join(sentences)
            token_count = self._count_tokens(block_text)

            if token_count <= max_tokens:
                concepts = self._extract_concepts(block_text)
                chunks.append(ChunkInfo(
                    text=block_text,
                    chunk_type=chunk_type,
                    semantic_tag=semantic_tag,
                    parent_title=parent_title,
                    concepts=concepts,
                    token_count=token_count
                ))
            else:
                # Block lớn, chia nhỏ nhưng giữ nguyên semantic completeness
                sub_chunks = self._split_large_block(sentences, chunk_type, semantic_tag, parent_title, max_tokens)
                chunks.extend(sub_chunks)

        return chunks

    def _split_large_block(self, sentences: List[str], chunk_type: str, semantic_tag: str,
                          parent_title: str, max_tokens: int) -> List[ChunkInfo]:
        """Chia block lớn thành chunks nhỏ hơn nhưng vẫn semantic complete"""
        chunks = []
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Nếu thêm câu này vượt quá limit
            if current_tokens + sentence_tokens > max_tokens and current_sentences:
                # Tạo chunk từ sentences hiện tại
                chunk_text = ' '.join(current_sentences)
                concepts = self._extract_concepts(chunk_text)

                chunks.append(ChunkInfo(
                    text=chunk_text,
                    chunk_type=chunk_type,
                    semantic_tag=semantic_tag,
                    parent_title=parent_title,
                    concepts=concepts,
                    token_count=current_tokens
                ))

                # Reset cho chunk mới
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Tạo chunk cuối
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            concepts = self._extract_concepts(chunk_text)

            chunks.append(ChunkInfo(
                text=chunk_text,
                chunk_type=chunk_type,
                semantic_tag=semantic_tag,
                parent_title=parent_title,
                concepts=concepts,
                token_count=current_tokens
            ))

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Đếm số tokens (ước lượng đơn giản)"""
        return len(text.split())

    def _extract_concepts(self, text: str) -> List[str]:
        """Trích xuất concepts từ text với focus vào thuật ngữ khoa học"""
        concepts = []

        # Pattern cho thuật ngữ khoa học (danh từ viết hoa)
        science_terms = re.findall(r'[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ][a-zàáạảãâầấậẩẫăằắặẳẵ]+(?:\s+[a-zàáạảãâầấậẩẫăằắặẳẵ]+)*', text)
        concepts.extend(science_terms[:5])

        # Pattern cho công thức hóa học
        chemical_formulas = re.findall(r'[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*', text)
        concepts.extend(chemical_formulas[:3])

        # Pattern cho số liệu quan trọng
        numbers = re.findall(r'\d+[.,]\d+(?:\s*x\s*10\^?-?\d+)?', text)
        concepts.extend(numbers[:2])

        return list(set(concepts))

    def _add_smart_overlap(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """
        Bước 4: Thêm overlap smart (1-2 câu trước đoạn mới)
        Đảm bảo ngữ cảnh không bị thiếu khi truy xuất
        """
        if len(chunks) <= 1:
            return chunks

        enhanced_chunks = []

        for i, chunk in enumerate(chunks):
            overlap_context = ""

            # Lấy overlap từ chunk trước (nếu có)
            if i > 0:
                prev_chunk = chunks[i - 1]
                prev_sentences = nltk.sent_tokenize(prev_chunk.text)

                # Lấy 1-2 câu cuối của chunk trước làm context
                overlap_sentences = prev_sentences[-self.overlap_sentences:] if len(prev_sentences) >= self.overlap_sentences else prev_sentences
                overlap_context = ' '.join(overlap_sentences)

            # Tạo enhanced chunk với overlap context
            enhanced_chunk = ChunkInfo(
                text=chunk.text,
                chunk_type=chunk.chunk_type,
                semantic_tag=chunk.semantic_tag,
                parent_title=chunk.parent_title,
                concepts=chunk.concepts,
                token_count=chunk.token_count,
                overlap_context=overlap_context
            )

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks


    def get_chunk_summary(self, chunk: ChunkInfo) -> str:
        """Tạo summary ngắn gọn cho chunk để debug/logging"""
        return f"[{chunk.chunk_type}|{chunk.semantic_tag}] {chunk.token_count} tokens: {chunk.text[:100]}..."


# Service instance
def get_smart_chunking_service() -> SmartChunkingService:
    """Get instance của SmartChunkingService (không dùng singleton)"""
    return SmartChunkingService()
