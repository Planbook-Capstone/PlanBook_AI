"""
Chunking Service - Xử lý chia nhỏ văn bản thành chunks
"""

import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service chuyên xử lý chunking văn bản"""

    def __init__(self):
        """Initialize chunking service"""
        pass

    def create_text_chunks(
        self, text: str, max_size: int = None, overlap: int = None
    ) -> List[str]:
        """
        Tạo chunks từ text content với semantic chunking tối ưu
        
        Args:
            text: Văn bản cần chia
            max_size: Kích thước tối đa của chunk (default từ settings)
            overlap: Độ chồng lấp giữa các chunks (default từ settings)
            
        Returns:
            List các text chunks
        """
        if not text or not text.strip():
            return []

        # Sử dụng default values từ settings nếu không được cung cấp
        if max_size is None:
            max_size = settings.MAX_CHUNK_SIZE
        if overlap is None:
            overlap = settings.CHUNK_OVERLAP

        # Sử dụng semantic chunking tối ưu với NLTK
        return self._semantic_chunking_with_nltk(text, max_size, overlap)

    def _semantic_chunking_with_nltk(self, text: str, max_size: int, overlap: int) -> List[str]:
        """
        Semantic chunking tối ưu sử dụng NLTK để đảm bảo ngữ nghĩa hoàn chỉnh

        Ưu điểm:
        - Sử dụng NLTK sent_tokenize để tách câu chính xác
        - Giữ nguyên cấu trúc paragraph và câu
        - Đảm bảo mỗi chunk có ngữ nghĩa hoàn chỉnh
        - Overlap thông minh dựa trên câu
        """
        try:
            from nltk import sent_tokenize
            import nltk

            # Download punkt tokenizer nếu chưa có
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
                nltk.download('punkt', quiet=True)

        except ImportError:
            logger.warning("NLTK not available, falling back to simple chunking")
            return self._simple_chunking(text, max_size, overlap)

        # Tiền xử lý text
        text = text.strip()
        if len(text) <= max_size:
            return [text]

        # Bước 1: Chia theo paragraphs để giữ cấu trúc
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Bước 2: Tách từng paragraph thành câu bằng NLTK
        all_sentences = []
        paragraph_boundaries = []  # Để track paragraph boundaries

        for para in paragraphs:
            sentences = sent_tokenize(para)
            start_idx = len(all_sentences)
            all_sentences.extend(sentences)
            paragraph_boundaries.append((start_idx, len(all_sentences) - 1))

        # Bước 3: Gộp câu thành chunks với kích thước phù hợp
        chunks = []
        current_chunk_sentences = []
        current_length = 0

        i = 0
        while i < len(all_sentences):
            sentence = all_sentences[i]
            sentence_length = len(sentence)

            # Kiểm tra nếu thêm câu này sẽ vượt quá max_size
            if current_length + sentence_length + 1 <= max_size:  # +1 cho space
                current_chunk_sentences.append(sentence)
                current_length += sentence_length + 1
                i += 1
            else:
                # Tạo chunk từ các câu hiện tại
                if current_chunk_sentences:
                    chunks.append(' '.join(current_chunk_sentences))

                # Bắt đầu chunk mới
                if sentence_length <= max_size:
                    current_chunk_sentences = [sentence]
                    current_length = sentence_length
                    i += 1
                else:
                    # Câu quá dài, phải chia nhỏ
                    long_sentence_chunks = self._split_long_sentence(sentence, max_size)
                    chunks.extend(long_sentence_chunks)
                    current_chunk_sentences = []
                    current_length = 0
                    i += 1

        # Thêm chunk cuối nếu có
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))

        # Bước 4: Áp dụng overlap nếu cần
        if overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks, overlap)

        # Bước 5: Làm sạch chunks
        final_chunks = []
        for chunk in chunks:
            cleaned_chunk = chunk.strip()
            if cleaned_chunk and len(cleaned_chunk) > 10:  # Loại bỏ chunks quá ngắn
                final_chunks.append(cleaned_chunk)

        logger.info(f"✅ Semantic chunking completed: {len(final_chunks)} chunks created")
        return final_chunks

    def _split_long_sentence(self, sentence: str, max_size: int) -> List[str]:
        """Chia câu dài thành các phần nhỏ hơn"""
        if len(sentence) <= max_size:
            return [sentence]

        chunks = []
        words = sentence.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length <= max_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """Áp dụng overlap giữa các chunks"""
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Lấy overlap từ chunk trước
            if len(prev_chunk) > overlap:
                overlap_text = prev_chunk[-overlap:]
                overlapped_chunk = overlap_text + " " + current_chunk
            else:
                overlapped_chunk = current_chunk

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _simple_chunking(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Fallback chunking method (original logic)"""
        chunks = []
        text = text.strip()

        # Nếu text ngắn hơn max_size, trả về luôn
        if len(text) <= max_size:
            return [text]

        # Chia text thành các chunks với overlap
        start = 0
        while start < len(text):
            end = start + max_size

            # Nếu không phải chunk cuối, tìm vị trí ngắt tự nhiên
            if end < len(text):
                # Tìm dấu câu hoặc khoảng trắng gần nhất
                for i in range(end, start + max_size // 2, -1):
                    if text[i] in ".!?\n ":
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Di chuyển start với overlap
            start = max(start + max_size - overlap, end)

            # Tránh vòng lặp vô hạn
            if start >= len(text):
                break

        return chunks


# Factory function - creates new instance each time
def get_chunking_service() -> ChunkingService:
    """
    Create new instance của ChunkingService (thread-safe)

    Returns:
        ChunkingService: New instance
    """
    return ChunkingService()
