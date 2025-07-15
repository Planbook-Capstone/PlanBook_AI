"""
RAG (Retrieval-Augmented Generation) Service
Xử lý logic RAG: semantic search + LLM generation
"""

import logging
import time
from typing import Dict, Any, Optional, List
from app.services.enhanced_textbook_service import get_enhanced_textbook_service

logger = logging.getLogger(__name__)

class RAGService:
    """Service xử lý RAG workflow"""
    
    def __init__(self):
        self.textbook_service = get_enhanced_textbook_service()
    
    async def process_rag_query(
        self,
        query: str,
        book_id: Optional[str] = None,
        lesson_id: Optional[str] = None,
        limit: int = 5,
        semantic_tags: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Xử lý RAG query hoàn chỉnh với semantic search

        Args:
            query: Câu hỏi của người dùng
            book_id: ID sách cụ thể (tùy chọn)
            lesson_id: ID bài học cụ thể (tùy chọn)
            limit: Số lượng kết quả tìm kiếm tối đa
            semantic_tags: Lọc theo semantic tags (phân cách bằng dấu phẩy)
            temperature: Temperature cho LLM response
            max_tokens: Số token tối đa cho response

        Returns:
            Dict chứa kết quả RAG với answer, sources và metadata
        """
        start_time = time.time()

        try:
            logger.info(f"🔍 Processing RAG query: {query[:100]}...")

            # Bước 1: Semantic Search
            search_result = await self._perform_semantic_search(
                query=query,
                book_id=book_id,
                lesson_id=lesson_id,
                limit=limit,
                semantic_tags=semantic_tags
            )

            if not search_result.get("success"):
                return {
                    "success": False,
                    "error": f"Search failed: {search_result.get('error')}",
                    "query": query
                }

            search_results = search_result.get("results", [])
            if not search_results:
                return {
                    "success": True,
                    "query": query,
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
                    "sources": [],
                    "context_used": "",
                    "total_sources": 0,
                    "processing_time": time.time() - start_time
                }

            # Bước 2: Tạo context từ search results
            context, sources = self._build_context_and_sources(search_results, book_id)

            # Bước 3: Generate answer với LLM
            answer = await self._generate_answer_with_llm(
                query=query,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )

            processing_time = time.time() - start_time

            return {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": sources,
                "context_used": context,
                "total_sources": len(sources),
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"❌ Error in RAG processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "processing_time": time.time() - start_time
            }

    async def _perform_semantic_search(
        self,
        query: str,
        book_id: Optional[str],
        lesson_id: Optional[str],
        limit: int,
        semantic_tags: Optional[str]
    ) -> Dict[str, Any]:
        """Thực hiện semantic search"""
        from app.services.qdrant_service import get_qdrant_service
        qdrant_service = get_qdrant_service()
            
        # Đảm bảo service được khởi tạo đầy đủ trước khi kiểm tra
        qdrant_service._ensure_service_initialized()
        search_params = {
            "query": query,
            "limit": limit
        }
        
        # Thêm filters nếu có
        semantic_filters = {}
        if semantic_tags:
            semantic_filters["semantic_tags"] = semantic_tags.split(",")
        
        # Thêm lesson_id filter nếu có
        if lesson_id:
            semantic_filters["lesson_id"] = lesson_id
        
        if semantic_filters:
            search_params["semantic_filters"] = semantic_filters
        
        # Tìm kiếm trong book cụ thể hoặc toàn bộ
        if book_id:
            return await qdrant_service.search_textbook(book_id, **search_params)
        else:
            # Global search across all books
            return await qdrant_service.global_search(**search_params)
    
    def _build_context_and_sources(self, search_results: List[Dict], book_id: Optional[str]) -> tuple:
        """Tạo context và sources từ search results"""
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[Nguồn {i}] {result.get('text', '')}")
            sources.append({
                "source_id": i,
                "text": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', ''),
                "score": result.get('score', 0),
                "lesson_id": result.get('lesson_id', ''),
                "semantic_tags": result.get('semantic_tags', []),
                "book_id": result.get('book_id', book_id)
            })
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def _build_rag_prompt(self, context: str, query: str) -> str:
        """Tạo prompt cho LLM để trả về nội dung HTML thuần túy"""
        return f"""Bạn là một trợ lý AI chuyên về giáo dục. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

NGUYÊN TẮC:
- Chỉ sử dụng thông tin từ các nguồn được cung cấp
- Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu
- Trả về nội dung HTML với style inline, KHÔNG cần thẻ div bao bọc bên ngoài
- Sử dụng các thẻ HTML phù hợp: <h3>, <p>, <ul>, <li>, <strong>, <em>
- Không sử dụng ký tự đặc biệt, chỉ HTML thuần
- Chỉ hiện thị trên cùng 1 dòng và không có bất kì khoảng cách nào
- Nếu thông tin không đủ để trả lời, hãy nói rõ

FORMAT TRẢ LỜI (chỉ nội dung HTML thuần):
<div style="padding: 15px; margin-bottom: 2px;line-height: 1.6; color: #333;">
    [Nội dung trả lời ở đây với các thẻ HTML phù hợp như <p>, <strong>, <em>]
</div>

<h4 style="color: #2c5aa0; margin-bottom: 2px; font-size: 16px;">Nguồn tham khảo:</h4>
<ul style="list-style-type: none; padding: 0;">
    [Danh sách nguồn nếu cần trích dẫn với <li> có style]
</ul>

THÔNG TIN TỪ TÀI LIỆU:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""

    async def _generate_answer_with_llm(
        self,
        query: str,
        context: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """Generate answer using LLM với context đã được chuẩn bị"""
        try:
            # Lấy LLM service
            from app.services.openrouter_service import get_openrouter_service
            llm_service = get_openrouter_service()

            # Đảm bảo service được khởi tạo
            llm_service._ensure_service_initialized()

            if not llm_service.available:
                return "Xin lỗi, dịch vụ AI hiện không khả dụng. Vui lòng thử lại sau."

            # Tạo prompt
            prompt = self._build_rag_prompt(context, query)

            # Gọi LLM
            llm_result = await llm_service.generate_content(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not llm_result.get("success"):
                logger.error(f"LLM generation failed: {llm_result.get('error')}")
                return "Xin lỗi, có lỗi xảy ra khi tạo câu trả lời. Vui lòng thử lại."

            # Làm sạch HTML response
            html_answer = llm_result.get("text", "")
            html_answer = self._clean_html_response(html_answer)

            return html_answer

        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."

    def _clean_html_response(self, html_text: str) -> str:
        """Làm sạch HTML response từ LLM"""
        if not html_text:
            return "Không có câu trả lời."

        # Làm sạch escape characters và ký tự đặc biệt
        cleaned = (html_text
                  .replace('\\n', '')
                  .replace('\\r', '')
                  .replace('\\t', '')
                  .replace('\\"', '"')
                  .replace('\\', '')
                  .replace('\n', '')
                  .replace('\r', '')
                  .replace('\t', '')
                  .replace('  ', ' ')
                  .strip())

        return cleaned

    def _build_filters_info(self, semantic_tags: Optional[str], lesson_id: Optional[str], book_id: Optional[str]) -> Dict[str, Any]:
        """Tạo thông tin filters đã áp dụng"""
        return {
            "semantic_tags": semantic_tags.split(",") if semantic_tags else None,
            "lesson_id": lesson_id,
            "book_id": book_id
        }

    async def search_with_semantic_filters(
        self,
        query: str,
        book_id: Optional[str] = None,
        semantic_tags: Optional[str] = None,
        difficulty: Optional[str] = None,
        has_examples: Optional[bool] = None,
        has_formulas: Optional[bool] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Tìm kiếm với semantic filters nâng cao

        Args:
            query: Câu truy vấn tìm kiếm
            book_id: ID sách cụ thể (optional)
            semantic_tags: Danh sách semantic tags để filter
            difficulty: Mức độ khó
            has_examples: Có ví dụ hay không
            has_formulas: Có công thức hay không
            min_confidence: Confidence tối thiểu
            limit: Số lượng kết quả

        Returns:
            Dict chứa kết quả tìm kiếm với semantic metadata
        """
        try:
            from app.services.qdrant_service import qdrant_service

            logger.info(f"Semantic search: query='{query}', book_id={book_id}, semantic_tags={semantic_tags}")

            # Chuẩn bị semantic filters
            semantic_filters = {}

            if semantic_tags:
                tag_list = [tag.strip() for tag in semantic_tags.split(",") if tag.strip()]
                if tag_list:
                    semantic_filters["semantic_tags"] = tag_list

            if difficulty:
                semantic_filters["difficulty"] = difficulty

            if has_examples is not None:
                semantic_filters["has_examples"] = has_examples

            if has_formulas is not None:
                semantic_filters["has_formulas"] = has_formulas

            # Thực hiện tìm kiếm
            if book_id:
                # Tìm kiếm trong sách cụ thể
                result = await qdrant_service.search_textbook(
                    book_id=book_id,
                    query=query,
                    limit=limit,
                    semantic_filters=semantic_filters if semantic_filters else None
                )
            else:
                # Tìm kiếm global (tất cả sách)
                result = await self._global_semantic_search(
                    query=query,
                    semantic_filters=semantic_filters,
                    min_confidence=min_confidence,
                    limit=limit
                )

            return result

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {
                "success": False,
                "error": f"Semantic search failed: {str(e)}"
            }

    async def _global_semantic_search(
        self,
        query: str,
        semantic_filters: Dict[str, Any],
        min_confidence: float,
        limit: int
    ) -> Dict[str, Any]:
        """Thực hiện global semantic search sử dụng unified collection"""
        from app.services.qdrant_service import get_qdrant_service

        qdrant_service = get_qdrant_service()

        if not qdrant_service.qdrant_client:
            return {
                "success": False,
                "error": "Qdrant service not available"
            }

        # Sử dụng global_search từ unified collection
        result = await qdrant_service.global_search(
            query=query,
            limit=limit,
            book_id=semantic_filters.get("book_id") if semantic_filters else None,
            lesson_id=semantic_filters.get("lesson_id") if semantic_filters else None
        )

        if not result.get("success"):
            return result

        # Filter by confidence if specified
        if min_confidence > 0.0:
            filtered_results = []
            for res in result.get("results", []):
                semantic_tags_data = res.get("semantic_tags", [])
                max_confidence = max([tag.get("confidence", 0.0) for tag in semantic_tags_data], default=0.0)
                if max_confidence >= min_confidence:
                    filtered_results.append(res)

            result["results"] = filtered_results
            result["total_results_found"] = len(filtered_results)
            result["message"] = f"Found {len(filtered_results)} results with confidence >= {min_confidence}"

        return result



# Factory function
def get_rag_service() -> RAGService:
    """
    Factory function để tạo RAGService instance

    Returns:
        RAGService: Instance của RAG service
    """
    return RAGService()
