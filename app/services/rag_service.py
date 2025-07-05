"""
RAG (Retrieval-Augmented Generation) Service
Xử lý logic RAG: semantic search + LLM generation
"""

import logging
from typing import Dict, Any, Optional, List
from app.services.enhanced_textbook_service import EnhancedTextbookService

logger = logging.getLogger(__name__)

class RAGService:
    """Service xử lý RAG workflow"""
    
    def __init__(self):
        self.textbook_service = EnhancedTextbookService()
    
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
        Xử lý RAG query hoàn chỉnh
        
        Args:
            query: Câu hỏi của người dùng
            book_id: ID sách cụ thể (tùy chọn)
            lesson_id: ID bài học cụ thể (tùy chọn)
            limit: Số lượng kết quả tìm kiếm tối đa
            semantic_tags: Lọc theo semantic tags
            temperature: Temperature cho LLM response
            max_tokens: Số token tối đa cho response
            
        Returns:
            Dict chứa kết quả RAG với answer đã được làm sạch
        """
        try:
            logger.info(f"Processing RAG query: {query[:100]}...")
            
            # Kiểm tra LLM service
            from app.services.openrouter_service import OpenRouterService
            llm_service = OpenRouterService()
            
            if not llm_service.available:
                return {
                    "success": False,
                    "error": "LLM service không khả dụng. Vui lòng kiểm tra cấu hình API key."
                }
            
            # Step 1: Semantic search để tìm context
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
                    "error": f"Search failed: {search_result.get('error')}"
                }
            
            search_results = search_result.get("results", [])
            
            if not search_results:
                return {
                    "success": True,
                    "query": query,
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
                    "sources": [],
                    "search_results_count": 0,
                    "filters_applied": self._build_filters_info(semantic_tags, lesson_id, book_id)
                }
            
            # Step 2: Tạo context từ search results
            context, sources = self._build_context_and_sources(search_results, book_id)
            
            # Step 3: Tạo prompt cho LLM
            rag_prompt = self._build_rag_prompt(context, query)
            
            # Step 4: Gọi LLM
            llm_result = await llm_service.generate_content(
                prompt=rag_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if not llm_result.get("success"):
                return {
                    "success": False,
                    "error": f"LLM generation failed: {llm_result.get('error')}"
                }
            
            # Step 5: Làm sạch answer text
            raw_answer = llm_result.get("text", "")
            clean_answer = self.textbook_service.clean_text_content(raw_answer)
            
            return {
                "success": True,
                "query": query,
                "answer": clean_answer,
                "sources": sources,
                "search_results_count": len(search_results),
                "filters_applied": self._build_filters_info(semantic_tags, lesson_id, book_id)
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "success": False,
                "error": f"RAG query failed: {str(e)}"
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
        from app.services.qdrant_service import qdrant_service
        
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
        """Tạo prompt cho LLM"""
        return f"""Bạn là một trợ lý AI chuyên về giáo dục. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

NGUYÊN TẮC:
- Chỉ sử dụng thông tin từ các nguồn được cung cấp
- Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu
- Nếu thông tin không đủ để trả lời, hãy nói rõ
- Trích dẫn nguồn khi cần thiết (ví dụ: "Theo nguồn 1...")

THÔNG TIN TỪ TÀI LIỆU:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""
    
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
        """Thực hiện global semantic search"""
        from app.services.qdrant_service import qdrant_service

        if not qdrant_service.qdrant_client:
            return {
                "success": False,
                "error": "Qdrant service not available"
            }

        collections = qdrant_service.qdrant_client.get_collections().collections
        textbook_collections = [c.name for c in collections if c.name.startswith("textbook_")]

        if not textbook_collections:
            return {
                "success": True,
                "query": query,
                "semantic_filters": semantic_filters,
                "results": [],
                "message": "No textbooks found"
            }

        all_results = []
        for collection_name in textbook_collections:
            book_id_temp = collection_name.replace("textbook_", "")
            book_result = await qdrant_service.search_textbook(
                book_id=book_id_temp,
                query=query,
                limit=limit,
                semantic_filters=semantic_filters if semantic_filters else None
            )

            if book_result.get("success") and book_result.get("results"):
                for res in book_result["results"]:
                    res["book_id"] = book_id_temp
                    # Filter by confidence if specified
                    if min_confidence > 0.0:
                        semantic_tags_data = res.get("semantic_tags", [])
                        max_confidence = max([tag.get("confidence", 0.0) for tag in semantic_tags_data], default=0.0)
                        if max_confidence >= min_confidence:
                            all_results.append(res)
                    else:
                        all_results.append(res)

        # Sort by score và limit
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            "success": True,
            "query": query,
            "semantic_filters": semantic_filters,
            "results": all_results[:limit],
            "total_found": len(all_results),
            "collections_searched": len(textbook_collections)
        }

# Singleton instance
rag_service = RAGService()
