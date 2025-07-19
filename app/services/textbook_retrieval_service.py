"""
Service để lấy nội dung textbook từ Qdrant - Tối ưu hóa cho việc lấy lesson content
"""
import logging
from typing import Dict, Any, Optional, List
from fastapi import HTTPException
from qdrant_client import models as qdrant_models

logger = logging.getLogger(__name__)


class TextbookRetrievalService:
    """Service chuyên dụng để lấy nội dung lesson từ Qdrant"""

    def __init__(self):
        self.qdrant_service = None

    def _get_qdrant_service(self):
        """Lazy loading của qdrant service"""
        if not self.qdrant_service:
            from app.services.qdrant_service import get_qdrant_service
            self.qdrant_service = get_qdrant_service()
        return self.qdrant_service
    
    async def _find_lesson_in_collections(self, lesson_id: str, book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Tìm lesson trong các textbook collections theo pattern textbook_bookId

        Args:
            lesson_id: ID của lesson cần tìm
            book_id: ID của book cụ thể (optional). Nếu có thì chỉ tìm trong collection textbook_{book_id}

        Returns:
            Dict chứa thông tin lesson hoặc error
        """
        qdrant_service = self._get_qdrant_service()
        qdrant_service._ensure_service_initialized()

        if not qdrant_service.qdrant_client:
            return {"success": False, "error": "Qdrant service not available"}

        try:
            # Nếu có book_id cụ thể, chỉ tìm trong collection đó
            if book_id:
                collection_name = f"textbook_{book_id}"
                try:
                    # Kiểm tra collection có tồn tại không
                    collections = qdrant_service.qdrant_client.get_collections().collections
                    collection_exists = any(col.name == collection_name for col in collections)

                    if not collection_exists:
                        return {"success": False, "error": f"Collection {collection_name} not found"}

                    result = await self._search_in_collection(collection_name, lesson_id)
                    if result["success"]:
                        return result
                    else:
                        return {"success": False, "error": f"Lesson {lesson_id} not found in book {book_id}"}

                except Exception as e:
                    logger.error(f"Error searching in specific collection {collection_name}: {e}")
                    return {"success": False, "error": str(e)}

            # Nếu không có book_id, tìm trong tất cả textbook collections
            else:
                collections = qdrant_service.qdrant_client.get_collections().collections
                for collection in collections:
                    # Chỉ tìm trong các collection có pattern textbook_
                    if collection.name.startswith("textbook_"):
                        try:
                            result = await self._search_in_collection(collection.name, lesson_id)
                            if result["success"]:
                                return result
                        except Exception as e:
                            logger.warning(f"Error searching in collection {collection.name}: {e}")
                            continue

                return {"success": False, "error": f"Lesson {lesson_id} not found in any textbook collection"}

        except Exception as e:
            logger.error(f"Error finding lesson {lesson_id}: {e}")
            return {"success": False, "error": str(e)}



    async def _search_in_collection(self, collection_name: str, lesson_id: str) -> Dict[str, Any]:
        """Tìm lesson trong collection cụ thể"""
        try:
            qdrant_service = self._get_qdrant_service()

            search_result = qdrant_service.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="lesson_id",
                            match=qdrant_models.MatchValue(value=lesson_id),
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
            )

            if search_result[0]:
                return self._process_search_result(search_result[0], collection_name, lesson_id)

            return {"success": False, "error": f"Lesson {lesson_id} not found in {collection_name}"}

        except Exception as e:
            logger.warning(f"Error searching in collection {collection_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_search_result(self, points: List, collection_name: str, lesson_id: str) -> Dict[str, Any]:
        """
        Xử lý kết quả tìm kiếm từ Qdrant

        Args:
            points: Danh sách points từ Qdrant
            collection_name: Tên collection
            lesson_id: ID của lesson

        Returns:
            Dict chứa lesson content đã được xử lý
        """
        try:
            chunks_with_index = []
            lesson_info = {}

            for point in points:
                payload = point.payload or {}
                text_content = payload.get("text", "")
                chunk_index = payload.get("chunk_index", 0)

                # Lấy thông tin lesson từ chunk đầu tiên
                if not lesson_info:
                    lesson_info = {
                        "lesson_id": payload.get("lesson_id", lesson_id),
                        "book_id": payload.get("book_id", ""),
                        "collection_name": collection_name,
                        "content_type": payload.get("content_type", "textbook")
                    }

                if text_content and text_content.strip():
                    chunks_with_index.append((chunk_index, {
                        "text": text_content.strip(),
                        "chunk_type": payload.get("chunk_type", "content"),
                        "semantic_tag": payload.get("semantic_tag", "theory"),
                        "concepts": payload.get("concepts", []),
                        "token_count": payload.get("token_count", 0)
                    }))

            # Sort by chunk_index để giữ thứ tự đúng
            chunks_with_index.sort(key=lambda x: x[0])

            # Tách text content và metadata
            content_chunks = [chunk_data for _, chunk_data in chunks_with_index]
            full_content = " ".join([chunk["text"] for chunk in content_chunks])

            return {
                "success": True,
                "lesson_info": lesson_info,
                "content_chunks": content_chunks,
                "full_content": full_content,
                "total_chunks": len(content_chunks),
                "content_length": len(full_content)
            }

        except Exception as e:
            logger.error(f"Error processing search result: {e}")
            return {"success": False, "error": str(e)}

    async def get_lesson_content(self, lesson_id: str, book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy nội dung của một lesson

        Args:
            lesson_id: ID của lesson cần lấy
            book_id: ID của book cụ thể (optional). Nếu có thì chỉ tìm trong collection textbook_{book_id}

        Returns:
            Dict chứa lesson content và metadata
        """
        try:
            result = await self._find_lesson_in_collections(lesson_id, book_id)

            if not result["success"]:
                raise HTTPException(status_code=404, detail=result["error"])

            return {
                "lesson_id": lesson_id,
                "book_id": result["lesson_info"]["book_id"],
                "lesson_content": result["full_content"],
                "collection_name": result["lesson_info"]["collection_name"],
                "total_chunks": result["total_chunks"],
                "content_length": result["content_length"],
                "content_chunks": result["content_chunks"]
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting lesson content for {lesson_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def get_multiple_lessons_content(self, lesson_ids: List[str], book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy nội dung của nhiều lessons

        Args:
            lesson_ids: Danh sách ID của các lessons cần lấy
            book_id: ID của book cụ thể (optional). Nếu có thì chỉ tìm trong collection textbook_{book_id}

        Returns:
            Dict chứa nội dung tất cả lessons
        """
        try:
            lessons_content = {}
            errors = []

            for lesson_id in lesson_ids:
                try:
                    result = await self._find_lesson_in_collections(lesson_id, book_id)

                    if result["success"]:
                        lessons_content[lesson_id] = {
                            "lesson_id": lesson_id,
                            "book_id": result["lesson_info"]["book_id"],
                            "lesson_content": result["full_content"],
                            "collection_name": result["lesson_info"]["collection_name"],
                            "total_chunks": result["total_chunks"],
                            "content_length": result["content_length"],
                            "content_chunks": result["content_chunks"]
                        }
                    else:
                        errors.append(f"Lesson {lesson_id}: {result['error']}")

                except Exception as e:
                    errors.append(f"Lesson {lesson_id}: {str(e)}")

            return {
                "success": len(lessons_content) > 0,
                "lessons_content": lessons_content,
                "total_lessons": len(lessons_content),
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Error getting multiple lessons content: {e}")
            return {
                "success": False,
                "lessons_content": {},
                "total_lessons": 0,
                "errors": [str(e)]
            }

    async def get_lesson_content_for_exam(self, lesson_id: str, book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy nội dung lesson tối ưu cho việc tạo đề thi
        Ưu tiên các chunks có concepts và semantic_tag phù hợp

        Args:
            lesson_id: ID của lesson cần lấy
            book_id: ID của book cụ thể (optional). Nếu có thì chỉ tìm trong collection textbook_{book_id}

        Returns:
            Dict chứa lesson content được tối ưu cho exam generation
        """
        try:
            result = await self._find_lesson_in_collections(lesson_id, book_id)

            if not result["success"]:
                return {"success": False, "error": result["error"]}

            # Lọc và ưu tiên chunks phù hợp cho tạo đề thi
            exam_chunks = []
            for chunk in result["content_chunks"]:
                # Ưu tiên chunks có concepts hoặc semantic_tag quan trọng
                if (chunk.get("concepts") and len(chunk["concepts"]) > 0) or \
                   chunk.get("semantic_tag") in ["definition", "theory", "example", "formula"]:
                    exam_chunks.append(chunk)

            # Nếu không có chunks phù hợp, lấy tất cả
            if not exam_chunks:
                exam_chunks = result["content_chunks"]

            exam_content = " ".join([chunk["text"] for chunk in exam_chunks])

            return {
                "success": True,
                "lesson_id": lesson_id,
                "book_id": result["lesson_info"]["book_id"],
                "collection_name": result["lesson_info"]["collection_name"],
                "lesson_content": exam_content,
                "content_chunks": exam_chunks,
                "total_chunks": len(exam_chunks),
                "content_length": len(exam_content),
                "optimized_for": "exam_generation"
            }

        except Exception as e:
            logger.error(f"Error getting lesson content for exam {lesson_id}: {e}")
            return {"success": False, "error": str(e)}

    async def get_lesson_content_for_lesson_plan(self, lesson_id: str, book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy nội dung lesson tối ưu cho việc tạo giáo án
        Ưu tiên các chunks có cấu trúc và thông tin giảng dạy

        Args:
            lesson_id: ID của lesson cần lấy
            book_id: ID của book cụ thể (optional). Nếu có thì chỉ tìm trong collection textbook_{book_id}

        Returns:
            Dict chứa lesson content được tối ưu cho lesson plan generation
        """
        try:
            result = await self._find_lesson_in_collections(lesson_id, book_id)

            if not result["success"]:
                return {"success": False, "error": result["error"]}

            # Lọc và ưu tiên chunks phù hợp cho tạo giáo án
            plan_chunks = []
            for chunk in result["content_chunks"]:
                # Ưu tiên chunks có cấu trúc giảng dạy
                if chunk.get("chunk_type") in ["heading", "section"] or \
                   chunk.get("semantic_tag") in ["theory", "definition", "example", "exercise"]:
                    plan_chunks.append(chunk)

            # Nếu không có chunks phù hợp, lấy tất cả
            if not plan_chunks:
                plan_chunks = result["content_chunks"]

            plan_content = " ".join([chunk["text"] for chunk in plan_chunks])

            return {
                "success": True,
                "lesson_id": lesson_id,
                "book_id": result["lesson_info"]["book_id"],
                "collection_name": result["lesson_info"]["collection_name"],
                "lesson_content": plan_content,
                "content_chunks": plan_chunks,
                "total_chunks": len(plan_chunks),
                "content_length": len(plan_content),
                "optimized_for": "lesson_plan_generation"
            }

        except Exception as e:
            logger.error(f"Error getting lesson content for lesson plan {lesson_id}: {e}")
            return {"success": False, "error": str(e)}

    async def get_multiple_lessons_content_for_exam(self, lesson_ids: List[str], book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy nội dung nhiều lessons tối ưu cho việc tạo đề thi

        Args:
            lesson_ids: Danh sách ID của các lessons cần lấy
            book_id: ID của book cụ thể (optional). Nếu có thì chỉ tìm trong collection textbook_{book_id}

        Returns:
            Dict chứa nội dung tất cả lessons được tối ưu cho exam generation
        """
        try:
            lessons_content = {}
            errors = []

            for lesson_id in lesson_ids:
                result = await self.get_lesson_content_for_exam(lesson_id, book_id)

                if result["success"]:
                    lessons_content[lesson_id] = result
                else:
                    errors.append(f"Lesson {lesson_id}: {result['error']}")

            return {
                "success": len(lessons_content) > 0,
                "lessons_content": lessons_content,
                "total_lessons": len(lessons_content),
                "errors": errors,
                "optimized_for": "exam_generation"
            }

        except Exception as e:
            logger.error(f"Error getting multiple lessons content for exam: {e}")
            return {
                "success": False,
                "lessons_content": {},
                "total_lessons": 0,
                "errors": [str(e)],
                "optimized_for": "exam_generation"
            }


# Factory function để tạo TextbookRetrievalService instance
def get_textbook_retrieval_service() -> TextbookRetrievalService:
    """
    Tạo TextbookRetrievalService instance mới

    Returns:
        TextbookRetrievalService: Fresh instance
    """
    return TextbookRetrievalService()
