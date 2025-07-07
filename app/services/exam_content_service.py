"""
Service để tìm kiếm và xử lý nội dung bài học cho việc tạo câu hỏi thi
"""
import logging
from typing import Dict, List, Any, Optional
from app.services.qdrant_service import get_qdrant_service
from app.models.exam_models import SearchContentRequest, LessonContentResponse
from qdrant_client import models as qdrant_models

logger = logging.getLogger(__name__)


class ExamContentService:
    """Service để tìm kiếm và xử lý nội dung cho việc tạo đề thi"""

    def __init__(self):
        self.qdrant_service = get_qdrant_service()

    async def get_multiple_lessons_content_for_exam(
        self, lesson_ids: List[str], search_terms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Lấy nội dung cho nhiều bài học từ Qdrant để tạo câu hỏi

        Args:
            lesson_ids: Danh sách ID của các bài học
            search_terms: Các từ khóa tìm kiếm bổ sung

        Returns:
            Dict chứa nội dung tất cả bài học đã được xử lý
        """
        try:
            # Ensure Qdrant service is initialized
            self.qdrant_service._ensure_service_initialized()

            if not self.qdrant_service.qdrant_client:
                return {
                    "success": False,
                    "error": "Qdrant service not available",
                    "content": None
                }

            logger.info(f"Searching for multiple lesson contents: {lesson_ids}")

            all_lessons_content = {}
            successful_lessons = []
            failed_lessons = []

            # Lấy nội dung cho từng lesson
            for lesson_id in lesson_ids:
                lesson_result = await self.get_lesson_content_for_exam(lesson_id, search_terms)

                if lesson_result.get("success", False):
                    all_lessons_content[lesson_id] = lesson_result
                    successful_lessons.append(lesson_id)
                    logger.info(f"Successfully retrieved content for lesson: {lesson_id}")
                else:
                    failed_lessons.append(lesson_id)
                    logger.warning(f"Failed to retrieve content for lesson: {lesson_id} - {lesson_result.get('error', 'Unknown error')}")

            # Tính toán chất lượng tổng thể
            total_quality = 0.0
            if successful_lessons:
                for lesson_id in successful_lessons:
                    lesson_quality = all_lessons_content[lesson_id].get("search_quality", 0.0)
                    total_quality += lesson_quality
                average_quality = total_quality / len(successful_lessons)
            else:
                average_quality = 0.0

            return {
                "success": len(successful_lessons) > 0,
                "lesson_ids": lesson_ids,
                "successful_lessons": successful_lessons,
                "failed_lessons": failed_lessons,
                "content": all_lessons_content,
                "search_quality": average_quality,
                "total_lessons": len(lesson_ids),
                "successful_count": len(successful_lessons),
                "failed_count": len(failed_lessons)
            }

        except Exception as e:
            logger.error(f"Error getting multiple lessons content for exam: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": None
            }

    async def get_lesson_content_for_exam(
        self, lesson_id: str, search_terms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Lấy nội dung bài học từ Qdrant để tạo câu hỏi

        Args:
            lesson_id: ID của bài học
            search_terms: Các từ khóa tìm kiếm bổ sung

        Returns:
            Dict chứa nội dung bài học đã được xử lý
        """
        try:
            # Ensure Qdrant service is initialized
            self.qdrant_service._ensure_service_initialized()

            if not self.qdrant_service.qdrant_client:
                return {
                    "success": False,
                    "error": "Qdrant service not available",
                    "content": None
                }

            logger.info(f"Searching for lesson content: {lesson_id}")

            # 1. Tìm kiếm lesson trong tất cả collections
            lesson_content = await self._find_lesson_in_collections(lesson_id)
            
            if not lesson_content["success"]:
                return lesson_content

            # 2. Nếu có search terms bổ sung, tìm kiếm thêm nội dung liên quan
            if search_terms:
                additional_content = await self._search_related_content(
                    lesson_content["collection_name"], search_terms
                )
                lesson_content["additional_content"] = additional_content

            # 3. Xử lý và cấu trúc nội dung cho việc tạo câu hỏi
            processed_content = await self._process_content_for_exam(lesson_content)

            return {
                "success": True,
                "lesson_id": lesson_id,
                "content": processed_content,
                "search_quality": self._calculate_search_quality(processed_content)
            }

        except Exception as e:
            logger.error(f"Error getting lesson content for exam: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": None
            }

    async def _find_lesson_in_collections(self, lesson_id: str) -> Dict[str, Any]:
        """Tìm kiếm lesson trong tất cả Qdrant collections"""
        try:
            collections = self.qdrant_service.qdrant_client.get_collections().collections
            
            for collection in collections:
                if collection.name.startswith("textbook_"):
                    try:
                        # Tìm kiếm lesson_id trong collection này
                        search_result = self.qdrant_service.qdrant_client.scroll(
                            collection_name=collection.name,
                            scroll_filter=qdrant_models.Filter(
                                must=[
                                    qdrant_models.FieldCondition(
                                        key="lesson_id",
                                        match=qdrant_models.MatchValue(value=lesson_id),
                                    )
                                ]
                            ),
                            limit=100,  # Lấy nhiều chunks của lesson
                            with_payload=True,
                        )

                        if search_result[0]:  # Tìm thấy lesson
                            lesson_chunks = []
                            lesson_info = {}

                            # Collect chunks with index for sorting
                            chunks_with_index = []
                            for point in search_result[0]:
                                payload = point.payload or {}

                                # Bỏ qua metadata points
                                if payload.get("type") == "metadata":
                                    continue

                                # Lưu thông tin lesson
                                if not lesson_info:
                                    lesson_info = {
                                        "lesson_id": payload.get("lesson_id", ""),
                                        "lesson_title": payload.get("lesson_title", ""),
                                        "chapter_title": payload.get("chapter_title", ""),
                                        "chapter_id": payload.get("chapter_id", ""),
                                    }

                                # Collect chunks with index for sorting
                                chunk_index = payload.get("chunk_index", 0)
                                chunks_with_index.append((chunk_index, {
                                    "text": payload.get("text", ""),
                                    "page": payload.get("page", 0),
                                    "type": payload.get("type", "content"),
                                    "section": payload.get("section", ""),
                                }))

                            # Sort by chunk_index to maintain correct order
                            chunks_with_index.sort(key=lambda x: x[0])

                            # Extract sorted chunks
                            lesson_chunks = [chunk for _, chunk in chunks_with_index]

                            return {
                                "success": True,
                                "collection_name": collection.name,
                                "lesson_info": lesson_info,
                                "content_chunks": lesson_chunks,
                                "total_chunks": len(lesson_chunks)
                            }

                    except Exception as e:
                        logger.warning(f"Error searching in collection {collection.name}: {e}")
                        continue

            # Fallback: Tìm trong MongoDB nếu không có trong Qdrant
            logger.info(f"Lesson {lesson_id} not found in Qdrant, searching in MongoDB...")
            mongodb_result = await self._find_lesson_in_mongodb(lesson_id)
            if mongodb_result["success"]:
                return mongodb_result

            return {
                "success": False,
                "error": f"Lesson {lesson_id} not found in any collection"
            }

        except Exception as e:
            logger.error(f"Error finding lesson in collections: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _find_lesson_in_mongodb(self, lesson_id: str) -> Dict[str, Any]:
        """Tìm kiếm lesson trong MongoDB collections"""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from app.core.config import settings

            # Kết nối MongoDB
            client = AsyncIOMotorClient(settings.MONGODB_URL)
            db = client[settings.MONGODB_DATABASE]

            # Tìm trong textbooks collection
            textbooks = await db.textbooks.find({}).to_list(length=None)

            for book in textbooks:
                chapters = book.get("chapters", [])
                for chapter in chapters:
                    lessons = chapter.get("lessons", [])
                    for lesson in lessons:
                        if lesson.get("lesson_id") == lesson_id:
                            # Tìm thấy lesson, tạo content chunks từ content
                            content = lesson.get("content", "")

                            # Chia content thành chunks
                            content_chunks = []
                            if content:
                                # Chia theo đoạn văn
                                paragraphs = content.split('\n\n')
                                for i, paragraph in enumerate(paragraphs):
                                    if paragraph.strip():
                                        content_chunks.append({
                                            "text": paragraph.strip(),
                                            "page": i + 1,
                                            "type": "content",
                                            "section": "main",
                                        })

                            lesson_info = {
                                "lesson_id": lesson.get("lesson_id", ""),
                                "lesson_title": lesson.get("title", ""),
                                "chapter_title": chapter.get("title", ""),
                                "chapter_id": chapter.get("chapter_id", ""),
                            }

                            return {
                                "success": True,
                                "collection_name": "mongodb_textbooks",
                                "lesson_info": lesson_info,
                                "content_chunks": content_chunks,
                                "total_chunks": len(content_chunks)
                            }

            return {
                "success": False,
                "error": f"Lesson {lesson_id} not found in MongoDB"
            }

        except Exception as e:
            logger.error(f"Error finding lesson in MongoDB: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _search_related_content(
        self, collection_name: str, search_terms: List[str]
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm nội dung liên quan dựa trên search terms"""
        try:
            related_content = []
            
            for term in search_terms:
                # Tạo embedding cho search term
                query_vector = self.qdrant_service.embedding_model.encode(term).tolist()
                
                # Tìm kiếm trong collection
                search_result = self.qdrant_service.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=5,  # Giới hạn kết quả cho mỗi term
                    with_payload=True,
                    score_threshold=0.4,  # Threshold cao hơn để đảm bảo chất lượng
                )

                for scored_point in search_result:
                    payload = scored_point.payload or {}
                    
                    # Bỏ qua metadata
                    if payload.get("type") == "metadata":
                        continue

                    related_content.append({
                        "text": payload.get("text", ""),
                        "score": scored_point.score,
                        "search_term": term,
                        "lesson_title": payload.get("lesson_title", ""),
                        "chapter_title": payload.get("chapter_title", ""),
                    })

            return related_content

        except Exception as e:
            logger.error(f"Error searching related content: {e}")
            return []

    async def _process_content_for_exam(self, lesson_content: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý nội dung để phù hợp cho việc tạo câu hỏi"""
        try:
            content_chunks = lesson_content.get("content_chunks", [])
            lesson_info = lesson_content.get("lesson_info", {})
            additional_content = lesson_content.get("additional_content", [])

            # Gộp và sắp xếp nội dung theo page
            all_content = sorted(content_chunks, key=lambda x: x.get("page", 0))
            
            # Tạo text liên tục từ các chunks
            main_content = "\n\n".join([chunk["text"] for chunk in all_content if chunk["text"]])
            
            # Phân loại nội dung theo section
            content_by_section = {}
            for chunk in all_content:
                section = chunk.get("section", "general")
                if section not in content_by_section:
                    content_by_section[section] = []
                content_by_section[section].append(chunk["text"])

            # Tạo summary cho từng section
            section_summaries = {}
            for section, texts in content_by_section.items():
                section_summaries[section] = "\n".join(texts)

            return {
                "lesson_info": lesson_info,
                "main_content": main_content,
                "content_chunks": all_content,
                "content_by_section": section_summaries,
                "additional_content": additional_content,
                "total_words": len(main_content.split()),
                "total_chunks": len(all_content),
                "available_sections": list(content_by_section.keys())
            }

        except Exception as e:
            logger.error(f"Error processing content for exam: {e}")
            return {}

    def _calculate_search_quality(self, processed_content: Dict[str, Any]) -> float:
        """Tính toán chất lượng search dựa trên nội dung tìm được"""
        try:
            total_words = processed_content.get("total_words", 0)
            total_chunks = processed_content.get("total_chunks", 0)
            sections_count = len(processed_content.get("available_sections", []))

            # Tính điểm dựa trên:
            # - Số lượng từ (nhiều từ = tốt hơn)
            # - Số lượng chunks (nhiều chunks = đa dạng hơn)
            # - Số lượng sections (nhiều sections = cấu trúc tốt hơn)
            
            word_score = min(total_words / 1000, 1.0)  # Tối đa 1000 từ = 1.0
            chunk_score = min(total_chunks / 20, 1.0)  # Tối đa 20 chunks = 1.0
            section_score = min(sections_count / 5, 1.0)  # Tối đa 5 sections = 1.0

            # Trọng số: nội dung (50%), đa dạng (30%), cấu trúc (20%)
            quality_score = (word_score * 0.5) + (chunk_score * 0.3) + (section_score * 0.2)
            
            return round(quality_score, 2)

        except Exception as e:
            logger.error(f"Error calculating search quality: {e}")
            return 0.0

    async def search_content_by_keywords(
        self, lesson_id: str, keywords: List[str], limit: int = 10
    ) -> Dict[str, Any]:
        """
        Tìm kiếm nội dung theo từ khóa cụ thể trong bài học

        Args:
            lesson_id: ID bài học
            keywords: Danh sách từ khóa
            limit: Số lượng kết quả tối đa

        Returns:
            Dict chứa kết quả tìm kiếm
        """
        try:
            # Tìm collection chứa lesson
            lesson_content = await self._find_lesson_in_collections(lesson_id)
            
            if not lesson_content["success"]:
                return lesson_content

            collection_name = lesson_content["collection_name"]
            
            # Tìm kiếm theo từng keyword
            search_results = []
            for keyword in keywords:
                related_content = await self._search_related_content(
                    collection_name, [keyword]
                )
                search_results.extend(related_content)

            # Loại bỏ duplicate và sắp xếp theo score
            unique_results = {}
            for result in search_results:
                text = result["text"]
                if text not in unique_results or result["score"] > unique_results[text]["score"]:
                    unique_results[text] = result

            sorted_results = sorted(
                unique_results.values(), 
                key=lambda x: x["score"], 
                reverse=True
            )[:limit]

            return {
                "success": True,
                "lesson_id": lesson_id,
                "keywords": keywords,
                "results": sorted_results,
                "total_found": len(sorted_results)
            }

        except Exception as e:
            logger.error(f"Error searching content by keywords: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }


# Lazy loading global instance để tránh khởi tạo ngay khi import
_exam_content_service_instance = None

def get_exam_content_service() -> ExamContentService:
    """
    Lấy singleton instance của ExamContentService
    Lazy initialization

    Returns:
        ExamContentService: Service instance
    """
    global _exam_content_service_instance
    if _exam_content_service_instance is None:
        _exam_content_service_instance = ExamContentService()
    return _exam_content_service_instance

# Backward compatibility - deprecated, sử dụng get_exam_content_service() thay thế
# Lazy loading để tránh khởi tạo ngay khi import
def _get_exam_content_service_lazy():
    """Lazy loading cho backward compatibility"""
    return get_exam_content_service()

# Tạo proxy object để lazy loading
class _ExamContentServiceProxy:
    def __getattr__(self, name):
        return getattr(_get_exam_content_service_lazy(), name)

    def __call__(self, *args, **kwargs):
        return _get_exam_content_service_lazy()(*args, **kwargs)

exam_content_service = _ExamContentServiceProxy()
