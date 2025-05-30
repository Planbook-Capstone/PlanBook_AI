from typing import List, Dict, Any, Optional
import logging
from app.services.embedding_service import embedding_service
from app.database.connection import get_database_sync, CHEMISTRY_LESSONS_COLLECTION, CHEMISTRY_CHAPTERS_COLLECTION
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    """
    Retrieval-Augmented Generation Service cho Chemistry content
    Tìm kiếm và retrieve relevant content từ MongoDB để hỗ trợ lesson plan generation
    """
    
    def __init__(self):
        self.top_k = settings.TOP_K_DOCUMENTS
        
    def search_relevant_content(
        self, 
        query: str, 
        grade: Optional[str] = None,
        content_types: List[str] = ["lesson", "chapter"],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tìm kiếm nội dung liên quan đến query
        
        Args:
            query: Câu hỏi/chủ đề cần tìm
            grade: Lớp (10, 11, 12)
            content_types: Loại content cần tìm
            top_k: Số lượng kết quả
            
        Returns:
            Dict chứa relevant content và metadata
        """
        try:
            if top_k is None:
                top_k = self.top_k
            
            all_results = []
            
            # Tìm kiếm cho từng content type
            for content_type in content_types:
                metadata_filter = {}
                if grade:
                    metadata_filter["grade"] = grade
                
                results = embedding_service.similarity_search(
                    query=query,
                    top_k=top_k,
                    content_type=content_type,
                    metadata_filter=metadata_filter
                )
                
                # Thêm content_type vào kết quả
                for result in results:
                    result["search_content_type"] = content_type
                
                all_results.extend(results)
            
            # Sort theo similarity score
            all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            # Lấy top_k kết quả tốt nhất
            top_results = all_results[:top_k]
            
            # Enrich với thông tin chi tiết từ database
            enriched_results = self._enrich_results(top_results)
            
            # Tổng hợp context
            context = self._build_context(enriched_results)
            
            logger.info(f"Found {len(enriched_results)} relevant documents for query: {query[:50]}...")
            
            return {
                "query": query,
                "total_results": len(enriched_results),
                "results": enriched_results,
                "context": context,
                "metadata": {
                    "grade_filter": grade,
                    "content_types": content_types,
                    "top_k": top_k
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to search relevant content: {e}")
            raise
    
    def _enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich kết quả với thông tin chi tiết từ database"""
        try:
            db = get_database_sync()
            enriched = []
            
            for result in results:
                content_id = result.get("content_id")
                content_type = result.get("content_type")
                
                if content_type == "lesson":
                    # Lấy thông tin lesson
                    lesson_collection = db[CHEMISTRY_LESSONS_COLLECTION]
                    lesson = lesson_collection.find_one({"_id": content_id})
                    
                    if lesson:
                        enriched_result = {
                            **result,
                            "lesson_info": {
                                "title": lesson.get("title"),
                                "lesson_number": lesson.get("lesson_number"),
                                "objectives": lesson.get("objectives", []),
                                "key_concepts": lesson.get("key_concepts", []),
                                "formulas": lesson.get("formulas", []),
                                "experiments": lesson.get("experiments", []),
                                "exercises": lesson.get("exercises", [])
                            }
                        }
                        enriched.append(enriched_result)
                
                elif content_type == "chapter":
                    # Lấy thông tin chapter
                    chapter_collection = db[CHEMISTRY_CHAPTERS_COLLECTION]
                    chapter = chapter_collection.find_one({"_id": content_id})
                    
                    if chapter:
                        enriched_result = {
                            **result,
                            "chapter_info": {
                                "title": chapter.get("title"),
                                "chapter_number": chapter.get("chapter_number"),
                                "description": chapter.get("description")
                            }
                        }
                        enriched.append(enriched_result)
                else:
                    enriched.append(result)
            
            return enriched
            
        except Exception as e:
            logger.error(f"Failed to enrich results: {e}")
            return results
    
    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Xây dựng context string từ kết quả tìm kiếm"""
        try:
            context_parts = []
            
            for i, result in enumerate(results, 1):
                similarity = result.get("similarity", 0)
                text_chunk = result.get("text_chunk", "")
                content_type = result.get("content_type", "")
                
                context_part = f"[Tài liệu {i} - {content_type.upper()} - Độ liên quan: {similarity:.3f}]\n"
                
                # Thêm thông tin chi tiết
                if "lesson_info" in result:
                    lesson_info = result["lesson_info"]
                    context_part += f"Bài học: {lesson_info.get('title', '')}\n"
                    
                    if lesson_info.get("key_concepts"):
                        context_part += f"Khái niệm chính: {', '.join(lesson_info['key_concepts'][:3])}\n"
                    
                    if lesson_info.get("formulas"):
                        context_part += f"Công thức: {', '.join(lesson_info['formulas'][:3])}\n"
                
                elif "chapter_info" in result:
                    chapter_info = result["chapter_info"]
                    context_part += f"Chương: {chapter_info.get('title', '')}\n"
                
                context_part += f"Nội dung: {text_chunk[:300]}...\n"
                context_part += "-" * 50 + "\n"
                
                context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            return ""
    
    def get_lesson_by_topic(self, topic: str, grade: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Tìm bài học cụ thể theo chủ đề"""
        try:
            db = get_database_sync()
            lesson_collection = db[CHEMISTRY_LESSONS_COLLECTION]
            
            # Tìm kiếm theo title
            query_filter = {"title": {"$regex": topic, "$options": "i"}}
            if grade:
                # Cần join với textbook để lấy grade
                pipeline = [
                    {"$match": query_filter},
                    {
                        "$lookup": {
                            "from": "chemistry_textbooks",
                            "localField": "textbook_id",
                            "foreignField": "_id",
                            "as": "textbook"
                        }
                    },
                    {"$match": {"textbook.grade": grade}},
                    {"$limit": 1}
                ]
                
                results = list(lesson_collection.aggregate(pipeline))
                if results:
                    return results[0]
            else:
                lesson = lesson_collection.find_one(query_filter)
                return lesson
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get lesson by topic: {e}")
            return None
    
    def get_related_lessons(self, lesson_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Lấy các bài học liên quan"""
        try:
            db = get_database_sync()
            lesson_collection = db[CHEMISTRY_LESSONS_COLLECTION]
            
            # Lấy lesson hiện tại
            current_lesson = lesson_collection.find_one({"_id": lesson_id})
            if not current_lesson:
                return []
            
            # Tìm lessons cùng chapter hoặc cùng textbook
            related_query = {
                "_id": {"$ne": lesson_id},
                "$or": [
                    {"chapter_id": current_lesson.get("chapter_id")},
                    {"textbook_id": current_lesson.get("textbook_id")}
                ]
            }
            
            related_lessons = list(lesson_collection.find(related_query).limit(limit))
            return related_lessons
            
        except Exception as e:
            logger.error(f"Failed to get related lessons: {e}")
            return []
    
    def search_by_concepts(self, concepts: List[str], grade: Optional[str] = None) -> List[Dict[str, Any]]:
        """Tìm kiếm theo khái niệm hóa học"""
        try:
            db = get_database_sync()
            lesson_collection = db[CHEMISTRY_LESSONS_COLLECTION]
            
            # Tìm lessons chứa các concepts
            query_filter = {
                "key_concepts": {"$in": concepts}
            }
            
            if grade:
                # Join với textbook để filter theo grade
                pipeline = [
                    {"$match": query_filter},
                    {
                        "$lookup": {
                            "from": "chemistry_textbooks",
                            "localField": "textbook_id",
                            "foreignField": "_id",
                            "as": "textbook"
                        }
                    },
                    {"$match": {"textbook.grade": grade}},
                    {"$limit": 10}
                ]
                
                results = list(lesson_collection.aggregate(pipeline))
            else:
                results = list(lesson_collection.find(query_filter).limit(10))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by concepts: {e}")
            return []

# Global instance
rag_service = RAGService()
