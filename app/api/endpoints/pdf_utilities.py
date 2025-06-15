import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def create_embeddings(enhanced_result: Dict[str, Any], book_metadata: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Create embeddings for textbook content

    Args:
        enhanced_result: Result from textbook processing
        book_metadata: Book metadata
        result: Current result dictionary

    Returns:
        Updated result dictionary
    """
    try:
        from app.services.qdrant_service import qdrant_service

        # Sử dụng trực tiếp dictionary từ enhanced_result
        book_structure_dict = enhanced_result.get("book")

        # Nếu book_structure là string JSON, parse nó
        if isinstance(book_structure_dict, str):
            book_structure_dict = json.loads(book_structure_dict)

        # Tạo embeddings và lưu vào Qdrant
        embedding_result = await qdrant_service.process_textbook(
            book_id=book_metadata.get("id"),
            book_structure=book_structure_dict,  # Đảm bảo gửi dictionary
        )

        # Thêm thông tin về embeddings vào kết quả
        result["embeddings_created"] = embedding_result.get("success", False)
        result["embeddings_info"] = {
            "collection_name": embedding_result.get("collection_name"),
            "vector_count": embedding_result.get("total_chunks", 0),
            "vector_dimension": embedding_result.get("vector_dimension"),
        }
        result["message"] += " with searchable embeddings in Qdrant"
    except Exception as e:
        logger.warning(f"Embeddings creation failed: {e}")
        result["embeddings_created"] = False
        result["embeddings_error"] = str(e)

    return result
