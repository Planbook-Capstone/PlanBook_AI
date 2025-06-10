from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings
from app.database.connection import get_database_sync, CHEMISTRY_EMBEDDINGS_COLLECTION
from app.database.models import ChemistryEmbedding
from bson import ObjectId
import re

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service xử lý embedding cho nội dung Hóa học
    Sử dụng sentence-transformers với model đã config sẵn
    """

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        self.chunk_size = settings.MAX_CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

        logger.info(f"Initializing EmbeddingService with model: {self.model_name}")

    def _load_model(self):
        """Load embedding model"""
        if self.model is None:
            try:
                # Lazy import to avoid startup issues
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def chunk_text(self, text: str) -> List[str]:
        """
        Chia text thành các chunks với overlap
        Tối ưu cho nội dung Hóa học (giữ nguyên công thức, phương trình)
        """
        # Làm sạch text
        text = re.sub(r'\s+', ' ', text.strip())

        # Tách theo câu trước
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Kiểm tra nếu thêm câu này có vượt quá chunk_size không
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Lưu chunk hiện tại nếu không rỗng
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Bắt đầu chunk mới
                if len(sentence) <= self.chunk_size:
                    current_chunk = sentence
                else:
                    # Câu quá dài, chia nhỏ hơn
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk + " " + word) <= self.chunk_size:
                            temp_chunk = temp_chunk + " " + word if temp_chunk else word
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    current_chunk = temp_chunk

        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Thêm overlap giữa các chunks
        if len(chunks) > 1 and self.chunk_overlap > 0:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    # Lấy overlap từ chunk trước
                    prev_chunk = chunks[i-1]
                    overlap_words = prev_chunk.split()[-self.chunk_overlap:]
                    overlap_text = " ".join(overlap_words)

                    overlapped_chunk = overlap_text + " " + chunk
                    overlapped_chunks.append(overlapped_chunk)

            return overlapped_chunks

        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Tạo embeddings cho list texts"""
        self._load_model()

        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")

        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def generate_single_embedding(self, text: str) -> List[float]:
        """Tạo embedding cho một text"""
        return self.generate_embeddings([text])[0]

    def store_embeddings(
        self,
        content_id: str,
        content_type: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Chia text thành chunks, tạo embeddings và lưu vào MongoDB

        Returns:
            List[str]: Danh sách IDs của embeddings đã lưu
        """
        try:
            # Chia text thành chunks
            chunks = self.chunk_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")

            # Tạo embeddings cho tất cả chunks
            embeddings = self.generate_embeddings(chunks)

            # Lưu vào MongoDB
            db = get_database_sync()
            if db is None:
                raise RuntimeError("Database connection failed")
            collection = db[CHEMISTRY_EMBEDDINGS_COLLECTION]

            embedding_ids = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Convert content_id to ObjectId if it's a string
                content_obj_id = ObjectId(content_id) if isinstance(content_id, str) else content_id

                embedding_doc = ChemistryEmbedding(
                    content_id=content_obj_id,
                    content_type=content_type,
                    text_chunk=chunk,
                    chunk_index=i,
                    embedding=embedding,
                    metadata=metadata or {}
                )

                result = collection.insert_one(embedding_doc.model_dump(by_alias=True))
                embedding_ids.append(str(result.inserted_id))

            logger.info(f"Stored {len(embedding_ids)} embeddings for content {content_id}")
            return embedding_ids

        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        content_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm similarity với query

        Args:
            query: Câu hỏi/query
            top_k: Số lượng kết quả trả về (default từ config)
            content_type: Lọc theo loại content
            metadata_filter: Lọc theo metadata

        Returns:
            List[Dict]: Danh sách kết quả với similarity score
        """
        if top_k is None:
            top_k = settings.TOP_K_DOCUMENTS

        try:
            # Tạo embedding cho query
            query_embedding = self.generate_single_embedding(query)

            # Tạo pipeline aggregation cho MongoDB
            pipeline = []

            # Match filter
            match_filter = {}
            if content_type:
                match_filter["content_type"] = content_type
            if metadata_filter:
                for key, value in metadata_filter.items():
                    match_filter[f"metadata.{key}"] = value

            if match_filter:
                pipeline.append({"$match": match_filter})

            # Vector search (sử dụng $expr để tính cosine similarity)
            pipeline.extend([
                {
                    "$addFields": {
                        "similarity": {
                            "$let": {
                                "vars": {
                                    "dot_product": {
                                        "$reduce": {
                                            "input": {"$range": [0, {"$size": "$embedding"}]},
                                            "initialValue": 0,
                                            "in": {
                                                "$add": [
                                                    "$$value",
                                                    {"$multiply": [
                                                        {"$arrayElemAt": ["$embedding", "$$this"]},
                                                        {"$arrayElemAt": [query_embedding, "$$this"]}
                                                    ]}
                                                ]
                                            }
                                        }
                                    },
                                    "query_norm": {
                                        "$sqrt": {
                                            "$reduce": {
                                                "input": query_embedding,
                                                "initialValue": 0,
                                                "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                            }
                                        }
                                    },
                                    "doc_norm": {
                                        "$sqrt": {
                                            "$reduce": {
                                                "input": "$embedding",
                                                "initialValue": 0,
                                                "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                            }
                                        }
                                    }
                                },
                                "in": {
                                    "$divide": [
                                        "$$dot_product",
                                        {"$multiply": ["$$query_norm", "$$doc_norm"]}
                                    ]
                                }
                            }
                        }
                    }
                },
                {"$sort": {"similarity": -1}},
                {"$limit": top_k}
            ])

            # Thực hiện search
            db = get_database_sync()
            if db is None:
                raise RuntimeError("Database connection failed")
            collection = db[CHEMISTRY_EMBEDDINGS_COLLECTION]

            results = list(collection.aggregate(pipeline))

            logger.info(f"Found {len(results)} similar documents for query")
            return results

        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise

    def get_embeddings_by_content(self, content_id: str) -> List[Dict[str, Any]]:
        """Lấy tất cả embeddings của một content"""
        try:
            db = get_database_sync()
            if db is None:
                raise RuntimeError("Database connection failed")
            collection = db[CHEMISTRY_EMBEDDINGS_COLLECTION]

            # Convert content_id to ObjectId if it's a string
            content_obj_id = ObjectId(content_id) if isinstance(content_id, str) else content_id

            results = list(collection.find(
                {"content_id": content_obj_id},
                sort=[("chunk_index", 1)]
            ))

            return results

        except Exception as e:
            logger.error(f"Failed to get embeddings for content {content_id}: {e}")
            raise

# Global instance
embedding_service = EmbeddingService()
