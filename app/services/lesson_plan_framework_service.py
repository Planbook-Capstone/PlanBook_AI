"""
Lesson Plan Framework Service - Xử lý khung giáo án template
"""

import asyncio
import logging
import uuid
import os
import tempfile
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
)
from fastapi import UploadFile

from app.core.config import settings
from app.services.simple_ocr_service import SimpleOCRService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class LessonPlanFrameworkService:
    """Service xử lý khung giáo án template"""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.frameworks_collection: Optional[AsyncIOMotorCollection] = None
        self.ocr_service = SimpleOCRService()
        self.llm_service = LLMService()

    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.db = self.client[settings.MONGODB_DATABASE]
            self.frameworks_collection = self.db["lesson_plan_frameworks"]

            # Create indexes
            await self.frameworks_collection.create_index("framework_id", unique=True)
            await self.frameworks_collection.create_index("name")
            await self.frameworks_collection.create_index("created_at")

            logger.info("LessonPlanFrameworkService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LessonPlanFrameworkService: {e}")
            raise

    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self.frameworks_collection:
            await self.initialize()

    async def process_framework_file(
        self, framework_name: str, framework_file: UploadFile
    ) -> Dict[str, Any]:
        """
        Xử lý file khung giáo án: OCR -> LLM phân tích -> lưu MongoDB

        Args:
            framework_name: Tên khung giáo án
            framework_file: File upload

        Returns:
            Dict chứa thông tin framework đã xử lý
        """
        try:
            # Ensure service is initialized
            await self._ensure_initialized()

            # Generate framework ID
            framework_id = str(uuid.uuid4())

            # Validate filename
            if not framework_file.filename:
                raise ValueError("Filename is required")

            # Read file content
            file_content = await framework_file.read()

            # Extract text from file
            extracted_text = await self._extract_text_from_file(
                file_content, framework_file.filename
            )

            if not extracted_text:
                raise ValueError("Không thể trích xuất text từ file")

            # Analyze framework structure with LLM
            framework_structure = await self._analyze_framework_structure(
                extracted_text
            )

            # Create framework document
            framework_doc = {
                "framework_id": framework_id,
                "name": framework_name,
                "filename": framework_file.filename,
                "original_text": extracted_text,
                "structure": framework_structure,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "status": "active",
            }  # Save to MongoDB
            if self.frameworks_collection is not None:
                await self.frameworks_collection.insert_one(framework_doc)
            else:
                raise RuntimeError("Database not initialized")

            logger.info(
                f"Framework {framework_name} processed and saved with ID: {framework_id}"
            )

            return {
                "framework_id": framework_id,
                "name": framework_name,
                "filename": framework_file.filename,
                "structure": framework_structure,
                "created_at": framework_doc["created_at"].isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing framework file: {e}")
            raise

    async def _extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Trích xuất text từ file"""
        try:
            file_extension = filename.lower().split(".")[-1]

            if file_extension == "pdf":
                return await self._extract_text_from_pdf(file_content)
            elif file_extension in ["docx", "doc"]:
                return await self._extract_text_from_word(file_content)
            elif file_extension == "txt":
                return file_content.decode("utf-8")
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        except Exception as e:
            logger.error(f"Error extracting text from file: {e}")
            raise

    async def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Trích xuất text từ PDF sử dụng OCR"""
        try:
            # Use OCR service to extract text
            text, metadata = await self.ocr_service.extract_text_from_pdf(
                file_content, "framework.pdf"
            )

            if text and len(text.strip()) > 0:
                return text
            else:
                raise ValueError("Could not extract text from PDF")

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    async def _extract_text_from_word(self, file_content: bytes) -> str:
        """Trích xuất text từ Word document"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(file_content)
                temp_file.flush()

                try:
                    import docx

                    doc = docx.Document(temp_file.name)
                    text = []
                    for paragraph in doc.paragraphs:
                        text.append(paragraph.text)
                    return "\n".join(text)
                except ImportError:
                    raise ValueError(
                        "python-docx not installed. Cannot process Word documents."
                    )
                finally:
                    os.unlink(temp_file.name)

        except Exception as e:
            logger.error(f"Error extracting text from Word document: {e}")
            raise

    async def _analyze_framework_structure(self, text: str) -> Dict[str, Any]:
        """Phân tích cấu trúc khung giáo án bằng LLM"""
        try:
            prompt = f"""
Phân tích văn bản khung giáo án sau và trích xuất cấu trúc thành JSON với các thành phần chính:

Văn bản khung giáo án:
{text}

Hãy phân tích và trả về JSON với cấu trúc sau:
{{
    "title_format": "Cách định dạng tiêu đề giáo án",
    "sections": [
        {{
            "name": "Tên phần",
            "description": "Mô tả phần này",
            "required": true/false,
            "subsections": [...]
        }}
    ],
    "metadata_fields": [
        {{
            "field_name": "Tên trường",
            "description": "Mô tả trường",
            "required": true/false,
            "type": "text/number/list/..."
        }}
    ],
    "content_structure": {{
        "introduction": "Mô tả phần mở đầu",
        "main_content": "Mô tả phần nội dung chính", 
        "activities": "Mô tả phần hoạt động",
        "assessment": "Mô tả phần đánh giá",
        "conclusion": "Mô tả phần kết thúc"
    }},
    "formatting_guidelines": [
        "Guideline 1",
        "Guideline 2"
    ]
}}

Chỉ trả về JSON, không thêm text giải thích.
"""

            # Use LLM format_document_text method
            response = await self.llm_service.format_document_text(
                prompt, "lesson_plan_framework"
            )

            if response.get("success", False):
                # Try to parse JSON from response
                try:
                    structure = json.loads(response["formatted_text"])
                    return structure
                except json.JSONDecodeError:
                    # If direct JSON parsing fails, extract JSON from text
                    content = response["formatted_text"]
                    start_idx = content.find("{")
                    end_idx = content.rfind("}") + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = content[start_idx:end_idx]
                        structure = json.loads(json_str)
                        return structure
                    else:
                        raise ValueError(
                            "Could not extract valid JSON from LLM response"
                        )
            else:
                raise ValueError(
                    f"LLM analysis failed: {response.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Error analyzing framework structure: {e}")
            # Return default structure if LLM fails
            return {
                "title_format": "Giáo án [Môn học] - [Chủ đề]",
                "sections": [
                    {
                        "name": "Thông tin chung",
                        "description": "Thông tin về môn học, lớp, thời gian",
                        "required": True,
                        "subsections": ["Môn học", "Lớp", "Thời gian", "Tiết học"],
                    },
                    {
                        "name": "Mục tiêu bài học",
                        "description": "Các mục tiêu kiến thức, kỹ năng, thái độ",
                        "required": True,
                        "subsections": ["Kiến thức", "Kỹ năng", "Thái độ"],
                    },
                    {
                        "name": "Hoạt động dạy học",
                        "description": "Các hoạt động trong giờ học",
                        "required": True,
                        "subsections": [
                            "Khởi động",
                            "Hình thành kiến thức",
                            "Luyện tập",
                            "Vận dụng",
                        ],
                    },
                ],
                "metadata_fields": [
                    {
                        "field_name": "subject",
                        "description": "Môn học",
                        "required": True,
                        "type": "text",
                    },
                    {
                        "field_name": "grade",
                        "description": "Khối lớp",
                        "required": True,
                        "type": "text",
                    },
                    {
                        "field_name": "topic",
                        "description": "Chủ đề bài học",
                        "required": True,
                        "type": "text",
                    },
                ],
                "content_structure": {
                    "introduction": "Phần mở đầu giới thiệu bài học",
                    "main_content": "Nội dung chính của bài học",
                    "activities": "Các hoạt động học tập",
                    "assessment": "Đánh giá kết quả học tập",
                    "conclusion": "Tổng kết và dặn dò",
                },
                "formatting_guidelines": [
                    "Sử dụng font Times New Roman, size 12",
                    "Căn lề trái cho nội dung",
                    "Đánh số thứ tự cho các phần",
                ],
            }

    async def get_framework_by_id(self, framework_id: str) -> Optional[Dict[str, Any]]:
        """Lấy framework theo ID"""
        try:
            await self._ensure_initialized()

            if self.frameworks_collection is not None:
                framework = await self.frameworks_collection.find_one(
                    {"framework_id": framework_id, "status": "active"}
                )
                if framework:
                    framework["_id"] = str(framework["_id"])
                    framework["created_at"] = framework["created_at"].isoformat()
                    framework["updated_at"] = framework["updated_at"].isoformat()
                return framework
            return None
        except Exception as e:
            logger.error(f"Error getting framework by ID: {e}")
            return None

    async def get_all_frameworks(self) -> List[Dict[str, Any]]:
        """Lấy tất cả frameworks"""
        try:
            logger.info("Getting all frameworks...")
            await self._ensure_initialized()

            logger.info(
                f"Database initialized: {self.frameworks_collection is not None}"
            )

            if self.frameworks_collection is not None:
                # Count total documents first
                total_count = await self.frameworks_collection.count_documents({})
                active_count = await self.frameworks_collection.count_documents(
                    {"status": "active"}
                )
                logger.info(f"Total documents in collection: {total_count}")
                logger.info(f"Active documents in collection: {active_count}")

                cursor = self.frameworks_collection.find({"status": "active"}).sort(
                    "created_at", -1
                )

                frameworks = []
                async for framework in cursor:
                    framework["_id"] = str(framework["_id"])
                    framework["created_at"] = framework["created_at"].isoformat()
                    framework["updated_at"] = framework["updated_at"].isoformat()
                    frameworks.append(framework)

                logger.info(f"Retrieved {len(frameworks)} frameworks")
                return frameworks
            else:
                logger.warning("Database collection is None")
                return []
        except Exception as e:
            logger.error(f"Error getting all frameworks: {e}")
            return []

    async def delete_framework(self, framework_id: str) -> bool:
        """Xóa framework (soft delete)"""
        try:
            await self._ensure_initialized()

            if self.frameworks_collection is not None:
                result = await self.frameworks_collection.update_one(
                    {"framework_id": framework_id},
                    {"$set": {"status": "deleted", "updated_at": datetime.utcnow()}},
                )
                return result.modified_count > 0
            return False
        except Exception as e:
            logger.error(f"Error deleting framework: {e}")
            return False

    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()


# Global instance
lesson_plan_framework_service = LessonPlanFrameworkService()
