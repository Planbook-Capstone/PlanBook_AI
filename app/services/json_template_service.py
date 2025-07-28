"""
JSON Template Processing Service
Xử lý slide generation với JSON template từ frontend thay vì Google Slides
"""

import logging
import re
import copy
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from app.services.llm_service import get_llm_service
from app.services.textbook_retrieval_service import get_textbook_retrieval_service

logger = logging.getLogger(__name__)


class JsonTemplateService:
    """Service xử lý JSON template từ frontend"""

    def __init__(self):
        self.llm_service = get_llm_service()
        self.textbook_service = get_textbook_retrieval_service()

    def is_available(self) -> bool:
        """Kiểm tra service có sẵn sàng không"""
        return (
            self.llm_service and self.llm_service.is_available() and
            self.textbook_service is not None
        )

    async def process_json_template(
        self,
        lesson_id: str,
        template_json: Dict[str, Any],
        config_prompt: Optional[str] = None,
        book_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Xử lý JSON template với workflow tối ưu hóa 3 bước:
        1. Xây dựng khung slide
        2. Chi tiết hóa từng slide
        3. Gắn placeholder

        Args:
            lesson_id: ID của bài học
            template_json: JSON template từ frontend đã được phân tích sẵn
            config_prompt: Prompt cấu hình tùy chỉnh
            book_id: ID của sách giáo khoa (optional)

        Returns:
            Dict chứa template đã được xử lý
        """
        try:
            logger.info(f"🔄 Starting optimized workflow for lesson: {lesson_id}")
            logger.info(f"🔍 Template JSON type: {type(template_json)}")
            logger.info(f"🔍 Config prompt: {config_prompt}")

            # Validation: Kiểm tra input rỗng hoặc thiếu dữ liệu quan trọng
            if not lesson_id or not lesson_id.strip():
                raise ValueError("lesson_id is empty or missing")

            if not template_json or not isinstance(template_json, dict):
                raise ValueError("template_json is empty or invalid")

            if not template_json.get("slides") or len(template_json.get("slides", [])) == 0:
                raise ValueError("template_json has no slides")

            # Bước 1: Lấy nội dung bài học
            lesson_content = await self._get_lesson_content(lesson_id, book_id)
            logger.info(f"🔍 Lesson content result type: {type(lesson_content)}")

            if not lesson_content.get("success", False):
                error_msg = lesson_content.get("error", "Unknown error in lesson content")
                raise Exception(error_msg)

            content_text = lesson_content.get("content", "")
            if not content_text or not content_text.strip():
                raise ValueError("lesson content is empty")

            # Bước 2: Sử dụng trực tiếp JSON đã được phân tích từ input
            # Input đã có sẵn description trong slides nên không cần phân tích thêm
            logger.info(f"📊 Using pre-analyzed template: {len(template_json['slides'])} slides")
            # Sử dụng trực tiếp template_json với format mới
            analyzed_template = template_json

            # Workflow tối ưu hóa 3 bước
            result = await self._execute_optimized_workflow(
                content_text,
                config_prompt,
                template_json,
                analyzed_template
            )

            # Format nội dung cho frontend (xuống dòng đẹp)
            formatted_result = self._format_content_for_frontend(result)

            # Trả về kết quả với success flag
            return {
                "success": True,
                "lesson_id": lesson_id,
                "processed_template": formatted_result,
                "slides_created": len(formatted_result.get("slides", []))
            }

        except ValueError as ve:
            logger.error(f"❌ Validation error: {ve}")
            return {
                "success": False,
                "error": f"Input validation failed: {str(ve)}",
                "lesson_id": lesson_id,
                "processed_template": {
                    "version": "1.0",
                    "createdAt": datetime.now().isoformat(),
                    "slideFormat": "16:9",
                    "slides": []
                },
                "slides_created": 0
            }
        except Exception as e:
            logger.error(f"❌ Error processing JSON template: {e}")
            return {
                "success": False,
                "error": f"Failed to process JSON template: {str(e)}",
                "lesson_id": lesson_id,
                "processed_template": {
                    "version": "1.0",
                    "createdAt": datetime.now().isoformat(),
                    "slideFormat": "16:9",
                    "slides": []
                },
                "slides_created": 0
            }

    async def process_json_template_with_progress(
        self,
        lesson_id: str,
        template_json: Dict[str, Any],
        config_prompt: Optional[str] = None,
        task_id: Optional[str] = None,
        task_service: Optional[Any] = None,
        user_id: Optional[str] = None,
        book_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Xử lý JSON template với progress tracking cho Celery
        Cập nhật progress theo từng slide hoàn thành
        Gửi Kafka notifications cho từng slide hoàn thành nếu có user_id
        """
        try:
            logger.info(f"🔄 Starting JSON template processing with progress tracking")
            logger.info(f"   Lesson ID: {lesson_id}")
            logger.info(f"   Task ID: {task_id}")
            logger.info(f"   Slides count: {len(template_json.get('slides', []))}")

            # Validation
            if not lesson_id or not lesson_id.strip():
                raise ValueError("lesson_id is empty or missing")

            if not template_json or not isinstance(template_json, dict):
                raise ValueError("template_json is empty or invalid")

            if not template_json.get("slides") or len(template_json.get("slides", [])) == 0:
                raise ValueError("template_json has no slides")

            # Bước 1: Lấy nội dung bài học
            if task_service and task_id:
                await task_service.update_task_progress(
                    task_id,
                    progress=50,
                    message="📚 Đang lấy nội dung bài học..."
                )

            lesson_content = await self._get_lesson_content(lesson_id, book_id)
            if not lesson_content.get("success", False):
                error_msg = lesson_content.get("error", "Unknown error in lesson content")
                raise Exception(error_msg)

            content_text = lesson_content.get("content", "")
            if not content_text or not content_text.strip():
                raise ValueError("lesson content is empty")

            # Bước 2: Sử dụng trực tiếp JSON đã được phân tích từ input
            analyzed_template = template_json

            if task_service and task_id:
                await task_service.update_task_progress(
                    task_id,
                    progress=60,
                    message="🔍 Đang phân tích cấu trúc template..."
                )

            # Workflow tối ưu hóa với progress tracking
            result = await self._execute_optimized_workflow_with_progress(
                content_text,
                config_prompt,
                template_json,
                analyzed_template,
                task_id,
                task_service,
                user_id
            )

            # Format nội dung cho frontend
            formatted_result = self._format_content_for_frontend(result)

            return {
                "success": True,
                "lesson_id": lesson_id,
                "processed_template": formatted_result,
                "slides_created": len(formatted_result.get("slides", []))
            }

        except Exception as e:
            logger.error(f"❌ Error processing JSON template with progress: {e}")
            return {
                "success": False,
                "error": f"Failed to process JSON template: {str(e)}",
                "lesson_id": lesson_id,
                "processed_template": {
                    "version": "1.0",
                    "createdAt": datetime.now().isoformat(),
                    "slideFormat": "16:9",
                    "slides": []
                },
                "slides_created": 0
            }

    async def _get_lesson_content(self, lesson_id: str, book_id: str = None) -> Dict[str, Any]:
        """Lấy nội dung bài học từ TextbookRetrievalService"""
        try:
            logger.info(f"📚 Getting lesson content for: {lesson_id}, book_id: {book_id}")

            # Sử dụng TextbookRetrievalService để lấy lesson content
            lesson_result = await self.textbook_service.get_lesson_content(lesson_id, book_id)

            logger.info(f"🔍 Lesson result keys: {list(lesson_result.keys())}")

            # Extract lesson content từ result
            lesson_content = lesson_result.get("lesson_content", "")

            if not lesson_content or not lesson_content.strip():
                logger.error(f"❌ No lesson content found for lesson_id: {lesson_id}")
                return {
                    "success": False,
                    "error": f"Empty lesson content for lesson_id: {lesson_id}"
                }

            logger.info(f"✅ Retrieved lesson content: {len(lesson_content)} characters")
            logger.info(f"📋 Additional info - Book ID: {lesson_result.get('book_id')}, Total chunks: {lesson_result.get('total_chunks')}")

            return {
                "success": True,
                "content": lesson_content.strip(),
                "book_id": lesson_result.get("book_id"),
                "total_chunks": lesson_result.get("total_chunks"),
                "content_length": lesson_result.get("content_length")
            }

        except Exception as e:
            logger.error(f"❌ Error getting lesson content: {e}")
            return {
                "success": False,
                "error": f"Failed to get lesson content: {str(e)}"
            }

    def _format_content_for_frontend(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format nội dung cho frontend - chuyển \\n thành xuống dòng thật và thêm gạch đầu dòng
        """
        try:
            logger.info("🎨 Formatting content for frontend...")

            # Deep copy để không ảnh hưởng data gốc
            formatted_data = copy.deepcopy(template_data)

            slides = formatted_data.get("slides", [])
            for slide in slides:
                elements = slide.get("elements", [])
                for element in elements:
                    text = element.get("text", "")
                    if text and isinstance(text, str):
                        # Format text đẹp cho frontend
                        formatted_text = self._format_text_content(text)
                        element["text"] = formatted_text

                        # Log để debug
                        if "\\n" in text or len(text.split('\n')) > 1:
                            logger.info(f"🎨 Formatted text in element {element.get('id', 'unknown')}:")
                            logger.info(f"   Before: {text[:100]}...")
                            logger.info(f"   After: {formatted_text[:100]}...")

            logger.info(f"✅ Content formatting complete for {len(slides)} slides")
            return formatted_data

        except Exception as e:
            logger.error(f"❌ Error formatting content for frontend: {e}")
            # Trả về data gốc nếu format lỗi
            return template_data

    def _format_text_content(self, text: str) -> str:
        """
        Format text content với gạch đầu dòng cho TẤT CẢ các câu
        """
        try:
            # Chuyển \\n thành xuống dòng thật
            formatted_text = text.replace("\\n", "\n")

            # Split thành các dòng
            lines = formatted_text.split('\n')

            # Nếu chỉ có 1 dòng, thêm gạch đầu dòng và trả về
            if len(lines) <= 1:
                line = formatted_text.strip()
                if not line:
                    return ""
                # Kiểm tra xem đã có gạch đầu dòng chưa
                if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                    return line
                else:
                    return f"- {line}"

            # Format từng dòng - THÊM GẠCH ĐẦU DÒNG CHO TẤT CẢ
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if not line:  # Bỏ qua dòng trống
                    continue

                # Kiểm tra xem dòng đã có gạch đầu dòng chưa
                if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                    formatted_lines.append(line)
                else:
                    # Thêm gạch đầu dòng cho TẤT CẢ các dòng
                    formatted_lines.append(f"- {line}")

            # Ghép lại với xuống dòng
            result = '\n'.join(formatted_lines)

            return result

        except Exception as e:
            logger.error(f"❌ Error formatting text content: {e}")
            # Trả về text gốc nếu lỗi
            return text.replace("\\n", "\n")



    async def _execute_optimized_workflow(
        self,
        lesson_content: str,
        config_prompt: Optional[str],
        template_json: Dict[str, Any],
        analyzed_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Thực hiện workflow tối ưu hóa 3 bước:
        1. Xây dựng khung slide
        2. Chi tiết hóa từng slide
        3. Gắn placeholder
        """
        try:
            logger.info("🚀 Starting optimized 3-step workflow...")

            # Bước 1: Xây dựng khung slide
            logger.info("📋 Step 1: Generating slide framework...")
            slide_framework = await self._generate_slide_framework(
                lesson_content,
                config_prompt
            )

            if not slide_framework.get("success", False):
                raise Exception(f"Step 1 failed: {slide_framework.get('error', 'Unknown error')}")

            framework_slides = slide_framework.get("slides", [])
            logger.info(f"✅ Step 1 complete: Generated {len(framework_slides)} slide frameworks")
            logger.info(f"---------slide: {framework_slides}")

            # Bước 2 & 3: Chi tiết hóa từng slide, gắn placeholder và map ngay lập tức
            final_template = {
                "version": template_json.get("version", "1.0"),
                "createdAt": datetime.now().isoformat(),
                "slideFormat": template_json.get("slideFormat", "16:9"),
                "slides": []
            }

            # Content index để track việc sử dụng content
            all_parsed_data = {
                "LessonName": [],
                "LessonDescription": [],
                "CreatedDate": [],
                "TitleName": [],
                "TitleContent": [],
                "SubtitleName": [],
                "SubtitleContent": [],
                "ImageName": [],
                "ImageContent": []
            }

            content_index = {
                "LessonName": 0,
                "LessonDescription": 0,
                "CreatedDate": 0,
                "TitleName": 0,
                "TitleContent": 0,
                "SubtitleName": 0,
                "SubtitleContent": 0,
                "ImageName": 0,
                "ImageContent": 0
            }

            # Track used slides để tránh duplicate
            used_slide_ids = set()
            # analyzed_template bây giờ chính là input JSON với format mới
            template_slides = analyzed_template.get("slides", [])

            for i, framework_slide in enumerate(framework_slides):
                slide_num = i + 1
                logger.info(f"🔄 Processing slide {slide_num}/{len(framework_slides)}")

                # Bước 2: Chi tiết hóa slide (bỏ qua slide đầu - slide giới thiệu)
                if slide_num == 1:
                    logger.info(f"⏭️ Skipping detailed processing for slide {slide_num} (introduction slide)")
                    # Sử dụng trực tiếp framework_slide content cho slide giới thiệu
                    detailed_slide = {
                        "success": True,
                        "content": framework_slide
                    }
                else:
                    detailed_slide = await self._detail_slide_content(
                        framework_slide,
                        lesson_content,
                        config_prompt,
                        slide_num
                    )

                    if not detailed_slide.get("success", False):
                        logger.error(f"❌ Step 2 failed for slide {slide_num}: {detailed_slide.get('error', 'Unknown error')}")
                        continue  # Skip slide này

                logger.info(f"---------detailed_slide: {detailed_slide}")

                # Bước 3: Gắn placeholder
                slide_with_placeholders = await self._map_placeholders(
                    detailed_slide.get("content", ""),
                    slide_num
                )

                if not slide_with_placeholders.get("success", False):
                    logger.error(f"❌ Step 3 failed for slide {slide_num}: {slide_with_placeholders.get('error', 'Unknown error')}")
                    continue  # Skip slide này

                slide_data = slide_with_placeholders.get("slide_data", {})
                logger.info(f"✅ Slide {slide_num} content processed successfully")

                # Bước 4: Map ngay lập tức vào template
                mapped_slide = await self._map_single_slide_to_template(
                    slide_data,
                    template_slides,
                    used_slide_ids,
                    all_parsed_data,
                    content_index,
                    slide_num
                )

                if mapped_slide:
                    final_template["slides"].append(mapped_slide)
                    logger.info(f"✅ Slide {slide_num} mapped to template successfully")
                else:
                    logger.error(f"❌ Failed to map slide {slide_num} to template")
                    continue

            logger.info(f"🎉 Optimized workflow complete: {len(final_template.get('slides', []))} slides created")
            return final_template

        except Exception as e:
            logger.error(f"❌ Error in optimized workflow: {e}")
            raise

    async def _execute_optimized_workflow_with_progress(
        self,
        lesson_content: str,
        config_prompt: Optional[str],
        template_json: Dict[str, Any],
        analyzed_template: Dict[str, Any],
        task_id: Optional[str] = None,
        task_service: Optional[Any] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Thực hiện workflow tối ưu hóa với progress tracking
        Cập nhật progress theo từng slide hoàn thành
        Gửi Kafka notifications cho từng slide hoàn thành nếu có user_id
        """
        try:
            logger.info("🚀 Starting optimized workflow with progress tracking...")

            # Bước 1: Xây dựng khung slide
            if task_service and task_id:
                await task_service.update_task_progress(
                    task_id,
                    progress=70,
                    message="📋 Đang tạo khung slide..."
                )

            slide_framework = await self._generate_slide_framework(
                lesson_content,
                config_prompt
            )

            if not slide_framework.get("success", False):
                raise Exception(f"Step 1 failed: {slide_framework.get('error', 'Unknown error')}")

            framework_slides = slide_framework.get("slides", [])
            logger.info(f"✅ Step 1 complete: Generated {len(framework_slides)} slide frameworks")

            # Tạo final template
            final_template = {
                "version": template_json.get("version", "1.0"),
                "createdAt": datetime.now().isoformat(),
                "slideFormat": template_json.get("slideFormat", "16:9"),
                "slides": []
            }

            # Content tracking
            all_parsed_data = {
                "LessonName": [],
                "LessonDescription": [],
                "CreatedDate": [],
                "TitleName": [],
                "TitleContent": [],
                "SubtitleName": [],
                "SubtitleContent": [],
                "ImageName": [],
                "ImageContent": []
            }

            content_index = {
                "LessonName": 0,
                "LessonDescription": 0,
                "CreatedDate": 0,
                "TitleName": 0,
                "TitleContent": 0,
                "SubtitleName": 0,
                "SubtitleContent": 0,
                "ImageName": 0,
                "ImageContent": 0
            }

            # Track used slides
            used_slide_ids = set()
            template_slides = analyzed_template.get("slides", [])

            total_slides = len(framework_slides)
            base_progress = 75  # Bắt đầu từ 75%
            progress_per_slide = 20 / total_slides if total_slides > 0 else 0  # 20% cho tất cả slides

            # Xử lý từng slide với progress tracking
            for i, framework_slide in enumerate(framework_slides):
                slide_num = i + 1
                logger.info(f"🔄 Processing slide {slide_num}/{total_slides}")

                # Cập nhật progress cho slide hiện tại
                current_progress = base_progress + (i * progress_per_slide)
                if task_service and task_id:
                    await task_service.update_task_progress(
                        task_id,
                        progress=int(current_progress),
                        message=f"🤖 Đang xử lý slide {slide_num}/{total_slides}..."
                    )

                # Bước 2: Chi tiết hóa slide (bỏ qua slide đầu - slide giới thiệu)
                if slide_num == 1:
                    logger.info(f"⏭️ Skipping detailed processing for slide {slide_num} (introduction slide)")
                    # Sử dụng trực tiếp framework_slide content cho slide giới thiệu
                    detailed_slide = {
                        "success": True,
                        "content": framework_slide
                    }
                else:
                    detailed_slide = await self._detail_slide_content(
                        framework_slide,
                        lesson_content,
                        config_prompt,
                        slide_num
                    )

                    if not detailed_slide.get("success", False):
                        logger.error(f"❌ Step 2 failed for slide {slide_num}: {detailed_slide.get('error', 'Unknown error')}")
                        continue

                # Bước 3: Gắn placeholder
                slide_with_placeholders = await self._map_placeholders(
                    detailed_slide.get("content", ""),
                    slide_num
                )

                if not slide_with_placeholders.get("success", False):
                    logger.error(f"❌ Step 3 failed for slide {slide_num}: {slide_with_placeholders.get('error', 'Unknown error')}")
                    continue

                slide_data = slide_with_placeholders.get("slide_data", {})

                # Bước 4: Map vào template
                mapped_slide = await self._map_single_slide_to_template(
                    slide_data,
                    template_slides,
                    used_slide_ids,
                    all_parsed_data,
                    content_index,
                    slide_num
                )

                if mapped_slide:
                    final_template["slides"].append(mapped_slide)
                    logger.info(f"✅ Slide {slide_num} completed and added to result")

                    # Cập nhật progress và result từng phần sau khi hoàn thành slide
                    completed_progress = base_progress + ((i + 1) * progress_per_slide)
                    if task_service and task_id:
                        logger.info(f"🔄 Updating partial result for slide {slide_num} - {len(final_template['slides'])} slides completed")

                        # Tạo partial result với slides đã hoàn thành
                        partial_result = {
                            "success": True,
                            "lesson_id": template_json.get("lesson_id", ""),
                            "processed_template": {
                                "version": final_template.get("version", "1.0"),
                                "createdAt": final_template.get("createdAt"),
                                "slideFormat": final_template.get("slideFormat", "16:9"),
                                "slides": final_template["slides"]  # Chứa tất cả slides đã hoàn thành
                            },
                            "slides_created": len(final_template["slides"]),
                            "total_slides": total_slides,
                            "completed_slides": len(final_template["slides"])
                        }

                        logger.info(f"🔄 Calling update_task_progress_with_result for task {task_id}")
                        await task_service.update_task_progress_with_result(
                            task_id,
                            progress=int(completed_progress),
                            message=f"✅ Đã hoàn thành slide {slide_num}/{total_slides}",
                            partial_result=partial_result
                        )
                        logger.info(f"✅ Successfully updated partial result for slide {slide_num}")

                        # Send Kafka notification for slide completion if user_id is available
                        if user_id:
                            from app.services.kafka_service import kafka_service
                            from app.services.kafka_service import safe_kafka_call

                            progress = int((slide_num / total_slides) * 100) if total_slides > 0 else 100
                            safe_kafka_call(
                                kafka_service.send_progress_update_sync,
                                tool_log_id=task_id,
                                task_id=task_id,
                                user_id=user_id,
                                progress=progress,
                                message=f"✅ Đã hoàn thành slide {slide_num}/{total_slides}",
                                status="processing",
                                additional_data={
                                    "slide_number": slide_num,
                                    "total_slides": total_slides,
                                    "completed_slides": partial_result.get("completed_slides", 0),
                                    "partial_result": partial_result
                                }
                            )
                else:
                    logger.error(f"❌ Failed to map slide {slide_num} to template")
                    continue

            # Hoàn thành - cập nhật final result
            if task_service and task_id:
                final_result = {
                    "success": True,
                    "lesson_id": template_json.get("lesson_id", ""),
                    "processed_template": final_template,
                    "slides_created": len(final_template.get("slides", [])),
                    "total_slides": total_slides,
                    "completed_slides": len(final_template.get("slides", []))
                }

                await task_service.update_task_progress_with_result(
                    task_id,
                    progress=95,
                    message=f"🎉 Đã tạo thành công {len(final_template.get('slides', []))} slides",
                    partial_result=final_result
                )

            logger.info(f"🎉 Optimized workflow with progress complete: {len(final_template.get('slides', []))} slides created")
            return final_template

        except Exception as e:
            logger.error(f"❌ Error in optimized workflow with progress: {e}")
            raise

    async def _map_single_slide_to_template(
        self,
        slide_data: Dict[str, Any],
        template_slides: List[Dict[str, Any]],
        used_slide_ids: set,
        all_parsed_data: Dict[str, List[Dict[str, Any]]],
        content_index: Dict[str, int],
        slide_number: int
    ) -> Dict[str, Any]:
        """
        Map một slide đơn lẻ vào template ngay lập tức
        """
        try:
            logger.info(f"🔧 Mapping slide {slide_number} to template...")

            # Lấy parsed data từ slide
            parsed_data = slide_data.get("parsed_data", {})
            placeholder_counts = slide_data.get("placeholder_counts", {})
            required_placeholders = list(placeholder_counts.keys())

            logger.info(f"🔍 Slide {slide_number} requirements:")
            logger.info(f"   Required placeholders: {required_placeholders}")
            logger.info(f"   Required counts: {placeholder_counts}")

            # Thêm parsed data vào all_parsed_data
            for placeholder_type, items in parsed_data.items():
                all_parsed_data[placeholder_type].extend(items)

            # Tìm template phù hợp CHÍNH XÁC
            best_template = self._find_exact_matching_template(
                required_placeholders,
                placeholder_counts,
                template_slides,
                used_slide_ids
            )

            # Nếu không tìm thấy template chưa sử dụng, cho phép reuse template
            if not best_template:
                logger.info(f"🔄 No unused template found for slide {slide_number}, trying to reuse...")
                best_template = self._find_exact_matching_template_with_reuse(
                    required_placeholders,
                    placeholder_counts,
                    template_slides
                )

            if not best_template:
                logger.error(f"❌ No matching template found for slide {slide_number}")
                return None

            template_id = best_template['id']  # Format mới sử dụng 'id' thay vì 'slideId'
            is_reused = template_id in used_slide_ids

            if is_reused:
                logger.info(f"✅ Found exact matching template (REUSED): {template_id}")
            else:
                logger.info(f"✅ Found exact matching template (NEW): {template_id}")

            # Tạo processed slide từ template
            processed_slide = await self._create_processed_slide_from_template(
                best_template,
                all_parsed_data,
                content_index,
                slide_number,
                is_reused
            )

            if processed_slide:
                # Chỉ thêm vào used_slide_ids nếu chưa được sử dụng
                if not is_reused:
                    used_slide_ids.add(template_id)
                logger.info(f"✅ Successfully mapped slide {slide_number} ({'reused' if is_reused else 'new'})")
                return processed_slide
            else:
                logger.error(f"❌ Failed to create processed slide {slide_number}")
                return None

        except Exception as e:
            logger.error(f"❌ Error mapping slide {slide_number} to template: {e}")
            return None

    async def _generate_slide_framework(
        self,
        lesson_content: str,
        config_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Bước 1: Xây dựng khung slide tổng quát
        Input: lesson_content, default_prompt, config_prompt
        Output: Khung slide tổng quát (mỗi slide thể hiện một chủ đề chính, ý định và kiến thức cần truyền đạt)
        """
        try:
            logger.info("📋 Generating slide framework...")

            # Tạo prompt cho việc xây dựng khung slide
            framework_prompt = self._create_framework_prompt(lesson_content, config_prompt)

            # Gọi LLM để tạo khung slide
            llm_response = await self.llm_service.generate_content(
                prompt=framework_prompt,
                max_tokens=20000,
                temperature=0.1
            )

            if not llm_response.get("success", False):
                return {
                    "success": False,
                    "error": f"LLM framework generation failed: {llm_response.get('error', 'Unknown error')}"
                }

            framework_content = llm_response.get("text", "").strip()
            logger.info(f"✅ Framework content generated: {len(framework_content)} characters")

            # Parse framework content thành danh sách slides
            slides = self._parse_framework_content(framework_content)

            if not slides:
                return {
                    "success": False,
                    "error": "No slides found in framework content"
                }

            logger.info(f"✅ Framework parsing complete: {len(slides)} slides")
            return {
                "success": True,
                "slides": slides,
                "raw_content": framework_content
            }

        except Exception as e:
            logger.error(f"❌ Error generating slide framework: {e}")
            return {
                "success": False,
                "error": f"Failed to generate framework: {str(e)}"
            }

    def _create_framework_prompt(
        self,
        lesson_content: str,
        config_prompt: Optional[str] = None
    ) -> str:
        """Tạo prompt cho việc xây dựng khung slide"""

        default_config = config_prompt if config_prompt else """
Bạn là chuyên gia thiết kế nội dung giáo dục. Hãy phân tích nội dung bài học và tạo khung slide logic, dễ theo dõi.
"""

        prompt = f"""
{default_config}

NHIỆM VỤ: Phân tích nội dung bài học và tạo KHUNG SLIDE tổng quát

NỘI DUNG BÀI HỌC:
{lesson_content}

YÊU CẦU KHUNG SLIDE:
1. Tách lesson_content thành các slide với mục đích và nội dung chính rõ ràng
2. Đảm bảo khung slide có tính logic, hợp lý và dễ theo dõi
3. Mỗi slide thể hiện một chủ đề chính, ý định và kiến thức cần truyền đạt
4. Không cần chi tiết, chỉ cần khung tổng quát
5. Slide đầu tiên bắt buộc là slide giới thiệu với ĐÚNG 3 dòng: tên bài học, mô tả ngắn và ngày tạo bài thuyết trình.

FORMAT OUTPUT:

SLIDE 1: [Tên bài thuyết trình]
Mô tả ngắn bài thuyết trình
Ngày thuyết trình: 12-07-2025
---

SLIDE 2: [Tiêu đề slide]
Mục đích: [Mục đích của slide này]
Nội dung chính: 
- [Nội dung chính 1 cần truyền đạt]
- [Nội dung chính 2 cần truyền đạt]
- ....
---

SLIDE 3: [Tiêu đề slide]
Mục đích: [Mục đích của slide này]
Nội dung chính:
- [Nội dung chính 1 cần truyền đạt]
- [Nội dung chính 2 cần truyền đạt]
- ....
---

... (tiếp tục cho các slide khác)

LƯU Ý:
- Chỉ tạo khung tổng quát, không chi tiết hóa
- Đảm bảo logic từ slide này sang slide khác
- Mỗi slide có mục đích rõ ràng trong chuỗi kiến thức
- Slide đầu tiên bắt buộc là slide giới thiệu với ĐÚNG 3 dòng: tên bài học, mô tả ngắn và ngày tạo bài thuyết trình.
"""

        return prompt

    def _parse_framework_content(self, framework_content: str) -> List[Dict[str, Any]]:
        """Parse framework content thành danh sách slides"""
        try:
            slides = []

            # Split theo dấu --- để tách các slide
            slide_blocks = framework_content.split('---')

            for i, block in enumerate(slide_blocks):
                block = block.strip()
                if not block:
                    continue

                slide_data = {
                    "slide_number": i + 1,
                    "title": "",
                    "purpose": "",
                    "main_content": "",
                    "raw_block": block
                }

                # Parse từng dòng trong block
                lines = block.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('SLIDE '):
                        # Extract title từ "SLIDE 1: [Tiêu đề]"
                        if ':' in line:
                            slide_data["title"] = line.split(':', 1)[1].strip()
                    elif line.startswith('Mục đích:'):
                        slide_data["purpose"] = line.replace('Mục đích:', '').strip()
                    elif line.startswith('Nội dung chính:'):
                        slide_data["main_content"] = line.replace('Nội dung chính:', '').strip()

                # Chỉ thêm slide nếu có đủ thông tin cơ bản
                if slide_data["title"] or slide_data["purpose"] or slide_data["main_content"]:
                    slides.append(slide_data)

            logger.info(f"📋 Parsed {len(slides)} slides from framework")
            return slides

        except Exception as e:
            logger.error(f"❌ Error parsing framework content: {e}")
            return []

    async def _detail_slide_content(
        self,
        framework_slide: Dict[str, Any],
        lesson_content: str,
        config_prompt: Optional[str],
        slide_number: int,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Bước 2: Chi tiết hóa nội dung cho từng slide cụ thể
        Input: lesson_content, default_prompt, config_prompt, khung_slide
        Output: Slide chi tiết với nội dung đầy đủ
        """
        try:
            logger.info(f"📝 Detailing slide {slide_number}: {framework_slide.get('title', 'Untitled')}")

            # Tạo prompt cho việc chi tiết hóa slide
            detail_prompt = self._create_detail_prompt(
                framework_slide,
                lesson_content,
                config_prompt,
                slide_number
            )

            # Retry logic cho LLM
            for attempt in range(max_retries):
                logger.info(f"🔄 Attempt {attempt + 1}/{max_retries} for slide {slide_number}")

                llm_response = await self.llm_service.generate_content(
                    prompt=detail_prompt,
                    max_tokens=30000,
                    temperature=0.1
                )

                if llm_response.get("success", False):
                    detailed_content = llm_response.get("text", "").strip()

                    if detailed_content:
                        logger.info(f"✅ Slide {slide_number} detailed successfully: {len(detailed_content)} characters")
                        return {
                            "success": True,
                            "content": detailed_content,
                            "slide_number": slide_number,
                            "framework": framework_slide
                        }
                    else:
                        logger.warning(f"⚠️ Empty content for slide {slide_number}, attempt {attempt + 1}")
                else:
                    logger.warning(f"⚠️ LLM failed for slide {slide_number}, attempt {attempt + 1}: {llm_response.get('error', 'Unknown error')}")

            # Fallback: Trả về content gốc nếu không thể chi tiết hóa
            logger.error(f"❌ Failed to detail slide {slide_number} after {max_retries} attempts")
            fallback_content = f"""
{framework_slide.get('title', 'Slide Content')}

{framework_slide.get('purpose', '')}

{framework_slide.get('main_content', '')}
"""

            return {
                "success": True,
                "content": fallback_content.strip(),
                "slide_number": slide_number,
                "framework": framework_slide,
                "fallback_used": True
            }

        except Exception as e:
            logger.error(f"❌ Error detailing slide {slide_number}: {e}")
            return {
                "success": False,
                "error": f"Failed to detail slide: {str(e)}",
                "slide_number": slide_number
            }

    def _create_detail_prompt(
        self,
        framework_slide: Dict[str, Any],
        lesson_content: str,
        config_prompt: Optional[str],
        slide_number: int
    ) -> str:
        """Tạo prompt cho việc chi tiết hóa slide"""

        default_config = config_prompt if config_prompt else """
Bạn là chuyên gia thiết kế nội dung slide giáo dục chuyên nghiệp. Hãy chi tiết hóa nội dung slide theo yêu cầu.
"""

        prompt = f"""

YÊU CẦU CỦA NGƯỜI DÙNG:
{default_config}

NHIỆM VỤ: Chi tiết hóa nội dung cho slide cụ thể

THÔNG TIN SLIDE CẦN CHI TIẾT HÓA:
- Số slide: {slide_number}
- Tiêu đề: {framework_slide.get('title', 'Không có tiêu đề')}
- Mục đích: {framework_slide.get('purpose', 'Không có mục đích')}
- Nội dung chính: {framework_slide.get('main_content', 'Không có nội dung chính')}

NỘI DUNG BÀI HỌC THAM KHẢO:
{lesson_content}

YÊU CẦU CHI TIẾT HÓA:
1. Chi tiết hóa nội dung cho slide cụ thể dựa trên nội dung bài học 
2. Điều chỉnh ngữ điệu, độ khó, độ chi tiết hoặc nâng cao sao cho phù hợp với đối tượng và bối cảnh thuyết trình theo mục YÊU CẦU CỦA NGƯỜI DÙNG
3. Tạo nội dung đầy đủ, chi tiết
4. Bao gồm định nghĩa, giải thích, ví dụ minh họa nếu cần
5. Đảm bảo nội dung phù hợp với mục đích của slide
6. 🚨 QUAN TRỌNG: Nếu có nhiều mục con, hãy GỘP CHÚNG LẠI để không vượt quá 6 mục

🚨 TUYỆT ĐỐI TRÁNH:
- KHÔNG sử dụng lời chào hỏi: "Chào mừng các em", "Xin chào", "Hôm nay chúng ta sẽ"
- KHÔNG sử dụng lời kết thúc: "Hãy cùng nhau bắt đầu", "Chúc các em học tốt"
- KHÔNG sử dụng ngôn ngữ nói chuyện: "Các em có biết không?", "Chúng ta hãy cùng tìm hiểu"
- KHÔNG sử dụng câu mở đầu dài dòng không cần thiết
- KHÔNG sử dụng emoji hoặc ký tự đặc biệt như **, *, •, -, etc.
- TUYỆT ĐỐI KHÔNG tạo bảng (table) với dấu | hoặc format bảng - chỉ viết text thuần túy
- 🚨 TUYỆT ĐỐI KHÔNG tạo quá 6 mục con trong 1 slide - hãy gộp nội dung nếu cần

✅ NỘI DUNG SLIDE PHẢI:
- Đi thẳng vào nội dung chính, tránh nội dung lan man hoặc không liên quan tới bài học
- Sử dụng ngôn ngữ khoa học, chính xác
- Trình bày thông tin một cách súc tích, rõ ràng
- Tập trung vào kiến thức cốt lõi
- Sử dụng định nghĩa, công thức, ví dụ cụ thể

FORMAT OUTPUT:
Trả về nội dung chi tiết cho slide này dưới dạng text thuần túy, không format đặc biệt.
Nội dung phải đầy đủ, chi tiết và phù hợp với mục đích của slide.

VÍ DỤ ĐÚNG:
"Nguyên tố hóa học là tập hợp các nguyên tử có cùng số proton trong hạt nhân. Số hiệu nguyên tử Z chính là số proton, xác định tính chất hóa học của nguyên tố. Ví dụ: Hydrogen có Z=1, Helium có Z=2. Các nguyên tố được sắp xếp trong bảng tuần hoàn theo thứ tự tăng dần của số hiệu nguyên tử."

VÍ DỤ SAI (TUYỆT ĐỐI KHÔNG LÀM):
"Chào mừng các em đến với bài học mới! Hôm nay chúng ta sẽ cùng nhau khám phá nguyên tố hóa học. **Nguyên tố hóa học** là một khái niệm rất quan trọng..."

VÍ DỤ SAI VỀ BẢNG (TUYỆT ĐỐI KHÔNG LÀM):
"| Kí hiệu | Số hiệu nguyên tử | Số khối |
|---|---|---|
| ⁴⁰₁₈Ar |  |  |
| ³⁹₁₉K |  |  |"

VÍ DỤ ĐÚNG THAY THẾ BẢNG:
"Phân tích các nguyên tử: Argon (⁴⁰₁₈Ar) có số hiệu nguyên tử ?, số khối ?, chứa ? proton, ? electron và ? neutron."

VÍ DỤ GỘP MỤC (TRÁNH VƯỢT QUÁ 6 MỤC):
❌ SAI (8 mục - vượt quá):
"Mục 1: Định nghĩa
Mục 2: Tính chất vật lý
Mục 3: Tính chất hóa học
Mục 4: Ứng dụng trong công nghiệp
Mục 5: Ứng dụng trong y học
Mục 6: Ứng dụng trong nông nghiệp
Mục 7: Tác hại với môi trường
Mục 8: Biện pháp bảo vệ"

✅ ĐÚNG (6 mục - đã gộp):
"Mục 1: Định nghĩa và cấu trúc
Mục 2: Tính chất vật lý và hóa học
Mục 3: Ứng dụng trong công nghiệp và y học
Mục 4: Ứng dụng trong nông nghiệp và đời sống
Mục 5: Tác động môi trường và sức khỏe
Mục 6: Biện pháp an toàn và bảo vệ"

LƯU Ý:
- Chỉ tập trung vào slide này, không đề cập đến slide khác
- Nội dung phải chi tiết và đầy đủ
- Sử dụng ngôn ngữ khoa học chính xác
- Có thể bao gồm ví dụ minh họa cụ thể
- 🚨 QUAN TRỌNG NHẤT: Nếu có nhiều hơn 6 mục con, hãy GỘP CHÚNG LẠI thành tối đa 6 mục
"""

        return prompt

    async def _map_placeholders(
        self,
        detailed_content: str,
        slide_number: int
    ) -> Dict[str, Any]:
        """
        Bước 3: Gắn placeholder cho từng slide chi tiết
        Input: slide_chi_tiet, default_prompt
        Output: Slide với placeholder được gắn theo quy tắc hiện tại
        """
        try:
            logger.info(f"🏷️ Mapping placeholders for slide {slide_number}")

            # Tạo prompt cho việc gắn placeholder
            placeholder_prompt = self._create_placeholder_prompt(detailed_content, slide_number)

            # Gọi LLM để gắn placeholder
            llm_response = await self.llm_service.generate_content(
                prompt=placeholder_prompt,
                max_tokens=20000,
                temperature=0.1
            )

            if not llm_response.get("success", False):
                return {
                    "success": False,
                    "error": f"LLM placeholder mapping failed: {llm_response.get('error', 'Unknown error')}"
                }

            placeholder_content = llm_response.get("text", "").strip()
            logger.info(f"Placeholder content generated: {placeholder_content}")

            if not placeholder_content:
                return {
                    "success": False,
                    "error": "Empty placeholder content"
                }

            # Parse placeholder content để tạo slide data
            slide_data = self._parse_placeholder_content(placeholder_content, slide_number)

            # Validate và fix 1:1 mapping
            validated_slide_data = self._validate_and_fix_mapping(slide_data, slide_number)

            logger.info(f"✅ Placeholders mapped for slide {slide_number}")
            logger.info(f"📋 Placeholder summary: {validated_slide_data}")

            return {
                "success": True,
                "slide_data": validated_slide_data,
                "raw_content": placeholder_content
            }

        except Exception as e:
            logger.error(f"❌ Error mapping placeholders for slide {slide_number}: {e}")
            return {
                "success": False,
                "error": f"Failed to map placeholders: {str(e)}"
            }

    def _create_placeholder_prompt(self, detailed_content: str, slide_number: int) -> str:
        """Tạo prompt cho việc gắn placeholder"""

        prompt = f"""
NHIỆM VỤ: Gắn placeholder cho slide chi tiết theo quy tắc 1:1 MAPPING NGHIÊM NGẶT

SLIDE CHI TIẾT CẦN GẮN PLACEHOLDER:
{detailed_content}

🚨 QUY TẮC 1:1 MAPPING BẮT BUỘC - CỰC KỲ QUAN TRỌNG:
1. MỖI TitleName CHỈ CÓ ĐÚNG 1 TitleContent duy nhất
2. MỖI SubtitleName CHỈ CÓ ĐÚNG 1 SubtitleContent duy nhất
3. TUYỆT ĐỐI KHÔNG tạo nhiều TitleContent riêng biệt cho 1 TitleName
4. TUYỆT ĐỐI KHÔNG tạo nhiều SubtitleContent riêng biệt cho 1 SubtitleName
5. Nếu có nhiều ý trong cùng 1 mục, hãy GỘP TẤT CẢ thành 1 khối duy nhất
6. Sử dụng \\n để xuống dòng giữa các ý trong cùng 1 khối content
7. GIỚI HẠN: Tối đa 6 SubtitleName mỗi slide (không được vượt quá) 
8. Hãy gộp nội dung nếu cần thiết để tránh vượt quá 

PLACEHOLDER TYPES:
- LessonName: Tên bài học (chỉ slide đầu tiên)
- LessonDescription: Mô tả bài học (chỉ slide đầu tiên)
- CreatedDate: Ngày tạo (chỉ slide đầu tiên)
- TitleName: Tiêu đề chính của slide
- TitleContent: Nội dung giải thích chi tiết cho TitleName (CHỈ 1 KHỐI)
- SubtitleName: Tiêu đề các mục con (TỐI ĐA 6 MỤC MỖI SLIDE)
- SubtitleContent: Nội dung chi tiết cho từng SubtitleName (CHỈ 1 KHỐI)
- ImageName: Tên hình ảnh minh họa
- ImageContent: Mô tả nội dung hình ảnh

SLIDE HIỆN TẠI: {slide_number}

🔥 VÍ DỤ SAI (TUYỆT ĐỐI KHÔNG LÀM):
Cấu trúc nguyên tử #*(TitleName)*#
Nguyên tử gồm hạt nhân và electron. #*(TitleContent)*#
Hạt nhân ở trung tâm. #*(TitleContent)*#  ❌ SAI - Có 2 TitleContent riêng biệt
Electron chuyển động xung quanh. #*(TitleContent)*#  ❌ SAI - Có 3 TitleContent riêng biệt

✅ VÍ DỤ ĐÚNG (BẮT BUỘC LÀM THEO):
Cấu trúc nguyên tử #*(TitleName)*#
Nguyên tử gồm hạt nhân và electron.\\nHạt nhân ở trung tâm, chứa proton và neutron.\\nElectron chuyển động xung quanh hạt nhân trong các orbital.\\nLực tĩnh điện giữ electron gần hạt nhân. #*(TitleContent)*#

✅ VÍ DỤ ĐÚNG VỚI SUBTITLE:
Bài toán tính toán #*(SubtitleName)*#
Gọi x là phần trăm số nguyên tử của ⁶³Cu và y là phần trăm số nguyên tử của ⁶⁵Cu.\\nTa có hệ phương trình: x + y = 100 (Tổng phần trăm là 100%).\\nVà (63x + 65y) / 100 = 63,54 (Công thức nguyên tử khối trung bình).\\nTừ (1), ta có y = 100 - x.\\nThay vào (2): (63x + 65(100 - x)) / 100 = 63,54.\\nGiải phương trình: 63x + 6500 - 65x = 6354, -2x = -146, x = 73.\\nVậy phần trăm số nguyên tử của ⁶³Cu là 73% và ⁶⁵Cu là 27%. #*(SubtitleContent)*#

FORMAT OUTPUT:
Trả về nội dung đã được gắn placeholder với \\n để xuống dòng:
content #*(PlaceholderType)*#

🔥 NHẮC NHỞ CUỐI CÙNG - CỰC KỲ QUAN TRỌNG:
- CHỈ 1 TitleContent cho mỗi TitleName (KHÔNG BAO GIỜ NHIỀU HỠN 1)
- CHỈ 1 SubtitleContent cho mỗi SubtitleName (KHÔNG BAO GIỜ NHIỀU HỠN 1)
- TỐI ĐA 6 SubtitleName mỗi slide (KHÔNG ĐƯỢC VƯỢT QUÁ)
- Hãy gộp nội dung nếu cần thiết để tránh vượt quá 
- Sử dụng \\n để xuống dòng trong cùng 1 khối content
- TUYỆT ĐỐI TUÂN THỦ QUY TẮC 1:1 MAPPING
- NẾU CÓ NHIỀU Ý TRONG CÙNG MỤC, HÃY GỘP TẤT CẢ THÀNH 1 KHỐI DUY NHẤT
- KIỂM TRA LẠI TRƯỚC KHI TRẢ VỀ: Mỗi TitleName chỉ có 1 TitleContent, mỗi SubtitleName chỉ có 1 SubtitleContent, tối đa 6 SubtitleName

🚨 VÍ DỤ CUỐI CÙNG - ĐÚNG 100%:
Cấu trúc nguyên tử #*(TitleName)*#
Nguyên tử gồm hạt nhân và electron.\\nHạt nhân ở trung tâm.\\nElectron chuyển động xung quanh. #*(TitleContent)*#
Proton #*(SubtitleName)*#
Proton mang điện dương.\\nCó khối lượng 1,67×10^-27 kg.\\nQuyết định nguyên tố hóa học. #*(SubtitleContent)*#
Neutron #*(SubtitleName)*#
Neutron không mang điện.\\nCó khối lượng gần bằng proton.\\nTạo thành đồng vị. #*(SubtitleContent)*#
"""

        return prompt

    def _parse_placeholder_content(self, placeholder_content: str, slide_number: int) -> Dict[str, Any]:
        """Parse placeholder content thành slide data"""
        try:
            # Parse content theo annotation format
            parsed_data = {
                "LessonName": [],
                "LessonDescription": [],
                "CreatedDate": [],
                "TitleName": [],
                "TitleContent": [],
                "SubtitleName": [],
                "SubtitleContent": [],
                "ImageName": [],
                "ImageContent": []
            }

            # Pattern để match: "content #*(PlaceholderType)*#"
            valid_placeholders = '|'.join(parsed_data.keys())
            pattern = rf'(.+?)\s*#\*\(({valid_placeholders})\)\*#'

            matches = re.findall(pattern, placeholder_content, re.IGNORECASE | re.DOTALL)

            for content, placeholder_type in matches:
                clean_content = content.strip()
                if clean_content:
                    parsed_data[placeholder_type].append({
                        "content": clean_content,
                        "length": len(clean_content)
                    })

            # Tạo slide summary
            placeholder_counts = {}
            for placeholder_type, items in parsed_data.items():
                if items:
                    placeholder_counts[placeholder_type] = len(items)

            slide_data = {
                "slide_number": slide_number,
                "parsed_data": parsed_data,
                "placeholder_counts": placeholder_counts,
                "raw_content": placeholder_content
            }

            logger.info(f"📋 Slide {slide_number} placeholder summary: {placeholder_counts}")
            return slide_data

        except Exception as e:
            logger.error(f"❌ Error parsing placeholder content: {e}")
            return {
                "slide_number": slide_number,
                "parsed_data": {},
                "placeholder_counts": {},
                "raw_content": placeholder_content,
                "error": str(e)
            }

    def _validate_and_fix_mapping(self, slide_data: Dict[str, Any], slide_number: int) -> Dict[str, Any]:
        """
        Validate và fix 1:1 mapping violations
        """
        try:
            logger.info(f"🔍 Validating 1:1 mapping for slide {slide_number}")

            parsed_data = slide_data.get("parsed_data", {})
            placeholder_counts = slide_data.get("placeholder_counts", {})

            # Log original counts
            logger.info(f"📋 Original placeholder counts: {placeholder_counts}")

            violations_fixed = []

            # Fix TitleName vs TitleContent mapping
            title_name_count = placeholder_counts.get('TitleName', 0)
            title_content_count = placeholder_counts.get('TitleContent', 0)

            if title_name_count > 0 and title_content_count != title_name_count:
                logger.warning(f"⚠️ TitleName={title_name_count} but TitleContent={title_content_count}")

                if title_content_count > title_name_count:
                    # Gộp multiple TitleContent thành 1
                    title_contents = parsed_data.get('TitleContent', [])
                    if len(title_contents) > 1:
                        combined_content = "\\n".join([item['content'] for item in title_contents])
                        parsed_data['TitleContent'] = [{
                            "content": combined_content,
                            "length": len(combined_content)
                        }]
                        placeholder_counts['TitleContent'] = 1
                        violations_fixed.append(f"Combined {title_content_count} TitleContent into 1")
                        logger.info(f"🔧 Fixed: Combined {title_content_count} TitleContent into 1")

            # Fix SubtitleName vs SubtitleContent mapping
            subtitle_name_count = placeholder_counts.get('SubtitleName', 0)
            subtitle_content_count = placeholder_counts.get('SubtitleContent', 0)

            if subtitle_name_count > 0 and subtitle_content_count != subtitle_name_count:
                logger.warning(f"⚠️ SubtitleName={subtitle_name_count} but SubtitleContent={subtitle_content_count}")

                if subtitle_content_count > subtitle_name_count:
                    # Gộp SubtitleContent theo tỷ lệ
                    subtitle_contents = parsed_data.get('SubtitleContent', [])
                    subtitle_names = parsed_data.get('SubtitleName', [])

                    if len(subtitle_contents) > len(subtitle_names) and len(subtitle_names) > 0:
                        # Chia đều SubtitleContent cho SubtitleName
                        contents_per_name = len(subtitle_contents) // len(subtitle_names)
                        remainder = len(subtitle_contents) % len(subtitle_names)

                        new_subtitle_contents = []
                        content_index = 0

                        for i in range(len(subtitle_names)):
                            # Số content cho subtitle này
                            num_contents = contents_per_name + (1 if i < remainder else 0)

                            # Gộp contents
                            contents_to_combine = subtitle_contents[content_index:content_index + num_contents]
                            combined_content = "\\n".join([item['content'] for item in contents_to_combine])

                            new_subtitle_contents.append({
                                "content": combined_content,
                                "length": len(combined_content)
                            })

                            content_index += num_contents

                        parsed_data['SubtitleContent'] = new_subtitle_contents
                        placeholder_counts['SubtitleContent'] = len(new_subtitle_contents)
                        violations_fixed.append(f"Redistributed {subtitle_content_count} SubtitleContent to match {subtitle_name_count} SubtitleName")
                        logger.info(f"🔧 Fixed: Redistributed SubtitleContent to match SubtitleName")

            # Update slide data
            slide_data["parsed_data"] = parsed_data
            slide_data["placeholder_counts"] = placeholder_counts

            # Log final counts
            logger.info(f"📋 Final placeholder counts: {placeholder_counts}")

            if violations_fixed:
                logger.info(f"🔧 Violations fixed: {violations_fixed}")
                slide_data["violations_fixed"] = violations_fixed
            else:
                logger.info(f"✅ No violations found for slide {slide_number}")

            return slide_data

        except Exception as e:
            logger.error(f"❌ Error validating mapping for slide {slide_number}: {e}")
            # Return original data if validation fails
            return slide_data










    def _generate_slide_description(self, placeholder_counts: Dict[str, int]) -> str:
        """
        Generate description for slide based on placeholder counts (từ luồng cũ)

        Args:
            placeholder_counts: Dictionary of placeholder type counts

        Returns:
            str: Generated description
        """
        try:
            if not placeholder_counts:
                return "Slide trống"

            descriptions = []
            for placeholder_type, count in placeholder_counts.items():
                if count > 0:
                    if count == 1:
                        descriptions.append(f"1 {placeholder_type}")
                    else:
                        descriptions.append(f"{count} {placeholder_type}")

            if descriptions:
                return f"Slide dành cho {', '.join(descriptions)}"
            else:
                return "Slide trống"

        except Exception as e:
            logger.warning(f"Error generating slide description: {e}")
            return "Slide không xác định"
    def _parse_description_to_counts(self, description: str) -> Dict[str, int]:
        """
        Parse description có sẵn thành placeholder counts
        Ví dụ: "1 TitleName, 1 TitleContent, 1 SubtitleName" -> {"TitleName": 1, "TitleContent": 1, "SubtitleName": 1}
        """
        try:
            placeholder_counts = {}

            if not description or not description.strip():
                return placeholder_counts

            # Pattern để match "số PlaceholderType"
            import re
            pattern = r'(\d+)\s+(\w+)'
            matches = re.findall(pattern, description)

            for count_str, placeholder_type in matches:
                try:
                    count = int(count_str)
                    placeholder_counts[placeholder_type] = count
                except ValueError:
                    continue

            logger.info(f"📋 Parsed description '{description}' -> {placeholder_counts}")
            return placeholder_counts

        except Exception as e:
            logger.error(f"❌ Error parsing description '{description}': {e}")
            return {}





    async def _handle_max_length_content(
        self,
        content: str,
        max_length: int,
        placeholder_type: str,
        max_retries: int = 3
    ) -> str:
        """Xử lý content vượt quá max_length"""
        try:
            if len(content) <= max_length:
                return content

            logger.info(f"⚠️ Content too long for {placeholder_type}: {len(content)} > {max_length}")

            # Retry với LLM để rút gọn
            for attempt in range(max_retries):
                logger.info(f"🔄 Retry {attempt + 1}/{max_retries} to shorten content...")

                shorten_prompt = f"""Hãy rút gọn nội dung sau để không vượt quá {max_length} ký tự, giữ nguyên ý nghĩa chính:

ORIGINAL CONTENT:
{content}

REQUIREMENTS:
- Tối đa {max_length} ký tự
- Giữ nguyên ý nghĩa chính
- Phù hợp với {placeholder_type}

SHORTENED CONTENT:"""

                llm_response = await self.llm_service.generate_content(
                    prompt=shorten_prompt,
                    max_tokens=12000,
                    temperature=0.1
                )

                if llm_response.get("success", False):
                    shortened_content = llm_response.get("text", "").strip()
                    if len(shortened_content) <= max_length:
                        logger.info(f"✅ Content shortened: {len(shortened_content)} chars")
                        return shortened_content

            # Không sử dụng fallback truncation
            logger.error(f"❌ Failed to shorten content for {placeholder_type} after {max_retries} retries")
            return content  # Trả về content gốc, để frontend xử lý

        except Exception as e:
            logger.error(f"❌ Error handling max_length content: {e}")
            return content  # Trả về content gốc, không truncate

    def _find_exact_matching_template(
        self,
        required_placeholders: List[str],
        required_counts: Dict[str, int],
        template_slides: List[Dict[str, Any]],
        used_slide_ids: set
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm template slide match chính xác với required placeholders và counts
        (Tương tự logic trong luồng cũ, không fallback)

        Args:
            required_placeholders: List placeholder types cần thiết
            required_counts: Dict số lượng từng placeholder type
            template_slides: List các template slides
            used_slide_ids: Set các slide IDs đã sử dụng

        Returns:
            Dict slide template match chính xác hoặc None
        """
        try:
            for slide in template_slides:
                slide_id = slide.get("id")  # Format mới sử dụng "id" thay vì "slideId"

                # Skip used slides
                if slide_id in used_slide_ids:
                    continue

                # Sử dụng description có sẵn thay vì phân tích lại
                description = slide.get("description", "")
                slide_placeholder_counts = self._parse_description_to_counts(description)

                # Check for EXACT match: same placeholder types and same counts
                required_set = set(required_placeholders)
                slide_set = set(slide_placeholder_counts.keys())

                if required_set == slide_set:
                    # Check if counts also match exactly
                    counts_match = True
                    for placeholder_type, required_count in required_counts.items():
                        slide_count = slide_placeholder_counts.get(placeholder_type, 0)
                        if slide_count != required_count:
                            counts_match = False
                            break

                    if counts_match:
                        logger.info(f"✅ Found EXACT matching template: {slide_id}")
                        logger.info(f"   Required: {required_counts}")
                        logger.info(f"   Template has: {slide_placeholder_counts}")
                        return slide
                    else:
                        logger.debug(f"❌ Template {slide_id}: placeholder types match but counts differ")
                        logger.debug(f"   Required: {required_counts}")
                        logger.debug(f"   Template has: {slide_placeholder_counts}")
                else:
                    logger.debug(f"❌ Template {slide_id}: placeholder types don't match")
                    logger.debug(f"   Required: {required_set}")
                    logger.debug(f"   Template has: {slide_set}")

            logger.info(f"❌ No EXACT matching template found for: {required_counts}")
            return None

        except Exception as e:
            logger.error(f"Error finding exact matching template: {e}")
            return None

    def _find_exact_matching_template_with_reuse(
        self,
        required_placeholders: List[str],
        required_counts: Dict[str, int],
        template_slides: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm template slide match chính xác với required placeholders (cho phép reuse)
        (Tương tự logic trong luồng cũ)

        Args:
            required_placeholders: List placeholder types cần thiết
            required_counts: Dict số lượng từng placeholder type
            template_slides: List các template slides

        Returns:
            Dict slide template match chính xác hoặc None
        """
        try:
            logger.info(f"🔍 Finding exact matching template with reuse support...")

            for slide in template_slides:
                slide_id = slide.get("id")  # Format mới sử dụng "id" thay vì "slideId"

                # Sử dụng description có sẵn thay vì phân tích lại
                description = slide.get("description", "")
                slide_placeholder_counts = self._parse_description_to_counts(description)

                # Check for EXACT match: same placeholder types and same counts
                required_set = set(required_placeholders)
                slide_set = set(slide_placeholder_counts.keys())

                if required_set == slide_set:
                    # Check if counts also match exactly
                    counts_match = True
                    for placeholder_type, required_count in required_counts.items():
                        slide_count = slide_placeholder_counts.get(placeholder_type, 0)
                        if slide_count != required_count:
                            counts_match = False
                            break

                    if counts_match:
                        logger.info(f"✅ Found EXACT matching template (reuse allowed): {slide_id}")
                        logger.info(f"   Required: {required_counts}")
                        logger.info(f"   Template has: {slide_placeholder_counts}")
                        return slide
                    else:
                        logger.debug(f"❌ Template {slide_id}: placeholder types match but counts differ")
                        logger.debug(f"   Required: {required_counts}")
                        logger.debug(f"   Template has: {slide_placeholder_counts}")
                else:
                    logger.debug(f"❌ Template {slide_id}: placeholder types don't match")
                    logger.debug(f"   Required: {required_set}")
                    logger.debug(f"   Template has: {slide_set}")

            logger.info(f"❌ No EXACT matching template found for reuse: {required_counts}")
            return None

        except Exception as e:
            logger.error(f"Error finding exact matching template with reuse: {e}")
            return None

    async def _create_processed_slide_from_template(
        self,
        template_slide: Dict[str, Any],
        parsed_data: Dict[str, List[Dict[str, Any]]],
        content_index: Dict[str, int],
        slide_number: int,
        is_reused: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Tạo processed slide từ template slide với content mapping
        (Tương tự logic trong luồng cũ, không fallback)

        Args:
            template_slide: Template slide để copy
            parsed_data: Parsed content từ LLM
            content_index: Index tracking cho content usage
            slide_number: Số thứ tự slide

        Returns:
            Dict processed slide hoặc None nếu fail
        """
        try:
            # Format mới: template_slide chính là slide từ input JSON
            template_slide_id = template_slide.get("id")
            slide_data = template_slide.get("slideData", {})
            template_elements = slide_data.get("elements", [])

            # Tạo slideId mới cho processed slide
            if is_reused:
                new_slide_id = f"slide_{slide_number:03d}_reused_from_{template_slide_id}"
                logger.info(f"📄 Creating processed slide (REUSED): {new_slide_id} (from template: {template_slide_id})")
            else:
                new_slide_id = f"slide_{slide_number:03d}_from_{template_slide_id}"
                logger.info(f"📄 Creating processed slide (NEW): {new_slide_id} (from template: {template_slide_id})")

            # Copy toàn bộ slide structure từ template (format mới)
            processed_slide = copy.deepcopy(template_slide)

            # Update slide ID và reset elements để fill content mới
            processed_slide["id"] = new_slide_id
            processed_slide["slideData"]["id"] = new_slide_id
            processed_slide["slideData"]["title"] = f"Slide {slide_number}"
            processed_slide["slideData"]["elements"] = []  # Reset elements để fill content mới

            # Placeholder patterns để detect từ text elements
            placeholder_patterns = {
                "LessonName": r"LessonName\s+(\d+)",
                "LessonDescription": r"LessonDescription\s+(\d+)",
                "CreatedDate": r"CreatedDate\s+(\d+)",
                "TitleName": r"TitleName\s+(\d+)",
                "TitleContent": r"TitleContent\s+(\d+)",
                "SubtitleName": r"SubtitleName\s+(\d+)",
                "SubtitleContent": r"SubtitleContent\s+(\d+)",
                "ImageName": r"ImageName\s+(\d+)",
                "ImageContent": r"ImageContent\s+(\d+)"
            }

            # Map content vào từng element (format mới)
            for element in template_elements:
                if element.get("type") == "text":
                    text = element.get("text", "").strip()
                    element_id = element.get("id")

                    # Detect placeholder type từ text
                    placeholder_result = self._detect_placeholder_type_from_text(text, placeholder_patterns)

                    if placeholder_result:
                        placeholder_type, max_length = placeholder_result

                        # Get content for this placeholder type
                        content_list = parsed_data.get(placeholder_type, [])
                        current_index = content_index.get(placeholder_type, 0)

                        logger.info(f"🔍 Mapping content for {placeholder_type}:")
                        logger.info(f"   Available content items: {len(content_list)}")
                        logger.info(f"   Current index: {current_index}")
                        logger.info(f"   Element ID: {element_id}")

                        if current_index < len(content_list):
                            content_item = content_list[current_index]
                            raw_content = content_item.get("content", "")
                            logger.info(f"   Raw content: {raw_content[:100]}...")

                            try:
                                # Check max_length and handle if needed
                                final_content = await self._handle_max_length_content(
                                    raw_content,
                                    max_length,
                                    placeholder_type
                                )

                                # Copy element và update content (format mới)
                                processed_element = copy.deepcopy(element)
                                processed_element["text"] = final_content  # Update content

                                processed_slide["slideData"]["elements"].append(processed_element)

                                # Increment content index
                                content_index[placeholder_type] = current_index + 1

                                logger.info(f"✅ Mapped {placeholder_type} to {element_id}: {final_content[:100]}...")
                                logger.info(f"   Final content length: {len(final_content)}")

                            except Exception as e:
                                logger.error(f"❌ Failed to handle content for {placeholder_type} in slide {slide_number}: {e}")
                                logger.error(f"   Content length: {len(raw_content)}, Max length: {max_length}")
                                logger.error(f"   SKIPPING this slide due to content length issue - NO FALLBACK")
                                return None  # Skip entire slide if any content fails
                        else:
                            logger.warning(f"❌ No more content available for {placeholder_type} in slide {slide_number}")
                            logger.warning(f"   Available content items: {len(content_list)}")
                            logger.warning(f"   Current index: {current_index}")
                            return None  # Skip slide if missing content
                    else:
                        # Copy element không phải placeholder (image, etc.)
                        processed_element = copy.deepcopy(element)
                        processed_slide["slideData"]["elements"].append(processed_element)

            logger.info(f"✅ Successfully created processed slide {slide_number} with {len(processed_slide['slideData']['elements'])} elements")
            return processed_slide

        except Exception as e:
            logger.error(f"❌ Error creating processed slide from template: {e}")
            return None

    def _detect_placeholder_type_from_text(self, text: str, placeholder_patterns: Dict[str, str]) -> Optional[Tuple[str, int]]:
        """
        Detect placeholder type từ text element và extract max length

        Args:
            text: Text content của element
            placeholder_patterns: Dict mapping placeholder types to regex patterns

        Returns:
            Tuple (placeholder_type, max_length) hoặc None nếu không match
        """
        try:
            import re

            for placeholder_type, pattern in placeholder_patterns.items():
                match = re.search(pattern, text)
                if match:
                    # Extract max_length từ captured group
                    max_length = int(match.group(1)) if match.group(1) else 0
                    logger.info(f"🎯 Detected placeholder: {placeholder_type} with max_length: {max_length}")
                    return (placeholder_type, max_length)

            # Không tìm thấy placeholder pattern
            return None

        except Exception as e:
            logger.error(f"❌ Error detecting placeholder type from text '{text}': {e}")
            return None




# Singleton instance
_json_template_service = None

def get_json_template_service() -> JsonTemplateService:
    """Get singleton instance của JsonTemplateService"""
    global _json_template_service
    if _json_template_service is None:
        _json_template_service = JsonTemplateService()
    return _json_template_service
