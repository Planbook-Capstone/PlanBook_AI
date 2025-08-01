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


    async def process_json_template_with_progress(
        self,
        lesson_id: str,
        template_json: Dict[str, Any],
        config_prompt: Optional[str] = None,
        task_id: Optional[str] = None,
        task_service: Optional[Any] = None,
        user_id: Optional[str] = None,
        book_id: Optional[str] = None,
        tool_log_id: Optional[str] = None
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
                user_id,
                tool_log_id
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

    async def _execute_optimized_workflow_with_progress(
        self,
        lesson_content: str,
        config_prompt: Optional[str],
        template_json: Dict[str, Any],
        analyzed_template: Dict[str, Any],
        task_id: Optional[str] = None,
        task_service: Optional[Any] = None,
        user_id: Optional[str] = None,
        tool_log_id: Optional[str] = None
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

            # Content tracking không còn cần thiết vì mỗi slide sử dụng data riêng



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

                # Bước 2: Chi tiết hóa slide (bỏ qua slide intro)
                if framework_slide.get("type") == "intro":
                    logger.info(f"⏭️ Skipping detailed processing for slide {slide_num} (intro slide type)")
                    # Tạo detailed_json cho slide intro từ framework_slide
                    intro_detailed_json = {
                        "slideId": framework_slide.get("slide_id", f"slide{slide_num}"),
                        "type": "intro",
                        "title": framework_slide.get("title", ""),
                        "description": framework_slide.get("description", ""),
                        "date": framework_slide.get("date", "")
                    }

                    detailed_slide = {
                        "success": True,
                        "content": framework_slide,
                        "detailed_json": intro_detailed_json
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
                    logger.info(f"======= Detailed slide {slide_num}: {detailed_slide.get('detailed_json', {})}")
                # Bước 3: Gắn placeholder
                detailed_json = detailed_slide.get("detailed_json")
                if detailed_json:
                    # Use JSON-based placeholder mapping
                    slide_with_placeholders = self._map_placeholders_from_json(
                        detailed_json,
                        slide_num
                    )
                else:
                    logger.error(f"❌ No detailed_json found for slide {slide_num}")
                    continue

                if not slide_with_placeholders.get("success", False):
                    logger.error(f"❌ Step 3 failed for slide {slide_num}: {slide_with_placeholders.get('error', 'Unknown error')}")
                    continue

                slide_data = slide_with_placeholders.get("slide_data", {})

                # Bước 4: Map vào template
                mapped_slide = await self._map_single_slide_to_template(
                    slide_data,
                    template_slides,
                    used_slide_ids,
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
                                tool_log_id=tool_log_id,
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

            # Sử dụng parsed_data riêng của slide này thay vì all_parsed_data chung
            # để tránh việc các slide sử dụng content của nhau
            slide_parsed_data = parsed_data

            # Tìm template phù hợp với exact matching requirements
            slide_description = slide_data.get("description", [])

            best_template = None
            try:
                best_template = self._find_best_matching_template_with_max_length(
                    slide_description,
                    template_slides,
                    used_slide_ids
                )
            except (ValueError, Exception) as e:
                # Nếu không tìm thấy template chưa sử dụng, thử reuse template
                logger.info(f"🔄 No unused exact template found for slide {slide_number}, trying to reuse...")
                logger.info(f"   Original error: {e}")
                try:
                    best_template = self._find_best_matching_template_with_max_length(
                        slide_description,
                        template_slides,
                        set()  # Allow reuse by passing empty used_slide_ids
                    )
                except (ValueError, Exception) as reuse_error:
                    logger.error(f"❌ No exact matching template found for slide {slide_number} (even with reuse)")
                    logger.error(f"   Reuse error: {reuse_error}")
                    return None

            # Kiểm tra best_template có hợp lệ không
            if not best_template or not isinstance(best_template, dict):
                logger.error(f"❌ Invalid template returned for slide {slide_number}: {best_template}")
                return None

            template_id = best_template.get('id')
            if not template_id:
                logger.error(f"❌ Template missing 'id' field for slide {slide_number}: {best_template}")
                return None

            is_reused = template_id in used_slide_ids

            if is_reused:
                logger.info(f"✅ Found exact matching template (REUSED): {template_id}")
            else:
                logger.info(f"✅ Found exact matching template (NEW): {template_id}")

            # Get template requirements for max_length handling
            template_description = best_template.get("description", "")
            template_requirements = self._parse_template_description(template_description)

            # Tạo processed slide từ template với parsed_data riêng của slide này
            processed_slide = await self._create_processed_slide_from_template(
                best_template,
                slide_parsed_data,  # Sử dụng data riêng của slide này
                slide_number,
                is_reused,
                template_requirements
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
                max_tokens=30000,
                temperature=0.1
            )

            if not llm_response.get("success", False):
                return {
                    "success": False,
                    "error": f"LLM framework generation failed: {llm_response.get('error', 'Unknown error')}"
                }

            framework_content = llm_response.get("text", "").strip()
            logger.info(f"✅ Framework content generated: {len(framework_content)} characters")
            logger.info(f"======================================  Framework content: {framework_content}")
            # Parse JSON framework content directly
            import json
            try:
                # Extract JSON from the response
                json_start = framework_content.find('{')
                json_end = framework_content.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_content = framework_content[json_start:json_end]
                    parsed_json = json.loads(json_content)

                    # Extract slides from JSON
                    slides = parsed_json.get("slides", [])

                    if not slides:
                        return {
                            "success": False,
                            "error": "No slides found in JSON framework content"
                        }

                    logger.info(f"✅ JSON Framework parsing complete: {len(slides)} slides")
                    return {
                        "success": True,
                        "slides": slides,
                        "raw_content": framework_content
                    }
                else:
                    logger.error("❌ No valid JSON found in framework content")
                    return {
                        "success": False,
                        "error": "No valid JSON found in framework content"
                    }

            except json.JSONDecodeError as je:
                logger.error(f"❌ JSON decode error in framework: {je}")
                # Fallback to old parsing method
                slides = self._parse_framework_content(framework_content)

                if not slides:
                    return {
                        "success": False,
                        "error": "No slides found in framework content (fallback parsing also failed)"
                    }

                logger.info(f"✅ Fallback framework parsing complete: {len(slides)} slides")
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

        # Get current date
        current_date = datetime.now().strftime("%d-%m-%Y")

        prompt = f"""Đóng vai trò người thiết kế bài thuyết trình giáo dục kinh nghiệm chuyên sâu.

NHIỆM VỤ:
- Hãy đọc JSON yêu cầu bên dưới và tạo danh sách các slide tổng quát dựa trên nội dung bài học.
- Chỉ sinh kết quả dưới dạng JSON theo định dạng đầu ra mẫu ở cuối .

NỘI DUNG BÀI HỌC:
{lesson_content}

JSON YÊU CẦU:
{{
  "instruction": "Phân tích nội dung bài học và tạo khung slide logic, dễ theo dõi.",
  "task":  "Phân tích nội dung bài học và chia thành các slide tổng quát, có mục đích rõ ràng và các ý chính phù hợp để trình bày.",
  "rules": [
    "Tách lesson_content thành các slide với tiêu đề, mục đích và các ý chính rõ ràng.",
    "Mỗi slide thể hiện một chủ đề lớn, với mục đích cụ thể và nội dung cốt lõi.",
    "Mỗi slide chứa tối đa 4 ý lớn. Linh hoạt trong 1-4 ý chính, không cố định.",
    "Nếu nội dung cần hơn 4 ý lớn, hãy tạo slide mới nhưng giữ tiêu đề tương tự (ví dụ: Phần 1, Phần 2) và đảm bảo sự liên kết giữa các phần",
    "Các ý chính cần được mô tả rõ ràng, không sơ sài.",
    "Sau mỗi ý chính, thêm một note thể hiện liệu có cần ví dụ minh họa, hình ảnh hỗ trợ hoặc giải thích thêm không.",
    "Slide đầu tiên phải là slide giới thiệu, gồm đúng 3 dòng: tên bài học, mô tả ngắn và ngày tạo bài thuyết trình.",
    "Đảm bảo trình tự các slide có tính logic, mạch lạc, dễ theo dõi.",
    "Tùy chỉnh kết quả theo personalize trong config bên dưới, ví dụ: điều chỉnh độ khó, văn phong, nội dung trình bày cho phù hợp đối tượng người học."
  ],
  "config": {{
    "language": "vi",
    "maxSlides": 20,
    "minSlides": 10,
    "outputFormat": "json",
    "date": "{current_date}",
    "personalize": "{config_prompt if config_prompt else 'Phân tích nội dung bài học và tạo khung slide logic, dễ theo dõi.'}"
  }}
}}

JSON ĐẦU RA:
{{
    "slides": [
      {{
        "slideId": "slide1",
        "type": "intro",
        "title": "[Tên bài học]",
        "description": "[Mô tả ngắn bài học]",
        "date": "{current_date}"
      }},
      {{
        "slideId": "slide2",
        "type": "content",
        "title": "[Tiêu đề slide]",
        "mainPoints": [
          {{
            "point": "[Ý chính 1]",
            "note": "[Có cần ví dụ minh họa / hình ảnh / cần giải thích thêm hay chi tiết gì không?]"
          }},
          {{
            "point": "[Ý chính 2]",
            "note": "[Gợi ý nếu cần hỗ trợ trực quan hoặc mở rộng nội dung]"
          }}
        ]  
      }}
    ]
  }},
  "_hint": {{
    "slideId": "Đặt ID duy nhất cho mỗi slide, dạng s2_abc",
    "type": "intro hoặc content",
    "title": "Tiêu đề chính của slide",
    "purpose": "Mục tiêu truyền đạt của slide",
    "mainPoints": "Tối đa 4 mục chính mỗi slide, mỗi mục có ghi chú đi kèm",
    "user_config": "Tùy chỉnh đầu ra theo đối tượng, phong cách, độ khó và yêu cầu trực quan"
  }}
}}"""

        return prompt

    def _parse_framework_content(self, framework_content: str) -> List[Dict[str, Any]]:
        """Parse framework content thành danh sách slides từ JSON format mới"""
        try:
            import json
            slides = []

            # Try to parse as JSON first
            try:
                # Clean the content to extract JSON
                json_start = framework_content.find('{')
                json_end = framework_content.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_content = framework_content[json_start:json_end]
                    parsed_json = json.loads(json_content)

                    # Extract slides from JSON
                    json_slides = parsed_json.get("slides", [])

                    for i, slide in enumerate(json_slides):
                        slide_data = {
                            "slide_number": i + 1,
                            "title": slide.get("title", ""),
                            "purpose": "",  # Will be derived from mainPoints
                            "main_content": "",  # Will be derived from mainPoints
                            "raw_block": json.dumps(slide, ensure_ascii=False),
                            "slide_id": slide.get("slideId", f"slide{i+1}"),
                            "type": slide.get("type", "content"),
                            "description": slide.get("description", ""),
                            "date": slide.get("date", ""),
                            "main_points": slide.get("mainPoints", []),
                            "lesson_content_used": slide.get("lessonContentUsed", "")
                        }

                        # For intro slides, use description as main_content
                        if slide_data["type"] == "intro":
                            slide_data["main_content"] = f"{slide_data['description']}\n{slide_data['date']}"
                        else:
                            # For content slides, combine main points
                            if slide_data["main_points"]:
                                points_text = []
                                for point in slide_data["main_points"]:
                                    point_text = point.get("point", "")
                                    note_text = point.get("note", "")
                                    if point_text:
                                        points_text.append(f"- {point_text}")
                                        if note_text:
                                            points_text.append(f"  Note: {note_text}")
                                slide_data["main_content"] = "\n".join(points_text)

                        slides.append(slide_data)

                    logger.info(f"✅ Successfully parsed JSON format: {len(slides)} slides")
                    return slides

            except json.JSONDecodeError as je:
                logger.warning(f"⚠️ JSON parsing failed, trying fallback parsing: {je}")

            # Fallback to old parsing method if JSON parsing fails
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

            logger.info(f"📋 Parsed {len(slides)} slides from framework (fallback method)")
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
            logger.info(f"📋 Framework slide: {framework_slide}")
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
                    max_tokens=35000,
                    temperature=0.1
                )
                logger.info(f"LLM response detail slide: {llm_response}")
                if llm_response.get("success", False):
                    detailed_content = llm_response.get("text", "").strip()

                    if detailed_content:
                        # Try to parse JSON response
                        parsed_detail = self._parse_detailed_json_response(detailed_content, slide_number)

                        if parsed_detail.get("success", False):
                            logger.info(f"✅ Slide {slide_number} detailed successfully with JSON format")
                            return {
                                "success": True,
                                "content": parsed_detail.get("content", detailed_content),
                                "slide_number": slide_number,
                                "framework": framework_slide,
                                "detailed_json": parsed_detail.get("detailed_json", {})
                            }
                        else:
                            # JSON parsing failed - this should not happen with new logic
                            logger.error(f"❌ JSON parsing failed for slide {slide_number} - this should not happen")
                            return {
                                "success": False,
                                "error": "JSON parsing failed",
                                "slide_number": slide_number,
                                "framework": framework_slide
                            }
                    else:
                        logger.warning(f"⚠️ Empty content for slide {slide_number}, attempt {attempt + 1}")
                else:
                    logger.warning(f"⚠️ LLM failed for slide {slide_number}, attempt {attempt + 1}: {llm_response.get('error', 'Unknown error')}")

            # No fallback - must have detailed_json for new logic
            logger.error(f"❌ Failed to detail slide {slide_number} after {max_retries} attempts")
            return {
                "success": False,
                "error": f"Failed to detail slide after {max_retries} attempts",
                "slide_number": slide_number,
                "framework": framework_slide
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
        """Tạo prompt cho việc chi tiết hóa slide với JSON format mới"""

        import json

        # Get current date
        current_date = datetime.now().strftime("%d-%m-%Y")

        # Create framework slide JSON for the prompt
        khung_slide_json = {
            "slideId": framework_slide.get("slide_id", f"slide{slide_number}"),
            "type": framework_slide.get("type", "content"),
            "title": framework_slide.get("title", ""),
            "mainPoints": []
        }

        # Handle different slide types
        if framework_slide.get("type") == "intro":
            khung_slide_json["description"] = framework_slide.get("description", "")
            khung_slide_json["date"] = framework_slide.get("date", current_date)
        else:
            # For content slides, use main_points if available, otherwise parse from main_content
            main_points = framework_slide.get("main_points", [])
            if main_points:
                khung_slide_json["mainPoints"] = main_points
            else:
                # Parse from main_content if main_points not available
                main_content = framework_slide.get("main_content", "")
                if main_content:
                    # Simple parsing of main content into points
                    lines = main_content.split('\n')
                    points = []
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith('- '):
                            point_text = line[2:].strip()
                            if point_text:
                                points.append({
                                    "point": point_text,
                                    "note": "Cần chi tiết hóa nội dung"
                                })
                    khung_slide_json["mainPoints"] = points

        khung_slide_str = json.dumps(khung_slide_json, ensure_ascii=False, indent=2)

        prompt = f"""Đóng vai trò người thiết kế bài thuyết trình giáo dục kinh nghiệm chuyên sâu.

NHIỆM VỤ:
Đọc JSON yêu cầu bên dưới và chi tiết hóa nội dung slide cụ thể dựa trên khung slide, nội dung bài học.
Chỉ sinh kết quả dưới dạng JSON theo định dạng đầu ra mẫu ở cuối .

KHUNG SLIDE:
{khung_slide_str}

NỘI DUNG BÀI HỌC THAM KHẢO:
{lesson_content}

JSON YÊU CẦU:
{{
  "instruction": "Viết nội dung chi tiết (viết vào field pointContent) cho mỗi ý chính \"point\" trong (`mainPoints`) từ khung slide dựa vào bài học",
  "rules": [
    "Mỗi ý phải trình bày rõ ràng, đúng kiến thức, có thể bao gồm định nghĩa, giải thích, công thức, ví dụ cụ thể.",
    "Kiến thức bám sát nội dung bài học, chi tiết và đầy đủ.",
    "Các dạng bảng có trong NỘI DUNG BÀI HỌC phải thay đổi thành dạng chữ",
    "Viết đúng và đủ các \"mainPoints\" trong KHUNG SLIDE",
    "TUYỆT ĐỐI KHÔNG tạo hay xóa \"mainPoints\" nào trong KHUNG SLIDE.",
    "Tùy chỉnh kết quả theo \"personalize\" trong \"config\" bên dưới, ví dụ: điều chỉnh độ khó, văn phong, nội dung trình bày cho phù hợp đối tượng người học."
  ],
  "avoid": [
     "Tạo thêm mainPoints mới",
     "Lời chào hỏi hoặc mở đầu như: 'Chào mừng các em'",
     "Câu kết thúc như: 'Chúc các em học tốt'",
     "Ngôn ngữ hội thoại: 'Chúng ta hãy cùng nhau tìm hiểu...'",
     "Emoji hoặc ký tự đặc biệt như *, •, |",
     "Định dạng bằng | hoặc markdown"
    ],
  "config": {{
    "language": "vi",
    "outputFormat": "json",
    "date": "{current_date}",
    "personalize": "{config_prompt if config_prompt else 'Nội dung slide logic, dễ theo dõi, chuyên nghiệp.'}"
  }}
}}

JSON ĐẦU RA:

[
    {{
        "slideId": "{khung_slide_json.get('slideId', f'slide{slide_number}')}",
        "type": "{khung_slide_json.get('type', 'content')}",
        "title": "[Tiêu đề slide]",
        "mainPoints": [
          {{
            "point": "[Ý chính 1]",
            "number": 1,
            "pointContent": "[Nội dung cho Ý chính 1]"
          }},
          {{
            "point": "[Ý chính 2]",
            "number": 2,
            "pointContent": "[Nội dung cho Ý chính 2]"
          }}
        ]
    }}
]"""

        return prompt

    def _parse_detailed_json_response(self, detailed_content: str, slide_number: int) -> Dict[str, Any]:
        """Parse detailed JSON response from LLM"""
        try:
            import json

            # Try to extract JSON from the response
            json_start = detailed_content.find('[')
            json_end = detailed_content.rfind(']') + 1

            if json_start != -1 and json_end > json_start:
                json_content = detailed_content[json_start:json_end]
                parsed_json = json.loads(json_content)

                if isinstance(parsed_json, list) and len(parsed_json) > 0:
                    slide_data = parsed_json[0]  # Get first slide from array

                    # Convert detailed JSON to text content for backward compatibility
                    text_content = self._convert_detailed_json_to_text(slide_data)

                    logger.info(f"✅ Successfully parsed detailed JSON for slide {slide_number}")
                    return {
                        "success": True,
                        "content": text_content,
                        "detailed_json": slide_data
                    }
                else:
                    logger.warning(f"⚠️ Invalid JSON structure for slide {slide_number}")
                    return {"success": False, "error": "Invalid JSON structure"}
            else:
                logger.warning(f"⚠️ No JSON found in response for slide {slide_number}")
                return {"success": False, "error": "No JSON found"}

        except json.JSONDecodeError as je:
            logger.warning(f"⚠️ JSON decode error for slide {slide_number}: {je}")
            return {"success": False, "error": f"JSON decode error: {str(je)}"}
        except Exception as e:
            logger.error(f"❌ Error parsing detailed JSON for slide {slide_number}: {e}")
            return {"success": False, "error": f"Parsing error: {str(e)}"}

    def _convert_detailed_json_to_text(self, slide_data: Dict[str, Any]) -> str:
        """Convert detailed JSON slide data to text format for backward compatibility"""
        try:
            text_parts = []

            # Add title
            title = slide_data.get("title", "")
            if title:
                text_parts.append(title)
                text_parts.append("")  # Empty line

            # Process main points
            main_points = slide_data.get("mainPoints", [])
            for main_point in main_points:
                point_text = main_point.get("point", "")
                if point_text:
                    text_parts.append(point_text)

                # Process point contents
                point_contents = main_point.get("pointContents", [])
                for content in point_contents:
                    sub_point = content.get("subPoint", "")
                    sub_content = content.get("subContent", "")

                    if sub_point and sub_content:
                        text_parts.append(f"{sub_point}: {sub_content}")
                    elif sub_content:
                        text_parts.append(sub_content)
                    elif sub_point:
                        text_parts.append(sub_point)

                text_parts.append("")  # Empty line between main points

            # Join all parts
            result = "\n".join(text_parts).strip()

            # Remove multiple consecutive empty lines
            while "\n\n\n" in result:
                result = result.replace("\n\n\n", "\n\n")

            return result

        except Exception as e:
            logger.error(f"❌ Error converting detailed JSON to text: {e}")
            return str(slide_data)  # Fallback to string representation

    def _map_placeholders_from_json(
        self,
        detailed_json: Dict[str, Any],
        slide_number: int
    ) -> Dict[str, Any]:
        """
        Bước 3: Gắn placeholder trực tiếp từ JSON chi tiết (không gọi LLM)
        Input: detailed_json từ bước chi tiết hóa
        Output: Slide với placeholder được gắn theo quy tắc
        """
        try:
            logger.info(f"🏷️ Mapping placeholders from JSON for slide {slide_number}")

            # Tạo slide data trực tiếp từ JSON
            slide_data = self._create_slide_data_from_json(detailed_json, slide_number)

            # Validate và fix 1:1 mapping
            validated_slide_data = self._validate_and_fix_mapping(slide_data, slide_number)

            logger.info(f"✅ Placeholders mapped from JSON for slide {slide_number}")
            logger.info(f"📋 Placeholder summary: {validated_slide_data}")

            return {
                "success": True,
                "slide_data": validated_slide_data,
                "raw_content": str(detailed_json)
            }

        except Exception as e:
            logger.error(f"❌ Error mapping placeholders from JSON for slide {slide_number}: {e}")
            return {
                "success": False,
                "error": f"Failed to map placeholders from JSON: {str(e)}"
            }

    def _create_slide_data_from_json(
        self,
        detailed_json: Dict[str, Any],
        slide_number: int
    ) -> Dict[str, Any]:
        """Tạo slide data với placeholder từ detailed JSON theo format mới"""
        try:
            slide_type = detailed_json.get("type", "content")

            # Initialize slide data structure
            slide_data = {
                "parsed_data": {
                    "LessonName": [],
                    "LessonDescription": [],
                    "CreatedDate": [],
                    "TitleName": [],
                    "MainPointName": [],
                    "MainPointContent": [],
                    "ImageName": [],
                    "ImageContent": []
                },
                "placeholder_counts": {},
                "description": []  # New field for placeholder descriptions
            }

            if slide_type == "intro":
                # Handle intro slide
                title = detailed_json.get("title", "")
                description = detailed_json.get("description", "")
                date = detailed_json.get("date", "")

                if title:
                    slide_data["parsed_data"]["LessonName"].append({"content": title})
                    slide_data["placeholder_counts"]["LessonName"] = 1
                    # Add description without number since there's only one
                    slide_data["description"].append(f"LessonName_{len(title)}")

                if description:
                    slide_data["parsed_data"]["LessonDescription"].append({"content": description})
                    slide_data["placeholder_counts"]["LessonDescription"] = 1
                    slide_data["description"].append(f"LessonDescription_{len(description)}")

                if date:
                    slide_data["parsed_data"]["CreatedDate"].append({"content": date})
                    slide_data["placeholder_counts"]["CreatedDate"] = 1
                    slide_data["description"].append(f"CreatedDate_{len(date)}")

            else:
                # Handle content slide - logic đơn giản với format mới
                title = detailed_json.get("title", "")
                main_points = detailed_json.get("mainPoints", [])

                # Add title -> TitleName
                if title:
                    slide_data["parsed_data"]["TitleName"].append({"content": title})
                    slide_data["placeholder_counts"]["TitleName"] = 1
                    slide_data["description"].append(f"TitleName_{len(title)}")

                # Process main points với format mới
                for main_point_idx, main_point in enumerate(main_points, 1):
                    point_text = main_point.get("point", "")
                    point_content = main_point.get("pointContent", "")

                    # point -> MainPointName
                    if point_text:
                        slide_data["parsed_data"]["MainPointName"].append({
                            "content": point_text,
                            "main_point": main_point_idx,
                            "position_key": f"MainPointName_{main_point_idx}"
                        })
                        slide_data["description"].append(f"MainPointName_{main_point_idx}_{len(point_text)}")

                    # pointContent -> MainPointContent
                    if point_content:
                        slide_data["parsed_data"]["MainPointContent"].append({
                            "content": point_content,
                            "main_point": main_point_idx,
                            "position_key": f"MainPointContent_{main_point_idx}"
                        })
                        slide_data["description"].append(f"MainPointContent_{main_point_idx}_{len(point_content)}")

                # Update placeholder counts
                slide_data["placeholder_counts"]["MainPointName"] = len(slide_data["parsed_data"]["MainPointName"])
                slide_data["placeholder_counts"]["MainPointContent"] = len(slide_data["parsed_data"]["MainPointContent"])

            logger.info(f"📊 Created slide data for slide {slide_number}:")
            logger.info(f"   Placeholder counts: {slide_data['placeholder_counts']}")
            logger.info(f"   Description: {slide_data['description']}")
            return slide_data

        except Exception as e:
            logger.error(f"❌ Error creating slide data from JSON for slide {slide_number}: {e}")
            raise



    def _parse_placeholder_content(self, placeholder_content: str, slide_number: int) -> Dict[str, Any]:
        """Parse placeholder content thành slide data"""
        try:
            # Parse content theo annotation format
            parsed_data = {
                "LessonName": [],
                "LessonDescription": [],
                "CreatedDate": [],
                "TitleName": [],
                "MainPointName": [],
                "SubPointName": [],
                "SubPointContent": [],
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
        Validate slide data - logic đơn giản
        """
        try:
            logger.info(f"🔍 Validating slide data for slide {slide_number}")

            placeholder_counts = slide_data.get("placeholder_counts", {})

            # Logic đơn giản - chỉ log placeholder counts
            logger.info(f"� Placeholder counts: {placeholder_counts}")
            logger.info(f"✅ Slide {slide_number} validation complete")

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
        Parse description từ Kafka format mới thành placeholder counts
        Ví dụ: "MainPointName_1_80, MainPointName_2_80, SubPointContent_1_1_80"
        -> {"MainPointName": 2, "SubPointContent": 1}
        """
        try:
            placeholder_counts = {}

            if not description or not description.strip():
                return placeholder_counts

            # Split by comma để lấy từng placeholder
            placeholders = [p.strip() for p in description.split(',')]

            for placeholder in placeholders:
                if not placeholder:
                    continue

                # Extract placeholder type từ format: PlaceholderType_numbers_maxlength
                parts = placeholder.split('_')
                if len(parts) >= 2:
                    placeholder_type = parts[0]

                    # Count occurrences of each placeholder type
                    if placeholder_type in placeholder_counts:
                        placeholder_counts[placeholder_type] += 1
                    else:
                        placeholder_counts[placeholder_type] = 1

            logger.info(f"📋 Parsed Kafka description '{description}' -> {placeholder_counts}")
            return placeholder_counts

        except Exception as e:
            logger.error(f"❌ Error parsing Kafka description '{description}': {e}")
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
                    max_tokens=15000,
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

    def _find_best_matching_template_with_max_length(
        self,
        slide_description: List[str],
        template_slides: List[Dict[str, Any]],
        used_slide_ids: set
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm template slide phù hợp nhất dựa trên description và max_length requirements

        QUY TẮC CHỌN SLIDE MỚI:
        1. Bắt buộc chọn đúng placeholder không dư không thiếu
        2. Nếu có trên 2 slide phù hợp -> chọn ra cái nào có max_length nhiều hơn
        3. Nếu max_length của slide nhiều hơn vẫn chưa đáp ứng được thì bỏ vào LLM làm ngắn
        4. Có thể dùng lại slide đã dùng

        Args:
            slide_description: List các placeholder descriptions từ slide (e.g., ["MainPointName_1_120", "TitleName_100"])
            template_slides: List các template slides
            used_slide_ids: Set các slide IDs đã sử dụng

        Returns:
            Dict slide template phù hợp nhất hoặc None
        """
        try:
            logger.info(f"🔍 Finding best template for description: {slide_description}")

            # Parse slide requirements from description
            slide_requirements = self._parse_slide_description(slide_description)

            matching_templates = []

            for template in template_slides:
                template_id = template.get("id")

                # Skip used templates (chỉ khi tìm lần đầu)
                if template_id in used_slide_ids:
                    continue

                # Parse template description (from Kafka format)
                template_description = template.get("description", "")
                template_requirements = self._parse_template_description(template_description)

                # Check if template matches slide requirements (EXACT MATCH - không dư không thiếu)
                match_score = self._calculate_template_match_score(
                    slide_requirements,
                    template_requirements
                )

                if match_score > 0:
                    # Tính tổng max_length của template để ưu tiên template có max_length lớn hơn
                    total_max_length = sum(req.get("max_length", 0) for req in template_requirements.values())

                    matching_templates.append({
                        "template": template,
                        "score": match_score,
                        "requirements": template_requirements,
                        "total_max_length": total_max_length
                    })

            if not matching_templates:
                logger.error(f"❌ No exact matching templates found for slide requirements!")
                logger.error(f"   Slide description: {slide_description}")
                logger.error(f"   Slide requirements: {slide_requirements}")
                logger.error(f"   Available templates checked: {len(template_slides)}")

                # Log all available templates for debugging
                for i, template in enumerate(template_slides):
                    template_desc = template.get("description", "")
                    template_id = template.get("id", "unknown")
                    logger.error(f"   Template {i+1}: {template_id} - {template_desc}")

                raise ValueError(f"No exact matching template found for slide requirements: {list(slide_requirements.keys())}")

            # QUY TẮC CHỌN SLIDE MỚI:
            # 1. Nếu có trên 2 slide phù hợp -> chọn ra cái nào có max_length nhiều hơn
            # 2. Nếu max_length bằng nhau thì chọn theo match_score cao hơn
            if len(matching_templates) >= 2:
                logger.info(f"🔍 Found {len(matching_templates)} matching templates, selecting by max_length priority")

                # Sort theo thứ tự ưu tiên:
                # 1. total_max_length (cao hơn = tốt hơn)
                # 2. match_score (cao hơn = tốt hơn)
                matching_templates.sort(key=lambda x: (x["total_max_length"], x["score"]), reverse=True)

                # Log thông tin các template để debug
                for i, match in enumerate(matching_templates[:3]):  # Log top 3
                    template_id = match["template"].get("id", "unknown")
                    logger.info(f"   Rank {i+1}: {template_id} - max_length: {match['total_max_length']}, score: {match['score']:.2f}")
            else:
                # Chỉ có 1 template phù hợp, sort theo score
                matching_templates.sort(key=lambda x: x["score"], reverse=True)

            best_match = matching_templates[0]
            template_id = best_match['template'].get('id', 'unknown')

            logger.info(f"✅ Selected best template: {template_id}")
            logger.info(f"   Total max_length: {best_match['total_max_length']}")
            logger.info(f"   Match score: {best_match['score']:.2f}")

            return best_match["template"]

        except ValueError as ve:
            # Re-raise ValueError để logic reuse có thể catch được
            logger.debug(f"🔍 ValueError in template matching: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"❌ Unexpected error finding best matching template: {e}")
            raise ValueError(f"Unexpected error in template matching: {str(e)}")

    def _parse_slide_description(self, slide_description: List[str]) -> Dict[str, Any]:
        """Parse slide description into structured requirements"""
        try:
            requirements = {}

            for desc in slide_description:
                parts = desc.split('_')
                if len(parts) >= 2:
                    placeholder_type = parts[0]
                    content_length = int(parts[-1])  # Last part is always length

                    # Handle numbered placeholders (e.g., MainPointName_1_120)
                    if len(parts) == 3 and parts[1].isdigit():
                        number = int(parts[1])
                        key = f"{placeholder_type}_{number}"
                    # Handle double-numbered placeholders (e.g., SubPointName_1_1_120)
                    elif len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
                        main_num = int(parts[1])
                        sub_num = int(parts[2])
                        key = f"{placeholder_type}_{main_num}_{sub_num}"
                    # Handle non-numbered placeholders (e.g., TitleName_100)
                    else:
                        key = placeholder_type

                    requirements[key] = {
                        "type": placeholder_type,
                        "length": content_length
                    }

            return requirements

        except Exception as e:
            logger.error(f"❌ Error parsing slide description: {e}")
            return {}

    def _parse_template_description(self, template_description: str) -> Dict[str, Any]:
        """Parse template description from Kafka format"""
        try:
            requirements = {}

            if not template_description:
                return requirements

            # Split by comma and parse each placeholder
            placeholders = [p.strip() for p in template_description.split(',')]

            for placeholder in placeholders:
                if not placeholder:
                    continue

                parts = placeholder.split('_')
                if len(parts) >= 2:
                    placeholder_type = parts[0]
                    max_length = int(parts[-1])  # Last part is max_length

                    # Handle numbered placeholders
                    if len(parts) == 3 and parts[1].isdigit():
                        number = int(parts[1])
                        key = f"{placeholder_type}_{number}"
                    # Handle double-numbered placeholders
                    elif len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
                        main_num = int(parts[1])
                        sub_num = int(parts[2])
                        key = f"{placeholder_type}_{main_num}_{sub_num}"
                    # Handle non-numbered placeholders
                    else:
                        key = placeholder_type

                    requirements[key] = {
                        "type": placeholder_type,
                        "max_length": max_length
                    }

            return requirements

        except Exception as e:
            logger.error(f"❌ Error parsing template description: {e}")
            return {}

    def _calculate_template_match_score(
        self,
        slide_requirements: Dict[str, Any],
        template_requirements: Dict[str, Any]
    ) -> float:
        """Calculate match score between slide and template requirements"""
        try:
            if not slide_requirements or not template_requirements:
                return 0.0

            # Check for EXACT placeholder match - no more, no less
            slide_keys = set(slide_requirements.keys())
            template_keys = set(template_requirements.keys())

            # Must have EXACT match - same placeholders, same count
            if slide_keys != template_keys:
                logger.debug(f"❌ Template placeholders don't match exactly:")
                logger.debug(f"   Slide needs: {slide_keys}")
                logger.debug(f"   Template has: {template_keys}")
                if slide_keys - template_keys:
                    logger.debug(f"   Template missing: {slide_keys - template_keys}")
                if template_keys - slide_keys:
                    logger.debug(f"   Template has extra: {template_keys - slide_keys}")
                return 0.0

            total_score = 0.0
            total_placeholders = len(slide_keys)

            for key in slide_keys:
                slide_req = slide_requirements[key]
                template_req = template_requirements[key]

                slide_length = slide_req["length"]
                template_max_length = template_req["max_length"]

                # Calculate score based on how well template accommodates content
                if slide_length <= template_max_length:
                    # Perfect fit or template has more space - good score
                    score = 1.0
                else:
                    # Content exceeds template max_length - lower score but still possible
                    # Score decreases as the excess increases
                    excess_ratio = (slide_length - template_max_length) / template_max_length
                    score = max(0.1, 1.0 - excess_ratio)  # Minimum score of 0.1

                total_score += score

            # Average score across all placeholders (exact match only)
            final_score = total_score / total_placeholders if total_placeholders > 0 else 0.0

            logger.debug(f"📊 Template match score: {final_score:.2f}")
            return final_score

        except Exception as e:
            logger.error(f"❌ Error calculating template match score: {e}")
            return 0.0

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



    def _create_placeholder_key(self, placeholder_type: str, index: int) -> str:
        """Create placeholder key for template lookup"""
        # For numbered placeholders like MainPointName_1, MainPointContent_1
        if placeholder_type in ["MainPointName", "MainPointContent"]:
            return f"{placeholder_type}_{index}"
        else:
            # For non-numbered placeholders like TitleName
            return placeholder_type

    async def _create_processed_slide_from_template(
        self,
        template_slide: Dict[str, Any],
        parsed_data: Dict[str, List[Dict[str, Any]]],
        slide_number: int,
        is_reused: bool = False,
        template_requirements: Dict[str, Any] = None
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

            # New placeholder patterns for the updated format - match both space and underscore formats
            placeholder_patterns = {
                "LessonName": r"LessonName[_\s]+(\d+)",
                "LessonDescription": r"LessonDescription[_\s]+(\d+)",
                "CreatedDate": r"CreatedDate[_\s]+(\d+)",
                "TitleName": r"TitleName[_\s]+(\d+)",
                "MainPointName": r"MainPointName[_\s]+(\d+)",
                "MainPointContent": r"MainPointContent[_\s]+(\d+)",
                "ImageName": r"ImageName[_\s]+(\d+)",
                "ImageContent": r"ImageContent[_\s]+(\d+)"
            }

            # Initialize template requirements if not provided
            if template_requirements is None:
                template_requirements = {}

            # Map content vào từng element (format mới)
            for element in template_elements:
                if element.get("type") == "text":
                    text = element.get("text", "").strip()
                    element_id = element.get("id")

                    # Detect placeholder type từ text
                    logger.info(f"🔍 Checking element text for placeholder: '{text}'")
                    placeholder_result = self._detect_placeholder_type_from_text(text, placeholder_patterns)
                    logger.info(f"🎯 Placeholder detection result: {placeholder_result}")

                    if placeholder_result:
                        placeholder_type, detected_max_length = placeholder_result

                        logger.info(f"🔍 Mapping content for {placeholder_type}:")
                        logger.info(f"   Element ID: {element_id}")

                        try:
                            # Extract position info from element text for precise mapping
                            placeholder_key = self._extract_placeholder_key_from_text(text, placeholder_type)
                            template_max_length = None

                            # Try to find max_length from template requirements using exact key
                            if placeholder_key in template_requirements:
                                template_max_length = template_requirements[placeholder_key].get("max_length")
                            elif placeholder_type in template_requirements:
                                template_max_length = template_requirements[placeholder_type].get("max_length")

                            # Use template max_length if available, otherwise use detected max_length
                            final_max_length = template_max_length if template_max_length is not None else detected_max_length

                            # Get content with precise position mapping
                            content_item = self._get_content_by_position(
                                parsed_data, placeholder_type, placeholder_key
                            )

                            if not content_item:
                                logger.warning(f"❌ No content found for {placeholder_key}")
                                return None  # Skip slide if missing positioned content

                            raw_content = content_item.get("content", "")
                            logger.info(f"   Raw content for {placeholder_key}: {raw_content[:100]}...")
                            logger.info(f"   Max length: {final_max_length} (template: {template_max_length}, detected: {detected_max_length})")

                            # Use existing _handle_max_length_content method
                            final_content = await self._handle_max_length_content(
                                raw_content,
                                final_max_length,
                                placeholder_type
                            )

                            # Copy element và update content (format mới)
                            processed_element = copy.deepcopy(element)
                            processed_element["text"] = final_content  # Update content

                            processed_slide["slideData"]["elements"].append(processed_element)

                            logger.info(f"✅ Mapped {placeholder_key} to {element_id}: {final_content[:100]}...")
                            logger.info(f"   Final content length: {len(final_content)}")

                        except Exception as e:
                            logger.error(f"❌ Failed to handle content for {placeholder_type} in slide {slide_number}: {e}")
                            logger.error(f"   SKIPPING this slide due to content length issue - NO FALLBACK")
                            return None  # Skip entire slide if any content fails
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

    def _extract_placeholder_key_from_text(self, text: str, placeholder_type: str) -> str:
        """
        Extract exact placeholder key from element text
        Ví dụ: "MainPointName 1 80" -> "MainPointName_1" hoặc "MainPointName_1" -> "MainPointName_1"
        """
        try:
            import re

            # Pattern để extract numbers từ text - support both space and underscore formats
            # Format 1: "MainPointName 1 80" (space separated)
            pattern1 = rf'{placeholder_type}\s+(\d+)\s+\d+'
            match1 = re.search(pattern1, text)

            if match1:
                main_num = match1.group(1)
                return f"{placeholder_type}_{main_num}"

            # Format 2: "MainPointName_1" (underscore format)
            pattern2 = rf'{placeholder_type}_(\d+)'
            match2 = re.search(pattern2, text)

            if match2:
                main_num = match2.group(1)
                return f"{placeholder_type}_{main_num}"

            # Non-numbered: TitleName
            return placeholder_type

        except Exception as e:
            logger.error(f"❌ Error extracting placeholder key from text '{text}': {e}")
            return placeholder_type

    def _get_content_by_position(
        self,
        parsed_data: Dict[str, List[Dict[str, Any]]],
        placeholder_type: str,
        placeholder_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get content by exact position based on placeholder key
        """
        try:
            content_list = parsed_data.get(placeholder_type, [])

            if not content_list:
                return None

            # For positioned placeholders, find by exact position_key match
            if placeholder_type in ["MainPointName", "MainPointContent"]:
                for item in content_list:
                    if item.get("position_key") == placeholder_key:
                        logger.info(f"✅ Found exact position match for {placeholder_key}")
                        return item

                # Fallback: try to find by position parsing
                logger.warning(f"⚠️ No exact position match for {placeholder_key}, using fallback")
                parts = placeholder_key.split('_')

                if placeholder_type in ["MainPointName", "MainPointContent"] and len(parts) >= 2:
                    target_main = int(parts[1])
                    for item in content_list:
                        if item.get("main_point") == target_main:
                            return item
            else:
                # Non-numbered placeholders: TitleName, LessonName, etc.
                if len(content_list) > 0:
                    return content_list[0]

            return None

        except Exception as e:
            logger.error(f"❌ Error getting content by position for {placeholder_key}: {e}")
            return None




# Singleton instance
_json_template_service = None

def get_json_template_service() -> JsonTemplateService:
    """Get singleton instance của JsonTemplateService"""
    global _json_template_service
    if _json_template_service is None:
        _json_template_service = JsonTemplateService()
    return _json_template_service
