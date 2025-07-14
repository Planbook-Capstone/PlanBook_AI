"""
Slide Generation Service
Xử lý logic sinh nội dung slide từ lesson content và template structure sử dụng LLM
"""

import logging
import threading
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.llm_service import get_llm_service
from app.services.textbook_retrieval_service import TextbookRetrievalService
from app.services.google_slides_service import get_google_slides_service

logger = logging.getLogger(__name__)


class SlideGenerationService:
    """
    Service để sinh nội dung slide từ lesson content và template
    Singleton pattern với Lazy Initialization
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation với thread-safe"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SlideGenerationService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Lazy initialization - chỉ khởi tạo một lần"""
        if self._initialized:
            return

        self.llm_service = None
        self.textbook_service = None
        self.slides_service = None
        self._service_initialized = False
        self._initialized = True

    def _ensure_service_initialized(self):
        """Ensure services are initialized"""
        if not self._service_initialized:
            logger.info("🔄 SlideGenerationService: First-time initialization triggered")
            self.llm_service = get_llm_service()
            self.textbook_service = TextbookRetrievalService()
            self.slides_service = get_google_slides_service()
            self._service_initialized = True
            logger.info("✅ SlideGenerationService: Initialization completed")

    def is_available(self) -> bool:
        """Kiểm tra service có sẵn sàng không"""
        self._ensure_service_initialized()
        return (self.llm_service and self.llm_service.is_available() and 
                self.slides_service and self.slides_service.is_available())

    async def generate_slides_from_lesson(
        self,
        lesson_id: str,
        template_id: str,
        config_prompt: Optional[str] = None,
        presentation_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tạo slides từ lesson_id và template_id (QUY TRÌNH MỚI)

        Args:
            lesson_id: ID của bài học
            template_id: ID của Google Slides template
            config_prompt: Prompt cấu hình tùy chỉnh (optional)
            presentation_title: Tiêu đề presentation tùy chỉnh (optional)

        Returns:
            Dict chứa kết quả tạo slides
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Slide generation service not available"
            }

        try:
            logger.info(f"Starting NEW slide generation process for lesson {lesson_id} with template {template_id}")

            # Bước 1: Lấy nội dung bài học
            lesson_result = await self.textbook_service.get_lesson_content(lesson_id)
            if not lesson_result:
                return {
                    "success": False,
                    "error": f"Could not retrieve lesson content for {lesson_id}"
                }

            lesson_content = lesson_result.get("lesson_content", "")
            if not lesson_content:
                return {
                    "success": False,
                    "error": f"Empty lesson content for {lesson_id}"
                }

            # Bước 2: Copy template và phân tích cấu trúc của bản sao (QUY TRÌNH MỚI)
            new_title = presentation_title or f"Bài học {lesson_id} - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
            copy_and_analyze_result = await self.slides_service.copy_and_analyze_template(template_id, new_title)
            if not copy_and_analyze_result["success"]:
                return {
                    "success": False,
                    "error": f"Could not copy and analyze template: {copy_and_analyze_result['error']}"
                }

            # Lưu template slide IDs ngay sau khi copy (trước khi tạo slides mới)
            original_template_slide_ids = [slide.get("slideId") for slide in copy_and_analyze_result.get("slides", [])]
            logger.info(f"📋 Saved original template slide IDs immediately after copy: {original_template_slide_ids}")

            # Bước 3: Sinh nội dung slides bằng LLM với cấu trúc của bản sao
            logger.info("🤖 Step 3: Generating slides content with LLM...")
            slides_content = await self._generate_slides_content(
                lesson_content,
                copy_and_analyze_result,
                config_prompt
            )
            if not slides_content["success"]:
                logger.error(f"❌ Failed to generate slides content: {slides_content.get('error', 'Unknown error')}")
                return slides_content

            logger.info(f"✅ Successfully generated slides content:")
            logger.info(f"   - Total slides: {len(slides_content.get('slides', []))}")
            logger.info(f"   - Original template slides: {slides_content.get('original_template_slide_ids', [])}")

            # Log chi tiết từng slide
            for i, slide in enumerate(slides_content.get('slides', [])):
                slide_id = slide.get('slideId')
                action = slide.get('action', 'update')
                slide_order = slide.get('slide_order', 'N/A')
                elements_count = len(slide.get('elements', []))
                logger.info(f"   Slide {i+1}: {slide_id} (order: {slide_order}, action: {action}, elements: {elements_count})")

            # Bước 4: Cập nhật nội dung vào bản sao đã tạo
            logger.info("📝 Step 4: Updating presentation content...")
            logger.info(f"   Presentation ID: {copy_and_analyze_result['copied_presentation_id']}")
            logger.info(f"   Slides to process: {len(slides_content['slides'])}")

            update_result = await self.slides_service.update_copied_presentation_content(
                copy_and_analyze_result["copied_presentation_id"],
                slides_content["slides"]
            )
            if not update_result["success"]:
                logger.error(f"❌ Failed to update presentation content: {update_result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": f"Could not update presentation content: {update_result['error']}"
                }

            logger.info("✅ Successfully updated presentation content")

            # Bước 5: Xóa TẤT CẢ template slides gốc (luồng mới)
            logger.info("🧹 Starting template cleanup - deleting ALL original template slides...")

            # Sử dụng template IDs đã lưu từ đầu (không phụ thuộc vào slides_content)
            logger.info(f"🗂️ Using saved template slide IDs: {original_template_slide_ids}")

            # Lấy danh sách slides mới đã tạo (tất cả đều là copy)
            created_slide_ids = []
            slides_content_ids = slides_content.get("original_template_slide_ids", [])

            for slide_data in slides_content["slides"]:
                slide_id = slide_data.get('slideId')
                if slide_id:
                    created_slide_ids.append(slide_id)
                    logger.info(f"📝 Created slide to keep: {slide_id}")

            logger.info(f"🗂️ Template slides to delete (saved from start): {original_template_slide_ids}")
            logger.info(f"🗂️ Template slides from content: {slides_content_ids}")
            logger.info(f"📝 Created slides to keep: {created_slide_ids}")

            # Debug: Kiểm tra trạng thái presentation trước khi xóa
            await self.slides_service.debug_presentation_state(
                copy_and_analyze_result["copied_presentation_id"],
                "Before template cleanup"
            )

            # Xóa TẤT CẢ template slides gốc (sử dụng IDs đã lưu từ đầu)
            if original_template_slide_ids:
                logger.info(f"🗑️ Attempting to delete template slides: {original_template_slide_ids}")

                delete_result = await self.slides_service.delete_all_template_slides(
                    copy_and_analyze_result["copied_presentation_id"],
                    original_template_slide_ids
                )
                logger.info(f"🧹 Template cleanup result: {delete_result}")

                # Log chi tiết kết quả
                if delete_result.get("success"):
                    deleted_count = delete_result.get("slides_deleted", 0)
                    not_found = delete_result.get("slides_not_found", [])
                    remaining = delete_result.get("remaining_slides", 0)

                    logger.info(f"✅ Template cleanup completed:")
                    logger.info(f"   - Slides deleted: {deleted_count}")
                    logger.info(f"   - Slides not found: {not_found}")
                    logger.info(f"   - Slides remaining: {remaining}")

                    if not_found:
                        logger.warning(f"⚠️ Some template slides were not found: {not_found}")
                        logger.warning("   This might indicate they were already deleted or IDs changed")
                else:
                    logger.error(f"❌ Template cleanup failed: {delete_result.get('error', 'Unknown error')}")

                # Debug: Kiểm tra trạng thái presentation sau khi xóa
                await self.slides_service.debug_presentation_state(
                    copy_and_analyze_result["copied_presentation_id"],
                    "After template cleanup"
                )
            else:
                logger.warning("⚠️ No original template slides found to delete")

            # Bước 6: Hoàn thành và trả về kết quả
            logger.info("🎉 Step 6: Slide generation completed successfully!")
            logger.info(f"   Final presentation ID: {copy_and_analyze_result['copied_presentation_id']}")
            logger.info(f"   Final slide count: {len(slides_content['slides'])}")
            logger.info(f"   Web view link: {copy_and_analyze_result['web_view_link']}")

            return {
                "success": True,
                "lesson_id": lesson_id,
                "original_template_id": template_id,
                "presentation_id": copy_and_analyze_result["copied_presentation_id"],
                "presentation_title": copy_and_analyze_result["presentation_title"],
                "web_view_link": copy_and_analyze_result["web_view_link"],
                "slides_created": update_result.get("slides_updated", 0) + update_result.get("slides_created", 0),
                "template_info": {
                    "title": copy_and_analyze_result["presentation_title"],
                    "layouts_count": copy_and_analyze_result["slide_count"]
                }
            }

        except Exception as e:
            logger.error(f"Error generating slides: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_slides_content(
        self,
        lesson_content: str,
        copied_presentation_info: Dict[str, Any],
        config_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sinh nội dung slides bằng LLM theo quy trình MỚI (chỉ 1 lần gọi AI + xử lý code)

        Args:
            lesson_content: Nội dung bài học
            copied_presentation_info: Thông tin presentation đã copy và phân tích
            config_prompt: Prompt cấu hình tùy chỉnh

        Returns:
            Dict chứa nội dung slides đã sinh
        """
        try:
            # Bước 1: Phân tích template và thêm placeholder types
            logger.info("🔍 Step 1: Analyzing template and detecting placeholder types...")
            analyzed_template = self._analyze_template_with_placeholders(copied_presentation_info)

            # Bước 2: Lần 1 gọi AI - Sinh presentation-content với annotation
            logger.info("🤖 Step 2: Single AI call - Generate annotated presentation content...")
            presentation_content = await self._generate_annotated_presentation_content(
                lesson_content,
                config_prompt
            )
            if not presentation_content["success"]:
                return presentation_content
            logger.info(f"-----------------------Generated presentation content: {presentation_content}")

            # Bước 3: Xử lý bằng code - Parse và map content vào template
            logger.info("🔧 Step 3: Code-based processing - Parse and map content to template...")
            mapped_slides = await self._parse_and_map_content_to_template(
                presentation_content["content"],
                analyzed_template
            )
            if not mapped_slides["success"]:
                return mapped_slides

            # Bước 4: Lọc và chỉ giữ slides được sử dụng
            logger.info("🧹 Step 4: Filter and keep only used slides...")
            final_slides = self._filter_used_slides(mapped_slides["slides"])

            return {
                "success": True,
                "slides": final_slides,
                "presentation_content": presentation_content["content"],  # For debugging
                "analyzed_template": analyzed_template  # For debugging
            }

        except Exception as e:
            logger.error(f"Error generating slides content: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _analyze_template_with_placeholders(self, copied_presentation_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phân tích template và thêm placeholder types theo enum yêu cầu

        Args:
            copied_presentation_info: Thông tin presentation đã copy

        Returns:
            Dict chứa template đã phân tích với placeholder types
        """
        try:
            analyzed_slides = []

            for slide in copied_presentation_info.get("slides", []):
                analyzed_elements = []
                placeholder_counts = {}

                for element in slide.get("elements", []):
                    text = element.get("text", "").strip()

                    if text:  # Chỉ xử lý elements có text
                        logger.info(f"🔍 Processing text in slide {slide.get('slideId')}: '{text}'")

                        # Detect placeholder type và max_length từ text
                        placeholder_result = self._detect_placeholder_type_from_text(text)

                        if placeholder_result:  # Chỉ xử lý nếu detect được placeholder
                            placeholder_type, max_length = placeholder_result

                            logger.info(f"✅ Found placeholder: {placeholder_type} <{max_length}>")

                            # Đếm số lượng placeholder types
                            placeholder_counts[placeholder_type] = placeholder_counts.get(placeholder_type, 0) + 1

                            # Tạo analyzed element với thông tin đầy đủ
                            analyzed_element = {
                                "objectId": element.get("objectId"),
                                "text": None,  # LLM sẽ insert nội dung sau
                                "Type": placeholder_type,
                                "max_length": max_length,
                            }

                            analyzed_elements.append(analyzed_element)
                        else:
                            # Bỏ qua text không phải placeholder format
                            logger.info(f"❌ Skipping non-placeholder text: '{text}'")
                            continue

                # Tạo description cho slide dựa trên placeholder counts
                description = self._generate_slide_description(placeholder_counts)

                analyzed_slide = {
                    "slideId": slide.get("slideId"),
                    "description": description,
                    "elements": analyzed_elements,
                    "placeholder_counts": placeholder_counts  # For logic selection
                }

                analyzed_slides.append(analyzed_slide)

            return {
                "slides": analyzed_slides,
                "total_slides": len(analyzed_slides),
                "original_info": copied_presentation_info
            }

        except Exception as e:
            logger.error(f"Error analyzing template with placeholders: {e}")
            return {"slides": [], "total_slides": 0, "original_info": copied_presentation_info}

    def _detect_placeholder_type_from_text(self, text: str) -> Optional[tuple]:
        """
        Detect placeholder type và max_length từ text format "PlaceholderName <max_length>"

        Args:
            text: Text từ element

        Returns:
            tuple: (placeholder_type, max_length) hoặc None nếu không detect được
        """
        try:
            # Tìm pattern "PlaceholderName max_length" (không có dấu < >)
            pattern = r'(\w+)\s+(\d+)'
            match = re.search(pattern, text)

            if match:
                placeholder_name = match.group(1)
                max_length = int(match.group(2))

                # Map placeholder name to enum
                placeholder_type = self._map_to_placeholder_enum(placeholder_name)
                if placeholder_type:  # Chỉ return nếu tìm thấy valid placeholder
                    return placeholder_type, max_length

            return None

        except Exception as e:
            logger.warning(f"Error detecting placeholder type: {e}")
            return None

    def _map_to_placeholder_enum(self, placeholder_name: str) -> Optional[str]:
        """
        Map placeholder name to enum values

        Args:
            placeholder_name: Name from text

        Returns:
            str: Enum placeholder type
        """
        # Mapping dictionary
        mapping = {
            "LessonName": "LessonName",
            "LessonDescription": "LessonDescription",
            "CreatedDate": "CreatedDate",
            "TitleName": "TitleName",
            "TitleContent": "TitleContent",
            "SubtitleName": "SubtitleName",
            "SubtitleContent": "SubtitleContent",
            "BulletItem": "BulletItem",
            "ImageName": "ImageName",
            "ImageContent": "ImageContent"
        }

        return mapping.get(placeholder_name)  # Return None if not found


    def _generate_slide_description(self, placeholder_counts: Dict[str, int]) -> str:
        """
        Generate description for slide based on placeholder counts

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

    async def _generate_annotated_presentation_content(
        self,
        lesson_content: str,
        config_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Lần 1 gọi AI: Sinh presentation-content với annotation rõ ràng

        Args:
            lesson_content: Nội dung bài học
            config_prompt: Prompt cấu hình tùy chỉnh

        Returns:
            Dict chứa presentation content đã sinh với annotation (text thuần túy)
        """
        try:
            # Tạo prompt cho lần gọi AI với annotation requirements
            prompt = self._create_annotated_presentation_prompt(lesson_content, config_prompt)

            logger.info(f"AI call prompt length: {len(prompt)} characters")

            # Gọi LLM với retry logic
            max_retries = 3
            for attempt in range(max_retries):
                logger.info(f"AI call attempt {attempt + 1}/{max_retries}")

                # Tăng max_tokens cho slide generation vì response có thể dài
                llm_result = await self.llm_service.generate_content(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=50000
                )

                if llm_result["success"] and llm_result.get("text") and llm_result["text"].strip():
                    logger.info(f"AI call successful on attempt {attempt + 1}")

                    # Return the annotated text content
                    presentation_content = llm_result["text"].strip()
                    logger.debug(f"AI response length: {len(presentation_content)} characters")
                    logger.debug(f"AI response preview: {presentation_content[:200]}...")

                    return {
                        "success": True,
                        "content": presentation_content
                    }
                else:
                    logger.warning(f"AI call attempt {attempt + 1} failed: {llm_result.get('error', 'Empty response')}")
                    if attempt == max_retries - 1:
                        return {
                            "success": False,
                            "error": f"AI call failed after {max_retries} attempts: {llm_result.get('error', 'Empty response')}"
                        }

                # Wait before retry
                import asyncio
                await asyncio.sleep(1)

            return {
                "success": False,
                "error": "AI call failed"
            }

        except Exception as e:
            logger.error(f"Error in AI call: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_annotated_presentation_prompt(
        self,
        lesson_content: str,
        config_prompt: Optional[str] = None
    ) -> str:
        """
        Tạo prompt cho lần gọi AI với annotation requirements

        Args:
            lesson_content: Nội dung bài học
            config_prompt: Prompt cấu hình tùy chỉnh

        Returns:
            str: Prompt cho AI với annotation requirements
        """
        default_config = """
Bạn là chuyên gia thiết kế nội dung thuyết trình giáo dục. Nhiệm vụ của bạn là phân tích nội dung bài học và tạo ra nội dung thuyết trình.
NGUYÊN TẮC THIẾT KẾ:
1. PHÂN TÍCH TOÀN DIỆN - Hiểu rõ nội dung bài học và chia thành các phần logic
2. CẤU TRÚC RÕ RÀNG - Từ tổng quan đến chi tiết, có thứ tự logic
3. NỘI DUNG PHONG PHÚ VÀ CHI TIẾT - Tạo ít nhất 6-8 slides với nội dung đầy đủ
4. ANNOTATION CHÍNH XÁC - Đánh dấu rõ ràng các placeholder type
5. KÝ HIỆU KHOA HỌC CHÍNH XÁC - Sử dụng Unicode cho công thức
6. SLIDE SUMMARIES CHI TIẾT - Ghi rõ số lượng từng placeholder type
YÊU CẦU ANNOTATION:
- PHẢI có annotation bằng #*(PlaceholderType)*# chỉ rõ placeholder type.
- Placeholder types hỗ trợ: LessonName, LessonDescription, CreatedDate, TitleName, TitleContent, SubtitleName, SubtitleContent, ImageName, ImageContent
- TẠM THỜI KHÔNG SỬ DỤNG BulletItem - chỉ dùng 9 placeholder types trên
- Annotation phải chính xác và nhất quán
- CẦN có slide summaries với SỐ LƯỢNG RÕ RÀNG để hỗ trợ chọn slide template phù hợp
"""

        # final_config = config_prompt if config_prompt else default_config
        final_config = default_config
        prompt = f"""
{final_config}

NỘI DUNG BÀI HỌC:
{lesson_content}

HƯỚNG DẪN TẠO PRESENTATION CONTENT VỚI ANNOTATION:
1. PHÂN TÍCH BÀI HỌC:
   - Xác định chủ đề chính và các chủ đề phụ
   - Chia nội dung thành các phần logic (slides)
   - Mỗi phần có nội dung đầy đủ, chi tiết
   - Xác định thông tin quan trọng cần nhấn mạnh
   - Tránh lược bỏ các thông tin quan trọng trong nội dung bài học được cung cấp
2. TẠO NỘI DUNG VỚI ANNOTATION:
   - PHẢI có annotation #*(PlaceholderType)*# ngay sau
   - Ví dụ: "Bài 1: Cấu hình phân tử #*(LessonName)*#"
   - Ví dụ: "Bài này cho chúng ta biết được cấu hình... #*(LessonDescription)*#"
   - Ví dụ: "Ngày thuyết trình: 12-07-2025 #*(CreatedDate)*#"
   - TẠM THỜI KHÔNG dùng BulletItem - chỉ dùng 9 placeholder types còn lại
3. HIỂU RÕ CẤU TRÚC PHÂN CẤP VÀ NHÓM NỘI DUNG:
   - TitleName: Tên mục lớn (tên nội dung chính của slide đó) - CHỈ LÀ TIÊU ĐỀ
   - TitleContent: Tất cả nội dung giải thích thuộc mục lớn đó - NHÓM TẤT CẢ NỘI DUNG CHUNG
   - SubtitleName: Tên mục nhỏ bên trong mục lớn - CHỈ LÀ TIÊU ĐỀ CON
   - SubtitleContent: Tất cả nội dung giải thích thuộc mục nhỏ (SubtitleName) đó - NHÓM TẤT CẢ NỘI DUNG CON CHUNG
4. Ví dụ CHI TIẾT VỚI CẤU TRÚC PHÂN CẤP RÕ RÀNG VÀ NHÓM NỘI DUNG:
SLIDE 1 - GIỚI THIỆU:
[Tên bài học] #*(LessonName)*#
[Tóm tắt ngắn gọn về bài học] #*(LessonDescription)*#
Ngày thuyết trình: 12-07-2025 #*(CreatedDate)*#
=== SLIDE 1 SUMMARY ===
Placeholders: 1xLessonName, 1xLessonDescription, 1xCreatedDate
===========================
SLIDE 2 - MỤC LỚN VỚI NỘI DUNG TỔNG QUÁT:
[Tên mục lớn] #*(TitleName)*#
[Tất cả nội dung tổng quát giải thích về mục lớn này, khái niệm chung, định nghĩa. Nếu có nhiều đoạn thì gộp tất cả thành một khối nội dung chung] #*(TitleContent)*#
=== SLIDE 2 SUMMARY ===
Placeholders: 1xTitleName, 1xTitleContent
===========================
SLIDE 3 - CHI TIẾT CÁC MỤC NHỎ TRONG MỤC LỚN:
[Tên mục lớn khác] #*(TitleName)*#
[Tên mục nhỏ thứ nhất] #*(SubtitleName)*#
[Tất cả nội dung chi tiết của mục nhỏ thứ Nhất được gộp chung thành một khối nội dung] #*(SubtitleContent)*#
[Tên mục nhỏ thứ hai] #*(SubtitleName)*#
[Tất cả nội dung chi tiết của mục nhỏ thứ HAI được gộp chung thành một khối nội dung] #*(SubtitleContent)*#
=== SLIDE 3 SUMMARY ===
Placeholders: 1xTitleName, 2xSubtitleName, 1xSubtitleContent
... (tiếp tục với các slide khác tùy theo nội dung bài học)
4. QUY TẮC ANNOTATION VÀ NHÓM NỘI DUNG:
   - LUÔN có annotation #*(PlaceholderType)*# sau mỗi câu/tiêu đề
   - Sử dụng đúng placeholder types: LessonName, LessonDescription, CreatedDate, TitleName, TitleContent, SubtitleName, SubtitleContent, BulletItem, ImageName, ImageContent
   - Annotation phải nhất quán và chính xác
   - Nội dung phải phù hợp với placeholder type
   - QUAN TRỌNG: Mỗi TitleName có thể có nhiều TitleContent thì tất cả TitleContent đều chung 1 TitleContent
   - QUAN TRỌNG: Mỗi SubtitleName có thể có nhiều SubtitleContent thì tất cả SubtitleContent đều chung 1 SubtitleContent
   VÍ DỤ : "Nguyên tố Hydro (H) có tính chất đặc biệt. #*(TitleContent)*#" 
   VÍ DỤ CẤU TRÚC ĐÚNG VỚI NHÓM NỘI DUNG:
   Slide 1: 
   Khái niệm nguyên tố #*(TitleName)*# ← Đây là tên mục lớn
   Nguyên tố hóa học là tập hợp các nguyên tử có cùng số proton. Mỗi nguyên tố có tính chất riêng biệt và được xác định bởi số hiệu nguyên tử. Các nguyên tố được sắp xếp trong bảng tuần hoàn theo thứ tự tăng dần của số hiệu nguyên tử. #*(TitleContent)*# ← Tất cả nội dung mục lớn gộp chung
   Slide 2: 
   Đặc điểm của nguyên tố #*(TitleName)*# ← Đây là tên mục lớn khác
    Định nghĩa #*(SubtitleName)*# ← Đây là tên mục nhỏ trong mục lớn
    Nguyên tố được định nghĩa là những chất không thể phân tách thành những chất đơn giản hơn bằng phương pháp hóa học thông thường. #*(SubtitleContent)*# ← Tất cả nội dung các mục nhỏ gộp chung
    Tính chất #*(SubtitleName)*# ← Đây là tên mục nhỏ khác
   Các tính chất của nguyên tố bao gồm tính chất vật lý như màu sắc, trạng thái và tính chất hóa học như khả năng phản ứng. #*(SubtitleContent)*# ← Tất cả nội dung các mục nhỏ gộp chung
5. SLIDE SUMMARIES:
   Cuối mỗi phần logic của presentation, thêm slide summary với SỐ LƯỢNG RÕ RÀNG:
   === SLIDE [Số] SUMMARY ===
   Placeholders: [Số lượng]x[PlaceholderType], [Số lượng]x[PlaceholderType], ...
   Ví dụ: 1xLessonName, 1xLessonDescription, 1xCreatedDate, 2xTitleName, 3xTitleContent
   ===========================
YÊU CẦU OUTPUT:
Tạo nội dung thuyết trình TEXT THUẦN TÚY với annotation rõ ràng, theo đúng format trên.
BẮT BUỘC có slide summaries để hỗ trợ việc chọn slide template phù hợp.
VÍ DỤ MINH HỌA CẤU TRÚC ĐÚNG VỚI NHÓM NỘI DUNG:
SLIDE 1: (Slide này là bắt buộc và luôn có)
Cấu hình electron #*(LessonName)*#
Bài này cho chúng ta biết được cấu hình electron trong nguyên tử và phân tử #*(LessonDescription)*#
Ngày thuyết trình: 12-07-2025 #*(CreatedDate)*#
=== SLIDE 1 SUMMARY ===
Placeholders: 1xLessonName, 1xLessonDescription, 1xCreatedDate
===========================
SLIDE 2: 
Khái niệm cấu hình electron #*(TitleName)*#
Cấu hình electron là cách sắp xếp các electron trong các orbital của nguyên tử. Cấu hình này quyết định tính chất hóa học của nguyên tố và khả năng tạo liên kết. Việc hiểu rõ cấu hình electron giúp dự đoán tính chất và hành vi của các nguyên tố trong phản ứng hóa học. #*(TitleContent)*#
=== SLIDE 2 SUMMARY ===
Placeholders: 1xTitleName, 1xTitleContent
===========================
SLIDE 3:
Các quy tắc sắp xếp electron #*(TitleName)*#
 Quy tắc Aufbau #*(SubtitleName)*#
  Electron điền vào orbital có mức năng lượng thấp trước, sau đó mới điền vào orbital có mức năng lượng cao hơn theo quy tắc Aufbau. #*(SubtitleContent)*#
 Nguyên lý Pauli #*(SubtitleName)*#
  Mỗi orbital chứa tối đa 2 electron và chúng phải có spin ngược chiều nhau theo nguyên lý Pauli. Các quy tắc này đảm bảo cấu hình electron ổn định nhất. #*(SubtitleContent)*#
=== SLIDE 3 SUMMARY ===
Placeholders: 1xTitleName, 2xSubtitleName, 2xSubtitleContent
===========================
SLIDE 4: 
Hình ảnh minh họa: Sơ đồ cấu hình electron #*(ImageName)*#
Sơ đồ thể hiện cách electron được sắp xếp trong các orbital 1s, 2s, 2p theo thứ tự năng lượng tăng dần #*(ImageContent)*#
=== SLIDE 4 SUMMARY ===
Placeholders: 1xImageName, 1xImageContent
===========================
QUY TẮC VIẾT VỚI NHÓM NỘI DUNG:
- LUÔN có annotation #*(PlaceholderType)*# sau mỗi nội dung
- Nội dung đầy đủ, chi tiết. Không được bỏ xót bất kì kiến thức nào trong bài học
- TẠMTHỜI KHÔNG sử dụng BulletItem - chỉ dùng 9 placeholder types còn lại
- PHÂN BIỆT RÕ RÀNG CẤU TRÚC PHÂN CẤP VÀ NHÓM NỘI DUNG:
  * TitleName: CHỈ là tiêu đề mục lớn (Tên nội dung chính của slide đó)
  * TitleContent: TẤT CẢ nội dung giải thích của mục lớn được gộp chung thành 1 khối
  * SubtitleName: CHỈ là tiêu đề mục nhỏ bên trong mục lớn 
  * SubtitleContent: TẤT CẢ nội dung giải thích của từng mục nhỏ được gộp chung thành 1 khối
- Ký hiệu khoa học chính xác: H₂O, CO₂, x², √x, π, α, β
- Logic trình bày từ tổng quan đến chi tiết
- Sử dụng ngày hiện tại cho CreatedDate
"""

        return prompt

    async def _parse_and_map_content_to_template(
        self,
        annotated_content: str,
        analyzed_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Xử lý bằng code: Parse annotated content và map vào template

        Args:
            annotated_content: Nội dung có annotation từ AI
            analyzed_template: Template đã phân tích

        Returns:
            Dict chứa slides đã map content
        """
        try:
            logger.info("🔧 Starting code-based content parsing and mapping...")

            # Bước 1: Parse annotated content
            parsed_content = self._parse_annotated_content(annotated_content)
            if not parsed_content:
                return {
                    "success": False,
                    "error": "Failed to parse annotated content"
                }

            # Bước 2: Map content to template slides
            mapped_slides = await self._map_parsed_content_to_slides(
                parsed_content,
                analyzed_template
            )
            if not mapped_slides["success"]:
                return mapped_slides

            return {
                "success": True,
                "slides": mapped_slides["slides"]
            }

        except Exception as e:
            logger.error(f"Error in code-based content mapping: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _parse_annotated_content(self, annotated_content: str) -> Dict[str, Any]:
        """
        Parse annotated content từ AI response

        Args:
            annotated_content: Nội dung có annotation

        Returns:
            Dict chứa parsed content theo placeholder types
        """
        try:
            import re

            logger.info("📝 Parsing annotated content...")

            # Dictionary để lưu parsed content (tạm thời loại bỏ BulletItem)
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

            # Slide summaries để track slide structure
            slide_summaries = []

            # Pattern để tìm annotation: text #*(PlaceholderType)*#
            # Chỉ match các placeholder types hợp lệ (tạm thời loại bỏ BulletItem)
            valid_placeholders = '|'.join([
                'LessonName', 'LessonDescription', 'CreatedDate',
                'TitleName', 'TitleContent', 'SubtitleName', 'SubtitleContent',
                'ImageName', 'ImageContent'
            ])
            annotation_pattern = rf'(.+?)\s*#\*\(({valid_placeholders})\)\*#'

            # Pattern để tìm slide summaries với format số lượng
            summary_pattern = r'=== SLIDE (\d+) SUMMARY ===\s*Placeholders:\s*([^=]+)\s*==='

            lines = annotated_content.split('\n')
            current_slide_content = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for slide summary
                summary_match = re.search(summary_pattern, line + '\n' + '\n'.join(lines[lines.index(line):lines.index(line)+3]))
                if summary_match:
                    slide_num = int(summary_match.group(1))
                    placeholder_text = summary_match.group(2).strip()

                    # Parse placeholder format: "2xTitleName, 3xSubtitleContent" hoặc "TitleName, SubtitleContent"
                    placeholders = []
                    placeholder_counts = {}

                    for item in placeholder_text.split(','):
                        item = item.strip()
                        if 'x' in item:
                            # Format: "2xTitleName"
                            count_str, placeholder_type = item.split('x', 1)
                            try:
                                count = int(count_str)
                                placeholders.append(placeholder_type.strip())
                                placeholder_counts[placeholder_type.strip()] = count
                            except ValueError:
                                # Fallback nếu không parse được số
                                placeholders.append(item)
                                placeholder_counts[item] = 1
                        else:
                            # Format cũ: "TitleName"
                            placeholders.append(item)
                            placeholder_counts[item] = 1

                    slide_summaries.append({
                        "slide_number": slide_num,
                        "placeholders": placeholders,
                        "placeholder_counts": placeholder_counts,
                        "content": current_slide_content.copy()
                    })
                    current_slide_content = []
                    continue

                # Find annotation matches
                matches = re.findall(annotation_pattern, line)
                for match in matches:
                    content = match[0].strip()
                    placeholder_type = match[1].strip()

                    if placeholder_type in parsed_data:
                        parsed_data[placeholder_type].append({
                            "content": content,
                            "original_line": line
                        })
                        current_slide_content.append({
                            "type": placeholder_type,
                            "content": content
                        })
                        logger.debug(f"✅ Parsed {placeholder_type}: {content[:50]}...")
                    else:
                        logger.warning(f"❌ Unknown placeholder type: {placeholder_type}")

            # If no slide summaries found, create default structure
            if not slide_summaries and any(parsed_data.values()):
                logger.info("No slide summaries found, creating default structure...")
                slide_summaries = [{"slide_number": 1, "placeholders": list(parsed_data.keys()), "content": current_slide_content}]

            result = {
                "parsed_data": parsed_data,
                "slide_summaries": slide_summaries,
                "total_items": sum(len(items) for items in parsed_data.values())
            }

            logger.info(f"✅ Parsing completed: {result['total_items']} items found, {len(slide_summaries)} slides")
            return result

        except Exception as e:
            logger.error(f"Error parsing annotated content: {e}")
            return None

    async def _map_parsed_content_to_slides(
        self,
        parsed_content: Dict[str, Any],
        analyzed_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map parsed content vào slides bằng cách tạo copy từ template (luồng mới)

        Args:
            parsed_content: Content đã parse
            analyzed_template: Template đã phân tích

        Returns:
            Dict chứa mapped slides
        """
        try:
            logger.info("🎯 Starting new mapping flow: copy-based template usage...")

            template_slides = analyzed_template.get("slides", [])
            parsed_data = parsed_content.get("parsed_data", {})

            mapped_slides = []
            content_index = {key: 0 for key in parsed_data.keys()}  # Track content usage

            # Lưu lại tất cả ID template gốc để xóa sau này
            original_template_slide_ids = [slide.get("slideId") for slide in template_slides]
            logger.info(f"📋 Saved original template slide IDs for cleanup: {original_template_slide_ids}")

            slide_summaries = parsed_content.get("slide_summaries", [])

            # Xử lý theo slide summaries
            if slide_summaries:
                logger.info(f"🎯 Processing {len(slide_summaries)} slide summaries...")

                # Xử lý từng slide summary
                for i, summary in enumerate(slide_summaries):
                    slide_num = i + 1
                    required_placeholders = summary.get("placeholders", [])
                    required_counts = summary.get("placeholder_counts", {})

                    logger.info(f"🔍 Processing slide {slide_num}:")
                    logger.info(f"   Required placeholders: {required_placeholders}")
                    logger.info(f"   Required counts: {required_counts}")

                    # Tìm template phù hợp CHÍNH XÁC
                    best_template = self._find_exact_matching_template(
                        required_placeholders,
                        required_counts,
                        template_slides
                    )

                    if best_template:
                        logger.info(f"✅ Found exact matching template: {best_template['slideId']}")

                        # Tạo slide copy từ template
                        copied_slide = await self._create_slide_copy_from_template(
                            best_template,
                            parsed_data,
                            content_index,
                            slide_num
                        )

                        if copied_slide:
                            mapped_slides.append(copied_slide)
                            logger.info(f"✅ Successfully created slide {slide_num}: {copied_slide['slideId']}")
                            logger.info(f"📊 Elements mapped: {len(copied_slide.get('elements', []))}")
                        else:
                            logger.warning(f"❌ Failed to create slide copy for slide {slide_num}")
                    else:
                        # Không có exact match - skip slide này
                        logger.warning(f"❌ No exact matching template found for slide {slide_num} - SKIPPING")
                        logger.warning(f"   Required: {required_counts}")
                        logger.warning(f"   Available templates do not match exactly - slide will be skipped")

                logger.info(f"🎯 Completed processing all {len(slide_summaries)} slides")
            else:
                # Không có slide summaries - không xử lý
                logger.error("❌ No slide summaries found - cannot process without structured content")
                logger.error("💡 LLM must generate proper slide summaries with placeholder counts")
                return {
                    "success": False,
                    "error": "No slide summaries found - cannot process without structured content",
                    "slides": [],
                    "original_template_slide_ids": original_template_slide_ids
                }

            logger.info(f"✅ Mapping completed: {len(mapped_slides)} slides created")
            logger.info(f"📋 Original template slides to cleanup: {original_template_slide_ids}")

            return {
                "success": True,
                "slides": mapped_slides,
                "original_template_slide_ids": original_template_slide_ids
            }

        except Exception as e:
            logger.error(f"Error mapping content to slides: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _find_exact_matching_slide(
        self,
        required_placeholders: List[str],
        required_counts: Dict[str, int],
        template_slides: List[Dict[str, Any]],
        used_slide_ids: set
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm slide template có placeholder CHÍNH XÁC với required placeholders (legacy method)

        Args:
            required_placeholders: List placeholder types cần thiết
            required_counts: Dict số lượng từng placeholder type
            template_slides: List template slides available
            used_slide_ids: Set slide IDs đã sử dụng

        Returns:
            Dict slide template match chính xác hoặc None
        """
        try:
            for slide in template_slides:
                slide_id = slide.get("slideId")

                # Skip used slides
                if slide_id in used_slide_ids:
                    continue

                # Get placeholder types and counts in this slide
                slide_elements = slide.get("elements", [])
                slide_placeholder_counts = {}

                for elem in slide_elements:
                    placeholder_type = elem.get("Type")
                    if placeholder_type in slide_placeholder_counts:
                        slide_placeholder_counts[placeholder_type] += 1
                    else:
                        slide_placeholder_counts[placeholder_type] = 1

                # Check for EXACT match: same placeholder types and same counts
                required_set = set(required_placeholders)
                slide_set = set(slide_placeholder_counts.keys())

                # Must have exactly the same placeholder types
                if required_set == slide_set:
                    # Check if counts match
                    counts_match = True
                    for placeholder_type in required_placeholders:
                        required_count = required_counts.get(placeholder_type, 1)
                        slide_count = slide_placeholder_counts.get(placeholder_type, 0)

                        if required_count != slide_count:
                            counts_match = False
                            break

                    if counts_match:
                        logger.info(f"✅ Found EXACT matching slide: {slide_id}")
                        logger.info(f"   Required: {required_counts}")
                        logger.info(f"   Slide has: {slide_placeholder_counts}")
                        return slide
                    else:
                        logger.debug(f"❌ Slide {slide_id}: placeholder types match but counts differ")
                        logger.debug(f"   Required: {required_counts}")
                        logger.debug(f"   Slide has: {slide_placeholder_counts}")
                else:
                    logger.debug(f"❌ Slide {slide_id}: placeholder types don't match")
                    logger.debug(f"   Required: {required_set}")
                    logger.debug(f"   Slide has: {slide_set}")

            logger.warning(f"❌ No EXACT matching slide found for: {required_counts}")
            return None

        except Exception as e:
            logger.error(f"Error finding exact matching slide: {e}")
            return None

    def _find_exact_matching_slide_with_reuse(
        self,
        required_placeholders: List[str],
        required_counts: Dict[str, int],
        template_slides: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm slide template có placeholder CHÍNH XÁC với required placeholders (cho phép reuse)

        Args:
            required_placeholders: List placeholder types cần thiết
            required_counts: Dict số lượng từng placeholder type
            template_slides: List template slides available

        Returns:
            Dict slide template match chính xác hoặc None
        """
        try:
            logger.info(f"🔍 Finding exact matching slide with reuse support...")

            for slide in template_slides:
                slide_id = slide.get("slideId")

                # Get placeholder types and counts in this slide
                slide_elements = slide.get("elements", [])
                slide_placeholder_counts = {}

                for elem in slide_elements:
                    placeholder_type = elem.get("Type")
                    if placeholder_type in slide_placeholder_counts:
                        slide_placeholder_counts[placeholder_type] += 1
                    else:
                        slide_placeholder_counts[placeholder_type] = 1

                # Check for EXACT match: same placeholder types and same counts
                required_set = set(required_placeholders)
                slide_set = set(slide_placeholder_counts.keys())

                # Must have exactly the same placeholder types
                if required_set == slide_set:
                    # Check if counts match
                    counts_match = True
                    for placeholder_type in required_placeholders:
                        required_count = required_counts.get(placeholder_type, 1)
                        slide_count = slide_placeholder_counts.get(placeholder_type, 0)

                        if required_count != slide_count:
                            counts_match = False
                            break

                    if counts_match:
                        logger.info(f"✅ Found EXACT matching slide (reuse allowed): {slide_id}")
                        logger.info(f"   Required: {required_counts}")
                        logger.info(f"   Slide has: {slide_placeholder_counts}")
                        return slide
                    else:
                        logger.debug(f"❌ Slide {slide_id}: placeholder types match but counts differ")
                        logger.debug(f"   Required: {required_counts}")
                        logger.debug(f"   Slide has: {slide_placeholder_counts}")
                else:
                    logger.debug(f"❌ Slide {slide_id}: placeholder types don't match")
                    logger.debug(f"   Required: {required_set}")
                    logger.debug(f"   Slide has: {slide_set}")

            logger.info(f"❌ No EXACT matching slide found for reuse: {required_counts}")
            return None

        except Exception as e:
            logger.error(f"Error finding exact matching slide with reuse: {e}")
            return None

    def _find_exact_matching_template(
        self,
        required_placeholders: List[str],
        required_counts: Dict[str, int],
        template_slides: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm template slide có placeholder CHÍNH XÁC (luồng mới - chỉ exact match)

        Args:
            required_placeholders: List placeholder types cần thiết
            required_counts: Dict số lượng từng placeholder type
            template_slides: List template slides available

        Returns:
            Dict template slide match chính xác hoặc None
        """
        try:
            logger.info(f"🔍 Finding exact matching template (strict mode)...")

            for slide in template_slides:
                slide_id = slide.get("slideId")

                # Get placeholder types and counts in this slide
                slide_elements = slide.get("elements", [])
                slide_placeholder_counts = {}

                for elem in slide_elements:
                    placeholder_type = elem.get("Type")
                    if placeholder_type in slide_placeholder_counts:
                        slide_placeholder_counts[placeholder_type] += 1
                    else:
                        slide_placeholder_counts[placeholder_type] = 1

                # Check for EXACT match: same placeholder types and same counts
                required_set = set(required_placeholders)
                slide_set = set(slide_placeholder_counts.keys())

                # Must have exactly the same placeholder types
                if required_set == slide_set:
                    # Check if counts match exactly
                    counts_match = True
                    for placeholder_type in required_placeholders:
                        required_count = required_counts.get(placeholder_type, 1)
                        slide_count = slide_placeholder_counts.get(placeholder_type, 0)

                        if required_count != slide_count:
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

    async def _create_mapped_slide(
        self,
        template_slide: Dict[str, Any],
        parsed_data: Dict[str, List[Dict[str, Any]]],
        content_index: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Tạo mapped slide từ template và parsed content

        Args:
            template_slide: Template slide
            parsed_data: Parsed content data
            content_index: Index tracking cho content usage

        Returns:
            Dict mapped slide hoặc None
        """
        try:
            slide_id = template_slide.get("slideId")
            template_elements = template_slide.get("elements", [])

            mapped_elements = []

            for element in template_elements:
                object_id = element.get("objectId")
                placeholder_type = element.get("Type")
                max_length = element.get("max_length", 1000)

                # Get content for this placeholder type
                content_list = parsed_data.get(placeholder_type, [])
                current_index = content_index.get(placeholder_type, 0)

                if current_index < len(content_list):
                    content_item = content_list[current_index]
                    raw_content = content_item.get("content", "")

                    try:
                        # Check max_length and handle if needed
                        final_content = await self._handle_max_length_content(
                            raw_content,
                            max_length,
                            placeholder_type
                        )

                        mapped_element = {
                            "objectId": object_id,
                            "text": final_content,
                            "Type": placeholder_type,
                            "max_length": max_length
                        }

                        mapped_elements.append(mapped_element)
                        content_index[placeholder_type] = current_index + 1

                        logger.debug(f"✅ Mapped {placeholder_type}: {final_content[:50]}...")
                    except Exception as e:
                        logger.error(f"❌ Failed to handle content for {placeholder_type} in slide {slide_id}: {e}")
                        logger.error(f"   Content length: {len(raw_content)}, Max length: {max_length}")
                        logger.error(f"   SKIPPING this slide due to content length issue - NO FALLBACK")
                        return None  # Skip entire slide if any content fails
                else:
                    logger.warning(f"❌ No content available for {placeholder_type} in slide {slide_id}")
                    return None  # Skip slide if missing content

            if mapped_elements:
                return {
                    "slideId": slide_id,
                    "elements": mapped_elements
                }
            else:
                logger.warning(f"❌ No elements mapped for slide {slide_id}")
                return None

        except Exception as e:
            logger.error(f"Error creating mapped slide: {e}")
            return None





    async def _create_slide_copy_from_template(
        self,
        template_slide: Dict[str, Any],
        parsed_data: Dict[str, List[Dict[str, Any]]],
        content_index: Dict[str, int],
        slide_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Tạo slide copy từ template

        Args:
            template_slide: Template slide để copy
            parsed_data: Parsed content data
            content_index: Index tracking cho content usage
            slide_number: Số thứ tự slide

        Returns:
            Dict slide copy hoặc None
        """
        try:
            template_slide_id = template_slide.get("slideId")
            template_elements = template_slide.get("elements", [])

            # Tạo slideId mới cho slide copy
            new_slide_id = f"slide_{slide_number:03d}_copy_of_{template_slide_id}"

            logger.info(f"📄 Creating slide copy: {new_slide_id} (from template: {template_slide_id})")

            mapped_elements = []
            updates = {}

            # Map content vào từng element
            for element in template_elements:
                object_id = element.get("objectId")
                placeholder_type = element.get("Type")
                max_length = element.get("max_length")

                content_list = parsed_data.get(placeholder_type, [])
                current_index = content_index.get(placeholder_type, 0)

                if current_index < len(content_list):
                    content_item = content_list[current_index]
                    raw_content = content_item.get("content", "")

                    try:
                        # Handle max_length if needed
                        final_content = await self._handle_max_length_content(
                            raw_content,
                            max_length,
                            placeholder_type
                        )

                        mapped_element = {
                            "objectId": object_id,
                            "text": final_content,
                            "Type": placeholder_type,
                            "max_length": max_length
                        }

                        mapped_elements.append(mapped_element)
                        updates[object_id] = final_content

                        # Increment content index
                        content_index[placeholder_type] = current_index + 1

                        logger.debug(f"✅ Mapped {placeholder_type} to {object_id}: {final_content[:50]}...")
                    except Exception as e:
                        logger.error(f"❌ Failed to handle content for {placeholder_type} in slide {slide_number}: {e}")
                        logger.error(f"   Content length: {len(raw_content)}, Max length: {max_length}")
                        logger.error(f"   SKIPPING this slide due to content length issue - NO FALLBACK")
                        return None  # Skip entire slide if any content fails
                else:
                    logger.warning(f"❌ No more content available for {placeholder_type} in slide {slide_number}")
                    return None  # Skip slide if missing content

            if mapped_elements:
                return {
                    "slideId": new_slide_id,
                    "elements": mapped_elements,
                    "action": "create",
                    "baseSlideId": template_slide_id,  # Template để duplicate
                    "updates": updates,
                    "template_source": template_slide_id
                }
            else:
                logger.warning(f"❌ No elements mapped for slide copy {new_slide_id}")
                return None

        except Exception as e:
            logger.error(f"Error creating slide copy from template: {e}")
            return None

    async def _handle_max_length_content(
        self,
        content: str,
        max_length: int,
        placeholder_type: str
    ) -> str:
        """
        Xử lý content vượt quá max_length bằng cách gọi LLM để viết lại

        Args:
            content: Nội dung gốc
            max_length: Giới hạn độ dài
            placeholder_type: Loại placeholder

        Returns:
            str: Nội dung đã xử lý (có thể đã được viết lại)
        """
        try:
            if len(content) <= max_length:
                return content

            logger.info(f"⚠️ Content exceeds max_length ({len(content)} > {max_length}) for {placeholder_type}")
            logger.info("🤖 Requesting LLM to rewrite content with 3 retry attempts...")

            # Thử 3 lần với LLM
            max_retries = 3
            for attempt in range(max_retries):
                logger.info(f"🔄 LLM rewrite attempt {attempt + 1}/{max_retries}")

                # Tạo prompt để LLM viết lại content với độ nghiêm ngặt tăng dần
                strictness_levels = [
                    "súc tích nhưng đầy đủ thông tin",
                    "rất súc tích, chỉ giữ thông tin cốt lõi",
                    "cực kỳ súc tích, chỉ giữ thông tin thiết yếu nhất"
                ]

                rewrite_prompt = f"""
Bạn cần viết lại nội dung sau để phù hợp với giới hạn độ dài NGHIÊM NGẶT, {strictness_levels[attempt]}.

NỘI DUNG GỐC:
{content}

YÊU CẦU NGHIÊM NGẶT:
- Độ dài tối đa: {max_length} ký tự (BẮT BUỘC)
- Lần thử {attempt + 1}/3: {strictness_levels[attempt]}
- Phù hợp với loại placeholder: {placeholder_type}
- Ngôn ngữ rõ ràng, súc tích
- Ký hiệu khoa học chính xác nếu có
- TUYỆT ĐỐI KHÔNG VƯỢT QUÁ {max_length} KÝ TỰ

CHỈ TRẢ VỀ NỘI DUNG ĐÃ VIẾT LẠI, KHÔNG CÓ GIẢI THÍCH THÊM.
"""

                # Gọi LLM để viết lại
                llm_result = await self.llm_service.generate_content(
                    prompt=rewrite_prompt,
                    temperature=0.1,
                    max_tokens=2000
                )

                if llm_result["success"] and llm_result.get("text"):
                    rewritten_content = llm_result["text"].strip()

                    # Kiểm tra độ dài sau khi viết lại
                    if len(rewritten_content) <= max_length:
                        logger.info(f"✅ Content rewritten successfully on attempt {attempt + 1}: {len(rewritten_content)} chars")
                        return rewritten_content
                    else:
                        logger.warning(f"⚠️ Attempt {attempt + 1} failed: rewritten content still exceeds max_length ({len(rewritten_content)} > {max_length})")
                        if attempt == max_retries - 1:
                            logger.error(f"❌ All {max_retries} attempts failed - LLM cannot reduce content to {max_length} chars")
                            logger.error("❌ NO FALLBACK - Slide will be skipped")
                            raise Exception(f"LLM failed after {max_retries} attempts: content still exceeds max_length")
                else:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed: LLM error - {llm_result.get('error', 'Unknown error')}")
                    if attempt == max_retries - 1:
                        logger.error(f"❌ All {max_retries} attempts failed - LLM errors")
                        logger.error("❌ NO FALLBACK - Slide will be skipped")
                        raise Exception(f"LLM failed after {max_retries} attempts: {llm_result.get('error', 'Unknown error')}")

                # Wait before retry (except for last attempt)
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"❌ Error handling max_length content: {e}")
            logger.error("❌ NO FALLBACK - Slide will be skipped")
            raise Exception(f"Content length handling failed: {e}")





    def _filter_used_slides(self, mapped_slides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Lọc và chỉ giữ slides được sử dụng

        Args:
            mapped_slides: Slides đã map content từ AI

        Returns:
            List slides đã lọc (chỉ giữ slides được sử dụng)
        """
        try:
            used_slide_ids = set()
            final_slides = []

            logger.info(f"🧹 Filtering {len(mapped_slides)} mapped slides...")

            # Convert mapped slides to final format (compatible with Google Slides API)
            for slide in mapped_slides:
                slide_id = slide.get("slideId")
                elements = slide.get("elements", [])
                action = slide.get("action", "update")  # Default to update for backward compatibility
                base_slide_id = slide.get("baseSlideId")
                updates = slide.get("updates", {})

                if slide_id and elements:
                    used_slide_ids.add(slide_id)

                    # Nếu không có updates sẵn, tạo từ elements
                    if not updates:
                        for element in elements:
                            object_id = element.get("objectId")
                            text = element.get("text")
                            if object_id and text is not None:
                                updates[object_id] = text

                    if updates:
                        final_slide = {
                            "slideId": slide_id,
                            "action": action,
                            "updates": updates
                        }

                        # Thêm baseSlideId nếu là slide được tạo mới
                        if action == "create" and base_slide_id:
                            final_slide["baseSlideId"] = base_slide_id
                            logger.info(f"✅ Prepared slide for creation: {slide_id} (from template: {base_slide_id})")
                        else:
                            logger.info(f"✅ Prepared slide for update: {slide_id}")

                        final_slides.append(final_slide)
                    else:
                        logger.warning(f"❌ No updates found for slide: {slide_id}")
                else:
                    logger.warning(f"❌ Invalid slide data: slideId={slide_id}, elements_count={len(elements)}")

            logger.info(f"🧹 Filtering completed:")
            logger.info(f"   - Used slide IDs: {list(used_slide_ids)}")
            logger.info(f"   - Final slides count: {len(final_slides)}")

            # Log action breakdown
            action_counts = {}
            for slide in final_slides:
                action = slide.get("action", "update")
                action_counts[action] = action_counts.get(action, 0) + 1

            logger.info(f"   - Action breakdown: {action_counts}")

            return final_slides

        except Exception as e:
            logger.error(f"Error filtering used slides: {e}")
            return mapped_slides  # Return original as fallback




# Hàm để lấy singleton instance
def get_slide_generation_service() -> SlideGenerationService:
    """
    Lấy singleton instance của SlideGenerationService
    Thread-safe lazy initialization

    Returns:
        SlideGenerationService: Singleton instance
    """
    return SlideGenerationService()
