"""
JSON Template Processing Service
Xử lý slide generation với JSON template từ frontend thay vì Google Slides
"""

import logging
import re
import copy
from typing import Dict, List, Any, Optional
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
        config_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Xử lý JSON template với nội dung bài học
        
        Args:
            lesson_id: ID của bài học
            template_json: JSON template từ frontend
            config_prompt: Prompt cấu hình tùy chỉnh
            
        Returns:
            Dict chứa template đã được xử lý
        """
        try:
            logger.info(f"🔄 Processing JSON template for lesson: {lesson_id}")
            logger.info(f"🔍 Template JSON type: {type(template_json)}")
            logger.info(f"🔍 Config prompt: {config_prompt}")

            # Bước 1: Lấy nội dung bài học
            lesson_content = await self._get_lesson_content(lesson_id)
            logger.info(f"🔍 Lesson content result type: {type(lesson_content)}")
            logger.info(f"🔍 Lesson content keys: {list(lesson_content.keys()) if isinstance(lesson_content, dict) else 'Not a dict'}")

            if not lesson_content.get("success", False):
                error_msg = lesson_content.get("error", "Unknown error in lesson content")
                raise Exception(error_msg)

            # Bước 2: Phân tích template và detect placeholders
            try:
                analyzed_template = self._analyze_json_template(template_json)
                logger.info(f"📊 Analyzed template: {len(analyzed_template['slides'])} slides")
            except Exception as e:
                raise Exception(f"Failed to analyze template: {str(e)}")

            # Bước 3: Sinh nội dung với LLM
            presentation_content = await self._generate_presentation_content(
                lesson_content.get("content", ""),
                config_prompt
            )
            logger.info(f"🔍 Presentation content result type: {type(presentation_content)}")
            logger.info(f"🔍 Presentation content keys: {list(presentation_content.keys()) if isinstance(presentation_content, dict) else 'Not a dict'}")

            if not presentation_content.get("success", False):
                error_msg = presentation_content.get("error", "Unknown error in presentation content")
                raise Exception(error_msg)

            # Bước 4: Map nội dung vào template
            try:
                processed_template = await self._map_content_to_json_template(
                    presentation_content.get("content", ""),
                    template_json,
                    analyzed_template
                )
            except Exception as e:
                raise Exception(f"Failed to map content to template: {str(e)}")

            # Trả về kết quả với success flag
            return {
                "success": True,
                "lesson_id": lesson_id,
                "processed_template": processed_template,
                "slides_created": len(processed_template.get("slides", []))
            }

        except Exception as e:
            logger.error(f"❌ Error processing JSON template: {e}")
            # Trả về lỗi với success flag
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
    
    async def _get_lesson_content(self, lesson_id: str) -> Dict[str, Any]:
        """Lấy nội dung bài học từ TextbookRetrievalService"""
        try:
            logger.info(f"📚 Getting lesson content for: {lesson_id}")

            # Sử dụng TextbookRetrievalService để lấy lesson content
            lesson_result = await self.textbook_service.get_lesson_content(lesson_id)

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
    
    def _analyze_json_template(self, template_json: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích JSON template và detect placeholders (theo logic cũ)"""
        try:
            logger.info("🔍 Analyzing JSON template structure...")
            logger.info(f"🔍 Template JSON type: {type(template_json)}")
            logger.info(f"🔍 Template JSON keys: {list(template_json.keys()) if isinstance(template_json, dict) else 'Not a dict'}")

            slides = template_json.get("slides", [])
            analyzed_slides = []

            # Placeholder patterns để detect
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

            for slide in slides:
                analyzed_elements = []
                placeholder_counts = {}

                # Phân tích elements
                for element in slide.get("elements", []):
                    text = element.get("text", "").strip()

                    # Detect placeholder type từ text
                    placeholder_result = self._detect_placeholder_type_from_text(text, placeholder_patterns)

                    if placeholder_result:  # Chỉ xử lý nếu detect được placeholder
                        placeholder_type, max_length = placeholder_result

                        logger.info(f"✅ Found placeholder: {placeholder_type} <{max_length}>")

                        # Đếm số lượng placeholder types
                        placeholder_counts[placeholder_type] = placeholder_counts.get(placeholder_type, 0) + 1

                        # Tạo analyzed element với thông tin đầy đủ
                        analyzed_element = {
                            "objectId": element.get("id"),
                            "text": None,  # LLM sẽ insert nội dung sau
                            "Type": placeholder_type,
                            "max_length": max_length,
                            "original_element": element  # Giữ thông tin gốc để mapping
                        }

                        analyzed_elements.append(analyzed_element)
                    else:
                        # Bỏ qua text không phải placeholder format
                        logger.info(f"❌ Skipping non-placeholder text: '{text}'")
                        continue

                # Tạo description cho slide dựa trên placeholder counts (như luồng cũ)
                description = self._generate_slide_description(placeholder_counts)

                analyzed_slide = {
                    "slideId": slide.get("id"),
                    "description": description,
                    "elements": analyzed_elements,
                    "placeholder_counts": placeholder_counts,  # For logic selection
                    "original_slide": slide  # Giữ thông tin gốc
                }

                analyzed_slides.append(analyzed_slide)

            result = {
                "slides": analyzed_slides,
                "total_slides": len(analyzed_slides),
                "slideFormat": template_json.get("slideFormat", "16:9"),
                "version": template_json.get("version", "1.0")
            }

            logger.info(f"✅ Template analysis complete: {len(analyzed_slides)} slides analyzed")
            return result

        except Exception as e:
            logger.error(f"❌ Error analyzing JSON template: {e}")
            raise
    
    async def _generate_presentation_content(
        self,
        lesson_content: str,
        config_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Sinh nội dung presentation với LLM"""
        try:
            logger.info("🤖 Generating presentation content with LLM...")

            # Tạo prompt cho LLM
            prompt = self._create_llm_prompt(lesson_content, config_prompt)
            
            # Gọi LLM
            llm_response = await self.llm_service.generate_content(
                prompt=prompt,
                max_tokens=60000,
                temperature=0.1
            )
            
            if not llm_response.get("success", False):
                return {
                    "success": False,
                    "error": f"LLM generation failed: {llm_response.get('error', 'Unknown error')}"
                }

            content = llm_response.get("text", "")  # LLMService trả về "text" chứ không phải "content"
            logger.info(f"✅ LLM content generated: {len(content)} characters")

            # Debug: Log first 500 chars of LLM content
            logger.info(f"🔍 LLM content preview: {content[:500]}...")

            # Debug: Log full LLM content for debugging
            logger.info(f"🔍 FULL LLM CONTENT DEBUG:")
            logger.info(f"Content length: {len(content)} characters")
            logger.info(f"Content: {content}")


            # Debug: Check for annotation patterns
            annotation_pattern = r'#\*\([^)]+\)\*#'
            annotation_matches = re.findall(annotation_pattern, content)
            logger.info(f"🔍 Found {len(annotation_matches)} annotation patterns: {annotation_matches[:10]}")  # First 10

            return {
                "success": True,
                "content": content
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating presentation content: {e}")
            return {
                "success": False,
                "error": f"Failed to generate content: {str(e)}"
            }
    
    def _create_llm_prompt(
        self,
        lesson_content: str,
        config_prompt: Optional[str] = None
    ) -> str:
        """Tạo prompt cho LLM theo format của luồng cũ (chi tiết và chính xác)"""


        # Cải thiện default config để tạo nội dung chi tiết hơn
        default_config = """
Bạn là chuyên gia thiết kế nội dung thuyết trình giáo dục chuyên nghiệp. Nhiệm vụ của bạn là phân tích sâu nội dung bài học và tạo ra bài thuyết trình chi tiết, đầy đủ và hấp dẫn.
NGUYÊN TẮC THIẾT KẾ CHẤT LƯỢNG CAO:
1. PHÂN TÍCH TOÀN DIỆN VÀ SÂU SẮC:
   - Hiểu rõ từng khái niệm, định nghĩa, công thức trong bài học
   - Xác định mối liên hệ giữa các khái niệm
   - Phân tích ví dụ minh họa và ứng dụng thực tế
   - Tìm ra các điểm quan trọng cần nhấn mạnh
2. CẤU TRÚC LOGIC VÀ KHOA HỌC:
   - Từ khái niệm cơ bản đến nâng cao
   - Từ lý thuyết đến ứng dụng thực tế
   - Mỗi slide có mục đích rõ ràng trong chuỗi kiến thức
   - Đảm bảo tính liên kết giữa các slide
3. NỘI DUNG PHONG PHÚ VÀ CHI TIẾT:
   - Tạo ít nhất 10-12 slides với nội dung đầy đủ và sâu sắc
   - Mỗi khái niệm được giải thích rõ ràng với ví dụ cụ thể
   - Bổ sung thông tin mở rộng, ứng dụng thực tế
   - Không bỏ sót bất kỳ thông tin quan trọng nào
4. NGÔN NGỮ KHOA HỌC CHÍNH XÁC:
   - Sử dụng thuật ngữ khoa học chính xác
   - Ký hiệu hóa học, công thức toán học đúng chuẩn Unicode
   - Giải thích thuật ngữ khó hiểu
   - Ngôn ngữ rõ ràng, dễ hiểu nhưng vẫn chuyên nghiệp
5. VÍ DỤ VÀ MINH HỌA PHONG PHÚ:
   - Mỗi khái niệm có thể thêm 1 ví dụ nếu cần thiết
   - Ví dụ từ đơn giản đến phức tạp
   - Liên hệ với thực tế, đời sống
   - Bài tập minh họa có lời giải chi tiết
YÊU CẦU ANNOTATION CHÍNH XÁC:
- PHẢI có annotation bằng #*(PlaceholderType)*# ngay sau mỗi nội dung
- Placeholder types: LessonName, LessonDescription, CreatedDate, TitleName, TitleContent, SubtitleName, SubtitleContent, ImageName, ImageContent
- Annotation phải chính xác 100% và nhất quán
- BẮT BUỘC có slide summaries với số lượng rõ ràng để chọn template phù hợp
"""

        

        prompt = f"""
{default_config}
CÁC LƯU Ý NGƯỜI TẠO (NẾU CÓ):
{config_prompt}
NỘI DUNG BÀI HỌC:
{lesson_content}
📚 HƯỚNG DẪN TẠO PRESENTATION CONTENT CHI TIẾT:

1. PHÂN TÍCH BÀI HỌC SÂU SẮC:
   - Đọc kỹ và hiểu rõ từng đoạn văn, khái niệm trong bài học
   - Xác định chủ đề chính và tất cả các chủ đề phụ
   - Phân loại thông tin: định nghĩa, công thức, ví dụ, ứng dụng
   - Tìm ra mối liên hệ logic giữa các khái niệm
   - Xác định độ khó và thứ tự trình bày hợp lý
   - TUYỆT ĐỐI KHÔNG được bỏ sót bất kỳ thông tin quan trọng nào

2. TẠO NỘI DUNG VỚI ANNOTATION CHÍNH XÁC:
   - PHẢI có annotation #*(PlaceholderType)*# ngay sau mỗi nội dung
   - Ví dụ: "Nguyên tố hóa học và cấu trúc nguyên tử #*(LessonName)*#"
   - Ví dụ: "Bài học này giúp học sinh hiểu rõ về cấu trúc nguyên tử, các hạt cơ bản và tính chất của nguyên tố hóa học #*(LessonDescription)*#"
   - Ví dụ: "Ngày thuyết trình: 18-07-2025 #*(CreatedDate)*#"
3. HIỂU RÕ CẤU TRÚC PHÂN CẤP VÀ NHÓM NỘI DUNG CHI TIẾT:
   📌 TitleName: Tiêu đề chính của slide (ngắn gọn, súc tích)
      - Chỉ là tên chủ đề, không phải nội dung giải thích
      - Ví dụ: "Cấu trúc nguyên tử", "Liên kết hóa học", "Phản ứng oxi hóa khử"
   📝 TitleContent: Nội dung giải thích chi tiết cho TitleName
      - Giải thích đầy đủ khái niệm, định nghĩa
      - Bao gồm ví dụ minh họa cụ thể
      - Có thể có nhiều đoạn văn nhưng gộp thành một khối
   🔸 SubtitleName: Tiêu đề các mục con trong chủ đề chính
      - Các khía cạnh nhỏ hơn của chủ đề chính
      - Ví dụ: "Proton", "Neutron", "Electron" (trong chủ đề Cấu trúc nguyên tử)
   📄 SubtitleContent: Nội dung chi tiết cho từng SubtitleName
      - Giải thích cụ thể cho từng mục con
      - Có ví dụ, công thức, ứng dụng
      - QUAN TRỌNG: Mỗi SubtitleContent tương ứng với ĐÚNG MỘT SubtitleName (1:1 mapping)
      - TUYỆT ĐỐI KHÔNG tạo nhiều SubtitleContent riêng biệt cho cùng 1 SubtitleName
      - Gộp tất cả nội dung của 1 mục con thành 1 khối SubtitleContent duy nhất
4. VÍ DỤ CHI TIẾT VỚI CẤU TRÚC PHÂN CẤP RÕ RÀNG VÀ NỘI DUNG PHONG PHÚ:
SLIDE 1 - GIỚI THIỆU TỔNG QUAN:
Nguyên tố hóa học và bảng tuần hoàn #*(LessonName)*#
Bài học này giúp học sinh hiểu rõ về khái niệm nguyên tố hóa học, cấu trúc bảng tuần hoàn và mối liên hệ giữa vị trí của nguyên tố với tính chất hóa học. Học sinh sẽ nắm được cách phân loại nguyên tố và dự đoán tính chất dựa vào vị trí trong bảng. #*(LessonDescription)*#
Ngày thuyết trình: 18-07-2025 #*(CreatedDate)*#
=== SLIDE 1 SUMMARY ===
Placeholders: 1xLessonName, 1xLessonDescription, 1xCreatedDate
===========================
SLIDE 2 - KHÁI NIỆM CƠ BẢN VỚI NỘI DUNG CHI TIẾT:
Khái niệm nguyên tố hóa học #*(TitleName)*#
Nguyên tố hóa học là tập hợp các nguyên tử có cùng số proton trong hạt nhân. Mỗi nguyên tố được xác định bởi số hiệu nguyên tử (Z) - chính là số proton trong hạt nhân. Hiện nay, có 118 nguyên tố đã được phát hiện, trong đó 94 nguyên tố tồn tại trong tự nhiên, còn lại là nguyên tố nhân tạo. Mỗi nguyên tố có ký hiệu hóa học riêng, thường là 1-2 chữ cái, ví dụ: H (hydro), He (heli), Li (lithi), Na (natri). Các nguyên tố cùng nhóm trong bảng tuần hoàn thường có tính chất hóa học tương tự nhau do có cấu hình electron hóa trị giống nhau. #*(TitleContent)*#
=== SLIDE 2 SUMMARY ===
Placeholders: 1xTitleName, 1xTitleContent
===========================
SLIDE 3 - CẤU TRÚC NGUYÊN TỬ CHI TIẾT:
Cấu trúc nguyên tử #*(TitleName)*#
Nguyên tử là đơn vị cấu tạo cơ bản của vật chất, gồm hạt nhân mang điện tích dương ở trung tâm và các electron mang điện tích âm chuyển động xung quanh. Hạt nhân chiếm phần lớn khối lượng nguyên tử nhưng thể tích rất nhỏ (khoảng 10^-14 m) so với kích thước nguyên tử (khoảng 10^-10 m). #*(TitleContent)*#
Hạt nhân nguyên tử #*(SubtitleName)*#
Hạt nhân nguyên tử được cấu tạo từ proton và neutron (gọi chung là nucleon). Proton mang điện tích dương (+1), có khối lượng khoảng 1,673 × 10^-27 kg. Neutron không mang điện, có khối lượng xấp xỉ proton. Lực hạt nhân mạnh giữ các nucleon lại với nhau, vượt qua lực đẩy tĩnh điện giữa các proton. Số proton trong hạt nhân xác định nguyên tố hóa học, còn số neutron có thể thay đổi tạo thành các đồng vị. #*(SubtitleContent)*#
Electron và đám mây electron #*(SubtitleName)*#
Electron là hạt mang điện tích âm (-1), có khối lượng rất nhỏ (khoảng 9,109 × 10^-31 kg), chỉ bằng 1/1836 khối lượng proton. Electron chuyển động xung quanh hạt nhân trong các orbital (đám mây electron) với xác suất xuất hiện khác nhau. Các orbital được sắp xếp thành các lớp (K, L, M, N...) và các phân lớp (s, p, d, f). Electron phân bố theo nguyên lý Pauli, quy tắc Hund và nguyên lý Aufbau. Cấu hình electron quyết định tính chất hóa học của nguyên tố. #*(SubtitleContent)*#
=== SLIDE 3 SUMMARY ===
Placeholders: 1xTitleName, 1xTitleContent, 2xSubtitleName, 2xSubtitleContent
===========================
SLIDE 4 - BẢNG TUẦN HOÀN VÀ XU HƯỚNG:
Bảng tuần hoàn các nguyên tố hóa học #*(TitleName)*#
Bảng tuần hoàn hiện đại #*(SubtitleName)*#
Bảng tuần hoàn hiện đại gồm 7 chu kỳ (hàng ngang) và 18 nhóm (cột dọc). Các nguyên tố được sắp xếp theo thứ tự tăng dần của số hiệu nguyên tử. Chu kỳ tương ứng với số lớp electron, nhóm tương ứng với số electron hóa trị. Bảng được chia thành các khối: s, p, d, f tương ứng với phân lớp electron ngoài cùng đang được điền. Các nguyên tố trong cùng nhóm có tính chất hóa học tương tự do có cùng cấu hình electron hóa trị. #*(SubtitleContent)*#
Xu hướng tính chất trong bảng tuần hoàn #*(SubtitleName)*#
Tính kim loại giảm dần từ trái sang phải trong chu kỳ và tăng dần từ trên xuống dưới trong nhóm. Bán kính nguyên tử giảm dần từ trái sang phải trong chu kỳ và tăng dần từ trên xuống dưới trong nhóm. Năng lượng ion hóa tăng dần từ trái sang phải trong chu kỳ và giảm dần từ trên xuống dưới trong nhóm. Độ âm điện tăng dần từ trái sang phải trong chu kỳ và giảm dần từ trên xuống dưới trong nhóm. Các xu hướng này giúp dự đoán tính chất và phản ứng hóa học của các nguyên tố. #*(SubtitleContent)*#
=== SLIDE 4 SUMMARY ===
Placeholders: 1xTitleName, 2xSubtitleName, 2xSubtitleContent
===========================
... (tiếp tục với các slide khác tùy theo nội dung bài học)
5. QUY TẮC ANNOTATION VÀ NHÓM NỘI DUNG - CỰC KỲ QUAN TRỌNG:

🚨 QUY TẮC NHÓM NỘI DUNG BẮT BUỘC - CỰC KỲ QUAN TRỌNG:
- TUYỆT ĐỐI KHÔNG tạo nhiều TitleContent riêng biệt trong 1 TitleName
- TUYỆT ĐỐI KHÔNG tạo nhiều SubtitleContent riêng biệt cho cùng 1 SubtitleName
- MỖI 1xTitleName CHỈ CÓ TỐI ĐA 1 TitleContent duy nhất (gộp tất cả nội dung lại)
- MỖI SubtitleName CHỈ CÓ ĐÚNG 1 SubtitleContent tương ứng (1:1 mapping)

🔥 VÍ DỤ SAI VỚI TITLECONTENT (TUYỆT ĐỐI KHÔNG LÀM):
Cấu trúc nguyên tử #*(TitleName)*#
Nguyên tử gồm hạt nhân và electron. #*(TitleContent)*#
Hạt nhân ở trung tâm. #*(TitleContent)*#  ❌ SAI - Có 2 TitleContent riêng biệt
Electron chuyển động xung quanh. #*(TitleContent)*#  ❌ SAI - Có 3 TitleContent riêng biệt

🔥 VÍ DỤ SAI VỚI SUBTITLECONTENT (TUYỆT ĐỐI KHÔNG LÀM):
Bài toán tính toán #*(SubtitleName)*#
Gọi x là phần trăm số nguyên tử của ⁶³Cu. #*(SubtitleContent)*#
Ta có hệ phương trình: x + y = 100. #*(SubtitleContent)*#  ❌ SAI - Có 2 SubtitleContent cho 1 SubtitleName
Từ (1), ta có y = 100 - x. #*(SubtitleContent)*#  ❌ SAI - Có 3 SubtitleContent cho 1 SubtitleName

✅ VÍ DỤ ĐÚNG VỚI TITLECONTENT (BẮT BUỘC LÀM THEO):
Cấu trúc nguyên tử #*(TitleName)*#
Nguyên tử gồm hạt nhân và electron. Hạt nhân ở trung tâm, chứa proton và neutron. Electron chuyển động xung quanh hạt nhân trong các orbital. Lực tĩnh điện giữ electron gần hạt nhân. #*(TitleContent)*#  ✅ ĐÚNG - Chỉ 1 TitleContent duy nhất

✅ VÍ DỤ ĐÚNG VỚI SUBTITLECONTENT (BẮT BUỘC LÀM THEO):
Bài toán tính toán #*(SubtitleName)*#
Gọi x là phần trăm số nguyên tử của ⁶³Cu và y là phần trăm số nguyên tử của ⁶⁵Cu. Ta có hệ phương trình: x + y = 100 (Tổng phần trăm là 100%) và (63x + 65y) / 100 = 63,54 (Công thức nguyên tử khối trung bình). Từ (1), ta có y = 100 - x. Thay vào (2): (63x + 65(100 - x)) / 100 = 63,54. Giải phương trình: 63x + 6500 - 65x = 6354, -2x = -146, x = 73. Vậy phần trăm số nguyên tử của ⁶³Cu là 73% và ⁶⁵Cu là 27%. #*(SubtitleContent)*#  ✅ ĐÚNG - Chỉ 1 SubtitleContent cho 1 SubtitleName
6. SLIDE SUMMARIES - ĐẾMCHÍNH XÁC:
   Cuối mỗi slide, thêm slide summary với SỐ LƯỢNG CHÍNH XÁC:
   === SLIDE [Số] SUMMARY ===
   Placeholders: [Số lượng]x[PlaceholderType], [Số lượng]x[PlaceholderType], ...

🚨 LƯU Ý QUAN TRỌNG KHI ĐẾM - QUY TẮC 1:1 MAPPING:
- TitleContent: LUÔN LUÔN chỉ có 1 cho mỗi TitleName (1 TitleName = 1 TitleContent)
- SubtitleContent: LUÔN LUÔN bằng số lượng SubtitleName (1 SubtitleName = 1 SubtitleContent)
- Ví dụ đúng: 1xTitleName, 1xTitleContent, 2xSubtitleName, 2xSubtitleContent
- Ví dụ sai: 1xTitleName, 3xTitleContent ❌ (không bao giờ có nhiều TitleContent)
- Ví dụ sai: 1xSubtitleName, 5xSubtitleContent ❌ (không bao giờ có nhiều SubtitleContent cho 1 SubtitleName)
   ===========================
7. YÊU CẦU OUTPUT CHẤT LƯỢNG CAO:
- Tạo nội dung thuyết trình TEXT THUẦN TÚY với annotation chính xác 100%
- Nội dung chi tiết, đầy đủ, không bỏ sót thông tin quan trọng
- Sử dụng ngôn ngữ khoa học chính xác, dễ hiểu
- Có ví dụ minh họa cụ thể cho mỗi khái niệm
- BẮT BUỘC có slide summaries chi tiết để chọn template phù hợp
- Không tạo ra bảng, sơ đồ - chỉ sử dụng text mô tả
- Đảm bảo tính logic và liên kết giữa các slide
🔍 VÍ DỤ MINH HỌA CẤU TRÚC ĐÚNG VỚI NHÓM NỘI DUNG:

SLIDE 1: (Slide này là bắt buộc và luôn có)
Cấu hình electron #*(LessonName)*#
Bài này cho chúng ta biết được cấu hình electron trong nguyên tử và phân tử #*(LessonDescription)*#
Ngày thuyết trình: 18-07-2025 #*(CreatedDate)*#
=== SLIDE 1 SUMMARY ===
Placeholders: 1xLessonName, 1xLessonDescription, 1xCreatedDate
===========================

SLIDE 2: (Slide đơn giản với 1 TitleName và 1 TitleContent)
Khái niệm cấu hình electron #*(TitleName)*#
Cấu hình electron là cách sắp xếp các electron trong các orbital của nguyên tử. Cấu hình này quyết định tính chất hóa học của nguyên tố và khả năng tạo liên kết. Việc hiểu rõ cấu hình electron giúp dự đoán tính chất và hành vi của các nguyên tố trong phản ứng hóa học. Mỗi orbital có mức năng lượng và hình dạng khác nhau. Các electron sẽ lấp đầy các orbital theo thứ tự năng lượng tăng dần. #*(TitleContent)*#
=== SLIDE 2 SUMMARY ===
Placeholders: 1xTitleName, 1xTitleContent
===========================

SLIDE 3: (Slide với TitleName, TitleContent và các SubtitleName, SubtitleContent)
Các quy tắc sắp xếp electron #*(TitleName)*#
Các electron trong nguyên tử tuân theo một số quy tắc nhất định khi sắp xếp vào các orbital. Việc hiểu rõ các quy tắc này giúp chúng ta xác định cấu hình electron chính xác và dự đoán tính chất hóa học của nguyên tố. #*(TitleContent)*#
Quy tắc Aufbau #*(SubtitleName)*#
Electron điền vào orbital có mức năng lượng thấp trước, sau đó mới điền vào orbital có mức năng lượng cao hơn theo quy tắc Aufbau. Thứ tự năng lượng tăng dần của các orbital là: 1s < 2s < 2p < 3s < 3p < 4s < 3d < 4p < 5s < 4d < 5p < 6s < 4f < 5d < 6p < 7s < 5f. #*(SubtitleContent)*#
Nguyên lý Pauli #*(SubtitleName)*#
Mỗi orbital chứa tối đa 2 electron và chúng phải có spin ngược chiều nhau theo nguyên lý Pauli. Điều này có nghĩa là không có hai electron trong một nguyên tử có thể có cả bốn số lượng tử giống nhau. Nguyên lý này giải thích tại sao các electron không thể tập trung hết vào orbital năng lượng thấp nhất. #*(SubtitleContent)*#
=== SLIDE 3 SUMMARY ===
Placeholders: 1xTitleName, 1xTitleContent, 2xSubtitleName, 2xSubtitleContent
===========================

SLIDE 4: (Slide với ImageName và ImageContent)
Hình ảnh minh họa: Sơ đồ cấu hình electron #*(ImageName)*#
Sơ đồ thể hiện cách electron được sắp xếp trong các orbital 1s, 2s, 2p theo thứ tự năng lượng tăng dần. Các mũi tên hướng lên và xuống biểu thị electron với spin khác nhau. Mỗi ô vuông đại diện cho một orbital. Các orbital cùng phân lớp có cùng mức năng lượng. #*(ImageContent)*#
=== SLIDE 4 SUMMARY ===
Placeholders: 1xImageName, 1xImageContent
===========================

SLIDE 5: (Slide phức tạp với nhiều SubtitleName và SubtitleContent)
Ứng dụng cấu hình electron #*(TitleName)*#
Cấu hình electron có nhiều ứng dụng quan trọng trong hóa học, vật lý và khoa học vật liệu. Hiểu rõ cấu hình electron giúp chúng ta giải thích và dự đoán nhiều hiện tượng trong tự nhiên. #*(TitleContent)*#
Dự đoán tính chất hóa học #*(SubtitleName)*#
Cấu hình electron của lớp ngoài cùng (electron hóa trị) quyết định tính chất hóa học của nguyên tố. Các nguyên tố có cấu hình electron hóa trị giống nhau thường có tính chất hóa học tương tự. Ví dụ: Na và K đều có 1 electron ở lớp ngoài cùng nên đều là kim loại kiềm có tính khử mạnh. #*(SubtitleContent)*#
Giải thích liên kết hóa học #*(SubtitleName)*#
Cấu hình electron giúp giải thích cách các nguyên tử liên kết với nhau. Nguyên tử có xu hướng đạt được cấu hình electron bền vững (8 electron ở lớp ngoài cùng) thông qua việc nhận, cho hoặc chia sẻ electron, tạo thành liên kết ion hoặc liên kết cộng hóa trị. #*(SubtitleContent)*#
Phát triển vật liệu mới #*(SubtitleName)*#
Hiểu biết về cấu hình electron giúp các nhà khoa học thiết kế và phát triển vật liệu mới với tính chất đặc biệt như chất bán dẫn, siêu dẫn, vật liệu từ tính và vật liệu quang học. #*(SubtitleContent)*#
=== SLIDE 5 SUMMARY ===
Placeholders: 1xTitleName, 1xTitleContent, 3xSubtitleName, 3xSubtitleContent
===========================
8. QUY TẮC VIẾT CHI TIẾT VÀ CHÍNH XÁC:

* ANNOTATION BẮT BUỘC:
- LUÔN có annotation #*(PlaceholderType)*# ngay sau mỗi nội dung
- Không được thiếu hoặc sai annotation
- Kiểm tra kỹ trước khi hoàn thành

* NỘI DUNG CHẤT LƯỢNG:
- Nội dung đầy đủ, chi tiết, không bỏ sót kiến thức nào
- Mỗi khái niệm có định nghĩa rõ ràng và ví dụ minh họa
- Giải thích từ cơ bản đến nâng cao
- Liên hệ với thực tế và ứng dụng

* CẤU TRÚC PHÂN CẤP RÕ RÀNG VÀ QUY TẮC 1:1 MAPPING:
- TitleName: CHỈ là tiêu đề chính
- TitleContent: Nội dung giải thích chi tiết (CHỈ 1 khối cho mỗi TitleName)
- SubtitleName: Tiêu đề mục con
- SubtitleContent: Nội dung chi tiết mục con (CHỈ 1 khối cho mỗi SubtitleName)

* SLIDE SUMMARIES CHÍNH XÁC:
- Đếm chính xác số lượng từng placeholder type
- Format: === SLIDE [Số] SUMMARY ===
- Ví dụ: Placeholders: 1xTitleName, 2xSubtitleName, 2xSubtitleContent
  * TitleContent: TẤT CẢ nội dung giải thích của mục lớn được gộp chung thành 1 khối
  * SubtitleName: CHỈ là tiêu đề mục nhỏ bên trong mục lớn
  * SubtitleContent: TẤT CẢ nội dung giải thích của từng mục nhỏ được gộp chung thành 1 khối
- Ký hiệu khoa học chính xác: H₂O, CO₂, x², √x, π, α, β
- Sử dụng ngày hiện tại cho CreatedDate

🔥 NHẮC NHỞ CUỐI CÙNG - QUY TẮC QUAN TRỌNG NHẤT:
*Không tạo ra bảng, sơ đồ - chỉ sử dụng text mô tả
*TUYỆT ĐỐI TUÂN THỦ QUY TẮC 1:1 MAPPING:
- Mỗi SubtitleName chỉ có ĐÚNG 1 SubtitleContent tương ứng
- Nếu có nhiều câu/đoạn văn cho 1 mục con, hãy gộp tất cả thành 1 SubtitleContent duy nhất
- Ví dụ: Thay vì tạo 5 SubtitleContent riêng biệt, hãy gộp thành 1 SubtitleContent dài
- Điều này đảm bảo template matching chính xác và tránh lỗi mapping
"""

        return prompt

    def _detect_placeholder_type_from_text(self, text: str, placeholder_patterns: Dict[str, str]) -> Optional[tuple]:
        """
        Detect placeholder type và max_length từ text format "PlaceholderName max_length"

        Args:
            text: Text từ element
            placeholder_patterns: Dictionary của patterns

        Returns:
            tuple: (placeholder_type, max_length) hoặc None nếu không detect được
        """
        try:
            for placeholder_type, pattern in placeholder_patterns.items():
                match = re.search(pattern, text)
                if match:
                    max_length = int(match.group(1))
                    return placeholder_type, max_length

            return None

        except Exception as e:
            logger.warning(f"Error detecting placeholder type: {e}")
            return None

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

    async def _map_content_to_json_template(
        self,
        llm_content: str,
        original_template: Dict[str, Any],
        analyzed_template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map nội dung LLM vào JSON template theo logic của luồng cũ với intelligent slide selection"""
        try:
            logger.info("🔧 Mapping LLM content to JSON template with intelligent slide selection...")

            # Parse LLM content với slide summaries
            parsed_data = self._parse_llm_content(llm_content)
            slide_summaries = parsed_data.get("_slide_summaries", [])

            if not slide_summaries:
                logger.error("❌ No slide summaries found in LLM content")
                raise ValueError("No slide summaries found - cannot perform intelligent slide selection")

            # Create processed template copy
            processed_template = {
                "version": original_template.get("version", "1.0"),
                "createdAt": datetime.now().isoformat(),
                "slideFormat": original_template.get("slideFormat", "16:9"),
                "slides": []
            }

            # Content index để track việc sử dụng content (như luồng cũ)
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
            template_slides = analyzed_template.get("slides", [])

            logger.info(f"� Processing {len(slide_summaries)} slide summaries with intelligent matching...")

            # Process từng slide summary với intelligent template selection
            for i, summary in enumerate(slide_summaries):
                slide_num = i + 1
                required_placeholders = summary.get("placeholders", [])
                required_counts = summary.get("placeholder_counts", {})

                logger.info(f"🔍 Processing slide {slide_num}:")
                logger.info(f"   Required placeholders: {required_placeholders}")
                logger.info(f"   Required counts: {required_counts}")



                # Tìm template phù hợp CHÍNH XÁC (không fallback)
                # Đầu tiên thử tìm template chưa sử dụng
                best_template = self._find_exact_matching_template(
                    required_placeholders,
                    required_counts,
                    template_slides,
                    used_slide_ids
                )

                # Nếu không tìm thấy template chưa sử dụng, cho phép reuse template
                if not best_template:
                    logger.info(f"🔄 No unused template found, trying to reuse existing template...")
                    best_template = self._find_exact_matching_template_with_reuse(
                        required_placeholders,
                        required_counts,
                        template_slides
                    )

                if best_template:
                    template_id = best_template['slideId']
                    is_reused = template_id in used_slide_ids

                    if is_reused:
                        logger.info(f"✅ Found exact matching template (REUSED): {template_id}")
                    else:
                        logger.info(f"✅ Found exact matching template (NEW): {template_id}")

                    # Tạo processed slide từ template
                    processed_slide = await self._create_processed_slide_from_template(
                        best_template,
                        parsed_data,
                        content_index,
                        slide_num,
                        is_reused
                    )

                    if processed_slide:
                        processed_template["slides"].append(processed_slide)
                        # Chỉ thêm vào used_slide_ids nếu chưa được sử dụng
                        if not is_reused:
                            used_slide_ids.add(template_id)
                        logger.info(f"✅ Successfully processed slide {slide_num} ({'reused' if is_reused else 'new'})")
                    else:
                        logger.error(f"❌ Failed to create processed slide {slide_num} - SKIPPING")
                        # Không fallback - skip slide này
                        continue
                else:
                    logger.error(f"❌ No exact matching template found for slide {slide_num} - SKIPPING")
                    # Không fallback - skip slide này
                    continue

            logger.info(f"✅ Template processing complete: {len(processed_template['slides'])} slides created")
            return processed_template

        except Exception as e:
            logger.error(f"❌ Error mapping content to template: {e}")
            raise

    def _parse_llm_content(self, llm_content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse nội dung từ LLM theo format của luồng cũ với slide summaries"""
        try:
            logger.info("📝 Parsing LLM content with slide summaries...")

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

            # Parse content theo annotation format - LLM sinh theo format: "content #*(PlaceholderType)*#"
            valid_placeholders = '|'.join(parsed_data.keys())

            # Tách content theo từng dòng và match từng dòng
            lines = llm_content.split('\n')
            matches = []

            for line in lines:
                # Pattern để match: "content #*(PlaceholderType)*#" trong một dòng
                pattern = rf'(.+?)\s*#\*\(({valid_placeholders})\)\*#'
                line_matches = re.findall(pattern, line, re.IGNORECASE)
                matches.extend(line_matches)

            logger.info(f"🔍 Found {len(matches)} annotation matches")
            logger.info(f"🔍 Pattern used: {pattern}")
            logger.info(f"🔍 Total lines processed: {len(lines)}")

            # Debug: Log some sample lines to see format
            logger.info(f"🔍 Sample lines with potential annotations:")
            for i, line in enumerate(lines[:20]):  # First 20 lines
                if '#*(' in line and ')*#' in line:
                    logger.info(f"  Line {i+1}: {line}")

            for content, placeholder_type in matches:
                clean_content = content.strip()
                if clean_content:
                    parsed_data[placeholder_type].append({
                        "content": clean_content,
                        "length": len(clean_content)
                    })
                    logger.info(f"✅ Parsed {placeholder_type}: {clean_content}...")
                else:
                    logger.warning(f"❌ Empty content for {placeholder_type}")

            # Debug: Log parsed data summary
            logger.info(f"🔍 PARSED DATA SUMMARY:")
            for placeholder_type, items in parsed_data.items():
                if items:
                    logger.info(f"  {placeholder_type}: {len(items)} items")
                    for i, item in enumerate(items[:3]):  # First 3 items
                        logger.info(f"    [{i+1}] {item['content']}...")
                else:
                    logger.info(f"  {placeholder_type}: 0 items")

            # Parse slide summaries để hiểu cấu trúc (như luồng cũ)
            slide_summaries = []
            summary_pattern = r'=== SLIDE (\d+) SUMMARY ===\s*Placeholders:\s*([^=]+)'
            summary_matches = re.findall(summary_pattern, llm_content, re.IGNORECASE)

            # Debug: Log LLM content và summary matches
            logger.info(f"🔍 LLM content length: {len(llm_content)} characters")
            logger.info(f"🔍 Summary pattern: {summary_pattern}")
            logger.info(f"🔍 Found {len(summary_matches)} summary matches")
            if len(summary_matches) == 0:
                logger.warning("❌ No slide summaries found! LLM content preview:")
                logger.warning(f"First 1000 chars: {llm_content[:1000]}")
                logger.warning(f"Last 1000 chars: {llm_content[-1000:]}")
            else:
                logger.info(f"✅ Summary matches: {summary_matches}")

            for slide_num_str, placeholder_text in summary_matches:
                slide_num = int(slide_num_str)
                placeholders = []
                placeholder_counts = {}

                # Parse placeholder counts từ text như "1xLessonName, 2xTitleContent"
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
                    "placeholder_counts": placeholder_counts
                })

            # Log parsed results
            logger.info(f"📋 Parsed {len(slide_summaries)} slide summaries")
            for placeholder_type, items in parsed_data.items():
                if items:
                    logger.info(f"📋 {placeholder_type}: {len(items)} items")

            # Store slide summaries for mapping logic
            parsed_data["_slide_summaries"] = slide_summaries

            return parsed_data

        except Exception as e:
            logger.error(f"❌ Error parsing LLM content: {e}")
            raise

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
                    max_tokens=5000,
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
                slide_id = slide.get("slideId")

                # Skip used slides
                if slide_id in used_slide_ids:
                    continue

                # Get placeholder types and counts in this slide
                slide_elements = slide.get("elements", [])
                slide_placeholder_counts = {}

                for elem in slide_elements:
                    placeholder_type = elem.get("Type")
                    if placeholder_type:
                        if placeholder_type in slide_placeholder_counts:
                            slide_placeholder_counts[placeholder_type] += 1
                        else:
                            slide_placeholder_counts[placeholder_type] = 1

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
                slide_id = slide.get("slideId")

                # Get placeholder types and counts in this slide
                slide_elements = slide.get("elements", [])
                slide_placeholder_counts = {}

                for elem in slide_elements:
                    placeholder_type = elem.get("Type")
                    if placeholder_type:
                        if placeholder_type in slide_placeholder_counts:
                            slide_placeholder_counts[placeholder_type] += 1
                        else:
                            slide_placeholder_counts[placeholder_type] = 1

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
            template_slide_id = template_slide.get("slideId")
            template_elements = template_slide.get("elements", [])
            original_slide = template_slide.get("original_slide", {})

            # Tạo slideId mới cho processed slide
            if is_reused:
                new_slide_id = f"slide_{slide_number:03d}_reused_from_{template_slide_id}"
                logger.info(f"📄 Creating processed slide (REUSED): {new_slide_id} (from template: {template_slide_id})")
            else:
                new_slide_id = f"slide_{slide_number:03d}_from_{template_slide_id}"
                logger.info(f"📄 Creating processed slide (NEW): {new_slide_id} (from template: {template_slide_id})")

            # Copy toàn bộ slide structure từ template (giống luồng cũ copy slide)
            processed_slide = copy.deepcopy(original_slide)

            # Chỉ update những field cần thiết
            processed_slide["id"] = new_slide_id  # Update slide ID
            processed_slide["elements"] = []  # Reset elements để fill content mới

            # Map content vào từng element
            for element in template_elements:
                element_id = element.get("objectId")
                placeholder_type = element.get("Type")
                max_length = element.get("max_length", 1000)
                original_element = element.get("original_element", {})

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
                    logger.info(f"   Raw content: {raw_content}...")

                    try:
                        # Check max_length and handle if needed
                        final_content = await self._handle_max_length_content(
                            raw_content,
                            max_length,
                            placeholder_type
                        )

                        # Copy toàn bộ JSON structure từ template (giống luồng cũ copy slide)
                        processed_element = copy.deepcopy(original_element)  # Deep copy toàn bộ structure

                        # Chỉ update những field cần thiết
                        processed_element["id"] = element_id  # Update ID
                        processed_element["text"] = final_content  # Update content

                        processed_slide["elements"].append(processed_element)

                        # Increment content index
                        content_index[placeholder_type] = current_index + 1

                        logger.info(f"✅ Mapped {placeholder_type} to {element_id}: {final_content}...")
                        logger.info(f"   Final content length: {len(final_content)}")
                        logger.info(f"   Element structure: {list(processed_element.keys())}")
                    except Exception as e:
                        logger.error(f"❌ Failed to handle content for {placeholder_type} in slide {slide_number}: {e}")
                        logger.error(f"   Content length: {len(raw_content)}, Max length: {max_length}")
                        logger.error(f"   SKIPPING this slide due to content length issue - NO FALLBACK")
                        return None  # Skip entire slide if any content fails
                else:
                    logger.warning(f"❌ No more content available for {placeholder_type} in slide {slide_number}")
                    logger.warning(f"   Available content items: {len(content_list)}")
                    logger.warning(f"   Current index: {current_index}")
                    logger.warning(f"   Content list: {[item.get('content', '') for item in content_list]}")
                    return None  # Skip slide if missing content

            logger.info(f"✅ Successfully created processed slide {slide_number} with {len(processed_slide['elements'])} elements")
            return processed_slide

        except Exception as e:
            logger.error(f"❌ Error creating processed slide from template: {e}")
            return None




# Singleton instance
_json_template_service = None

def get_json_template_service() -> JsonTemplateService:
    """Get singleton instance của JsonTemplateService"""
    global _json_template_service
    if _json_template_service is None:
        _json_template_service = JsonTemplateService()
    return _json_template_service
