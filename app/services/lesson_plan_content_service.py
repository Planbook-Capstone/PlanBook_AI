"""
Lesson Plan Content Generation Service
Xử lý sinh nội dung giáo án chi tiết bằng LLM theo cấu trúc JSON đầu vào
"""

import logging
import json
from typing import Dict, Any, List, Optional, Set
from copy import deepcopy

from app.services.llm_service import LLMService
from app.services.textbook_retrieval_service import TextbookRetrievalService
from app.services.enhanced_textbook_service import EnhancedTextbookService

logger = logging.getLogger(__name__)


class LessonPlanContentService:
    """
    Service xử lý sinh nội dung chi tiết cho giáo án từ cấu trúc JSON
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.textbook_service = TextbookRetrievalService()
        self.enhanced_textbook_service = EnhancedTextbookService()
        
        # Giới hạn độ sâu để tránh vòng lặp vô hạn
        self.MAX_DEPTH = 10
        
        # Các type cần sinh nội dung chi tiết
        self.CONTENT_TYPES = {"PARAGRAPH", "LIST_ITEM"}

        # Các type là section (container)
        self.SECTION_TYPES = {"SECTION", "SUBSECTION"}
    
    async def generate_lesson_plan_content(
        self, 
        lesson_plan_json: Dict[str, Any],
        lesson_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sinh nội dung chi tiết cho giáo án từ cấu trúc JSON
        
        Args:
            lesson_plan_json: Cấu trúc JSON của giáo án
            lesson_id: ID của bài học để lấy nội dung tham khảo (optional)
            
        Returns:
            Dict chứa kết quả xử lý và JSON đã được sinh nội dung
        """
        try:
            logger.info("Starting lesson plan content generation...")
            
            # 1. Validate đầu vào
            validation_result = self._validate_input_json(lesson_plan_json)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"Invalid input JSON: {validation_result['error']}",
                    "lesson_plan": lesson_plan_json
                }
            
            # 2. Lấy lesson content nếu có lesson_id
            lesson_content = ""
            if lesson_id:
                try:
                    content_result = await self.textbook_service.get_lesson_content(lesson_id)
                    lesson_content = content_result.get("lesson_content", "")
                    logger.info(f"Retrieved lesson content: {len(lesson_content)} characters")
                except Exception as e:
                    logger.warning(f"Could not retrieve lesson content for {lesson_id}: {e}")
            
            # 3. Tạo deep copy để không thay đổi input gốc
            processed_json = deepcopy(lesson_plan_json)
            
            # 4. Xử lý đệ quy để sinh nội dung
            processing_result = await self._process_lesson_plan_recursive(
                processed_json, 
                lesson_content,
                depth=0
            )
            
            if not processing_result["success"]:
                return {
                    "success": False,
                    "error": processing_result["error"],
                    "lesson_plan": lesson_plan_json
                }
            
            # 5. Validate kết quả cuối cùng
            final_validation = self._validate_output_json(
                lesson_plan_json, 
                processed_json
            )
            
            if not final_validation["valid"]:
                return {
                    "success": False,
                    "error": f"Output validation failed: {final_validation['error']}",
                    "lesson_plan": lesson_plan_json
                }
            
            logger.info("Lesson plan content generation completed successfully")
            
            return {
                "success": True,
                "lesson_plan": processed_json,
                "statistics": {
                    "total_nodes": self._count_nodes(processed_json),
                    "content_nodes_processed": processing_result.get("nodes_processed", 0),
                    "lesson_content_used": len(lesson_content) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in lesson plan content generation: {e}")
            return {
                "success": False,
                "error": str(e),
                "lesson_plan": lesson_plan_json
            }
    
    async def _process_lesson_plan_recursive(
        self, 
        node: Dict[str, Any], 
        lesson_content: str,
        depth: int = 0,
        visited_ids: Optional[Set[int]] = None
    ) -> Dict[str, Any]:
        """
        Xử lý đệ quy từng node trong cây JSON
        
        Args:
            node: Node hiện tại cần xử lý
            lesson_content: Nội dung bài học tham khảo
            depth: Độ sâu hiện tại
            visited_ids: Set các ID đã thăm để tránh vòng lặp
            
        Returns:
            Dict chứa kết quả xử lý
        """
        try:
            # Khởi tạo visited_ids nếu chưa có
            if visited_ids is None:
                visited_ids = set()
            
            # Kiểm tra độ sâu tối đa
            if depth > self.MAX_DEPTH:
                return {
                    "success": False,
                    "error": f"Maximum depth {self.MAX_DEPTH} exceeded"
                }
            
            # Kiểm tra node có hợp lệ không
            if not isinstance(node, dict):
                return {
                    "success": False,
                    "error": "Node is not a dictionary"
                }
            
            # Kiểm tra các trường bắt buộc
            required_fields = ["id", "type", "status"]
            for field in required_fields:
                if field not in node:
                    return {
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }
            
            # Chỉ xử lý node có status ACTIVE
            if node.get("status") != "ACTIVE":
                return {"success": True, "nodes_processed": 0}
            
            # Kiểm tra vòng lặp
            node_id = node.get("id")
            if node_id in visited_ids:
                return {
                    "success": False,
                    "error": f"Circular reference detected for node ID: {node_id}"
                }
            
            visited_ids.add(node_id)
            nodes_processed = 0
            
            try:
                # Xử lý node hiện tại
                node_result = await self._process_single_node(node, lesson_content)
                if not node_result["success"]:
                    return node_result
                
                if node_result.get("content_generated", False):
                    nodes_processed += 1
                
                # Xử lý các node con
                children = node.get("children", [])
                if children:
                    for child in children:
                        child_result = await self._process_lesson_plan_recursive(
                            child, 
                            lesson_content, 
                            depth + 1,
                            visited_ids.copy()  # Tạo copy để tránh ảnh hưởng giữa các nhánh
                        )
                        
                        if not child_result["success"]:
                            return child_result
                        
                        nodes_processed += child_result.get("nodes_processed", 0)
                
                return {
                    "success": True,
                    "nodes_processed": nodes_processed
                }
                
            finally:
                # Loại bỏ node_id khỏi visited_ids khi hoàn thành
                visited_ids.discard(node_id)
            
        except Exception as e:
            logger.error(f"Error processing node at depth {depth}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _process_single_node(
        self,
        node: Dict[str, Any],
        lesson_content: str
    ) -> Dict[str, Any]:
        """
        Xử lý một node đơn lẻ để sinh nội dung

        Args:
            node: Node cần xử lý
            lesson_content: Nội dung bài học tham khảo

        Returns:
            Dict chứa kết quả xử lý
        """
        try:
            node_type = node.get("type")
            node_title = node.get("title", "")
            current_content = node.get("content", "")

            # Quy tắc chung: Kiểm tra có children ACTIVE không
            children = node.get("children", [])
            has_active_children = any(
                child.get("status") == "ACTIVE"
                for child in children
            )

            if has_active_children:
                # Có children ACTIVE → giữ content rỗng (áp dụng cho tất cả loại node)
                node["content"] = ""
                logger.info(f"Node {node.get('id')} ({node_type}) has active children - keeping content empty")
                return {
                    "success": True,
                    "content_generated": False
                }

            # Không có children ACTIVE → xử lý theo type
            if node_type in self.CONTENT_TYPES:
                # PARAGRAPH hoặc LIST_ITEM: sinh nội dung chi tiết
                if not current_content.strip():  # Chỉ sinh nếu content rỗng
                    logger.info(f"Node {node.get('id')} ({node_type}) has no children - generating content")
                    generated_content = await self._generate_content_for_node(
                        node, lesson_content
                    )

                    if generated_content["success"]:
                        node["content"] = generated_content["content"]
                        return {
                            "success": True,
                            "content_generated": True
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Failed to generate content for node {node.get('id')}: {generated_content['error']}"
                        }
                else:
                    # Đã có content, không cần sinh thêm
                    return {
                        "success": True,
                        "content_generated": False
                    }

            elif node_type in self.SECTION_TYPES:
                # SECTION/SUBSECTION: sinh content cho section này
                if not current_content.strip():
                    logger.info(f"Node {node.get('id')} ({node_type}) has no children - generating content")
                    generated_content = await self._generate_content_for_node(
                        node, lesson_content
                    )

                    if generated_content["success"]:
                        node["content"] = generated_content["content"]
                        return {
                            "success": True,
                            "content_generated": True
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Failed to generate content for section {node.get('id')}: {generated_content['error']}"
                        }
                else:
                    # Đã có content, không cần sinh thêm
                    return {
                        "success": True,
                        "content_generated": False
                    }

            else:
                # Type không xác định, bỏ qua
                logger.warning(f"Unknown node type: {node_type} for node {node.get('id')}")
                return {
                    "success": True,
                    "content_generated": False
                }

        except Exception as e:
            logger.error(f"Error processing single node {node.get('id')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_content_for_node(
        self,
        node: Dict[str, Any],
        lesson_content: str
    ) -> Dict[str, Any]:
        """
        Sinh nội dung chi tiết cho một node bằng LLM

        Args:
            node: Node cần sinh nội dung
            lesson_content: Nội dung bài học tham khảo

        Returns:
            Dict chứa nội dung đã sinh
        """
        try:
            # Tạo prompt cho LLM
            prompt = self._create_content_generation_prompt(node, lesson_content)

            # Gọi LLM để sinh nội dung
            llm_result = await self.llm_service._generate_content(prompt)

            if llm_result["success"]:
                generated_content = llm_result["text"].strip()

                # Làm sạch nội dung từ LLM
                cleaned_content = self._clean_generated_content(generated_content, node.get('id'))

                # Validate nội dung đã làm sạch
                if len(cleaned_content) < 10:
                    return {
                        "success": False,
                        "error": "Generated content too short after cleaning"
                    }

                return {
                    "success": True,
                    "content": cleaned_content
                }
            else:
                return {
                    "success": False,
                    "error": llm_result["error"]
                }

        except Exception as e:
            logger.error(f"Error generating content for node {node.get('id')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_content_generation_prompt(
        self,
        node: Dict[str, Any],
        lesson_content: str
    ) -> str:
        """
        Tạo prompt cho LLM để sinh nội dung chi tiết

        Args:
            node: Node cần sinh nội dung
            lesson_content: Nội dung bài học tham khảo

        Returns:
            String prompt cho LLM
        """
        node_type = node.get("type", "")
        node_title = node.get("title", "")
        node_id = node.get("id", "")

        # Lấy context từ parent và siblings nếu có
        context_info = self._extract_context_info(node)

        # Tạo prompt dựa trên type
        if node_type == "PARAGRAPH":
            prompt = f"""
Bạn là một giáo viên trung học phổ thông Việt Nam giàu kinh nghiệm, chuyên soạn giáo án chi tiết và thực tiễn.
NHIỆM VỤ: Phát triển nội dung chi tiết cho đoạn văn trong giáo án.

THÔNG TIN NODE:
- ID: {node_id}
- Tiêu đề: "{node_title}"
- Loại: Đoạn văn mô tả (PARAGRAPH)

NGỮ CẢNH GIÁO ÁN:
{context_info}

NỘI DUNG BÀI HỌC THAM KHẢO:
{lesson_content[:4000] if lesson_content else "Không có nội dung tham khảo"}

YÊU CẦU:
1. Viết nội dung chi tiết, cụ thể cho đoạn văn này
2. Nội dung phải phù hợp với tiêu đề và ngữ cảnh giáo án
3. Sử dụng ngôn ngữ chuyên nghiệp, phù hợp với giáo viên
4. Tập trung vào nội dung thực tiễn, có thể áp dụng được
5. Các thuật ngữ trong sách giáo khoa không được thay đổi

ĐỊNH DẠNG ĐẦU RA:
Chỉ trả về nội dung đoạn văn, không có tiêu đề hay định dạng khác.
"""

        elif node_type == "LIST_ITEM":
            prompt = f"""
Bạn là một giáo viên Việt Nam giàu kinh nghiệm, chuyên soạn giáo án chi tiết và thực tiễn.

NHIỆM VỤ: Phát triển nội dung chi tiết cho mục liệt kê trong giáo án.

THÔNG TIN NODE:
- ID: {node_id}
- Tiêu đề: "{node_title}"
- Loại: Mục liệt kê (LIST_ITEM)

NGỮ CẢNH GIÁO ÁN:
{context_info}

NỘI DUNG BÀI HỌC THAM KHẢO:
{lesson_content[:4000] if lesson_content else "Không có nội dung tham khảo"}

YÊU CẦU:
1. Viết nội dung chi tiết, cụ thể cho đoạn văn này
2. Nội dung phải phù hợp với tiêu đề và ngữ cảnh giáo án
3. Sử dụng ngôn ngữ chuyên nghiệp, phù hợp với giáo viên
4. Tập trung vào nội dung thực tiễn, có thể áp dụng được
5. Các thuật ngữ trong sách giáo khoa không được thay đổi



ĐỊNH DẠNG ĐẦU RA:
Chỉ trả về nội dung mục liệt kê, không có tiêu đề hay định dạng khác.
Có thể sử dụng dấu gạch đầu dòng (-) nếu có nhiều điểm con.
"""

        elif node_type == "SECTION":
            prompt = f"""
Bạn là một giáo viên Việt Nam giàu kinh nghiệm, chuyên soạn giáo án chi tiết và thực tiễn.

NHIỆM VỤ: Phát triển nội dung tổng quan cho phần/mục trong giáo án.

THÔNG TIN NODE:
- ID: {node_id}
- Tiêu đề: "{node_title}"
- Loại: Phần/Mục (SECTION)

NGỮ CẢNH GIÁO ÁN:
{context_info}

NỘI DUNG BÀI HỌC THAM KHẢO:
{lesson_content[:4000] if lesson_content else "Không có nội dung tham khảo"}

YÊU CẦU:
1. Viết nội dung chi tiết, cụ thể cho đoạn văn này
2. Nội dung phải phù hợp với tiêu đề và ngữ cảnh giáo án
3. Sử dụng ngôn ngữ chuyên nghiệp, phù hợp với giáo viên
4. Tập trung vào nội dung thực tiễn, có thể áp dụng được
5. Các thuật ngữ trong sách giáo khoa không được thay đổi


ĐỊNH DẠNG ĐẦU RA:
Chỉ trả về nội dung phần/mục, không có tiêu đề hay định dạng khác.
"""

        else:
            # Fallback cho các type khác
            prompt = f"""
Bạn là một giáo viên Việt Nam giàu kinh nghiệm, chuyên soạn giáo án chi tiết và thực tiễn.

NHIỆM VỤ: Phát triển nội dung chi tiết cho thành phần trong giáo án.

THÔNG TIN NODE:
- ID: {node_id}
- Tiêu đề: "{node_title}"
- Loại: {node_type}

NGỮ CẢNH GIÁO ÁN:
{context_info}

NỘI DUNG BÀI HỌC THAM KHẢO:
{lesson_content[:4000] if lesson_content else "Không có nội dung tham khảo"}

YÊU CẦU:
1. Viết nội dung chi tiết, cụ thể cho đoạn văn này
2. Nội dung phải phù hợp với tiêu đề và ngữ cảnh giáo án
3. Sử dụng ngôn ngữ chuyên nghiệp, phù hợp với giáo viên
4. Tập trung vào nội dung thực tiễn, có thể áp dụng được
5. Các thuật ngữ trong sách giáo khoa không được thay đổi


ĐỊNH DẠNG ĐẦU RA:
Chỉ trả về nội dung, không có tiêu đề hay định dạng khác.
"""

        return prompt.strip()

    def _extract_context_info(self, node: Dict[str, Any]) -> str:
        """
        Trích xuất thông tin ngữ cảnh từ node để tạo prompt tốt hơn

        Args:
            node: Node cần trích xuất context

        Returns:
            String mô tả ngữ cảnh
        """
        try:
            context_parts = []

            # Thông tin cơ bản
            if node.get("lessonPlanId"):
                context_parts.append(f"Giáo án ID: {node.get('lessonPlanId')}")

            if node.get("parentId"):
                context_parts.append(f"Thuộc phần cha ID: {node.get('parentId')}")

            if node.get("orderIndex") is not None:
                context_parts.append(f"Thứ tự: {node.get('orderIndex')}")

            # Metadata nếu có
            metadata = node.get("metadata")
            if metadata and isinstance(metadata, dict):
                for key, value in metadata.items():
                    if value:
                        context_parts.append(f"{key}: {value}")

            return "\n".join(context_parts) if context_parts else "Không có thông tin ngữ cảnh"

        except Exception as e:
            logger.warning(f"Error extracting context info: {e}")
            return "Không thể trích xuất thông tin ngữ cảnh"

    def _validate_input_json(self, lesson_plan_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate cấu trúc JSON đầu vào

        Args:
            lesson_plan_json: JSON cần validate

        Returns:
            Dict chứa kết quả validation
        """
        try:
            # Kiểm tra có phải là dict không
            if not isinstance(lesson_plan_json, dict):
                return {
                    "valid": False,
                    "error": "Input must be a dictionary"
                }

            # Kiểm tra các trường bắt buộc ở root level
            required_fields = ["id", "type", "status"]
            for field in required_fields:
                if field not in lesson_plan_json:
                    return {
                        "valid": False,
                        "error": f"Missing required field at root level: {field}"
                    }

            # Validate đệ quy
            validation_result = self._validate_node_recursive(lesson_plan_json, set())

            return validation_result

        except Exception as e:
            logger.error(f"Error validating input JSON: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def _validate_node_recursive(
        self,
        node: Dict[str, Any],
        visited_ids: Set[int]
    ) -> Dict[str, Any]:
        """
        Validate một node và các children của nó

        Args:
            node: Node cần validate
            visited_ids: Set các ID đã thăm

        Returns:
            Dict chứa kết quả validation
        """
        try:
            # Kiểm tra node có phải là dict không
            if not isinstance(node, dict):
                return {
                    "valid": False,
                    "error": "Node must be a dictionary"
                }

            # Kiểm tra các trường bắt buộc
            required_fields = ["id", "type", "status"]
            for field in required_fields:
                if field not in node:
                    return {
                        "valid": False,
                        "error": f"Missing required field in node: {field}"
                    }

            # Kiểm tra ID hợp lệ
            node_id = node.get("id")
            if not isinstance(node_id, int):
                return {
                    "valid": False,
                    "error": f"Node ID must be integer, got: {type(node_id)}"
                }

            # Kiểm tra vòng lặp
            if node_id in visited_ids:
                return {
                    "valid": False,
                    "error": f"Circular reference detected for node ID: {node_id}"
                }

            visited_ids.add(node_id)

            # Kiểm tra type hợp lệ
            node_type = node.get("type")
            valid_types = self.CONTENT_TYPES | self.SECTION_TYPES
            if node_type not in valid_types:
                logger.warning(f"Unknown node type: {node_type} for node {node_id}")

            # Kiểm tra status hợp lệ
            status = node.get("status")
            valid_statuses = {"ACTIVE", "INACTIVE", "DELETED"}
            if status not in valid_statuses:
                return {
                    "valid": False,
                    "error": f"Invalid status: {status} for node {node_id}"
                }

            # Validate children nếu có
            children = node.get("children", [])
            if children:
                if not isinstance(children, list):
                    return {
                        "valid": False,
                        "error": f"Children must be a list for node {node_id}"
                    }

                for child in children:
                    child_validation = self._validate_node_recursive(child, visited_ids.copy())
                    if not child_validation["valid"]:
                        return child_validation

            return {"valid": True}

        except Exception as e:
            logger.error(f"Error validating node: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def _validate_output_json(
        self,
        original_json: Dict[str, Any],
        processed_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate JSON output so với JSON gốc

        Args:
            original_json: JSON gốc
            processed_json: JSON đã xử lý

        Returns:
            Dict chứa kết quả validation
        """
        try:
            # So sánh cấu trúc cơ bản
            structure_check = self._compare_json_structure(original_json, processed_json)
            if not structure_check["valid"]:
                return structure_check

            # Kiểm tra tất cả node có field content
            content_check = self._check_all_nodes_have_content(processed_json)
            if not content_check["valid"]:
                return content_check

            return {"valid": True}

        except Exception as e:
            logger.error(f"Error validating output JSON: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def _compare_json_structure(
        self,
        original: Dict[str, Any],
        processed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        So sánh cấu trúc giữa JSON gốc và JSON đã xử lý

        Args:
            original: JSON gốc
            processed: JSON đã xử lý

        Returns:
            Dict chứa kết quả so sánh
        """
        try:
            # Kiểm tra các trường cơ bản
            basic_fields = ["id", "lessonPlanId", "parentId", "title", "type", "orderIndex", "status"]

            for field in basic_fields:
                if original.get(field) != processed.get(field):
                    return {
                        "valid": False,
                        "error": f"Field '{field}' changed: {original.get(field)} -> {processed.get(field)}"
                    }

            # Kiểm tra children
            original_children = original.get("children", [])
            processed_children = processed.get("children", [])

            if len(original_children) != len(processed_children):
                return {
                    "valid": False,
                    "error": f"Number of children changed: {len(original_children)} -> {len(processed_children)}"
                }

            # So sánh từng child
            for i, (orig_child, proc_child) in enumerate(zip(original_children, processed_children)):
                child_comparison = self._compare_json_structure(orig_child, proc_child)
                if not child_comparison["valid"]:
                    return {
                        "valid": False,
                        "error": f"Child {i}: {child_comparison['error']}"
                    }

            return {"valid": True}

        except Exception as e:
            logger.error(f"Error comparing JSON structure: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def _check_all_nodes_have_content(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kiểm tra tất cả node có field content

        Args:
            node: Node cần kiểm tra

        Returns:
            Dict chứa kết quả kiểm tra
        """
        try:
            # Kiểm tra node hiện tại
            if "content" not in node:
                return {
                    "valid": False,
                    "error": f"Node {node.get('id')} missing 'content' field"
                }

            # Kiểm tra children
            children = node.get("children", [])
            for child in children:
                child_check = self._check_all_nodes_have_content(child)
                if not child_check["valid"]:
                    return child_check

            return {"valid": True}

        except Exception as e:
            logger.error(f"Error checking content fields: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def _count_nodes(self, node: Dict[str, Any]) -> int:
        """
        Đếm tổng số node trong cây JSON

        Args:
            node: Node gốc

        Returns:
            Số lượng node
        """
        try:
            count = 1  # Node hiện tại

            children = node.get("children", [])
            for child in children:
                count += self._count_nodes(child)

            return count

        except Exception as e:
            logger.warning(f"Error counting nodes: {e}")
            return 1

    def _clean_generated_content(self, content: str, node_id: int) -> str:
        """
        Làm sạch nội dung được sinh từ LLM

        Args:
            content: Nội dung cần làm sạch
            node_id: ID của node để logging

        Returns:
            Nội dung đã được làm sạch
        """
        try:
            if not content or not content.strip():
                return content

            logger.info(f"🧹 Cleaning generated content for node {node_id}")
            logger.debug(f"Original content length: {len(content)} chars")

            # Sử dụng hàm clean_text_content từ enhanced_textbook_service
            cleaned_content = self.enhanced_textbook_service.clean_text_content(content)

            logger.info(f"🧹 Content cleaned for node {node_id}: {len(content)} → {len(cleaned_content)} chars")

            return cleaned_content

        except Exception as e:
            logger.error(f"Error cleaning content for node {node_id}: {e}")
            # Trả về nội dung gốc nếu có lỗi khi làm sạch
            return content.strip()


# Global instance
lesson_plan_content_service = LessonPlanContentService()
