"""
Lesson Plan Content Generation Service
Xử lý sinh nội dung giáo án chi tiết bằng LLM theo cấu trúc JSON đầu vào
"""

import logging
import json
from typing import Dict, Any, List, Optional, Set
from copy import deepcopy

from app.services.llm_service import get_llm_service
from app.services.textbook_retrieval_service import TextbookRetrievalService
from app.services.enhanced_textbook_service import EnhancedTextbookService

logger = logging.getLogger(__name__)


class LessonPlanContentService:
    """
    Service xử lý sinh nội dung chi tiết cho giáo án từ cấu trúc JSON
    """
    
    def __init__(self):
        self.llm_service = get_llm_service()
        self.textbook_service = TextbookRetrievalService()
        self.enhanced_textbook_service = EnhancedTextbookService()
        
        # Giới hạn độ sâu để tránh vòng lặp vô hạn
        self.MAX_DEPTH = 10
        
        # Các type cần sinh nội dung chi tiết
        self.CONTENT_TYPES = {"PARAGRAPH", "LIST_ITEM", "TABLE"}

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
                    print(f"DEBUG: content_result: {content_result}")
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

    async def generate_lesson_plan_content_with_progress(
        self,
        lesson_plan_json: Dict[str, Any],
        lesson_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        total_nodes: int = 0
    ) -> Dict[str, Any]:
        """
        Sinh nội dung chi tiết cho giáo án từ cấu trúc JSON với progress tracking

        Args:
            lesson_plan_json: JSON cấu trúc giáo án
            lesson_id: ID bài học để lấy nội dung tham khảo (optional)
            progress_callback: Callback function để update progress
            total_nodes: Tổng số nodes để tính progress

        Returns:
            Dict với kết quả xử lý
        """
        try:
            logger.info("Starting lesson plan content generation with progress tracking...")

            # Validate đầu vào
            if not lesson_plan_json:
                return {
                    "success": False,
                    "error": "lesson_plan_json cannot be empty"
                }

            # Lấy nội dung bài học để tham khảo (nếu có lesson_id)
            lesson_content = ""
            if lesson_id:
                try:
                    # TODO: Implement _get_lesson_content method to retrieve lesson content from database
                    # For now, skip lesson content retrieval
                    lesson_content = ""
                    logger.info(f"Lesson content retrieval skipped for lesson_id: {lesson_id}")
                except Exception as e:
                    logger.warning(f"Could not retrieve lesson content for {lesson_id}: {e}")
                    lesson_content = ""

            # Tạo bản sao để xử lý
            processed_json = deepcopy(lesson_plan_json)

            # Xử lý đệ quy từ root với progress tracking
            processed_nodes = [0]  # Use list to make it mutable in nested function

            async def progress_wrapper():
                processed_nodes[0] += 1
                if progress_callback and total_nodes > 0:
                    await progress_callback(
                        processed_nodes[0],
                        total_nodes,
                        f"Processing node {processed_nodes[0]}/{total_nodes}..."
                    )

            processing_result = await self._process_lesson_plan_recursive_with_progress(
                processed_json,
                lesson_content,
                progress_wrapper
            )

            if processing_result["success"]:
                return {
                    "success": True,
                    "lesson_plan": processed_json,
                    "statistics": {
                        "total_nodes": self._count_nodes(processed_json),
                        "content_nodes_processed": processing_result.get("nodes_processed", 0),
                        "lesson_content_used": len(lesson_content) > 0
                    }
                }
            else:
                return {
                    "success": False,
                    "error": processing_result["error"],
                    "lesson_plan": processed_json
                }

        except Exception as e:
            logger.error(f"Error in lesson plan content generation with progress: {e}")
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

    async def _process_lesson_plan_recursive_with_progress(
        self,
        node: Dict[str, Any],
        lesson_content: str,
        progress_callback: Optional[callable] = None,
        depth: int = 0,
        visited_ids: Optional[Set[int]] = None
    ) -> Dict[str, Any]:
        """
        Xử lý đệ quy từng node trong cây JSON với progress tracking

        Args:
            node: Node hiện tại cần xử lý
            lesson_content: Nội dung bài học tham khảo
            progress_callback: Callback function để update progress
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
                    "error": f"Cycle detected at node {node_id}"
                }

            # Thêm node_id vào visited_ids
            visited_ids.add(node_id)

            try:
                # Update progress callback
                if progress_callback:
                    await progress_callback()

                # Xử lý node hiện tại
                single_node_result = await self._process_single_node(node, lesson_content)

                if not single_node_result["success"]:
                    return single_node_result

                nodes_processed = 1 if single_node_result.get("content_generated", False) else 0

                # Xử lý children nếu có
                children = node.get("children", [])
                for child in children:
                    if child.get("status") == "ACTIVE":
                        child_result = await self._process_lesson_plan_recursive_with_progress(
                            child, lesson_content, progress_callback, depth + 1, visited_ids
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
            logger.error(f"Error processing node with progress at depth {depth}: {e}")
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

            # Không có children ACTIVE → xử lý theo fieldType hoặc type
            fieldType = node.get("fieldType")

            if fieldType == "TABLE":
                # TABLE: xử lý đặc biệt cho bảng dựa trên fieldType
                return await self._process_table_node(node, lesson_content)

            elif node_type in self.CONTENT_TYPES:
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

    async def _process_table_node(
        self,
        node: Dict[str, Any],
        lesson_content: str
    ) -> Dict[str, Any]:
        """
        Xử lý đặc biệt cho node TABLE - sinh nội dung cho các cell trong bảng

        Args:
            node: Node TABLE cần xử lý
            lesson_content: Nội dung bài học tham khảo

        Returns:
            Dict chứa kết quả xử lý
        """
        try:
            current_content = node.get("content", "")

            # Xử lý content - có thể là dict hoặc string JSON
            table_data = None
            if isinstance(current_content, dict) and "rows" in current_content:
                table_data = current_content
            elif isinstance(current_content, str) and current_content.strip():
                try:
                    # Parse JSON string thành dict
                    parsed_content = json.loads(current_content)
                    if isinstance(parsed_content, dict) and "rows" in parsed_content:
                        table_data = parsed_content
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse table content JSON for node {node.get('id')}: {e}")

            if table_data:
                content_generated = False

                # Duyệt qua từng row
                for row in table_data.get("rows", []):
                    if not isinstance(row, dict) or "cells" not in row:
                        continue

                    # Duyệt qua từng cell trong row
                    for cell in row.get("cells", []):
                        if not isinstance(cell, dict):
                            continue

                        # Bỏ qua header cells
                        if cell.get("isHeader", False):
                            continue

                        # Chỉ sinh nội dung cho cell có title nhưng content rỗng
                        cell_title = cell.get("title", "").strip()
                        cell_content = cell.get("content", "").strip()

                        if cell_title and not cell_content:
                            logger.info(f"Generating content for table cell: {cell.get('id')} with title: {cell_title}")

                            # Tạo context cho cell
                            cell_context = {
                                "id": cell.get("id", ""),
                                "title": cell_title,
                                "type": "TABLE_CELL",
                                "table_context": node.get("title", "")
                            }

                            # Sinh nội dung cho cell
                            generated_content = await self._generate_content_for_table_cell(
                                cell_context, lesson_content
                            )

                            if generated_content["success"]:
                                cell["content"] = generated_content["content"]
                                content_generated = True
                                logger.info(f"Generated content for cell {cell.get('id')}")
                            else:
                                logger.warning(f"Failed to generate content for cell {cell.get('id')}: {generated_content['error']}")

                # Cập nhật lại content của node - giữ nguyên format gốc
                original_content = node.get("content", "")
                if isinstance(original_content, str):
                    # Nếu ban đầu là string, convert lại thành string JSON
                    node["content"] = json.dumps(table_data, ensure_ascii=False)
                else:
                    # Nếu ban đầu là dict, giữ nguyên dict
                    node["content"] = table_data

                return {
                    "success": True,
                    "content_generated": content_generated
                }

            else:
                # Content không phải là table structure, bỏ qua
                logger.warning(f"TABLE fieldType node {node.get('id')} does not have valid table structure in content")
                return {
                    "success": True,
                    "content_generated": False
                }

        except Exception as e:
            logger.error(f"Error processing table node {node.get('id')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_content_for_table_cell(
        self,
        cell_context: Dict[str, Any],
        lesson_content: str
    ) -> Dict[str, Any]:
        """
        Sinh nội dung cho một cell trong bảng

        Args:
            cell_context: Context của cell (id, title, type, table_context)
            lesson_content: Nội dung bài học tham khảo

        Returns:
            Dict chứa nội dung đã sinh
        """
        try:
            # Ensure LLM service is initialized
            self.llm_service._ensure_service_initialized()

            # Kiểm tra LLM service availability
            if not self.llm_service.is_available():
                return {
                    "success": False,
                    "error": "No LLM service available"
                }

            # Tạo prompt cho table cell
            prompt = self._create_table_cell_prompt(cell_context, lesson_content)

            # Gọi LLM để sinh nội dung
            llm_result = await self.llm_service._generate_content(prompt)

            if llm_result["success"]:
                generated_content = llm_result["text"].strip()

                # Làm sạch nội dung từ LLM
                cleaned_content = self._clean_generated_content(generated_content, cell_context.get('id'))

                # Validate nội dung đã làm sạch
                if len(cleaned_content) < 5:
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
            logger.error(f"Error generating content for table cell {cell_context.get('id')}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_table_cell_prompt(
        self,
        cell_context: Dict[str, Any],
        lesson_content: str
    ) -> str:
        """
        Tạo prompt đặc biệt cho việc sinh nội dung table cell

        Args:
            cell_context: Context của cell
            lesson_content: Nội dung bài học tham khảo

        Returns:
            String prompt cho LLM
        """
        cell_title = cell_context.get("title", "")
        cell_id = cell_context.get("id", "")
        table_context = cell_context.get("table_context", "")

        # Làm sạch HTML tags từ title để lấy text thuần
        import re
        clean_title = re.sub(r'<[^>]+>', '', cell_title).strip()

        prompt = f"""
Bạn là một giáo viên trung học phổ thông Việt Nam giàu kinh nghiệm, chuyên soạn giáo án chi tiết.

NHIỆM VỤ: Viết nội dung ngắn gọn cho ô trong bảng giáo án.

THÔNG TIN Ô BẢNG:
- ID: {cell_id}
- Tiêu đề ô: "{clean_title}"
- Ngữ cảnh bảng: "{table_context}"

NỘI DUNG BÀI HỌC THAM KHẢO:
{lesson_content[:1500] if lesson_content else "Không có nội dung tham khảo"}

YÊU CẦU QUAN TRỌNG:
1. Nội dung PHẢI ngắn gọn, chỉ 1-2 câu (tối đa 80 từ)
2. PHẢI dựa trên nội dung bài học tham khảo ở trên
3. PHẢI cụ thể với tiêu đề ô "{clean_title}"
4. Tránh hoàn toàn các cụm từ mở đầu như "Để bắt đầu", "Chúng ta cần"
5. Đi thẳng vào nội dung chính, không dẫn dắt
6. Sử dụng thuật ngữ chính xác từ sách giáo khoa
7. Phù hợp với ngữ cảnh của bảng "{table_context}"
8. Nếu là hoạt động giáo viên: mô tả hành động cụ thể của GV
9. Nếu là hoạt động học sinh: mô tả hành động cụ thể của HS
10. Nếu là bước thực hiện: nêu rõ các bước cụ thể

ĐỊNH DẠNG ĐẦU RA:
Trả về 1-2 câu ngắn gọn, cụ thể. Không có tiêu đề, không có cụm từ mở đầu dài.
"""

        return prompt.strip()

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
            # Ensure LLM service is initialized
            self.llm_service._ensure_service_initialized()

            # Kiểm tra LLM service availability
            if not self.llm_service.is_available():
                return {
                    "success": False,
                    "error": "No LLM service available"
                }

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

        # Tạo prompt sử dụng template cơ sở
        base_prompt = self._build_prompt_from_template(
            node_type=node_type,
            node_id=node_id,
            node_title=node_title,
            context_info=context_info,
            lesson_content=lesson_content
        )

        # Tùy chỉnh prompt dựa trên ngữ cảnh cụ thể
        customized_prompt = self._customize_prompt_by_context(base_prompt, node)

        return customized_prompt

    def _build_prompt_from_template(
        self,
        node_type: str,
        node_id: str,
        node_title: str,
        context_info: str,
        lesson_content: str
    ) -> str:
        """
        Xây dựng prompt từ template cơ sở để tránh trùng lặp

        Args:
            node_type: Loại node
            node_id: ID của node
            node_title: Tiêu đề node
            context_info: Thông tin ngữ cảnh
            lesson_content: Nội dung bài học tham khảo

        Returns:
            String prompt hoàn chỉnh
        """
        # Template cơ sở chung cho tất cả loại node
        base_template = """
Bạn là một giáo viên trung học phổ thông Việt Nam giàu kinh nghiệm, chuyên soạn giáo án ngắn gọn và cụ thể.

NHIỆM VỤ: {task_description}

THÔNG TIN NODE:
- ID: {node_id}
- Tiêu đề: "{node_title}"
- Loại: {node_type_description}

NGỮ CẢNH GIÁO ÁN:
{context_info}

NỘI DUNG BÀI HỌC THAM KHẢO:
{lesson_content}

YÊU CẦU QUAN TRỌNG:
1. Nội dung PHẢI ngắn gọn, chỉ 2-3 câu (tối đa 100 từ)
2. PHẢI dựa trên nội dung bài học tham khảo ở trên, không được tự sáng tác
3. PHẢI cụ thể với bài học, không được nói chung chung
4. Tránh hoàn toàn các cụm từ mở đầu như "Để bắt đầu", "Để giúp học sinh", "Chúng ta cần"
5. Đi thẳng vào nội dung chính, không dẫn dắt dài dòng
6. Sử dụng thuật ngữ chính xác từ sách giáo khoa
7. Tập trung vào {content_target} cụ thể, không lan man
{specific_requirements}

ĐỊNH DẠNG ĐẦU RA:
{output_format}
"""

        # Cấu hình cụ thể cho từng loại node
        node_configs = {
            "PARAGRAPH": {
                "task_description": "Viết nội dung ngắn gọn cho mục tiêu học tập này.",
                "node_type_description": "Mục tiêu học tập (PARAGRAPH)",
                "content_target": "mục tiêu này",
                "specific_requirements": "",
                "output_format": "Trả về 2-3 câu ngắn gọn, cụ thể. Không có tiêu đề, không có cụm từ mở đầu dài."
            },
            "LIST_ITEM": {
                "task_description": "Viết nội dung ngắn gọn cho mục liệt kê này.",
                "node_type_description": "Mục liệt kê (LIST_ITEM)",
                "content_target": "mục liệt kê này",
                "specific_requirements": "",
                "output_format": "Trả về 1-2 câu ngắn gọn hoặc danh sách ngắn với dấu gạch đầu dòng (-). Không có tiêu đề."
            },
            "SECTION": {
                "task_description": "Viết nội dung tổng quan ngắn gọn cho phần này.",
                "node_type_description": "Phần/Mục (SECTION)",
                "content_target": "phần này",
                "specific_requirements": "",
                "output_format": "Trả về 2-3 câu tổng quan ngắn gọn. Không có tiêu đề, không dẫn dắt dài."
            },
            "SUBSECTION": {
                "task_description": "Viết nội dung ngắn gọn cho phần con này.",
                "node_type_description": "Phần con (SUBSECTION)",
                "content_target": "phần con này",
                "specific_requirements": "",
                "output_format": "Trả về 2-3 câu ngắn gọn, cụ thể. Không có tiêu đề, không có cụm từ mở đầu."
            },
            "TABLE": {
                "task_description": "Xử lý nội dung bảng trong giáo án.",
                "node_type_description": "Bảng (TABLE)",
                "content_target": "bảng này",
                "specific_requirements": "",
                "output_format": "Xử lý từng ô trong bảng dựa trên tiêu đề ô."
            }
        }

        # Lấy cấu hình cho node type, fallback nếu không tìm thấy
        config = node_configs.get(node_type, {
            "task_description": "Phát triển nội dung chi tiết cho thành phần trong giáo án.",
            "node_type_description": f"{node_type}",
            "content_target": "thành phần này",
            "specific_requirements": "",
            "output_format": "Chỉ trả về nội dung, không có tiêu đề hay định dạng khác."
        })

        # Điền thông tin vào template
        prompt = base_template.format(
            task_description=config["task_description"],
            node_id=node_id,
            node_title=node_title,
            node_type_description=config["node_type_description"],
            context_info=context_info,
            lesson_content=lesson_content[:2000] if lesson_content else "Không có nội dung tham khảo",
            content_target=config["content_target"],
            specific_requirements=config["specific_requirements"],
            output_format=config["output_format"]
        )

        return prompt.strip()

    def _get_enhanced_requirements_for_node(self, node: Dict[str, Any]) -> str:
        """
        Lấy yêu cầu bổ sung dựa trên ngữ cảnh cụ thể của node

        Args:
            node: Node cần phân tích

        Returns:
            String chứa yêu cầu bổ sung
        """
        node_title = node.get("title", "").lower()
        node_type = node.get("type", "")
        enhanced_requirements = []

        # Yêu cầu dựa trên mức độ nhận thức (Bloom's Taxonomy)
        if "nhận biết" in node_title or "trình bày được" in node_title:
            enhanced_requirements.append("8. Liệt kê hoặc định nghĩa ngắn gọn các khái niệm chính từ bài học")
            enhanced_requirements.append("9. Sử dụng chính xác thuật ngữ từ sách giáo khoa")

        elif "hiểu được" in node_title or "giải thích" in node_title:
            enhanced_requirements.append("8. Giải thích ngắn gọn bằng từ ngữ đơn giản, dễ hiểu")
            enhanced_requirements.append("9. Nêu ví dụ cụ thể từ nội dung bài học")

        elif "phân biệt" in node_title or "so sánh" in node_title:
            enhanced_requirements.append("8. Chỉ ra 2-3 điểm khác nhau chính, không liệt kê quá nhiều")
            enhanced_requirements.append("9. Dựa trên tiêu chí rõ ràng từ bài học")

        elif "vận dụng" in node_title or "áp dụng" in node_title:
            enhanced_requirements.append("8. Đưa ra ví dụ cụ thể từ bài học hoặc thực tế")
            enhanced_requirements.append("9. Mô tả ngắn gọn cách thực hiện, không lý thuyết dài")

        elif "phân tích" in node_title:
            enhanced_requirements.append("8. Chia nhỏ thành 2-3 thành phần chính")
            enhanced_requirements.append("9. Giải thích mối quan hệ giữa các thành phần")

        elif "mục tiêu" in node_title:
            enhanced_requirements.append("8. Mục tiêu phải cụ thể, đo lường được")
            enhanced_requirements.append("9. Phù hợp với nội dung bài học cụ thể")

        elif "hoạt động" in node_title:
            enhanced_requirements.append("8. Mô tả ngắn gọn các bước chính")
            enhanced_requirements.append("9. Nêu rõ thời gian và phương pháp")

        elif "câu hỏi" in node_title or "đặt vấn đề" in node_title:
            enhanced_requirements.append("8. Câu hỏi phải liên quan trực tiếp đến bài học")
            enhanced_requirements.append("9. Phù hợp với trình độ học sinh")

        elif "bài tập" in node_title or "luyện tập" in node_title:
            enhanced_requirements.append("8. Nêu rõ số trang, bài tập cụ thể trong SGK/SBT")
            enhanced_requirements.append("9. Ước tính thời gian làm bài")

        elif "đánh giá" in node_title or "kiểm tra" in node_title:
            enhanced_requirements.append("8. Đưa ra tiêu chí đánh giá cụ thể")
            enhanced_requirements.append("9. Phương pháp đánh giá phù hợp")

        # Yêu cầu dựa trên loại node
        if node_type == "LIST_ITEM":
            enhanced_requirements.append("10. Sử dụng format danh sách ngắn với dấu gạch đầu dòng (-)")

        elif node_type == "PARAGRAPH":
            enhanced_requirements.append("10. Viết thành 2-3 câu ngắn, không dài dòng")

        return "\n".join(enhanced_requirements) if enhanced_requirements else ""

    def _customize_prompt_by_context(self, base_prompt: str, node: Dict[str, Any]) -> str:
        """
        Tùy chỉnh prompt dựa trên ngữ cảnh cụ thể

        Args:
            base_prompt: Prompt cơ sở
            node: Node cần xử lý

        Returns:
            Prompt đã được tùy chỉnh
        """
        enhanced_requirements = self._get_enhanced_requirements_for_node(node)

        if enhanced_requirements:
            # Thêm yêu cầu bổ sung vào prompt
            base_prompt = base_prompt.replace(
                "7. Không mở rộng ra các chủ đề không liên quan trực tiếp đến bài học",
                f"7. Không mở rộng ra các chủ đề không liên quan trực tiếp đến bài học\n{enhanced_requirements}"
            )

        return base_prompt

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


# Lazy loading global instance để tránh khởi tạo ngay khi import
_lesson_plan_content_service_instance = None

def get_lesson_plan_content_service() -> LessonPlanContentService:
    """
    Lấy singleton instance của LessonPlanContentService
    Lazy initialization

    Returns:
        LessonPlanContentService: Service instance
    """
    global _lesson_plan_content_service_instance
    if _lesson_plan_content_service_instance is None:
        _lesson_plan_content_service_instance = LessonPlanContentService()
    return _lesson_plan_content_service_instance

# Backward compatibility - deprecated, sử dụng get_lesson_plan_content_service() thay thế
# Lazy loading để tránh khởi tạo ngay khi import
def _get_lesson_plan_content_service_lazy():
    """Lazy loading cho backward compatibility"""
    return get_lesson_plan_content_service()

# Tạo proxy object để lazy loading
class _LessonPlanContentServiceProxy:
    def __getattr__(self, name):
        return getattr(_get_lesson_plan_content_service_lazy(), name)

    def __call__(self, *args, **kwargs):
        return _get_lesson_plan_content_service_lazy()(*args, **kwargs)

lesson_plan_content_service = _LessonPlanContentServiceProxy()
