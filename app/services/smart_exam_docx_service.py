"""
Service tạo file DOCX cho đề thi thông minh theo chuẩn THPT 2025
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.shared import OxmlElement, qn

from app.models.smart_exam_models import SmartExamRequest

logger = logging.getLogger(__name__)


class SmartExamDocxService:
    """Service tạo file DOCX cho đề thi thông minh theo chuẩn THPT 2025"""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "smart_exams"
        self.temp_dir.mkdir(exist_ok=True)

    async def create_smart_exam_docx(
        self, exam_data: Dict[str, Any], exam_request: Union[SmartExamRequest, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Tạo file DOCX cho đề thi thông minh theo chuẩn THPT 2025

        Args:
            exam_data: Dữ liệu đề thi đã được tạo
            exam_request: Request gốc chứa thông tin đề thi

        Returns:
            Dict chứa thông tin file đã tạo
        """
        try:
            # Tạo document mới
            doc = Document()

            # Thiết lập style
            self._setup_document_style(doc)

            # Tạo trang bìa
            self._create_cover_page(doc, exam_request, exam_data)

            # Tạo thông tin đề thi
            self._create_exam_info(doc, exam_request, exam_data)

            # Tạo bảng hóa trị cho môn Hóa học
            subject = self._get_field(exam_request, "subject", "")

            # Kiểm tra cả "hóa" và "hoa" để đảm bảo
            if "hóa" in subject.lower() or "hoa" in subject.lower():
                self._create_chemistry_valence_table(doc, exam_data.get("questions", []))

            # Tạo nội dung đề thi theo 3 phần
            self._create_exam_content_by_parts(doc, exam_data.get("questions", []))

            # Thêm chữ "Hết"
            self._add_exam_ending(doc)

            # Tạo đáp án theo chuẩn THPT 2025
            self._create_thpt_2025_answer_section(doc, exam_data.get("questions", []))

            # Lưu file
            filename = self._generate_filename(exam_request)
            filepath = self.temp_dir / filename
            doc.save(str(filepath))

            logger.info(f"Created smart exam DOCX: {filepath}")
            return {
                "success": True,
                "file_path": str(filepath),
                "filename": filename,
                "file_size": os.path.getsize(filepath)
            }

        except Exception as e:
            logger.error(f"Error creating smart exam DOCX: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _get_field(self, exam_request: Union[SmartExamRequest, Dict[str, Any]], field: str, default: Any = None) -> Any:
        """Helper method to get field from either Pydantic model or dict"""
        if isinstance(exam_request, dict):
            return exam_request.get(field, default)
        else:
            return getattr(exam_request, field, default)

    def _setup_document_style(self, doc: Document):
        """Thiết lập style cho document"""
        try:
            # Thiết lập margins
            sections = doc.sections
            for section in sections:
                section.top_margin = Inches(0.8)
                section.bottom_margin = Inches(0.8)
                section.left_margin = Inches(0.8)
                section.right_margin = Inches(0.8)

            # Thiết lập font mặc định
            style = doc.styles['Normal']
            font = style.font
            font.name = 'Times New Roman'
            font.size = Pt(11)

        except Exception as e:
            logger.error(f"Error setting up document style: {e}")

    def _create_cover_page(self, doc: Document, exam_request: Union[SmartExamRequest, Dict[str, Any]], exam_data: Dict[str, Any]):
        """Tạo trang bìa theo chuẩn THPT 2025"""
        try:
            # Header với logo và thông tin trường
            header_table = doc.add_table(rows=1, cols=2)

            # Cột trái - Logo và thông tin bộ
            left_cell = header_table.cell(0, 0)
            left_para = left_cell.paragraphs[0]
            left_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            left_para.add_run("BỘ GIÁO DỤC VÀ ĐÀO TẠO").bold = True
            left_para.add_run("\n")
            left_para.add_run(self._get_field(exam_request, "school", "TRƯỜNG THPT ABC")).bold = True

            # Cột phải - Thông tin đề thi
            right_cell = header_table.cell(0, 1)
            right_para = right_cell.paragraphs[0]
            right_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            right_para.add_run("ĐỀ KIỂM TRA LỚP 10").bold = True
            right_para.add_run(f"\nMôn: {self._get_field(exam_request, 'subject', 'HÓA HỌC').upper()}")
            right_para.add_run(f"\nThời gian làm bài: {self._get_field(exam_request, 'duration', 50)} phút, không kể thời gian phát đề")

            # Khoảng trắng
            doc.add_paragraph()

            # Thông tin thí sinh
            doc.add_paragraph()
            info_para = doc.add_paragraph()
            info_para.add_run("Họ, tên thí sinh: ").bold = True
            info_para.add_run("." * 50)
            
            info_para2 = doc.add_paragraph()
            info_para2.add_run("Số báo danh: ").bold = True
            info_para2.add_run("." * 50)

            # Thống kê đề thi
            doc.add_paragraph()
            statistics = exam_data.get("statistics", {})
            if hasattr(statistics, 'total_questions'):
                total_questions = statistics.total_questions
            elif isinstance(statistics, dict):
                total_questions = statistics.get("total_questions", 0)
            else:
                total_questions = 0
            
            stats_para = doc.add_paragraph()
            stats_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            stats_para.add_run(f"(Đề thi gồm {total_questions} câu)")


        except Exception as e:
            logger.error(f"Error creating cover page: {e}")

    def _create_exam_info(self, doc: Document, exam_request: Union[SmartExamRequest, Dict[str, Any]], exam_data: Dict[str, Any]):
        """Tạo thông tin đề thi"""
        try:
            # Thông tin cơ bản
            info_para = doc.add_paragraph()
            info_para.add_run("Họ và tên thí sinh: ").bold = True
            info_para.add_run("." * 40)
            info_para.add_run("  Số báo danh: ").bold = True
            info_para.add_run("." * 20)

        except Exception as e:
            logger.error(f"Error creating exam info: {e}")

    def _create_chemistry_valence_table(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo bảng nguyên tử khối cho môn Hóa học dựa trên nội dung đề thi"""
        try:
            # Phân tích nội dung đề thi để tìm các nguyên tố
            used_elements = self._extract_chemical_elements_from_questions(questions)

            if not used_elements:
                # Nếu không tìm thấy nguyên tố nào, sử dụng các nguyên tố phổ biến cho hóa học THPT
                used_elements = ["H", "C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Fe", "Cu", "Zn", "Br", "I", "Ag", "Ba"]

            # Tiêu đề
            valence_title = doc.add_paragraph()
            valence_title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            valence_run = valence_title.add_run("BẢNG NGUYÊN TỬ KHỐI CỦA CÁC NGUYÊN TỐ HÓA HỌC")
            valence_run.bold = True

            # Tạo bảng nguyên tử khối chỉ cho các nguyên tố được sử dụng
            atomic_masses_text = self._get_atomic_masses_for_elements(used_elements)

            valence_para = doc.add_paragraph()
            valence_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            valence_para.add_run(atomic_masses_text)

            # Thêm lưu ý
            # note_para = doc.add_paragraph()
            # note_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            # if len(used_elements) <= 20:  # Nếu ít nguyên tố thì ghi chú là từ đề thi
            #     # note_run = note_para.add_run("(Chỉ ghi các nguyên tố có trong đề thi)")
            #     print("DEBUG: Not enough elements to create custom valence table")
            # else:  # Nếu nhiều nguyên tố thì ghi chú là nguyên tố phổ biến
            #     note_run = note_para.add_run("(Các nguyên tố hóa học phổ biến)")
            # note_run.italic = True
            # note_run.font.size = Pt(10)

            doc.add_paragraph()

        except Exception as e:
            logger.error(f"Error creating chemistry valence table: {e}")

    def _extract_chemical_elements_from_questions(self, questions: List[Dict[str, Any]]) -> List[str]:
        """Trích xuất các nguyên tố hóa học từ nội dung câu hỏi"""
        import re

        # Danh sách các nguyên tố hóa học phổ biến
        all_elements = {
            "H": 1, "He": 4, "Li": 7, "Be": 9, "B": 11, "C": 12, "N": 14, "O": 16,
            "F": 19, "Ne": 20, "Na": 23, "Mg": 24, "Al": 27, "Si": 28, "P": 31, "S": 32,
            "Cl": 35.5, "Ar": 40, "K": 39, "Ca": 40, "Sc": 45, "Ti": 48, "V": 51, "Cr": 52,
            "Mn": 55, "Fe": 56, "Co": 59, "Ni": 59, "Cu": 64, "Zn": 65, "Ga": 70, "Ge": 73,
            "As": 75, "Se": 79, "Br": 80, "Kr": 84, "Rb": 85, "Sr": 88, "Y": 89, "Zr": 91,
            "Nb": 93, "Mo": 96, "Tc": 98, "Ru": 101, "Rh": 103, "Pd": 106, "Ag": 108, "Cd": 112,
            "In": 115, "Sn": 119, "Sb": 122, "Te": 128, "I": 127, "Xe": 131, "Cs": 133, "Ba": 137,
            "La": 139, "Ce": 140, "Pr": 141, "Nd": 144, "Pm": 145, "Sm": 150, "Eu": 152, "Gd": 157,
            "Tb": 159, "Dy": 163, "Ho": 165, "Er": 167, "Tm": 169, "Yb": 173, "Lu": 175, "Hf": 178,
            "Ta": 181, "W": 184, "Re": 186, "Os": 190, "Ir": 192, "Pt": 195, "Au": 197, "Hg": 201,
            "Tl": 204, "Pb": 207, "Bi": 209, "Po": 209, "At": 210, "Rn": 222, "Fr": 223, "Ra": 226,
            "Ac": 227, "Th": 232, "Pa": 231, "U": 238, "Np": 237, "Pu": 244, "Am": 243, "Cm": 247,
            "Bk": 247, "Cf": 251, "Es": 252, "Fm": 257, "Md": 258, "No": 259, "Lr": 262
        }

        found_elements = set()

        # Tìm kiếm trong tất cả nội dung câu hỏi
        for i, question in enumerate(questions):
            question_text = str(question.get("question", ""))
            answer_text = str(question.get("answer", ""))
            explanation_text = str(question.get("explanation", ""))

            # Kết hợp tất cả text
            full_text = f"{question_text} {answer_text} {explanation_text}"

            # Tìm các ký hiệu nguyên tố với pattern chính xác hơn
            # Tìm các pattern như: H2O, NaCl, CaCO3, Fe2O3, etc.
            element_patterns = [
                r'\b([A-Z][a-z]?)(?:\d+)?',  # H, H2, Na, Cl, etc.
                r'([A-Z][a-z]?)(?=\d)',      # H trong H2O, Na trong NaCl
                r'([A-Z][a-z]?)(?=[A-Z])',   # Na trong NaCl, Ca trong CaCO3
                r'\b([A-Z][a-z]?)\b'         # Standalone elements
            ]

            question_elements = set()
            for pattern in element_patterns:
                matches = re.findall(pattern, full_text)
                for match in matches:
                    if match in all_elements:
                        found_elements.add(match)
                        question_elements.add(match)
        return sorted(list(found_elements))

    def _get_atomic_masses_for_elements(self, elements: List[str]) -> str:
        """Tạo chuỗi nguyên tử khối cho các nguyên tố được chỉ định"""
        atomic_masses = {
            "H": 1, "He": 4, "Li": 7, "Be": 9, "B": 11, "C": 12, "N": 14, "O": 16,
            "F": 19, "Ne": 20, "Na": 23, "Mg": 24, "Al": 27, "Si": 28, "P": 31, "S": 32,
            "Cl": 35.5, "Ar": 40, "K": 39, "Ca": 40, "Sc": 45, "Ti": 48, "V": 51, "Cr": 52,
            "Mn": 55, "Fe": 56, "Co": 59, "Ni": 59, "Cu": 64, "Zn": 65, "Ga": 70, "Ge": 73,
            "As": 75, "Se": 79, "Br": 80, "Kr": 84, "Rb": 85, "Sr": 88, "Y": 89, "Zr": 91,
            "Nb": 93, "Mo": 96, "Tc": 98, "Ru": 101, "Rh": 103, "Pd": 106, "Ag": 108, "Cd": 112,
            "In": 115, "Sn": 119, "Sb": 122, "Te": 128, "I": 127, "Xe": 131, "Cs": 133, "Ba": 137,
            "La": 139, "Ce": 140, "Pr": 141, "Nd": 144, "Pm": 145, "Sm": 150, "Eu": 152, "Gd": 157,
            "Tb": 159, "Dy": 163, "Ho": 165, "Er": 167, "Tm": 169, "Yb": 173, "Lu": 175, "Hf": 178,
            "Ta": 181, "W": 184, "Re": 186, "Os": 190, "Ir": 192, "Pt": 195, "Au": 197, "Hg": 201,
            "Tl": 204, "Pb": 207, "Bi": 209, "Po": 209, "At": 210, "Rn": 222, "Fr": 223, "Ra": 226,
            "Ac": 227, "Th": 232, "Pa": 231, "U": 238, "Np": 237, "Pu": 244, "Am": 243, "Cm": 247,
            "Bk": 247, "Cf": 251, "Es": 252, "Fm": 257, "Md": 258, "No": 259, "Lr": 262
        }

        mass_strings = []
        for element in elements:
            if element in atomic_masses:
                mass = atomic_masses[element]
                if mass == int(mass):
                    mass_strings.append(f"{element} = {int(mass)}")
                else:
                    mass_strings.append(f"{element} = {mass}")

        return "; ".join(mass_strings)

    def _create_exam_content_by_parts(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo nội dung đề thi theo 3 phần chuẩn THPT 2025"""
        try:
            print(f"DEBUG DOCX: Total questions received: {len(questions)}")
            for i, q in enumerate(questions[:3]):  # Show first 3 questions
                print(f"DEBUG DOCX: Question {i+1} keys: {list(q.keys())}")
                print(f"DEBUG DOCX: Question {i+1} part: {q.get('part', 'NO_PART')}")

            # Phân loại câu hỏi theo phần
            part_1_questions = [q for q in questions if q.get("part") == 1]
            part_2_questions = [q for q in questions if q.get("part") == 2]
            part_3_questions = [q for q in questions if q.get("part") == 3]

            print(f"DEBUG DOCX: Part 1 questions: {len(part_1_questions)}")
            print(f"DEBUG DOCX: Part 2 questions: {len(part_2_questions)}")
            print(f"DEBUG DOCX: Part 3 questions: {len(part_3_questions)}")

            # PHẦN I: Câu trắc nghiệm nhiều phương án lựa chọn
            if part_1_questions:
                self._create_part_1_section(doc, part_1_questions)

            # PHẦN II: Câu trắc nghiệm đúng sai
            if part_2_questions:
                self._create_part_2_section(doc, part_2_questions)

            # PHẦN III: Câu trắc nghiệm trả lời ngắn
            if part_3_questions:
                self._create_part_3_section(doc, part_3_questions)

        except Exception as e:
            logger.error(f"Error creating exam content by parts: {e}")

    def _create_part_1_section(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo PHẦN I: Câu trắc nghiệm nhiều phương án lựa chọn"""
        try:
            # Tiêu đề phần
            part_title = doc.add_paragraph()
            part_title_run = part_title.add_run("PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn. ")
            part_title_run.bold = True
            part_title.add_run(f"Thí sinh trả lời từ câu 1 đến câu {len(questions)}.")

            note_para = doc.add_paragraph()
            note_para.add_run("(Mỗi câu trả lời đúng thí sinh được 0,25 điểm)")

            # Tạo câu hỏi
            for i, question in enumerate(questions, 1):
                self._create_multiple_choice_question(doc, question, i)

        except Exception as e:
            logger.error(f"Error creating part 1 section: {e}")

    def _create_part_2_section(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo PHẦN II: Câu trắc nghiệm đúng sai"""
        try:
            # Tiêu đề phần
            part_title = doc.add_paragraph()
            part_title_run = part_title.add_run("PHẦN II. Câu trắc nghiệm đúng sai. ")
            part_title_run.bold = True
            part_title.add_run(f"Thí sinh trả lời từ câu 1 đến câu {len(questions)}. Trong mỗi ý a), b), c), d) ở mỗi câu, thí sinh chọn đúng hoặc sai.")

            # Hướng dẫn chấm điểm
   

            # Tạo câu hỏi
            for i, question in enumerate(questions, 1):
                self._create_true_false_question_with_statements(doc, question, i)

        except Exception as e:
            logger.error(f"Error creating part 2 section: {e}")

    def _create_part_3_section(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo PHẦN III: Câu trắc nghiệm trả lời ngắn"""
        try:
            # Tiêu đề phần
            part_title = doc.add_paragraph()
            part_title_run = part_title.add_run("PHẦN III. Câu trắc nghiệm trả lời ngắn. ")
            part_title_run.bold = True
            part_title.add_run(f"Thí sinh trả lời từ câu 1 đến câu {len(questions)}")

            note_para = doc.add_paragraph()
            note_para.add_run("Mỗi câu trả lời đúng thí sinh được 0,25 điểm.")

            doc.add_paragraph()

            # Tạo câu hỏi
            for i, question in enumerate(questions, 1):
                self._create_short_answer_question(doc, question, i)

        except Exception as e:
            logger.error(f"Error creating part 3 section: {e}")

    def _create_multiple_choice_question(self, doc: Document, question: Dict[str, Any], question_num: int):
        """Tạo câu hỏi trắc nghiệm nhiều phương án"""
        try:
            # Câu hỏi
            q_para = doc.add_paragraph()
            q_para.add_run(f"Câu {question_num}. ").bold = True
            # Sử dụng field "question" thay vì "cau_hoi"
            q_para.add_run(question.get("question", question.get("cau_hoi", "")))

            # Các phương án
            # Sử dụng field "answer" thay vì "dap_an"
            dap_an = question.get("answer", question.get("dap_an", {}))
            for option in ["A", "B", "C", "D"]:
                if option in dap_an:
                    option_para = doc.add_paragraph()
                    option_para.add_run(f"{option}. {dap_an[option]}")

            doc.add_paragraph()

        except Exception as e:
            logger.error(f"Error creating multiple choice question: {e}")

    def _create_true_false_question(self, doc: Document, question: Dict[str, Any], question_num: int):
        """Tạo câu hỏi đúng sai"""
        try:
            # Câu hỏi chính
            q_para = doc.add_paragraph()
            q_para.add_run(f"Câu {question_num}. ").bold = True
            # Sử dụng field "question" thay vì "cau_hoi"
            q_para.add_run(question.get("question", question.get("cau_hoi", "")))

            # Các ý a, b, c, d
            # Sử dụng field "answer" thay vì "dap_an"
            dap_an = question.get("answer", question.get("dap_an", {}))
            for option in ["a", "b", "c", "d"]:
                if option in dap_an:
                    option_para = doc.add_paragraph()
                    option_para.add_run(f"{option}) {dap_an[option]}")

            doc.add_paragraph()

        except Exception as e:
            logger.error(f"Error creating true false question: {e}")

    def _create_true_false_question_with_statements(self, doc: Document, question: Dict[str, Any], question_num: int):
        """Tạo câu hỏi đúng sai với các statement a), b), c), d) theo mẫu THPT"""
        try:
            # Câu hỏi chính
            q_para = doc.add_paragraph()
            q_para.add_run(f"Câu {question_num}. ").bold = True
            # Sử dụng field "question" thay vì "cau_hoi"
            main_question = question.get("question", question.get("cau_hoi", ""))
            q_para.add_run(main_question)

            # Các statement a), b), c), d) - lấy từ explanation hoặc tạo từ answer
            answer_data = question.get("answer", question.get("dap_an", {}))
            explanation = question.get("explanation", "")

            # Tạo các statement từ explanation hoặc answer
            statements = self._extract_statements_from_question(answer_data, explanation)

            for option in ["a", "b", "c", "d"]:
                if option in statements:
                    option_para = doc.add_paragraph()
                    option_para.add_run(f"{option}) {statements[option]}")

            doc.add_paragraph()

        except Exception as e:
            logger.error(f"Error creating true false question with statements: {e}")

    def _extract_statements_from_question(self, answer_data: Dict[str, Any], explanation: str) -> Dict[str, str]:
        """Trích xuất các statement từ dữ liệu câu hỏi"""
        try:
            statements = {}

            # Nếu answer_data đã có các statement a, b, c, d
            for option in ["a", "b", "c", "d"]:
                if option in answer_data:
                    option_data = answer_data[option]

                    # Xử lý cấu trúc mới với content và evaluation
                    if isinstance(option_data, dict) and "content" in option_data:
                        statements[option] = option_data["content"]
                    elif isinstance(option_data, str):
                        # Nếu là đáp án Đúng/Sai, tạo statement mặc định
                        if option_data in ["Đúng", "Sai"]:
                            statements[option] = f"Statement {option.upper()}"
                        else:
                            # Nếu là statement thực tế (cấu trúc cũ)
                            statements[option] = option_data

            # Nếu không có statement, tạo mặc định
            if not statements:
                statements = {
                    "a": "Statement A",
                    "b": "Statement B",
                    "c": "Statement C",
                    "d": "Statement D"
                }

            return statements

        except Exception as e:
            logger.error(f"Error extracting statements: {e}")
            return {"a": "Statement A", "b": "Statement B", "c": "Statement C", "d": "Statement D"}

    def _create_short_answer_question(self, doc: Document, question: Dict[str, Any], question_num: int):
        """Tạo câu hỏi trả lời ngắn"""
        try:
            # Câu hỏi
            q_para = doc.add_paragraph()
            q_para.add_run(f"Câu {question_num}. ").bold = True
            # Sử dụng field "question" thay vì "cau_hoi"
            q_para.add_run(question.get("question", question.get("cau_hoi", "")))

            doc.add_paragraph()

        except Exception as e:
            logger.error(f"Error creating short answer question: {e}")

    def _add_exam_ending(self, doc: Document):
        """Thêm chữ 'Hết' vào cuối đề thi"""
        try:
            doc.add_paragraph()
            ending_para = doc.add_paragraph()
            ending_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            ending_run = ending_para.add_run("----- HẾT -----")
            ending_run.bold = True
            ending_run.font.size = Pt(14)
            doc.add_page_break()
        except Exception as e:
            logger.error(f"Error adding exam ending: {e}")

    def _create_thpt_2025_answer_section(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo phần đáp án theo chuẩn THPT 2025"""
        try:
            doc.add_paragraph()
            doc.add_paragraph()

            # Tiêu đề đáp án
            title_para = doc.add_paragraph()
            title_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            title_run = title_para.add_run("ĐÁP ÁN")
            title_run.bold = True
            title_run.font.size = Pt(14)

            doc.add_paragraph()

            # Phân loại câu hỏi theo phần
            part_1_questions = [q for q in questions if q.get("part") == 1]
            part_2_questions = [q for q in questions if q.get("part") == 2]
            part_3_questions = [q for q in questions if q.get("part") == 3]

            # Tạo đáp án cho từng phần
            if part_1_questions:
                self._create_part_1_answer_table(doc, part_1_questions)

            if part_2_questions:
                self._create_part_2_answer_table(doc, part_2_questions)

            if part_3_questions:
                self._create_part_3_answer_table(doc, part_3_questions)

        except Exception as e:
            logger.error(f"Error creating THPT 2025 answer section: {e}")

    def _create_part_1_answer_table(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo bảng đáp án Phần I"""
        try:
            # Tiêu đề
            section_para = doc.add_paragraph()
            section_run = section_para.add_run("PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn. ")
            section_run.bold = True
            section_para.add_run(f"Thí sinh trả lời từ câu 1 đến câu {len(questions)}")

            note_para = doc.add_paragraph()
            note_para.add_run("(Mỗi câu trả lời đúng thí sinh được 0,25 điểm)")

            # Tạo bảng đáp án
            cols_per_row = 10
            questions_per_row = 9
            num_rows = (len(questions) + questions_per_row - 1) // questions_per_row

            table = doc.add_table(rows=num_rows * 2, cols=cols_per_row)
            table.style = 'Table Grid'

            # Điền dữ liệu
            for row_idx in range(num_rows):
                header_row = row_idx * 2
                answer_row = row_idx * 2 + 1

                table.cell(header_row, 0).text = "Câu"
                table.cell(answer_row, 0).text = "Chọn"

                for col_idx in range(1, cols_per_row):
                    question_idx = row_idx * questions_per_row + col_idx - 1
                    if question_idx < len(questions):
                        table.cell(header_row, col_idx).text = str(question_idx + 1)
                        
                        # Lấy đáp án đúng
                        dap_an = questions[question_idx].get("dap_an", {})
                        correct_answer = dap_an.get("dung", "A")
                        table.cell(answer_row, col_idx).text = correct_answer

        except Exception as e:
            logger.error(f"Error creating part 1 answer table: {e}")

    def _create_part_2_answer_table(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo bảng đáp án Phần II theo chuẩn THPT - format gộp a,b,c,d trong 1 cột"""
        try:
            doc.add_paragraph()

            section_para = doc.add_paragraph()
            section_run = section_para.add_run("PHẦN II. Câu trắc nghiệm đúng sai. ")
            section_run.bold = True
            section_para.add_run(f"Thí sinh trả lời từ câu 1 đến câu {len(questions)}.")
            note_para = doc.add_paragraph()
            note_para.add_run("- Thí sinh chỉ lựa chọn chính xác 01 ý trong 01 câu hỏi được 0,1 điểm;")
            note_para.add_run("\n- Thí sinh chỉ lựa chọn chính xác 02 ý trong 01 câu hỏi được 0,25 điểm;")
            note_para.add_run("\n- Thí sinh chỉ lựa chọn chính xác 03 ý trong 01 câu hỏi được 0,5 điểm;")
            note_para.add_run("\n- Thí sinh lựa chọn chính xác cả 04 ý trong 01 câu hỏi được 1 điểm.")    
            doc.add_paragraph()

            # Tạo bảng đáp án với format gộp: 1 hàng cho header, 1 hàng cho đáp án
            num_questions = len(questions)
            table = doc.add_table(rows=2, cols=num_questions + 1)
            table.style = 'Table Grid'

            # Header row
            table.cell(0, 0).text = "Câu"
            for i in range(num_questions):
                table.cell(0, i + 1).text = str(i + 1)

            # Đáp án row - gộp tất cả a,b,c,d trong 1 cell
            table.cell(1, 0).text = "Đáp án"

            for i, question in enumerate(questions):
                dap_an = question.get("answer", question.get("dap_an", {}))

                # Tạo text gộp cho a,b,c,d với đánh giá Đúng/Sai
                answer_lines = []
                for option in ["a", "b", "c", "d"]:
                    option_data = dap_an.get(option, {})

                    # Xử lý cấu trúc mới với content và evaluation
                    if isinstance(option_data, dict) and "evaluation" in option_data:
                        evaluation = option_data.get("evaluation", "Đúng")
                    else:
                        # Fallback cho cấu trúc cũ (chỉ có nội dung phát biểu)
                        evaluation = "Đúng"  # Default value

                    # Chỉ hiển thị Đúng/Sai, không hiển thị nội dung phát biểu
                    answer_lines.append(f"{option}) {evaluation}")

                # Gộp tất cả đáp án trong 1 cell
                combined_answer = "\n".join(answer_lines)
                table.cell(1, i + 1).text = combined_answer

        except Exception as e:
            logger.error(f"Error creating part 2 answer table: {e}")

    def _create_part_3_answer_table(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo bảng đáp án Phần III"""
        try:
            doc.add_paragraph()
            
            section_para = doc.add_paragraph()
            section_run = section_para.add_run("PHẦN III. Câu trắc nghiệm trả lời ngắn. ")
            section_run.bold = True
            section_para.add_run(f"Thí sinh trả lời từ câu 1 đến câu {len(questions)}")

            note_para = doc.add_paragraph()
            note_para.add_run("Mỗi câu trả lời đúng thí sinh được 0,25 điểm.")

            # Tạo bảng đáp án
            table = doc.add_table(rows=2, cols=len(questions) + 1)
            table.style = 'Table Grid'

            # Header
            table.cell(0, 0).text = "Câu"
            table.cell(1, 0).text = "Đáp án"

            # Đáp án
            for i, question in enumerate(questions):
                table.cell(0, i + 1).text = str(i + 1)
                # Sử dụng field "answer" thay vì "dap_an"
                dap_an = question.get("answer", question.get("dap_an", {}))
                # Cho Part 3, đáp án có thể ở field "dap_an" trong answer object
                answer = dap_an.get("dap_an", dap_an.get("answer", ""))
                table.cell(1, i + 1).text = str(answer)

        except Exception as e:
            logger.error(f"Error creating part 3 answer table: {e}")

    def _generate_filename(self, exam_request: Union[SmartExamRequest, Dict[str, Any]]) -> str:
        """Tạo tên file"""
        try:
            subject = self._get_field(exam_request, "subject", "exam")
            grade = self._get_field(exam_request, "grade", "")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Làm sạch tên file
            safe_subject = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            return f"smart_exam_{safe_subject}_lop{grade}_{timestamp}.docx"
            
        except Exception as e:
            logger.error(f"Error generating filename: {e}")
            return f"smart_exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"


# Singleton instance
smart_exam_docx_service = SmartExamDocxService()
