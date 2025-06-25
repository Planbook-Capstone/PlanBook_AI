"""
Service để xuất đề thi ra file DOCX
"""

import logging
import os
import re
import tempfile
from typing import Dict, List, Any, Optional
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from docx.oxml.shared import OxmlElement, qn
from datetime import datetime

logger = logging.getLogger(__name__)


class ExamDocxService:
    """Service để tạo file DOCX cho đề thi"""

    def __init__(self):
        # Sử dụng thư mục tạm thời của hệ thống
        self.temp_dir = tempfile.gettempdir()

    async def create_exam_docx(
        self, exam_data: Dict[str, Any], exam_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tạo file DOCX cho đề thi

        Args:
            exam_data: Dữ liệu đề thi đã tạo
            exam_request: Thông tin request gốc

        Returns:
            Dict chứa thông tin file đã tạo
        """
        try:
            # Tạo document mới
            doc = Document()

            # Thiết lập margins và font
            self._setup_document_style(doc)

            # Tạo header đề thi
            self._create_exam_header(doc, exam_request, exam_data)

            # Tạo thông tin đề thi
            self._create_exam_info(doc, exam_request, exam_data)

            # Tạo câu hỏi
            self._create_questions_section(doc, exam_data.get("questions", []))

            # Tạo đáp án (trang riêng)
            self._create_answer_key(doc, exam_data.get("questions", []))

            # Lưu file với filename an toàn (không có ký tự đặc biệt) trong thư mục tạm thời
            lesson_id_safe = self._sanitize_filename(
                exam_request.get("lesson_id", "unknown")
            )
            filename = (
                f"exam_{lesson_id_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            )
            filepath = os.path.join(self.temp_dir, filename)

            doc.save(filepath)

            return {
                "success": True,
                "filename": filename,
                "filepath": filepath,
                "file_size": os.path.getsize(filepath),
                "created_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating exam DOCX: {e}")
            return {"success": False, "error": str(e)}

    def _sanitize_filename(self, filename: str) -> str:
        """
        Làm sạch filename để tránh lỗi encoding

        Args:
            filename: Tên file gốc

        Returns:
            Tên file đã được làm sạch (chỉ chứa ASCII)
        """
        try:
            # Loại bỏ ký tự đặc biệt và dấu tiếng Việt
            # Chỉ giữ lại chữ cái, số, dấu gạch dưới và gạch ngang
            sanitized = re.sub(r"[^\w\-_]", "_", filename)

            # Loại bỏ nhiều dấu gạch dưới liên tiếp
            sanitized = re.sub(r"_+", "_", sanitized)

            # Loại bỏ dấu gạch dưới ở đầu và cuối
            sanitized = sanitized.strip("_")

            # Nếu kết quả rỗng, dùng default
            if not sanitized:
                sanitized = "lesson"

            return sanitized

        except Exception as e:
            logger.warning(f"Error sanitizing filename '{filename}': {e}")
            return "lesson"

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
            style = doc.styles["Normal"]
            font = style.font
            font.name = "Times New Roman"
            font.size = Pt(12)

        except Exception as e:
            logger.error(f"Error setting up document style: {e}")

    def _create_exam_header(
        self, doc: Document, exam_request: Dict[str, Any], exam_data: Dict[str, Any]
    ):
        """Tạo header cho đề thi"""
        try:
            # Tiêu đề chính
            title = doc.add_heading("ĐỀ KIỂM TRA", level=1)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_run = title.runs[0]
            title_run.font.name = "Times New Roman"
            title_run.font.size = Pt(16)
            title_run.bold = True

            # Môn học và lớp
            subtitle = doc.add_paragraph()
            subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subtitle_run = subtitle.add_run(
                f"Môn: {exam_request.get('mon_hoc', '')} - Lớp {exam_request.get('lop', '')}"
            )
            subtitle_run.font.name = "Times New Roman"
            subtitle_run.font.size = Pt(14)
            subtitle_run.bold = True

            # Thông tin thời gian
            time_info = doc.add_paragraph()
            time_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
            time_run = time_info.add_run(
                f"Thời gian: 45 phút (không kể thời gian phát đề)"
            )
            time_run.font.name = "Times New Roman"
            time_run.font.size = Pt(12)

            # Đường kẻ phân cách
            doc.add_paragraph("─" * 80).alignment = WD_ALIGN_PARAGRAPH.CENTER

        except Exception as e:
            logger.error(f"Error creating exam header: {e}")

    def _create_exam_info(
        self, doc: Document, exam_request: Dict[str, Any], exam_data: Dict[str, Any]
    ):
        """Tạo thông tin đề thi"""
        try:
            # Thông tin học sinh
            info_para = doc.add_paragraph()
            info_para.add_run("Họ và tên: ").bold = True
            info_para.add_run("." * 40)
            info_para.add_run("  Lớp: ").bold = True
            info_para.add_run("." * 15)

            # Thống kê đề thi
            stats = exam_data.get("statistics", {})
            stats_para = doc.add_paragraph()
            stats_run = stats_para.add_run(
                f"Đề thi gồm {stats.get('tong_so_cau', 0)} câu hỏi: "
            )
            stats_run.font.name = "Times New Roman"
            stats_run.font.size = Pt(11)

            # Chi tiết phân bố câu hỏi
            phan_bo = stats.get("phan_bo_theo_loai", {})
            phan_bo_text = []
            for loai, so_luong in phan_bo.items():
                loai_name = self._get_question_type_name(loai)
                phan_bo_text.append(f"{loai_name}: {so_luong} câu")

            if phan_bo_text:
                stats_para.add_run("; ".join(phan_bo_text))

            doc.add_paragraph()  # Khoảng trống

        except Exception as e:
            logger.error(f"Error creating exam info: {e}")

    def _create_questions_section(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo phần câu hỏi"""
        try:
            # Tiêu đề phần câu hỏi
            questions_title = doc.add_heading("PHẦN I: CÂU HỎI", level=2)
            questions_title_run = questions_title.runs[0]
            questions_title_run.font.name = "Times New Roman"
            questions_title_run.font.size = Pt(14)

            # Tạo từng câu hỏi
            for i, question in enumerate(questions, 1):
                self._create_single_question(doc, question, i)

        except Exception as e:
            logger.error(f"Error creating questions section: {e}")

    def _create_single_question(
        self, doc: Document, question: Dict[str, Any], question_num: int
    ):
        """Tạo một câu hỏi"""
        try:
            loai_cau = question.get("loai_cau", "")

            # Số thứ tự và nội dung câu hỏi
            q_para = doc.add_paragraph()
            q_para.add_run(f"Câu {question_num}: ").bold = True
            q_para.add_run(question.get("noi_dung_cau_hoi", ""))

            # Tạo đáp án theo loại câu hỏi
            if loai_cau == "TN":
                self._create_multiple_choice_answers(doc, question.get("dap_an", {}))
            elif loai_cau == "DT":
                self._create_fill_blank_answer(doc)
            elif loai_cau == "DS":
                self._create_true_false_answers(doc, question.get("dap_an", {}))
            elif loai_cau == "TL":
                self._create_essay_answer_space(doc)

            doc.add_paragraph()  # Khoảng trống giữa các câu

        except Exception as e:
            logger.error(f"Error creating single question: {e}")

    def _create_multiple_choice_answers(self, doc: Document, dap_an: Dict[str, Any]):
        """Tạo đáp án trắc nghiệm"""
        try:
            options = ["A", "B", "C", "D"]
            for option in options:
                if option in dap_an:
                    option_para = doc.add_paragraph()
                    option_para.add_run(f"   {option}. ").bold = True
                    option_para.add_run(dap_an[option])

        except Exception as e:
            logger.error(f"Error creating multiple choice answers: {e}")

    def _create_fill_blank_answer(self, doc: Document):
        """Tạo chỗ trống cho câu điền từ"""
        try:
            answer_para = doc.add_paragraph()
            answer_para.add_run("   Đáp án: ")
            answer_para.add_run("_" * 30)

        except Exception as e:
            logger.error(f"Error creating fill blank answer: {e}")

    def _create_true_false_answers(self, doc: Document, dap_an: Dict[str, Any]):
        """Tạo đáp án đúng/sai"""
        try:
            options = ["a", "b", "c", "d"]
            for option in options:
                option_para = doc.add_paragraph()
                option_para.add_run(f"   {option}) ").bold = True
                option_para.add_run("Đúng ☐    Sai ☐")

        except Exception as e:
            logger.error(f"Error creating true/false answers: {e}")

    def _create_essay_answer_space(self, doc: Document):
        """Tạo chỗ trống cho câu tự luận"""
        try:
            # Thêm dòng kẻ cho học sinh viết
            for i in range(5):
                line_para = doc.add_paragraph()
                line_para.add_run("_" * 80)

        except Exception as e:
            logger.error(f"Error creating essay answer space: {e}")

    def _create_answer_key(self, doc: Document, questions: List[Dict[str, Any]]):
        """Tạo đáp án (trang riêng)"""
        try:
            # Ngắt trang
            doc.add_page_break()

            # Tiêu đề đáp án
            answer_title = doc.add_heading("ĐÁP ÁN VÀ HƯỚNG DẪN CHẤM", level=2)
            answer_title_run = answer_title.runs[0]
            answer_title_run.font.name = "Times New Roman"
            answer_title_run.font.size = Pt(14)

            # Tạo bảng đáp án cho câu trắc nghiệm
            tn_questions = [q for q in questions if q.get("loai_cau") == "TN"]
            if tn_questions:
                self._create_answer_table(doc, tn_questions)

            # Đáp án chi tiết cho các loại câu khác
            other_questions = [q for q in questions if q.get("loai_cau") != "TN"]
            if other_questions:
                doc.add_paragraph()
                detail_title = doc.add_heading("Đáp án chi tiết:", level=3)
                detail_title_run = detail_title.runs[0]
                detail_title_run.font.name = "Times New Roman"
                detail_title_run.font.size = Pt(12)

                for question in other_questions:
                    question_num = question.get("stt", 1)  # Sử dụng số thứ tự thực tế từ dữ liệu
                    self._create_detailed_answer(doc, question, question_num)

        except Exception as e:
            logger.error(f"Error creating answer key: {e}")

    def _create_answer_table(self, doc: Document, tn_questions: List[Dict[str, Any]]):
        """Tạo danh sách đáp án trắc nghiệm theo format dọc"""
        try:
            if not tn_questions:
                return

            # Tạo danh sách đáp án theo format dọc
            doc.add_paragraph()
            answer_list_title = doc.add_heading("Đáp án trắc nghiệm:", level=3)
            answer_list_title_run = answer_list_title.runs[0]
            answer_list_title_run.font.name = "Times New Roman"
            answer_list_title_run.font.size = Pt(12)

            # Tạo danh sách đáp án dọc
            for i, question in enumerate(tn_questions, 1):
                dap_an = question.get("dap_an", {})
                correct_answer = dap_an.get("dung", "")

                # Nếu không có đáp án đúng, thử trích xuất từ giải thích
                if not correct_answer:
                    giai_thich = question.get("giai_thich", "")
                    correct_answer = self._extract_correct_answer_from_explanation(giai_thich, dap_an)
                    if correct_answer:
                        logger.info(f"Extracted missing answer for question {i}: {correct_answer}")
                    else:
                        logger.warning(f"No correct answer found for question {i}")
                        correct_answer = "?"  # Placeholder để báo hiệu lỗi

                answer_para = doc.add_paragraph()
                answer_para.add_run(f"{i} ").bold = True
                answer_para.add_run(correct_answer)

            # Thêm giải thích chi tiết cho từng câu
            doc.add_paragraph()
            detail_title = doc.add_heading("Giải thích chi tiết:", level=3)
            detail_title_run = detail_title.runs[0]
            detail_title_run.font.name = "Times New Roman"
            detail_title_run.font.size = Pt(12)

            for i, question in enumerate(tn_questions, 1):
                # Câu hỏi và đáp án đúng
                answer_para = doc.add_paragraph()
                answer_para.add_run(f"Câu {i}: ").bold = True

                dap_an = question.get("dap_an", {})
                correct_answer = dap_an.get("dung", "")
                answer_para.add_run(f"Đáp án: {correct_answer}")

                # Giải thích
                giai_thich = question.get("giai_thich", "")
                if giai_thich:
                    explain_para = doc.add_paragraph()
                    explain_para.add_run("Giải thích: ").italic = True
                    explain_para.add_run(giai_thich)

                doc.add_paragraph()  # Khoảng trống

        except Exception as e:
            logger.error(f"Error creating answer list: {e}")

    def _create_detailed_answer(
        self, doc: Document, question: Dict[str, Any], question_num: int
    ):
        """Tạo đáp án chi tiết"""
        try:
            # Số câu
            answer_para = doc.add_paragraph()
            answer_para.add_run(f"Câu {question_num}: ").bold = True

            # Đáp án theo loại
            loai_cau = question.get("loai_cau", "")
            dap_an = question.get("dap_an", {})

            if loai_cau == "DT":
                answer_para.add_run(dap_an.get("dap_an_chinh", ""))
            elif loai_cau == "DS":
                ds_answers = []
                for key in ["a", "b", "c", "d"]:
                    if key in dap_an:
                        status = "Đúng" if dap_an[key] else "Sai"
                        ds_answers.append(f"{key}) {status}")
                answer_para.add_run("; ".join(ds_answers))
            elif loai_cau == "TL":
                y_chinh = dap_an.get("y_chinh", [])
                if y_chinh:
                    answer_para.add_run("\n".join([f"- {y}" for y in y_chinh]))

            # Giải thích
            giai_thich = question.get("giai_thich", "")
            if giai_thich:
                explain_para = doc.add_paragraph()
                explain_para.add_run("Giải thích: ").italic = True
                explain_para.add_run(giai_thich)

            doc.add_paragraph()  # Khoảng trống

        except Exception as e:
            logger.error(f"Error creating detailed answer: {e}")

    def _get_question_type_name(self, loai_cau: str) -> str:
        """Lấy tên đầy đủ của loại câu hỏi"""
        names = {
            "TN": "Trắc nghiệm",
            "DT": "Điền từ",
            "DS": "Đúng/Sai",
            "TL": "Tự luận",
        }
        return names.get(loai_cau, loai_cau)

    def _extract_correct_answer_from_explanation(self, explanation: str, dap_an: dict) -> str:
        """Trích xuất đáp án đúng từ giải thích (copy từ exam_generation_service)"""
        try:
            import re

            if not explanation or not isinstance(dap_an, dict):
                return ""

            explanation_lower = explanation.lower()

            # Tìm các pattern rõ ràng nhất trước
            strong_patterns = [
                r"đáp án ([abcd]) đúng",
                r"đáp án đúng là ([abcd])",
                r"([abcd]) đúng vì",
                r"([abcd]) là đáp án đúng",
                r"([abcd]) đúng",
                r"chọn đáp án ([abcd])",
                r"đáp án:\s*([abcd])",
                r"đáp án\s+([abcd])"
            ]

            for pattern in strong_patterns:
                match = re.search(pattern, explanation_lower)
                if match:
                    answer = match.group(1).upper()
                    if answer in dap_an:
                        return answer

            # Tìm pattern yếu hơn
            weak_patterns = [
                r"đáp án ([abcd])",
                r"chọn ([abcd])",
                r"([abcd])\s*[:\-\.]",
                r"^([abcd])\s"
            ]

            for pattern in weak_patterns:
                match = re.search(pattern, explanation_lower)
                if match:
                    answer = match.group(1).upper()
                    if answer in dap_an:
                        return answer

            # Phân tích nội dung đáp án
            option_scores = {}
            for option, content in dap_an.items():
                if option in ['A', 'B', 'C', 'D'] and isinstance(content, str):
                    content_words = content.lower().split()
                    score = 0
                    for word in content_words:
                        if len(word) > 2 and word in explanation_lower:
                            score += 1
                    option_scores[option] = score

            if option_scores:
                best_option = max(option_scores.keys(), key=lambda x: option_scores[x])
                if option_scores[best_option] > 0:
                    return best_option

            return ""

        except Exception as e:
            logger.error(f"Error extracting correct answer: {e}")
            return ""


# Tạo instance global
exam_docx_service = ExamDocxService()
