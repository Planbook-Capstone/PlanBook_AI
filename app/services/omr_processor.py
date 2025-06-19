import cv2
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class OMRProcessor:
    """Xử lý OMR (Optical Mark Recognition) cho phiếu trả lời trắc nghiệm"""

    def __init__(self):
        # Cấu hình các tham số cho việc nhận dạng
        self.bubble_threshold = 0.6  # Ngưỡng để xác định bubble được tô
        self.min_contour_area = 50  # Diện tích tối thiểu của contour

    async def process_image(self, image_content: bytes) -> Tuple[Dict, Dict[int, str]]:
        """
        Xử lý ảnh phiếu trả lời và trích xuất thông tin
        """
        try:
            # Chuyển đổi bytes thành image
            nparr = np.frombuffer(image_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise Exception("Cannot decode image")

            # Tiền xử lý ảnh
            processed_image = self.preprocess_image(image)

            # Phát hiện và căn chỉnh ảnh
            aligned_image = self.align_image(processed_image)

            # Trích xuất thông tin sinh viên
            student_info = self.extract_student_info(aligned_image)

            # Trích xuất câu trả lời
            answers = self.extract_answers(aligned_image)

            return student_info, answers

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise Exception(f"Image processing failed: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh"""
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Khử nhiễu
        denoised = cv2.medianBlur(gray, 3)

        # Tăng cường độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Nhị phân hóa
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def align_image(self, image: np.ndarray) -> np.ndarray:
        """Căn chỉnh ảnh dựa vào các điểm đánh dấu"""
        # Tìm các hình vuông đen (markers) ở góc
        contours, _ = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc các contour có dạng hình vuông
        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Kích thước tối thiểu của marker
                # Xấp xỉ contour thành đa giác
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # Hình chữ nhật
                    markers.append(cv2.boundingRect(contour))

        # Nếu tìm thấy đủ markers, thực hiện căn chỉnh
        if len(markers) >= 3:
            # Sắp xếp markers theo vị trí
            markers.sort(key=lambda x: (x[1], x[0]))  # Sắp xếp theo y rồi x

            # Thực hiện perspective correction (đơn giản hóa)
            # Trong thực tế, cần tính toán ma trận transformation chính xác hơn
            return image

        return image

    def extract_student_info(self, image: np.ndarray) -> Dict:
        """Trích xuất thông tin sinh viên từ phần header"""
        # Vùng thông tin sinh viên (ước lượng dựa trên layout)
        height, width = image.shape
        header_region = image[int(height * 0.05):int(height * 0.25), int(width * 0.3):int(width * 0.9)]

        # Trích xuất mã đề thi từ vùng bên phải
        test_code_region = image[int(height * 0.05):int(height * 0.15), int(width * 0.8):int(width * 0.95)]
        test_code = self.extract_test_code(test_code_region)

        # Trích xuất số báo danh (SBD) từ các bubble số
        sbd_region = image[int(height * 0.05):int(height * 0.25), int(width * 0.05):int(width * 0.3)]
        sbd = self.extract_student_id(sbd_region)

        return {
            "test_code": test_code,
            "student_id": sbd,
            "name": "",  # Có thể mở rộng để OCR tên
            "class": ""  # Có thể mở rộng để OCR lớp
        }

    def extract_test_code(self, region: np.ndarray) -> str:
        """Trích xuất mã đề từ vùng bubble mã đề"""
        # Phân tích các bubble trong vùng mã đề
        # Thông thường mã đề có 4 chữ số, mỗi chữ số có 10 bubble (0-9)

        # Tìm các hàng bubble
        rows = self.find_bubble_rows(region, expected_cols=4, expected_rows=10)

        test_code = ""
        for col in range(4):  # 4 chữ số
            digit = self.get_selected_digit(rows, col)
            test_code += str(digit) if digit != -1 else "0"

        return test_code if test_code else "0000"

    def extract_student_id(self, region: np.ndarray) -> str:
        """Trích xuất số báo danh từ vùng bubble SBD"""
        # Tương tự như mã đề, nhưng có thể có nhiều chữ số hơn
        rows = self.find_bubble_rows(region, expected_cols=8, expected_rows=10)

        sbd = ""
        for col in range(8):  # Giả sử SBD có 8 chữ số
            digit = self.get_selected_digit(rows, col)
            sbd += str(digit) if digit != -1 else "0"

        return sbd if sbd else "00000000"

    def extract_answers(self, image: np.ndarray) -> Dict[int, str]:
        """Trích xuất câu trả lời từ các phần bubble"""
        height, width = image.shape
        answers = {}

        # Phần I: Câu 1-40 (4 cột x 10 câu)
        part1_region = image[int(height * 0.3):int(height * 0.65), int(width * 0.05):int(width * 0.95)]
        part1_answers = self.extract_part_answers(part1_region, start_question=1, total_questions=40, layout="4x10")
        answers.update(part1_answers)

        # Phần II: Câu tự luận dạng trắc nghiệm
        part2_region = image[int(height * 0.65):int(height * 0.8), int(width * 0.05):int(width * 0.95)]
        part2_answers = self.extract_part2_answers(part2_region)
        answers.update(part2_answers)

        return answers

    def extract_part_answers(self, region: np.ndarray, start_question: int, total_questions: int, layout: str) -> Dict[
        int, str]:
        """Trích xuất câu trả lời từ một phần cụ thể"""
        answers = {}

        if layout == "4x10":  # 4 cột, mỗi cột 10 câu
            # Chia region thành 4 cột
            height, width = region.shape
            col_width = width // 4

            for col in range(4):
                col_region = region[:, col * col_width:(col + 1) * col_width]

                # Tìm 10 hàng câu hỏi trong cột này
                question_rows = self.find_question_rows(col_region, 10)

                for row, (y_start, y_end) in enumerate(question_rows):
                    question_num = start_question + col * 10 + row
                    if question_num > total_questions:
                        break

                    # Trích xuất câu trả lời từ hàng này
                    row_region = col_region[y_start:y_end, :]
                    answer = self.get_selected_answer(row_region)
                    answers[question_num] = answer

        return answers

    def extract_part2_answers(self, region: np.ndarray) -> Dict[int, str]:
        """Trích xuất câu trả lời phần II (dạng bảng)"""
        # Phần II thường có layout khác, cần xử lý riêng
        # Tạm thời return empty dict
        return {}

    def find_bubble_rows(self, region: np.ndarray, expected_cols: int, expected_rows: int) -> List[List]:
        """Tìm các hàng bubble trong region"""
        # Tìm contours
        contours, _ = cv2.findContours(255 - region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc các contour có kích thước phù hợp (bubble)
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < 500:  # Kích thước bubble
                x, y, w, h = cv2.boundingRect(contour)
                if 0.7 < w / h < 1.3:  # Tỷ lệ gần hình tròn
                    bubbles.append((x + w // 2, y + h // 2, x, y, w, h))

        # Sắp xếp bubbles thành lưới
        bubbles.sort(key=lambda b: (b[1], b[0]))  # Sắp xếp theo y rồi x

        # Nhóm thành các hàng
        rows = []
        current_row = []
        last_y = -1

        for bubble in bubbles:
            if last_y == -1 or abs(bubble[1] - last_y) < 20:  # Cùng hàng
                current_row.append(bubble)
                last_y = bubble[1]
            else:  # Hàng mới
                if current_row:
                    rows.append(current_row)
                current_row = [bubble]
                last_y = bubble[1]

        if current_row:
            rows.append(current_row)

        return rows

    def find_question_rows(self, region: np.ndarray, expected_rows: int) -> List[Tuple[int, int]]:
        """Tìm các hàng câu hỏi trong region"""
        height, width = region.shape
        row_height = height // expected_rows

        rows = []
        for i in range(expected_rows):
            y_start = i * row_height
            y_end = (i + 1) * row_height
            rows.append((y_start, y_end))

        return rows

    def get_selected_digit(self, rows: List[List], col_index: int) -> int:
        """Lấy chữ số được chọn từ cột bubble"""
        if col_index >= len(rows[0]) if rows else True:
            return -1

        # Kiểm tra từng hàng (0-9) trong cột
        for digit in range(10):
            if digit < len(rows):
                # Kiểm tra bubble tại vị trí [digit][col_index]
                if self.is_bubble_filled(rows[digit], col_index):
                    return digit

        return -1

    def get_selected_answer(self, row_region: np.ndarray) -> str:
        """Lấy câu trả lời được chọn từ một hàng"""
        # Chia hàng thành 4 phần (A, B, C, D)
        height, width = row_region.shape
        option_width = width // 4

        answers = ['A', 'B', 'C', 'D']

        for i, answer in enumerate(answers):
            option_region = row_region[:, i * option_width:(i + 1) * option_width]

            # Kiểm tra xem bubble có được tô không
            if self.is_region_filled(option_region):
                return answer

        return ""  # Không có câu trả lời

    def is_bubble_filled(self, bubble_row: List, bubble_index: int) -> bool:
        """Kiểm tra bubble có được tô không"""
        if bubble_index >= len(bubble_row):
            return False

        # Lấy thông tin bubble
        bubble = bubble_row[bubble_index]
        x, y, w, h = bubble[2], bubble[3], bubble[4], bubble[5]

        # Tính tỷ lệ pixel đen trong bubble
        # (Cần có access đến ảnh gốc - đơn giản hóa ở đây)
        return True  # Placeholder

    def is_region_filled(self, region: np.ndarray) -> bool:
        """Kiểm tra vùng có được tô không"""
        # Tính tỷ lệ pixel đen
        total_pixels = region.size
        black_pixels = np.sum(region == 0)
        fill_ratio = black_pixels / total_pixels

        return fill_ratio > self.bubble_threshold