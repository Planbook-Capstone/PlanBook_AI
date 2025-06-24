"""
AI Trắc Nghiệm – Pipeline Processor
Xử lý phiếu trắc nghiệm theo pipeline mô tả chi tiết:
- Section I: 40 câu ABCD (4 cột x 10 hàng)
- Section II: 8 câu đúng/sai với sub-questions
- Section III: 6 câu điền số 0-9
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json

logger = logging.getLogger(__name__)


class OMRPipelineProcessor:
    """
    OMR Processor theo pipeline AI Trắc Nghiệm
    """

    def __init__(self):
        self.debug_dir = Path("data/grading/debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Xóa debug images cũ
        for file in self.debug_dir.glob("*.jpg"):
            file.unlink()

    def save_debug_image(self, image: np.ndarray, step_name: str, description: str = ""):
        """Lưu ảnh debug với tên mô tả"""
        try:
            filename = f"{step_name}.jpg"
            filepath = self.debug_dir / filename
            cv2.imwrite(str(filepath), image)
            logger.info(f"Debug image saved: {filename} - {description}")
        except Exception as e:
            logger.error(f"Error saving debug image {step_name}: {e}")

    def process_omr_sheet(self, image_path: str) -> Dict[str, Any]:
        """
        Xử lý phiếu trắc nghiệm theo pipeline 7 bước
        
        Returns:
            Dict chứa kết quả xử lý theo format mới
        """
        try:
            # Bước 1: Đọc ảnh đầu vào
            original_image = self.read_and_preprocess_image(image_path)
            
            # Bước 2: Phát hiện các marker đen lớn (vuông)
            markers, aligned_image = self.detect_and_align_markers(original_image)
            
            # Bước 3: Cắt ảnh thành 2 vùng chính
            top_region, bottom_region = self.split_regions(aligned_image)

            # Bước 3.1: Xử lý top region với marker detection
            top_regions = self.process_top_region_with_markers(top_region)

            # Bước 4: Phân tích vùng trả lời (bottom region) với marker detection
            section_results = self.analyze_answer_regions(bottom_region)

            # Bước 4.1: Xử lý student code và test code từ top region
            if 'student_code' in top_regions:
                student_id = self.extract_student_id_from_region(top_regions['student_code'])
                section_results['student_id'] = student_id

            if 'test_code' in top_regions:
                test_code = self.extract_test_code_from_region(top_regions['test_code'])
                section_results['test_code'] = test_code
            
            # Bước 5: Tổng hợp kết quả
            final_results = self.consolidate_results(section_results)
            
            # Bước 6: Tạo ảnh kết quả đánh dấu
            result_image = self.create_result_visualization(aligned_image, final_results)
            self.save_debug_image(result_image, "99_final_result", "Kết quả cuối cùng")
            
            return {
                "success": True,
                "results": final_results,
                "debug_dir": str(self.debug_dir),
                "total_markers": len(markers),
                "processing_steps": 7
            }
            
        except Exception as e:
            logger.error(f"Error processing OMR sheet: {e}")
            return {
                "success": False,
                "error": str(e),
                "debug_dir": str(self.debug_dir)
            }

    def read_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Bước 1: Đọc ảnh đầu vào và tiền xử lý
        """
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Cannot read image: {image_path}")
        
        self.save_debug_image(image, "01_original", "Ảnh gốc")
        
        # Convert grayscale + làm mờ nhẹ
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        self.save_debug_image(blurred, "02_preprocessed", "Ảnh sau tiền xử lý")
        
        return image

    def detect_and_align_markers(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Bước 2: Phát hiện các marker đen lớn (vuông) và căn chỉnh ảnh
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold để tìm marker đen
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        markers = []
        marked_image = image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Lọc contour theo diện tích lớn
            if 500 < area < 10000:  # Marker lớn
                # Xấp xỉ contour thành đa giác
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Kiểm tra có phải hình vuông không
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Tỉ lệ khung hình ≈ 1.0 (vuông)
                    if 0.8 <= aspect_ratio <= 1.2:
                        center = (x + w // 2, y + h // 2)
                        markers.append(center)
                        
                        # Đánh dấu trên ảnh
                        cv2.drawContours(marked_image, [contour], -1, (0, 255, 0), 3)
                        cv2.circle(marked_image, center, 10, (255, 0, 0), -1)
                        cv2.putText(marked_image, f"M{len(markers)}", 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        self.save_debug_image(marked_image, "03_markers_detected", f"Phát hiện {len(markers)} markers")
        
        # Perspective transform nếu có đủ 4 markers
        if len(markers) >= 4:
            aligned_image = self.apply_perspective_transform(image, markers)
        else:
            logger.warning("Không đủ markers, sử dụng ảnh gốc")
            aligned_image = image
        
        self.save_debug_image(aligned_image, "04_aligned", "Ảnh đã căn chỉnh")
        
        return markers, aligned_image

    def apply_perspective_transform(self, image: np.ndarray, markers: List) -> np.ndarray:
        """
        Áp dụng perspective transform để chuẩn hóa ảnh
        """
        # Sắp xếp markers theo thứ tự: top-left, top-right, bottom-right, bottom-left
        markers = np.array(markers)
        
        # Tìm 4 góc xa nhất
        sum_coords = markers.sum(axis=1)
        top_left = markers[np.argmin(sum_coords)]
        bottom_right = markers[np.argmax(sum_coords)]
        
        diff_coords = np.diff(markers, axis=1).flatten()
        top_right = markers[np.argmin(diff_coords)]
        bottom_left = markers[np.argmax(diff_coords)]
        
        # Sắp xếp corners
        src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        
        # Kích thước ảnh đích (A4 ratio)
        width, height = 1200, 1600
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Tính ma trận transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Áp dụng transform
        aligned = cv2.warpPerspective(image, matrix, (width, height))
        
        return aligned

    def split_regions(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bước 3: Cắt ảnh thành 2 vùng chính
        """
        height, width = image.shape[:2]

        # Chia theo tỷ lệ: 30% top (thông tin), 70% bottom (câu trả lời)
        split_y = int(height * 0.3)

        top_region = image[0:split_y, :]
        bottom_region = image[split_y:, :]

        self.save_debug_image(top_region, "05_top_region", "Vùng thông tin (top)")
        self.save_debug_image(bottom_region, "06_bottom_region", "Vùng câu trả lời (bottom)")

        return top_region, bottom_region

    def detect_small_markers(self, region: np.ndarray, region_name: str) -> List[Tuple[int, int]]:
        """
        Phát hiện các marker vuông nhỏ trong một vùng

        Args:
            region: Vùng ảnh cần quét marker
            region_name: Tên vùng (để debug)

        Returns:
            List các tọa độ (x, y) của marker nhỏ
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

        # Threshold để tìm marker đen nhỏ
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        small_markers = []
        marked_image = region.copy()

        for contour in contours:
            area = cv2.contourArea(contour)

            # Lọc contour theo diện tích nhỏ (marker nhỏ)
            if 50 < area < 500:  # Marker nhỏ
                # Xấp xỉ contour thành đa giác
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Kiểm tra có phải hình vuông không
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h

                    # Tỉ lệ khung hình ≈ 1.0 (vuông)
                    if 0.7 <= aspect_ratio <= 1.3:
                        center = (x + w // 2, y + h // 2)
                        small_markers.append(center)

                        # Đánh dấu trên ảnh
                        cv2.drawContours(marked_image, [contour], -1, (0, 255, 0), 2)
                        cv2.circle(marked_image, center, 5, (255, 0, 0), -1)
                        cv2.putText(marked_image, f"S{len(small_markers)}",
                                  (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        self.save_debug_image(marked_image, f"05b_{region_name}_small_markers",
                             f"Small markers in {region_name}: {len(small_markers)} found")

        logger.info(f"Found {len(small_markers)} small markers in {region_name}")
        return small_markers

    def process_top_region_with_markers(self, top_region: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Xử lý top region với marker nhỏ để cắt student code và test code
        """
        # Phát hiện marker nhỏ trong top region
        small_markers = self.detect_small_markers(top_region, "top_region")

        # Sắp xếp markers theo tọa độ x (từ trái sang phải)
        small_markers.sort(key=lambda m: m[0])

        height, width = top_region.shape[:2]
        regions = {}

        if len(small_markers) >= 2:
            # Có đủ marker để chia vùng
            marker1_x = small_markers[0][0]
            marker2_x = small_markers[1][0] if len(small_markers) > 1 else width

            # Student code: từ đầu đến marker đầu tiên
            student_code_region = top_region[:, 0:marker1_x]
            regions['student_code'] = student_code_region
            self.save_debug_image(student_code_region, "05c_student_code_region", "Student Code region")

            # Test code: từ marker đầu tiên đến marker thứ hai (hoặc cuối)
            test_code_region = top_region[:, marker1_x:marker2_x]
            regions['test_code'] = test_code_region
            self.save_debug_image(test_code_region, "05d_test_code_region", "Test Code region")

        else:
            # Không đủ marker, chia theo tỷ lệ
            logger.warning("Not enough small markers in top region, using ratio-based division")
            split_x = int(width * 0.6)

            regions['student_code'] = top_region[:, 0:split_x]
            regions['test_code'] = top_region[:, split_x:]

            self.save_debug_image(regions['student_code'], "05c_student_code_region_ratio", "Student Code (ratio)")
            self.save_debug_image(regions['test_code'], "05d_test_code_region_ratio", "Test Code (ratio)")

        return regions

    def process_bottom_region_with_markers(self, bottom_region: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Xử lý bottom region với marker nhỏ để cắt các section answers theo block
        """
        # Phát hiện marker nhỏ trong bottom region
        small_markers = self.detect_small_markers(bottom_region, "bottom_region")

        # Sắp xếp markers theo tọa độ x và y
        small_markers.sort(key=lambda m: (m[1], m[0]))  # Sắp xếp theo y trước, x sau

        height, width = bottom_region.shape[:2]
        regions = {}

        if len(small_markers) >= 4:
            # Có đủ marker để chia vùng chính xác

            # Chia theo hàng dọc (y-coordinate) để tách các section
            y_coords = [m[1] for m in small_markers]
            y_coords = sorted(list(set(y_coords)))  # Loại bỏ duplicate và sắp xếp

            # Section I: Phần đầu (40 câu ABCD)
            if len(y_coords) >= 1:
                section1_end_y = y_coords[0] if len(y_coords) > 1 else int(height * 0.6)
                section1_region = bottom_region[0:section1_end_y, :]
                regions['section1'] = section1_region
                self.save_debug_image(section1_region, "06a_section1_region", "Section I region")

                # Chia Section I thành 4 cột dựa trên markers
                section1_markers = [m for m in small_markers if m[1] <= section1_end_y]
                section1_markers.sort(key=lambda m: m[0])  # Sắp xếp theo x

                regions.update(self.divide_section1_by_markers(section1_region, section1_markers))

            # Section II: Phần giữa (8 câu True/False)
            if len(y_coords) >= 2:
                section2_start_y = y_coords[0]
                section2_end_y = y_coords[1] if len(y_coords) > 2 else int(height * 0.8)
                section2_region = bottom_region[section2_start_y:section2_end_y, :]
                regions['section2'] = section2_region
                self.save_debug_image(section2_region, "06b_section2_region", "Section II region")

            # Section III: Phần cuối (6 câu digits)
            if len(y_coords) >= 2:
                section3_start_y = y_coords[1] if len(y_coords) > 2 else int(height * 0.8)
                section3_region = bottom_region[section3_start_y:, :]
                regions['section3'] = section3_region
                self.save_debug_image(section3_region, "06c_section3_region", "Section III region")

        else:
            # Không đủ marker, chia theo tỷ lệ
            logger.warning("Not enough small markers in bottom region, using ratio-based division")

            section1_end = int(height * 0.6)
            section2_end = int(height * 0.8)

            regions['section1'] = bottom_region[0:section1_end, :]
            regions['section2'] = bottom_region[section1_end:section2_end, :]
            regions['section3'] = bottom_region[section2_end:, :]

            self.save_debug_image(regions['section1'], "06a_section1_region_ratio", "Section I (ratio)")
            self.save_debug_image(regions['section2'], "06b_section2_region_ratio", "Section II (ratio)")
            self.save_debug_image(regions['section3'], "06c_section3_region_ratio", "Section III (ratio)")

        return regions

    def divide_section1_by_markers(self, section1_region: np.ndarray, markers: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        """
        Chia Section I thành 4 cột dựa trên markers
        """
        height, width = section1_region.shape[:2]
        regions = {}

        if len(markers) >= 3:
            # Có đủ marker để chia 4 cột
            x_coords = [m[0] for m in markers]
            x_coords = [0] + sorted(x_coords) + [width]  # Thêm biên trái và phải

            for i in range(len(x_coords) - 1):
                col_start = x_coords[i]
                col_end = x_coords[i + 1]
                col_region = section1_region[:, col_start:col_end]

                regions[f'section1_col{i+1}'] = col_region
                self.save_debug_image(col_region, f"06a{i+1}_section1_col{i+1}", f"Section I Column {i+1}")
        else:
            # Chia đều thành 4 cột
            col_width = width // 4
            for i in range(4):
                col_start = i * col_width
                col_end = (i + 1) * col_width if i < 3 else width
                col_region = section1_region[:, col_start:col_end]

                regions[f'section1_col{i+1}'] = col_region
                self.save_debug_image(col_region, f"06a{i+1}_section1_col{i+1}_ratio", f"Section I Column {i+1} (ratio)")

        return regions

    def process_section1_with_markers(self, section1_region: np.ndarray, all_regions: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Xử lý Section I với marker-based column division
        """
        results = {}

        # Tìm các cột đã được chia bởi markers
        column_regions = {k: v for k, v in all_regions.items() if k.startswith('section1_col')}

        if column_regions:
            # Xử lý từng cột
            for col_name, col_region in column_regions.items():
                col_number = int(col_name.split('col')[1])
                col_results = self.process_abcd_column_enhanced(col_region, (col_number - 1) * 10 + 1)
                results.update(col_results)

                # Debug image cho từng cột
                self.save_debug_image(col_region, f"07_section1_{col_name}", f"Section I {col_name}")
        else:
            # Fallback to original method
            results = self.process_section1_abcd(section1_region)

        return results

    def process_section2_with_markers(self, section2_region: np.ndarray) -> Dict[str, Dict[str, str]]:
        """
        Xử lý Section II với marker-based division
        """
        self.save_debug_image(section2_region, "08_section2_true_false_enhanced", "Section II - Enhanced")

        # Phát hiện marker nhỏ trong section2 để chia 8 khối
        small_markers = self.detect_small_markers(section2_region, "section2")

        results = {}
        height, width = section2_region.shape[:2]

        if len(small_markers) >= 6:  # Cần ít nhất 6 marker để chia 8 khối
            # Sắp xếp markers theo x (từ trái sang phải)
            small_markers.sort(key=lambda m: m[0])

            # Chia thành 8 khối dựa trên markers
            x_coords = [m[0] for m in small_markers[:7]]  # Lấy 7 marker đầu
            x_coords = [0] + x_coords + [width]  # Thêm biên

            for i in range(8):
                if i < len(x_coords) - 1:
                    block_start = x_coords[i]
                    block_end = x_coords[i + 1]
                    block_region = section2_region[:, block_start:block_end]

                    # Xử lý khối True/False
                    block_results = self.process_true_false_block_enhanced(block_region)
                    results[f"Q{i + 1}"] = block_results

                    # Debug image
                    self.save_debug_image(block_region, f"08a_section2_block{i+1}", f"Section II Block {i+1}")
        else:
            # Fallback: chia đều thành 8 khối
            block_width = width // 8
            for i in range(8):
                block_start = i * block_width
                block_end = (i + 1) * block_width if i < 7 else width
                block_region = section2_region[:, block_start:block_end]

                block_results = self.process_true_false_block_enhanced(block_region)
                results[f"Q{i + 1}"] = block_results

                # Debug image
                self.save_debug_image(block_region, f"08a_section2_block{i+1}_ratio", f"Section II Block {i+1} (ratio)")

        return results

    def process_section3_with_markers(self, section3_region: np.ndarray) -> Dict[str, str]:
        """
        Xử lý Section III với marker-based division
        """
        self.save_debug_image(section3_region, "09_section3_digits_enhanced", "Section III - Enhanced")

        # Phát hiện marker nhỏ trong section3 để chia 6 cột
        small_markers = self.detect_small_markers(section3_region, "section3")

        results = {}
        height, width = section3_region.shape[:2]

        if len(small_markers) >= 5:  # Cần ít nhất 5 marker để chia 6 cột
            # Sắp xếp markers theo x
            small_markers.sort(key=lambda m: m[0])

            # Chia thành 6 cột dựa trên markers
            x_coords = [m[0] for m in small_markers[:5]]  # Lấy 5 marker đầu
            x_coords = [0] + x_coords + [width]  # Thêm biên

            for i in range(6):
                if i < len(x_coords) - 1:
                    col_start = x_coords[i]
                    col_end = x_coords[i + 1]
                    col_region = section3_region[:, col_start:col_end]

                    # Phát hiện digit
                    digit = self.detect_digit_bubble_enhanced(col_region)
                    results[f"Q{i + 1}"] = digit

                    # Debug image
                    self.save_debug_image(col_region, f"09a_section3_col{i+1}", f"Section III Column {i+1}")
        else:
            # Fallback: chia đều thành 6 cột
            col_width = width // 6
            for i in range(6):
                col_start = i * col_width
                col_end = (i + 1) * col_width if i < 5 else width
                col_region = section3_region[:, col_start:col_end]

                digit = self.detect_digit_bubble_enhanced(col_region)
                results[f"Q{i + 1}"] = digit

                # Debug image
                self.save_debug_image(col_region, f"09a_section3_col{i+1}_ratio", f"Section III Column {i+1} (ratio)")

        return results

    def process_abcd_column_enhanced(self, col_region: np.ndarray, start_question: int) -> Dict[str, str]:
        """
        Xử lý một cột 10 câu ABCD với enhanced detection
        """
        height, width = col_region.shape[:2]
        row_height = height // 10

        results = {}

        for row in range(10):
            question_num = start_question + row
            y1 = row * row_height
            y2 = (row + 1) * row_height

            question_region = col_region[y1:y2, :]

            # Enhanced bubble detection
            answer = self.detect_abcd_bubble_enhanced(question_region)
            results[f"Q{question_num}"] = answer

        return results

    def detect_abcd_bubble_enhanced(self, question_region: np.ndarray) -> str:
        """
        Enhanced ABCD bubble detection với contour analysis
        """
        height, width = question_region.shape[:2]

        # Chia thành 4 phần cho A, B, C, D
        option_width = width // 4
        options = ["A", "B", "C", "D"]

        max_filled_ratio = 0
        selected_answer = "A"

        for i, option in enumerate(options):
            x1 = i * option_width
            x2 = (i + 1) * option_width
            option_region = question_region[:, x1:x2]

            # Sử dụng cả pixel counting và contour detection
            filled_ratio = self.calculate_fill_ratio(option_region)
            contour_score = self.calculate_contour_score(option_region)

            # Kết hợp cả hai điểm số
            combined_score = filled_ratio * 0.7 + contour_score * 0.3

            if combined_score > max_filled_ratio and combined_score > 0.15:  # Threshold 15%
                max_filled_ratio = combined_score
                selected_answer = option

        return selected_answer

    def process_true_false_block_enhanced(self, block_region: np.ndarray) -> Dict[str, str]:
        """
        Enhanced True/False block processing
        """
        height, width = block_region.shape[:2]

        # Chia thành 3 sub-questions: a, b, c
        sub_height = height // 3
        results = {}

        for sub in range(3):
            sub_letter = chr(ord('a') + sub)
            y1 = sub * sub_height
            y2 = (sub + 1) * sub_height

            sub_region = block_region[y1:y2, :]

            # Enhanced True/False detection
            answer = self.detect_true_false_bubble_enhanced(sub_region)
            results[sub_letter] = answer

        return results

    def detect_true_false_bubble_enhanced(self, sub_region: np.ndarray) -> str:
        """
        Enhanced True/False bubble detection
        """
        height, width = sub_region.shape[:2]

        # Chia đôi cho Đúng và Sai
        half_width = width // 2

        # Kiểm tra phần Đúng
        true_region = sub_region[:, :half_width]
        false_region = sub_region[:, half_width:]

        true_ratio = self.calculate_fill_ratio(true_region)
        false_ratio = self.calculate_fill_ratio(false_region)

        # Enhanced với contour detection
        true_contour = self.calculate_contour_score(true_region)
        false_contour = self.calculate_contour_score(false_region)

        true_score = true_ratio * 0.7 + true_contour * 0.3
        false_score = false_ratio * 0.7 + false_contour * 0.3

        if true_score > false_score and true_score > 0.15:
            return "Đúng"
        elif false_score > 0.15:
            return "Sai"
        else:
            return "Không xác định"

    def detect_digit_bubble_enhanced(self, col_region: np.ndarray) -> str:
        """
        Enhanced digit bubble detection
        """
        height, width = col_region.shape[:2]

        # Chia thành 10 hàng cho số 0-9
        row_height = height // 10

        max_score = 0
        selected_digit = "0"

        for digit in range(10):
            y1 = digit * row_height
            y2 = (digit + 1) * row_height

            digit_region = col_region[y1:y2, :]

            # Enhanced scoring
            filled_ratio = self.calculate_fill_ratio(digit_region)
            contour_score = self.calculate_contour_score(digit_region)
            combined_score = filled_ratio * 0.7 + contour_score * 0.3

            if combined_score > max_score and combined_score > 0.15:
                max_score = combined_score
                selected_digit = str(digit)

        return selected_digit

    def calculate_contour_score(self, region: np.ndarray) -> float:
        """
        Tính điểm số dựa trên contour detection (phát hiện hình tròn được tô)
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Lọc contour nhỏ
                # Tính độ tròn
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Threshold cho hình tròn
                        score = min(area / (region.shape[0] * region.shape[1]), 1.0)
                        max_score = max(max_score, score)

        return max_score

    def extract_student_id_from_region(self, student_code_region: np.ndarray) -> str:
        """
        Trích xuất Student ID từ vùng student code đã được cắt bởi marker
        """
        self.save_debug_image(student_code_region, "10a_student_id_extraction", "Student ID extraction")

        # Sử dụng grid detection để trích xuất 8 digits
        height, width = student_code_region.shape[:2]

        # Chia thành 8 cột cho 8 digits
        col_width = width // 8
        student_id = ""

        for col in range(8):
            x1 = col * col_width
            x2 = (col + 1) * col_width if col < 7 else width
            col_region = student_code_region[:, x1:x2]

            # Phát hiện digit trong cột này (0-9)
            digit = self.detect_digit_in_column(col_region, f"student_id_col{col+1}")
            student_id += digit

            # Debug image cho từng cột
            self.save_debug_image(col_region, f"10a{col+1}_student_id_col{col+1}", f"Student ID Column {col+1}")

        logger.info(f"Extracted Student ID: {student_id}")
        return student_id

    def extract_test_code_from_region(self, test_code_region: np.ndarray) -> str:
        """
        Trích xuất Test Code từ vùng test code đã được cắt bởi marker
        """
        self.save_debug_image(test_code_region, "10b_test_code_extraction", "Test Code extraction")

        # Sử dụng grid detection để trích xuất 4 digits
        height, width = test_code_region.shape[:2]

        # Chia thành 4 cột cho 4 digits
        col_width = width // 4
        test_code = ""

        for col in range(4):
            x1 = col * col_width
            x2 = (col + 1) * col_width if col < 3 else width
            col_region = test_code_region[:, x1:x2]

            # Phát hiện digit trong cột này (0-9)
            digit = self.detect_digit_in_column(col_region, f"test_code_col{col+1}")
            test_code += digit

            # Debug image cho từng cột
            self.save_debug_image(col_region, f"10b{col+1}_test_code_col{col+1}", f"Test Code Column {col+1}")

        logger.info(f"Extracted Test Code: {test_code}")
        return test_code

    def detect_digit_in_column(self, col_region: np.ndarray, col_name: str) -> str:
        """
        Phát hiện digit (0-9) trong một cột của student ID hoặc test code
        """
        height, width = col_region.shape[:2]

        # Chia thành 10 hàng cho số 0-9
        row_height = height // 10

        max_score = 0
        selected_digit = "0"

        # Tạo ảnh debug để hiển thị detection
        debug_image = col_region.copy()

        for digit in range(10):
            y1 = digit * row_height
            y2 = (digit + 1) * row_height

            digit_region = col_region[y1:y2, :]

            # Enhanced scoring với cả fill ratio và contour
            filled_ratio = self.calculate_fill_ratio(digit_region)
            contour_score = self.calculate_contour_score(digit_region)
            combined_score = filled_ratio * 0.6 + contour_score * 0.4

            # Vẽ score lên debug image
            cv2.putText(debug_image, f"{digit}:{combined_score:.2f}",
                       (5, y1 + row_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            if combined_score > max_score and combined_score > 0.2:  # Threshold 20%
                max_score = combined_score
                selected_digit = str(digit)

                # Highlight selected digit
                cv2.rectangle(debug_image, (0, y1), (width, y2), (0, 255, 0), 2)

        # Save debug image
        self.save_debug_image(debug_image, f"10c_{col_name}_detection", f"{col_name} digit detection")

        return selected_digit

    def analyze_answer_regions(self, bottom_region: np.ndarray) -> Dict[str, Any]:
        """
        Bước 4: Phân tích vùng trả lời (bottom region) với marker-based division
        """
        results = {}

        # Bước 4.1: Phát hiện marker nhỏ và chia vùng chính xác
        region_divisions = self.process_bottom_region_with_markers(bottom_region)

        # Bước 4.2: Xử lý từng section dựa trên vùng đã chia

        # Section I (PHẦN I – 40 câu ABCD)
        if 'section1' in region_divisions:
            section1_results = self.process_section1_with_markers(region_divisions['section1'], region_divisions)
            results["Section I"] = section1_results
        else:
            # Fallback to original method
            section1_results = self.process_section1_abcd(bottom_region)
            results["Section I"] = section1_results

        # Section II (PHẦN II – 8 câu đúng/sai)
        if 'section2' in region_divisions:
            section2_results = self.process_section2_with_markers(region_divisions['section2'])
            results["Section II"] = section2_results
        else:
            # Fallback to original method
            section2_results = self.process_section2_true_false(bottom_region)
            results["Section II"] = section2_results

        # Section III (PHẦN III – điền số 0–9)
        if 'section3' in region_divisions:
            section3_results = self.process_section3_with_markers(region_divisions['section3'])
            results["Section III"] = section3_results
        else:
            # Fallback to original method
            section3_results = self.process_section3_digits(bottom_region)
            results["Section III"] = section3_results

        return results

    def process_section1_abcd(self, region: np.ndarray) -> Dict[str, str]:
        """
        4.1. Section I (PHẦN I – 40 câu ABCD)
        Chia thành 4 cột, mỗi cột 10 hàng, mỗi hàng 4 ô (A,B,C,D)
        """
        height, width = region.shape[:2]
        
        # Lấy 60% đầu của bottom region cho Section I
        section1_height = int(height * 0.6)
        section1_region = region[0:section1_height, :]
        
        self.save_debug_image(section1_region, "07_section1_abcd", "Section I - 40 câu ABCD")
        
        results = {}
        
        # Chia thành 4 cột
        col_width = width // 4
        
        for col in range(4):
            x1 = col * col_width
            x2 = (col + 1) * col_width
            col_region = section1_region[:, x1:x2]
            
            # Xử lý 10 câu trong cột này
            col_results = self.process_abcd_column(col_region, col * 10 + 1)
            results.update(col_results)
        
        return results

    def process_abcd_column(self, col_region: np.ndarray, start_question: int) -> Dict[str, str]:
        """
        Xử lý một cột 10 câu ABCD
        """
        height, width = col_region.shape[:2]
        row_height = height // 10
        
        results = {}
        
        for row in range(10):
            question_num = start_question + row
            y1 = row * row_height
            y2 = (row + 1) * row_height
            
            question_region = col_region[y1:y2, :]
            
            # Phát hiện bubble A, B, C, D
            answer = self.detect_abcd_bubble(question_region)
            results[f"Q{question_num}"] = answer
        
        return results

    def detect_abcd_bubble(self, question_region: np.ndarray) -> str:
        """
        Phát hiện bubble A, B, C, D được tô trong một câu hỏi
        """
        height, width = question_region.shape[:2]
        
        # Chia thành 4 phần cho A, B, C, D
        option_width = width // 4
        options = ["A", "B", "C", "D"]
        
        max_filled_ratio = 0
        selected_answer = "A"
        
        for i, option in enumerate(options):
            x1 = i * option_width
            x2 = (i + 1) * option_width
            option_region = question_region[:, x1:x2]
            
            # Tính tỷ lệ pixel đen (bubble được tô)
            gray = cv2.cvtColor(option_region, cv2.COLOR_BGR2GRAY) if len(option_region.shape) == 3 else option_region
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            filled_pixels = np.sum(binary == 255)
            total_pixels = binary.size
            filled_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
            
            if filled_ratio > max_filled_ratio and filled_ratio > 0.1:  # Threshold 10%
                max_filled_ratio = filled_ratio
                selected_answer = option
        
        return selected_answer

    def process_section2_true_false(self, region: np.ndarray) -> Dict[str, Dict[str, str]]:
        """
        4.2. Section II (PHẦN II – 8 câu đúng/sai)
        """
        height, width = region.shape[:2]
        
        # Lấy 20% giữa của bottom region cho Section II
        section2_start = int(height * 0.6)
        section2_end = int(height * 0.8)
        section2_region = region[section2_start:section2_end, :]
        
        self.save_debug_image(section2_region, "08_section2_true_false", "Section II - 8 câu đúng/sai")
        
        results = {}
        
        # Chia thành 8 khối câu hỏi
        block_width = width // 8
        
        for block in range(8):
            question_num = block + 1
            x1 = block * block_width
            x2 = (block + 1) * block_width
            block_region = section2_region[:, x1:x2]
            
            # Xử lý câu đúng/sai với sub-questions
            block_results = self.process_true_false_block(block_region)
            results[f"Q{question_num}"] = block_results
        
        return results

    def process_true_false_block(self, block_region: np.ndarray) -> Dict[str, str]:
        """
        Xử lý một khối câu đúng/sai với sub-questions a, b, c
        """
        height, width = block_region.shape[:2]
        
        # Giả sử có 3 sub-questions: a, b, c
        sub_height = height // 3
        results = {}
        
        for sub in range(3):
            sub_letter = chr(ord('a') + sub)
            y1 = sub * sub_height
            y2 = (sub + 1) * sub_height
            
            sub_region = block_region[y1:y2, :]
            
            # Phát hiện Đúng/Sai
            answer = self.detect_true_false_bubble(sub_region)
            results[sub_letter] = answer
        
        return results

    def detect_true_false_bubble(self, sub_region: np.ndarray) -> str:
        """
        Phát hiện bubble Đúng/Sai
        """
        height, width = sub_region.shape[:2]
        
        # Chia đôi cho Đúng và Sai
        half_width = width // 2
        
        # Kiểm tra phần Đúng
        true_region = sub_region[:, :half_width]
        false_region = sub_region[:, half_width:]
        
        true_ratio = self.calculate_fill_ratio(true_region)
        false_ratio = self.calculate_fill_ratio(false_region)
        
        if true_ratio > false_ratio and true_ratio > 0.1:
            return "Đúng"
        elif false_ratio > 0.1:
            return "Sai"
        else:
            return "Không xác định"

    def process_section3_digits(self, region: np.ndarray) -> Dict[str, str]:
        """
        4.3. Section III (PHẦN III – điền số 0–9)
        """
        height, width = region.shape[:2]
        
        # Lấy 20% cuối của bottom region cho Section III
        section3_start = int(height * 0.8)
        section3_region = region[section3_start:, :]
        
        self.save_debug_image(section3_region, "09_section3_digits", "Section III - 6 câu điền số")
        
        results = {}
        
        # Chia thành 6 cột cho 6 câu
        col_width = width // 6
        
        for col in range(6):
            question_num = col + 1
            x1 = col * col_width
            x2 = (col + 1) * col_width
            col_region = section3_region[:, x1:x2]
            
            # Phát hiện số 0-9
            digit = self.detect_digit_bubble(col_region)
            results[f"Q{question_num}"] = digit
        
        return results

    def detect_digit_bubble(self, col_region: np.ndarray) -> str:
        """
        Phát hiện bubble số 0-9 được tô
        """
        height, width = col_region.shape[:2]
        
        # Chia thành 10 hàng cho số 0-9
        row_height = height // 10
        
        max_filled_ratio = 0
        selected_digit = "0"
        
        for digit in range(10):
            y1 = digit * row_height
            y2 = (digit + 1) * row_height
            
            digit_region = col_region[y1:y2, :]
            filled_ratio = self.calculate_fill_ratio(digit_region)
            
            if filled_ratio > max_filled_ratio and filled_ratio > 0.1:
                max_filled_ratio = filled_ratio
                selected_digit = str(digit)
        
        return selected_digit

    def calculate_fill_ratio(self, region: np.ndarray) -> float:
        """
        Tính tỷ lệ pixel được tô (đen) trong vùng
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        filled_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        return filled_pixels / total_pixels if total_pixels > 0 else 0

    def consolidate_results(self, section_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bước 5: Tổng hợp kết quả
        """
        consolidated = {
            "Section I": section_results.get("Section I", {}),
            "Section II": section_results.get("Section II", {}),
            "Section III": section_results.get("Section III", {}),
            "summary": {
                "total_section1": len(section_results.get("Section I", {})),
                "total_section2": len(section_results.get("Section II", {})),
                "total_section3": len(section_results.get("Section III", {})),
                "total_questions": (
                    len(section_results.get("Section I", {})) +
                    len(section_results.get("Section II", {})) +
                    len(section_results.get("Section III", {}))
                )
            }
        }
        
        return consolidated

    def create_result_visualization(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Bước 7: Tạo ảnh kết quả đánh dấu
        """
        result_image = image.copy()
        
        # Vẽ thông tin tổng quan
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 50
        
        summary = results.get("summary", {})
        
        cv2.putText(result_image, f"Total Questions: {summary.get('total_questions', 0)}", 
                   (50, y_pos), font, 1.0, (0, 255, 0), 2)
        y_pos += 40
        
        cv2.putText(result_image, f"Section I (ABCD): {summary.get('total_section1', 0)}", 
                   (50, y_pos), font, 0.8, (0, 255, 0), 2)
        y_pos += 30
        
        cv2.putText(result_image, f"Section II (T/F): {summary.get('total_section2', 0)}", 
                   (50, y_pos), font, 0.8, (0, 255, 0), 2)
        y_pos += 30
        
        cv2.putText(result_image, f"Section III (Digits): {summary.get('total_section3', 0)}", 
                   (50, y_pos), font, 0.8, (0, 255, 0), 2)
        
        return result_image


# Global instance
omr_pipeline_processor = OMRPipelineProcessor()
