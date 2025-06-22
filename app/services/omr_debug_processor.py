import cv2
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OMRDebugProcessor:
    """
    OMR Processor với debug chi tiết từng bước
    Xử lý phiếu trắc nghiệm Việt Nam với layout cố định
    """

    def __init__(self):
        self.debug_dir = Path("data/grading/debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Xóa debug images cũ
        for file in self.debug_dir.glob("*.jpg"):
            file.unlink()

    def save_debug_image(
        self, image: np.ndarray, step_name: str, description: str = ""
    ):
        """Lưu ảnh debug với tên mô tả"""
        try:
            filename = f"{step_name}.jpg"
            filepath = self.debug_dir / filename
            cv2.imwrite(str(filepath), image)
            logger.info(f"Debug image saved: {filename} - {description}")
        except Exception as e:
            logger.error(f"Error saving debug image {step_name}: {e}")

    def process_answer_sheet(self, image_path: str) -> Dict:
        """
        Xử lý phiếu trả lời với debug từng bước

        Args:
            image_path: Đường dẫn đến ảnh phiếu trả lời

        Returns:
            Dict chứa kết quả xử lý
        """
        try:
            # Bước 1: Đọc ảnh gốc
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise Exception(f"Cannot read image: {image_path}")

            self.save_debug_image(original_image, "01_original", "Ảnh gốc")
            logger.info(f"Original image size: {original_image.shape}")

            # Bước 2: Tiền xử lý ảnh
            preprocessed = self.preprocess_image(original_image)
            self.save_debug_image(preprocessed, "02_preprocessed", "Ảnh sau tiền xử lý")

            # Bước 3: Tìm và đánh dấu 4 góc markers
            corners, marked_image = self.find_corner_markers(
                original_image, preprocessed
            )
            self.save_debug_image(
                marked_image,
                "03_corners_detected",
                f"Phát hiện {len(corners)} góc markers",
            )

            # Bước 4: Căn chỉnh ảnh bằng perspective transform
            if len(corners) >= 4:
                aligned_image = self.align_image_with_corners(original_image, corners)
                self.save_debug_image(aligned_image, "04_aligned", "Ảnh đã căn chỉnh")
            else:
                logger.warning("Không đủ corner markers, sử dụng ảnh gốc")
                aligned_image = original_image
                self.save_debug_image(
                    aligned_image, "04_aligned", "Không căn chỉnh được - dùng ảnh gốc"
                )

            # Bước 5: Trích xuất các vùng ROI
            regions = self.extract_regions(aligned_image)

            # Bước 6: Xử lý từng vùng
            results = {}

            # Xử lý Student ID
            if "student_id" in regions:
                student_id = self.process_student_id_region(regions["student_id"])
                results["student_id"] = student_id

            # Xử lý Test Code
            if "test_code" in regions:
                test_code = self.process_test_code_region(regions["test_code"])
                results["test_code"] = test_code

            # Xử lý Answer regions
            answers = {}
            for region_name in [
                "answers_01_15",
                "answers_16_30",
                "answers_31_45",
                "answers_46_60",
            ]:
                if region_name in regions:
                    region_answers = self.process_answer_region(
                        regions[region_name], region_name
                    )
                    answers.update(region_answers)

            results["answers"] = answers

            # Tạo ảnh tổng hợp kết quả
            result_image = self.create_result_visualization(aligned_image, results)
            self.save_debug_image(result_image, "99_final_result", "Kết quả cuối cùng")

            return {
                "success": True,
                "student_id": results.get("student_id", ""),
                "test_code": results.get("test_code", ""),
                "answers": results.get("answers", {}),
                "debug_dir": str(self.debug_dir),
            }

        except Exception as e:
            logger.error(f"Error processing answer sheet: {e}")
            return {"success": False, "error": str(e), "debug_dir": str(self.debug_dir)}

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh"""
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Khử nhiễu
        denoised = cv2.medianBlur(gray, 3)

        # Tăng cường độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Nhị phân hóa adaptive
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def find_corner_markers(
        self, original: np.ndarray, binary: np.ndarray
    ) -> Tuple[List, np.ndarray]:
        """Tìm 4 góc markers (hình vuông đen)"""
        # Tìm contours
        contours, _ = cv2.findContours(
            255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Tạo ảnh để đánh dấu
        marked = original.copy()

        corners = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Lọc theo kích thước (markers phải đủ lớn nhưng không quá lớn)
            if 200 < area < 8000:
                # Xấp xỉ contour thành đa giác
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Kiểm tra có phải hình chữ nhật không
                if len(approx) == 4:
                    # Tính tỷ lệ khung hình
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h

                    # Markers phải gần vuông
                    if 0.7 <= aspect_ratio <= 1.3:
                        corners.append((x + w // 2, y + h // 2))  # Lưu tâm marker

                        # Đánh dấu trên ảnh
                        cv2.drawContours(marked, [contour], -1, (0, 255, 0), 3)
                        cv2.circle(
                            marked, (x + w // 2, y + h // 2), 10, (255, 0, 0), -1
                        )
                        cv2.putText(
                            marked,
                            f"M{len(corners)}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                        )

        logger.info(f"Found {len(corners)} corner markers")
        return corners, marked

    def align_image_with_corners(self, image: np.ndarray, corners: List) -> np.ndarray:
        """Căn chỉnh ảnh dựa vào 4 góc markers với thuật toán sắp xếp theo khoảng cách"""
        if len(corners) < 4:
            return image

        # Chuyển đổi corners thành numpy array để tính toán dễ hơn
        corners = np.array(corners)

        # Tìm 4 corners xa nhất (tạo thành hình chữ nhật lớn nhất)
        # Tính tổng tọa độ để tìm top-left (tổng nhỏ nhất) và bottom-right (tổng lớn nhất)
        sum_coords = corners.sum(axis=1)
        top_left = corners[np.argmin(sum_coords)]
        bottom_right = corners[np.argmax(sum_coords)]

        # Tính hiệu tọa độ để tìm top-right và bottom-left
        diff_coords = np.diff(corners, axis=1).flatten()
        top_right = corners[np.argmin(diff_coords)]  # x lớn, y nhỏ -> diff nhỏ nhất
        bottom_left = corners[np.argmax(diff_coords)]  # x nhỏ, y lớn -> diff lớn nhất

        # Sắp xếp corners theo thứ tự: top-left, top-right, bottom-right, bottom-left
        ordered_corners = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
        )

        # Tạo debug image để kiểm tra corners
        debug_corners = image.copy()
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]  # Blue, Green, Red, Yellow
        labels = ["TL", "TR", "BR", "BL"]

        for i, (corner, color, label) in enumerate(
            zip(ordered_corners, colors, labels)
        ):
            cv2.circle(debug_corners, tuple(corner.astype(int)), 15, color, -1)
            cv2.putText(
                debug_corners,
                f"{label}{i+1}",
                (int(corner[0]) - 20, int(corner[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

        self.save_debug_image(
            debug_corners, "03b_corners_ordered", "Corners đã sắp xếp"
        )

        # Kích thước ảnh đích (A4 ratio)
        width, height = 1086, 1536  # Kích thước chuẩn cho phiếu trả lời
        dst_points = np.array(
            [
                [0, 0],  # top-left
                [width - 1, 0],  # top-right
                [width - 1, height - 1],  # bottom-right
                [0, height - 1],  # bottom-left
            ],
            dtype=np.float32,
        )

        # Tính ma trận transform
        matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)

        # Áp dụng perspective transform
        aligned = cv2.warpPerspective(image, matrix, (width, height))

        return aligned

    def align_image(self, image: np.ndarray) -> np.ndarray:
        """
        Căn chỉnh ảnh dựa trên 4 góc đen
        """
        try:
            # Chuyển sang grayscale để tìm góc
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Nhị phân hóa để tìm góc
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Tìm 4 góc đen
            corners, marked = self.find_corner_markers(image, binary)

            if len(corners) >= 4:
                # Sử dụng method có sẵn để căn chỉnh
                aligned = self.align_image_with_corners(image, corners)
                self.save_debug_image(aligned, "03_aligned", "Ảnh sau khi căn chỉnh")
                return aligned
            else:
                print(f"⚠️ Không tìm đủ 4 góc, sử dụng ảnh gốc")
                return image

        except Exception as e:
            logger.error(f"Error in image alignment: {e}")
            return image

    def extract_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Trích xuất các vùng ROI theo layout cố định
        Sử dụng tọa độ tỷ lệ phần trăm như trong code mẫu
        """
        height, width = image.shape[:2]
        print(f"Image size: {width}x{height}")
        regions = {}

        # Kích thước tham chiếu (max_weight, max_height)
        max_weight = 1807
        max_height = 2555

        # Tính tỷ lệ ảnh hiện tại
        img_width = width
        img_height = height

        # Tạo ảnh tổng quan với tất cả vùng ROI được đánh dấu
        overview_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Student ID: Mở rộng vùng để dài hơn và cao hơn
        # Tọa độ gốc: (951, 54, 1430, 821) -> Mở rộng sang trái, lên trên, sang phải, xuống dưới
        crop_sbd = (
            int(
                1300 / max_weight * img_width
            ),  # Mở rộng sang trái: 951 -> 900 (-51 pixels)
            int(0 / max_height * img_height),  # Mở rộng lên trên: 54 -> 30 (-24 pixels)
            int(
                1590 / max_weight * img_width
            ),  # Mở rộng sang phải: 1430 -> 1480 (+50 pixels)
            int(
                1012 / max_height * img_height
            ),  # Mở rộng xuống dưới: 821 -> 900 (+79 pixels)
        )
        student_id_region = image[crop_sbd[1] : crop_sbd[3], crop_sbd[0] : crop_sbd[2]]
        regions["student_id"] = student_id_region

        # Vẽ khung cho Student ID
        cv2.rectangle(
            overview_image,
            (crop_sbd[0], crop_sbd[1]),
            (crop_sbd[2], crop_sbd[3]),
            (0, 255, 0),
            2,
        )
        sid_size = f"{crop_sbd[2]-crop_sbd[0]}x{crop_sbd[3]-crop_sbd[1]}"
        cv2.putText(
            overview_image,
            f"Student ID: {sid_size}",
            (crop_sbd[0], crop_sbd[1] - 10),
            font,
            0.6,
            (0, 255, 0),
            2,
        )
        self.save_debug_image(
            student_id_region, "05_region_student_id", f"Vùng Student ID {sid_size}"
        )

        # Test Code: crop_mdt = (1418, 254, 1726, 821)
        crop_mdt = (
            int(1418 / max_weight * img_width),
            int(254 / max_height * img_height),
            int(1726 / max_weight * img_width),
            int(821 / max_height * img_height),
        )
        test_code_region = image[crop_mdt[1] : crop_mdt[3], crop_mdt[0] : crop_mdt[2]]
        regions["test_code"] = test_code_region

        # Vẽ khung cho Test Code
        cv2.rectangle(
            overview_image,
            (crop_mdt[0], crop_mdt[1]),
            (crop_mdt[2], crop_mdt[3]),
            (255, 0, 0),
            2,
        )
        tc_size = f"{crop_mdt[2]-crop_mdt[0]}x{crop_mdt[3]-crop_mdt[1]}"
        cv2.putText(
            overview_image,
            f"Test Code: {tc_size}",
            (crop_mdt[0], crop_mdt[1] - 10),
            font,
            0.6,
            (255, 0, 0),
            2,
        )
        self.save_debug_image(
            test_code_region, "06_region_test_code", f"Vùng Test Code {tc_size}"
        )

        # Answers Q01-15: crop_1_30 = (41, 833, 480, 2470)
        crop_1_30 = (
            int(41 / max_weight * img_width),
            int(833 / max_height * img_height),
            int(480 / max_weight * img_width),
            int(2470 / max_height * img_height),
        )
        answers_01_15 = image[crop_1_30[1] : crop_1_30[3], crop_1_30[0] : crop_1_30[2]]
        regions["answers_01_15"] = answers_01_15

        # Vẽ khung cho Q01-15
        cv2.rectangle(
            overview_image,
            (crop_1_30[0], crop_1_30[1]),
            (crop_1_30[2], crop_1_30[3]),
            (0, 0, 255),
            2,
        )
        q1_size = f"{crop_1_30[2]-crop_1_30[0]}x{crop_1_30[3]-crop_1_30[1]}"
        cv2.putText(
            overview_image,
            f"Q01-15: {q1_size}",
            (crop_1_30[0], crop_1_30[1] - 10),
            font,
            0.6,
            (0, 0, 255),
            2,
        )
        self.save_debug_image(
            answers_01_15, "07_region_answers_01_15", f"Câu 01-15 {q1_size}"
        )

        # Answers Q16-30: crop_31_60 = (466, 833, 870, 2470)
        crop_16_30 = (
            int(466 / max_weight * img_width),
            int(833 / max_height * img_height),
            int(870 / max_weight * img_width),
            int(2470 / max_height * img_height),
        )
        answers_16_30 = image[
            crop_16_30[1] : crop_16_30[3], crop_16_30[0] : crop_16_30[2]
        ]
        regions["answers_16_30"] = answers_16_30

        # Vẽ khung cho Q16-30
        cv2.rectangle(
            overview_image,
            (crop_16_30[0], crop_16_30[1]),
            (crop_16_30[2], crop_16_30[3]),
            (255, 255, 0),
            2,
        )
        q2_size = f"{crop_16_30[2]-crop_16_30[0]}x{crop_16_30[3]-crop_16_30[1]}"
        cv2.putText(
            overview_image,
            f"Q16-30: {q2_size}",
            (crop_16_30[0], crop_16_30[1] - 10),
            font,
            0.6,
            (255, 255, 0),
            2,
        )
        self.save_debug_image(
            answers_16_30, "08_region_answers_16_30", f"Câu 16-30 {q2_size}"
        )

        # Answers Q31-45: Sử dụng cùng vùng với student_id (951, 254, 1430, 821) nhưng phần dưới
        crop_31_45 = (
            int(951 / max_weight * img_width),
            int(833 / max_height * img_height),  # Bắt đầu từ y=833 thay vì 254
            int(1430 / max_weight * img_width),
            int(1400 / max_height * img_height),  # Kết thúc ở giữa
        )
        answers_31_45 = image[
            crop_31_45[1] : crop_31_45[3], crop_31_45[0] : crop_31_45[2]
        ]
        regions["answers_31_45"] = answers_31_45

        # Vẽ khung cho Q31-45
        cv2.rectangle(
            overview_image,
            (crop_31_45[0], crop_31_45[1]),
            (crop_31_45[2], crop_31_45[3]),
            (0, 255, 255),
            2,
        )
        q3_size = f"{crop_31_45[2]-crop_31_45[0]}x{crop_31_45[3]-crop_31_45[1]}"
        cv2.putText(
            overview_image,
            f"Q31-45: {q3_size}",
            (crop_31_45[0], crop_31_45[1] - 10),
            font,
            0.6,
            (0, 255, 255),
            2,
        )
        self.save_debug_image(
            answers_31_45, "09_region_answers_31_45", f"Câu 31-45 {q3_size}"
        )

        # Answers Q46-60: Sử dụng cùng vùng với test_code (1418, 254, 1726, 821) nhưng phần dưới
        crop_46_60 = (
            int(1418 / max_weight * img_width),
            int(833 / max_height * img_height),  # Bắt đầu từ y=833 thay vì 254
            int(1726 / max_weight * img_width),
            int(1400 / max_height * img_height),  # Kết thúc ở giữa
        )
        answers_46_60 = image[
            crop_46_60[1] : crop_46_60[3], crop_46_60[0] : crop_46_60[2]
        ]
        regions["answers_46_60"] = answers_46_60

        # Vẽ khung cho Q46-60
        cv2.rectangle(
            overview_image,
            (crop_46_60[0], crop_46_60[1]),
            (crop_46_60[2], crop_46_60[3]),
            (255, 0, 255),
            2,
        )
        q4_size = f"{crop_46_60[2]-crop_46_60[0]}x{crop_46_60[3]-crop_46_60[1]}"
        cv2.putText(
            overview_image,
            f"Q46-60: {q4_size}",
            (crop_46_60[0], crop_46_60[1] - 10),
            font,
            0.6,
            (255, 0, 255),
            2,
        )
        self.save_debug_image(
            answers_46_60, "10_region_answers_46_60", f"Câu 46-60 {q4_size}"
        )

        # Tạo vùng tổng hợp tất cả câu trả lời
        full_answers_coords = (
            int(41 / max_weight * img_width),
            int(254 / max_height * img_height),
            int(1726 / max_weight * img_width),
            int(2470 / max_height * img_height),
        )
        answers_full = image[
            full_answers_coords[1] : full_answers_coords[3],
            full_answers_coords[0] : full_answers_coords[2],
        ]
        regions["answers_full"] = answers_full

        # Vẽ khung cho All Answers
        cv2.rectangle(
            overview_image,
            (full_answers_coords[0], full_answers_coords[1]),
            (full_answers_coords[2], full_answers_coords[3]),
            (128, 128, 128),
            3,
        )
        qall_size = f"{full_answers_coords[2]-full_answers_coords[0]}x{full_answers_coords[3]-full_answers_coords[1]}"
        cv2.putText(
            overview_image,
            f"All Answers: {qall_size}",
            (full_answers_coords[0], full_answers_coords[1] - 50),
            font,
            0.8,
            (128, 128, 128),
            3,
        )
        self.save_debug_image(
            answers_full, "11_region_answers_full", f"Tất cả câu trả lời {qall_size}"
        )

        # Thêm thông tin tổng quan lên ảnh overview
        cv2.putText(
            overview_image,
            f"Image Size: {width}x{height}",
            (50, 50),
            font,
            1.0,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            overview_image,
            "ROI Regions Overview",
            (50, 100),
            font,
            1.2,
            (255, 255, 255),
            3,
        )

        # Lưu ảnh overview với tất cả vùng ROI
        self.save_debug_image(
            overview_image, "04b_roi_overview", "Tổng quan các vùng ROI"
        )

        return regions

    def process_image_file(self, image_path: str) -> Dict:
        """
        Xử lý ảnh OMR từ file path
        """
        try:
            # Load ảnh từ file
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "student_id": "000000",
                    "test_code": "000",
                    "answers": {},
                    "total_questions": 0,
                    "processing_status": "error",
                    "error_message": f"Cannot load image from {image_path}",
                }

            return self.process_image(image)

        except Exception as e:
            logger.error(f"Error processing image file {image_path}: {e}")
            return {
                "student_id": "000000",
                "test_code": "000",
                "answers": {},
                "total_questions": 0,
                "processing_status": "error",
                "error_message": str(e),
            }

    def process_image(self, image: np.ndarray) -> Dict:
        """
        Xử lý toàn bộ ảnh OMR và trả về kết quả
        """
        try:
            print(f"DEBUG Starting OMR processing for image size: {image.shape}")

            # Bước 1: Tiền xử lý ảnh
            preprocessed = self.preprocess_image(image)

            # Bước 2: Căn chỉnh ảnh
            aligned = self.align_image(preprocessed)

            # Bước 3: Trích xuất các vùng ROI
            regions = self.extract_regions(aligned)

            # Bước 4: Xử lý từng vùng
            student_id = self.process_student_id_region(regions["student_id"])
            test_code = self.process_test_code_region(regions["test_code"])

            # Xử lý câu trả lời
            answers = {}

            # Q01-15
            answers_01_15 = self.process_answer_region(
                regions["answers_01_15"], "answers_01_15"
            )
            answers.update(answers_01_15)

            # Q16-30
            answers_16_30 = self.process_answer_region(
                regions["answers_16_30"], "answers_16_30"
            )
            answers.update(answers_16_30)

            # Q31-45
            answers_31_45 = self.process_answer_region(
                regions["answers_31_45"], "answers_31_45"
            )
            answers.update(answers_31_45)

            # Q46-60
            answers_46_60 = self.process_answer_region(
                regions["answers_46_60"], "answers_46_60"
            )
            answers.update(answers_46_60)

            result = {
                "student_id": student_id,
                "test_code": test_code,
                "answers": answers,
                "total_questions": len(answers),
                "processing_status": "success",
            }

            print(f"SUCCESS OMR processing completed successfully")
            print(f"   Student ID: {student_id}")
            print(f"   Test Code: {test_code}")
            print(f"   Total answers: {len(answers)}")

            return result

        except Exception as e:
            logger.error(f"Error in OMR processing: {e}")
            return {
                "student_id": "000000",
                "test_code": "000",
                "answers": {},
                "total_questions": 0,
                "processing_status": "error",
                "error_message": str(e),
            }

    def process_student_id_region(self, region: np.ndarray) -> str:
        """Xử lý vùng Student ID (6 cột số theo layout mới)"""
        try:
            # Chuyển sang grayscale nếu cần
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region

            # Nhị phân hóa
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            self.save_debug_image(binary, "12_student_id_binary", "Student ID binary")

            # Tìm bubbles đã tô - Student ID có 6 cột theo layout mới
            student_id = self.extract_digits_from_grid(
                binary=binary,
                cols=6,
                rows=10,
                region_name="student_id",
                reverse_order=False,  # Thử cả True và False
            )
            logger.info(f"Extracted Student ID: {student_id}")
            return student_id

        except Exception as e:
            logger.error(f"Error processing student ID: {e}")
            return "Chưa thể xác định được Student ID"

    def process_test_code_region(self, region: np.ndarray) -> str:
        """Xử lý vùng Test Code (3 cột số theo layout mới)"""
        try:
            # Chuyển sang grayscale nếu cần
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region

            # Nhị phân hóa
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            self.save_debug_image(binary, "13_test_code_binary", "Test Code binary")

            # Tìm bubbles đã tô - Test Code có 3 cột
            test_code = self.extract_digits_from_grid(
                binary, cols=3, rows=10, region_name="test_code"
            )

            logger.info(f"Extracted Test Code: {test_code}")
            return test_code

        except Exception as e:
            logger.error(f"Error processing test code: {e}")
            return "000"

    # Dán code này để thay thế cho hàm extract_digits_from_grid cũ của bạn

    def extract_digits_from_grid(
        self,
        binary: np.ndarray,
        cols: int,
        rows: int,
        region_name: str,
        reverse_order: bool = False,
    ) -> str:
        """
        Trích xuất số từ grid bubbles với thuật toán cải tiến:
        - StudentID: Sử dụng contour có diện tích lớn nhất
        - Answer bubbles: Sử dụng pixel counting
        - Layout StudentID: 6 cột x 10 hàng (0-9), đọc theo thứ tự cột
        """
        try:
            height, width = binary.shape
            if height == 0 or width == 0:
                logger.warning(f"Region {region_name} is empty.")
                return "?" * cols

            print(
                f"DEBUG {region_name.upper()} - Grid size: {width}x{height}, Layout: {cols}cols x {rows}rows"
            )

            col_width = width // cols
            row_height = height // rows
            result = ""

            # Xác định loại region để áp dụng thuật toán phù hợp
            is_student_id = "student_id" in region_name.lower()

            debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            for col in range(cols):
                if is_student_id:
                    # Thuật toán cho StudentID: Sử dụng contour có diện tích lớn nhất
                    max_contour_area = 0
                    selected_digit = -1  # ← SỬA: Khởi tạo thành -1

                    print(
                        f"    DEBUG Processing StudentID Col{col} (6 cols total, 10 rows 0-9)"
                    )

                    for row in range(rows):
                        # Tính toán vùng cell với offset để bỏ qua dãy số bên trên
                        # Chỉ lấy khu vực hình vuông lớn bên dưới (70% chiều cao)
                        effective_height = int(
                            height * 0.7
                        )  # Chỉ lấy 70% chiều cao bên dưới
                        y_offset = (
                            height - effective_height
                        )  # Offset để bỏ qua phần trên

                        effective_row_height = effective_height // rows

                        # Padding nhỏ hơn để khớp chính xác với bubble tròn
                        padding = 8  # Tăng padding để thu nhỏ vùng xanh
                        x1 = col * col_width + padding
                        x2 = (col + 1) * col_width - padding
                        y1 = y_offset + row * effective_row_height + padding
                        y2 = y_offset + (row + 1) * effective_row_height - padding

                        if x2 <= x1 or y2 <= y1:
                            continue
                        cell = binary[y1:y2, x1:x2]
                        if cell.size == 0:
                            continue

                        # Tìm contour có diện tích lớn nhất và tròn nhất trong cell
                        contours, _ = cv2.findContours(
                            cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        largest_area = 0
                        best_circularity = 0
                        total_white_pixels = np.sum(cell == 255)

                        for contour in contours:
                            area = cv2.contourArea(contour)

                            if area > 20:  # Giảm ngưỡng diện tích tối thiểu
                                perimeter = cv2.arcLength(contour, True)
                                if perimeter > 0:
                                    # Tính độ tròn (circularity)
                                    circularity = (
                                        4 * np.pi * area / (perimeter * perimeter)
                                    )

                                    # Ưu tiên contour có diện tích lớn và tròn
                                    if circularity > 0.3:  # Giảm ngưỡng độ tròn
                                        if area > largest_area:
                                            largest_area = area
                                            best_circularity = circularity

                        # Debug chi tiết từng cell
                        digit_value = row  # Hàng 0 = số 0, hàng 1 = số 1, ...
                        print(
                            f"      Row{row}(digit={digit_value}): area={largest_area:.0f}, circularity={best_circularity:.2f}, pixels={total_white_pixels}"
                        )

                        # Đánh dấu trên debug image với vùng nhỏ hơn khớp với bubble
                        is_filled = largest_area > 30 and best_circularity > 0.3
                        color = (0, 255, 0) if is_filled else (0, 0, 255)
                        # Vẽ rectangle nhỏ hơn để khớp với bubble tròn
                        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 1)
                        result += str(selected_digit)
                        cv2.putText(
                            debug_image,
                            str(digit_value),
                            (x1 + 2, y1 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.25,
                            color,
                            1,
                        )

                        # Cập nhật bubble tốt nhất (diện tích lớn + tròn)
                        if largest_area > max_contour_area and best_circularity > 0.3:
                            max_contour_area = largest_area
                            selected_digit = digit_value

                    # Quyết định kết quả cho cột StudentID với ngưỡng giảm
                    MIN_AREA_THRESHOLD = 20  # Giảm ngưỡng area để detect nhiều hơn
                    MIN_CIRCULARITY = 0.3  # Giảm ngưỡng độ tròn

                    # Tính best_circularity cho cột này với cùng logic
                    best_col_circularity = 0
                    for row in range(rows):
                        # Sử dụng cùng logic tính toán như trên
                        effective_height = int(height * 0.7)
                        y_offset = height - effective_height
                        effective_row_height = effective_height // rows

                        padding = 8
                        x1 = col * col_width + padding
                        x2 = (col + 1) * col_width - padding
                        y1 = y_offset + row * effective_row_height + padding
                        y2 = y_offset + (row + 1) * effective_row_height - padding

                        if x2 <= x1 or y2 <= y1:
                            continue
                        cell = binary[y1:y2, x1:x2]
                        if cell.size == 0:
                            continue

                        contours, _ = cv2.findContours(
                            cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area == max_contour_area and area > 20:
                                perimeter = cv2.arcLength(contour, True)
                                if perimeter > 0:
                                    circularity = (
                                        4 * np.pi * area / (perimeter * perimeter)
                                    )
                                    best_col_circularity = max(
                                        best_col_circularity, circularity
                                    )

                    # if max_contour_area > MIN_AREA_THRESHOLD and selected_digit != -1:
                    #     result += str(selected_digit)
                    #     print(
                    #         f"  SUCCESS Col{col}: Selected digit = {selected_digit} (area: {max_contour_area:.0f}, circularity: {best_col_circularity:.2f})"
                    #     )
                    # else:
                    #     result += "?"
                    #     print(
                    #         f"  FAILED Col{col}: No valid bubble found (max_area: {max_contour_area:.0f}, circularity: {best_col_circularity:.2f}). Resulting in '?'"
                    #     )

                else:
                    # Thuật toán cho Answer bubbles: Sử dụng pixel counting
                    max_pixel_count = 0
                    selected_digit = -1

                    for row in range(rows):
                        padding = 2
                        x1 = col * col_width + padding
                        x2 = (col + 1) * col_width - padding
                        y1 = row * row_height + padding
                        y2 = (row + 1) * row_height - padding

                        if x2 <= x1 or y2 <= y1:
                            continue
                        cell = binary[y1:y2, x1:x2]
                        if cell.size == 0:
                            continue

                        # Đếm số pixel đen (255 trong binary inverted)
                        white_pixels = np.sum(cell == 255)

                        # Cập nhật bubble có số pixel đen nhiều nhất
                        if white_pixels > max_pixel_count:
                            max_pixel_count = white_pixels
                            if reverse_order:
                                selected_digit = (rows - 1) - row
                            else:
                                selected_digit = row

                    # Quyết định kết quả cho cột Answer
                    MIN_PIXEL_THRESHOLD = 100  # Ngưỡng pixel tối thiểu
                    if max_pixel_count > MIN_PIXEL_THRESHOLD and selected_digit != -1:
                        # result += str(selected_digit)
                        print(
                            f"  SUCCESS Col{col}: Selected digit = {selected_digit} (pixels: {max_pixel_count})"
                        )
                    else:
                        result += "?"
                        print(
                            f"  FAILED Col{col}: No valid bubble found (max_pixels: {max_pixel_count}). Resulting in '?'"
                        )

            # Lưu ảnh debug cuối cùng của grid
            self.save_debug_image(
                debug_image,
                f"14_{region_name}_grid_analysis",
                f"Grid Analysis {region_name}: {result}",
            )

            print(f"FINAL {region_name.upper()} Result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error extracting digits from grid {region_name}: {e}")
            return "?" * cols

    def process_answer_region(
        self, region: np.ndarray, region_name: str
    ) -> Dict[int, str]:
        """Xử lý vùng câu trả lời (15 câu x 4 đáp án)"""
        try:
            # Chuyển sang grayscale nếu cần
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region

            # Nhị phân hóa
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            # Xác định số câu bắt đầu dựa vào tên region
            if "01_15" in region_name:
                start_question = 1
            elif "16_30" in region_name:
                start_question = 16
            elif "31_45" in region_name:
                start_question = 31
            elif "46_60" in region_name:
                start_question = 46
            else:
                start_question = 1

            answers = {}
            height, width = binary.shape

            # 15 câu, mỗi câu 4 đáp án (A, B, C, D)
            row_height = height // 15
            col_width = width // 4

            # Tạo debug image
            debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            for question in range(15):
                question_num = start_question + question
                max_pixel_count = 0
                selected_answer = "A"

                y1 = question * row_height
                y2 = (question + 1) * row_height

                for option in range(4):  # A, B, C, D
                    x1 = option * col_width
                    x2 = (option + 1) * col_width

                    # Trích xuất ô đáp án
                    cell = binary[y1:y2, x1:x2]

                    # Đếm số pixel đen (255 trong binary inverted) - thuật toán pixel counting
                    white_pixels = np.sum(cell == 255)

                    # Kiểm tra contour để xác định bubble hợp lệ
                    contours, _ = cv2.findContours(
                        cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    has_significant_contour = False

                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 15:  # Bubble phải có diện tích tối thiểu
                            has_significant_contour = True
                            break

                    # Đánh dấu trên debug image
                    MIN_PIXEL_THRESHOLD = 50  # Ngưỡng pixel tối thiểu
                    is_filled = (
                        white_pixels > MIN_PIXEL_THRESHOLD and has_significant_contour
                    )
                    color = (0, 255, 0) if is_filled else (0, 0, 255)
                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 1)

                    option_letter = chr(ord("A") + option)
                    cv2.putText(
                        debug_image,
                        option_letter,
                        (x1 + 3, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        color,
                        1,
                    )

                    # Hiển thị số pixel đen
                    cv2.putText(
                        debug_image,
                        f"{white_pixels}px",
                        (x1 + 3, y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25,
                        color,
                        1,
                    )

                    # Chọn đáp án có số pixel đen nhiều nhất
                    if white_pixels > max_pixel_count and is_filled:
                        max_pixel_count = white_pixels
                        selected_answer = option_letter

                answers[question_num] = selected_answer

                # Đánh dấu đáp án được chọn
                selected_option = ord(selected_answer) - ord("A")
                cv2.rectangle(
                    debug_image,
                    (selected_option * col_width, y1),
                    ((selected_option + 1) * col_width, y2),
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    debug_image,
                    f"Q{question_num}:{selected_answer}({max_pixel_count}px)",
                    (5, y1 + row_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 0),
                    1,
                )

            # Lưu debug image
            self.save_debug_image(
                debug_image, f"15_{region_name}_answers", f"Answers {region_name}"
            )

            return answers

        except Exception as e:
            logger.error(f"Error processing answer region {region_name}: {e}")
            return {}

    def create_result_visualization(
        self, image: np.ndarray, results: Dict
    ) -> np.ndarray:
        """Tạo ảnh visualization kết quả"""
        try:
            # Tạo bản copy để vẽ
            result_image = image.copy()

            # Vẽ thông tin lên ảnh
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 255, 0)  # Xanh lá
            thickness = 2

            # Vẽ Student ID
            student_id = results.get("student_id", "")
            cv2.putText(
                result_image,
                f"Student ID: {student_id}",
                (50, 50),
                font,
                font_scale,
                color,
                thickness,
            )

            # Vẽ Test Code
            test_code = results.get("test_code", "")
            cv2.putText(
                result_image,
                f"Test Code: {test_code}",
                (50, 100),
                font,
                font_scale,
                color,
                thickness,
            )

            # Vẽ số câu trả lời
            answers = results.get("answers", {})
            cv2.putText(
                result_image,
                f"Total Answers: {len(answers)}",
                (50, 150),
                font,
                font_scale,
                color,
                thickness,
            )

            # Vẽ một số câu trả lời mẫu
            y_pos = 200
            for i, (q_num, answer) in enumerate(list(answers.items())[:10]):
                cv2.putText(
                    result_image,
                    f"Q{q_num}: {answer}",
                    (50, y_pos + i * 30),
                    font,
                    0.6,
                    color,
                    1,
                )

            if len(answers) > 10:
                cv2.putText(
                    result_image, "...", (50, y_pos + 10 * 30), font, 0.6, color, 1
                )

            return result_image

        except Exception as e:
            logger.error(f"Error creating result visualization: {e}")
            return image


if __name__ == "__main__":
    import os
    import cv2

    # Test với file ảnh mẫu
    test_image_path = "../../data/grading/1.jpeg"

    if os.path.exists(test_image_path):
        # Load ảnh
        image = cv2.imread(test_image_path)
        if image is not None:
            processor = OMRDebugProcessor()
            results = processor.process_image(image)
            print("\n=== FINAL RESULTS ===")
            print(f"Student ID: {results.get('student_id', 'Not detected')}")
            print(f"Test Code: {results.get('test_code', 'Not detected')}")
            print(f"Total Answers: {len(results.get('answers', {}))}")
        else:
            print(f"Cannot load image: {test_image_path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Available files in data/grading:")
        grading_dir = "../../data/grading"
        if os.path.exists(grading_dir):
            for file in os.listdir(grading_dir):
                if file.endswith((".jpg", ".png", ".jpeg")):
                    print(f"  - {file}")
