"""
Marker Block Detector Service
Detect marker nhỏ và chia block trong phiếu trả lời trắc nghiệm

Pipeline 7 bước:
1. Đọc ảnh bottom_region (đã chuẩn hóa từ marker lớn)
2. Phát hiện tất cả contour marker nhỏ
3. Sắp xếp marker nhỏ theo tọa độ y và x
4. Phân cụm marker nhỏ thành nhóm theo phần (I, II, III)
5. Tính bounding box cho mỗi nhóm và crop ảnh
6. Trả kết quả dict với các block ảnh
7. Vẽ marker + bounding box để xác minh
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.cluster import DBSCAN
import json

logger = logging.getLogger(__name__)


class MarkerBlockDetector:
    """
    Service chuyên biệt để detect marker nhỏ và chia block
    """

    def __init__(self):
        self.debug_dir = Path("data/grading/marker_debug")
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
            logger.info(f"Marker debug image saved: {filename} - {description}")
        except Exception as e:
            logger.error(f"Error saving marker debug image {step_name}: {e}")

    def detect_and_divide_blocks(self, bottom_region: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Pipeline chính: Detect marker nhỏ và chia block
        
        Args:
            bottom_region: Ảnh vùng bottom đã chuẩn hóa
            
        Returns:
            Dict chứa các block ảnh đã chia theo phần
        """
        try:
            # Bước 1: Đọc và tiền xử lý ảnh bottom_region
            processed_image = self.preprocess_bottom_region(bottom_region)
            
            # Bước 2: Phát hiện tất cả contour marker nhỏ
            small_markers = self.detect_small_markers(processed_image, bottom_region)
            
            # Bước 3: Sắp xếp marker nhỏ theo tọa độ
            sorted_markers = self.sort_markers_by_coordinates(small_markers)
            
            # Bước 4: Phân cụm marker nhỏ thành nhóm theo phần
            marker_groups = self.cluster_markers_by_sections(sorted_markers)
            
            # Bước 5: Tính bounding box và crop ảnh cho mỗi nhóm
            block_images = self.crop_blocks_from_groups(bottom_region, marker_groups)

            # Fallback: Nếu không có block nào, sử dụng ratio-based division
            if not any(block_images.values()):
                logger.warning("No markers detected, using ratio-based fallback division")
                block_images = self.fallback_ratio_division(bottom_region)

            # Bước 6: Trả kết quả
            result = self.format_result(block_images)
            
            # Bước 7: Vẽ marker + bounding box để xác minh
            self.visualize_markers_and_blocks(bottom_region, sorted_markers, marker_groups)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in marker block detection: {e}")
            return {
                "part1_blocks": [],
                "part2_blocks": [],
                "part3_blocks": [],
                "error": str(e)
            }

    def preprocess_bottom_region(self, bottom_region: np.ndarray) -> np.ndarray:
        """
        Bước 1: Tiền xử lý ảnh bottom_region
        """
        # Convert grayscale nếu cần
        if len(bottom_region.shape) == 3:
            gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = bottom_region.copy()
        
        # Adaptive threshold để làm nổi bật marker nhỏ
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # Thêm threshold thông thường để so sánh
        _, binary_simple = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations để làm sạch
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        self.save_debug_image(gray, "01_bottom_region_gray", "Bottom region grayscale")
        self.save_debug_image(binary_simple, "02a_bottom_region_binary_simple", "Bottom region binary simple")
        self.save_debug_image(binary, "02b_bottom_region_binary_adaptive", "Bottom region binary adaptive")
        self.save_debug_image(cleaned, "03_bottom_region_cleaned", "Bottom region cleaned")
        
        return cleaned

    def detect_small_markers(self, processed_image: np.ndarray, original_image: np.ndarray) -> List[Dict]:
        """
        Bước 2: Phát hiện tất cả contour marker nhỏ
        """
        # Tìm contours
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        small_markers = []
        marked_image = original_image.copy()
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Lọc các contour có diện tích trung bình (giữa bubble và marker lớn)
            # Mở rộng range để phát hiện nhiều marker hơn
            if 50 < area < 1500:  # Marker nhỏ - range rộng hơn
                # Tính aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Aspect ratio gần vuông (0.7–1.4) - linh hoạt hơn
                if 0.7 <= aspect_ratio <= 1.4:
                    # Kiểm tra không nằm quá sâu trong phần trả lời
                    image_height = original_image.shape[0]
                    if y < image_height * 0.9:  # Không ở 10% cuối ảnh
                        
                        center = (x + w // 2, y + h // 2)
                        marker_info = {
                            'center': center,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'contour': contour
                        }
                        small_markers.append(marker_info)
                        
                        # Vẽ marker lên ảnh debug
                        cv2.drawContours(marked_image, [contour], -1, (0, 255, 0), 2)
                        cv2.circle(marked_image, center, 5, (255, 0, 0), -1)
                        cv2.putText(marked_image, f"M{len(small_markers)}", 
                                  (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        self.save_debug_image(marked_image, "04_small_markers_detected", 
                             f"Small markers detected: {len(small_markers)}")
        
        logger.info(f"Detected {len(small_markers)} small markers")
        return small_markers

    def sort_markers_by_coordinates(self, markers: List[Dict]) -> List[Dict]:
        """
        Bước 3: Sắp xếp marker nhỏ theo tọa độ y và x
        """
        # Sắp xếp theo y trước (để chia hàng), sau đó theo x (để chia cột)
        sorted_markers = sorted(markers, key=lambda m: (m['center'][1], m['center'][0]))
        
        # Tạo ảnh debug với số thứ tự
        if markers:
            debug_image = np.zeros((markers[0]['bbox'][1] + 500, markers[0]['bbox'][0] + 500, 3), dtype=np.uint8)
            for i, marker in enumerate(sorted_markers):
                center = marker['center']
                cv2.circle(debug_image, center, 10, (0, 255, 0), -1)
                cv2.putText(debug_image, str(i+1), 
                          (center[0] - 10, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.save_debug_image(debug_image, "05_markers_sorted", "Markers sorted by coordinates")
        
        return sorted_markers

    def cluster_markers_by_sections(self, sorted_markers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Bước 4: Phân cụm marker nhỏ thành nhóm theo phần (I, II, III)
        """
        if not sorted_markers:
            return {"part1": [], "part2": [], "part3": []}
        
        # Lấy tọa độ y của tất cả markers
        y_coords = [marker['center'][1] for marker in sorted_markers]
        
        # Sử dụng DBSCAN để cluster theo y-coordinate
        y_array = np.array(y_coords).reshape(-1, 1)
        
        # Tham số DBSCAN: eps = khoảng cách tối đa giữa các điểm trong cùng cluster
        eps = 50  # 50 pixels
        dbscan = DBSCAN(eps=eps, min_samples=2)
        clusters = dbscan.fit_predict(y_array)
        
        # Nhóm markers theo cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(sorted_markers[i])
        
        # Sắp xếp clusters theo y trung bình
        sorted_clusters = []
        for cluster_id, markers in cluster_groups.items():
            if cluster_id != -1:  # Bỏ qua noise points
                avg_y = np.mean([m['center'][1] for m in markers])
                sorted_clusters.append((avg_y, markers))
        
        sorted_clusters.sort(key=lambda x: x[0])
        
        # Phân chia thành 3 phần
        marker_groups = {"part1": [], "part2": [], "part3": []}

        if len(sorted_clusters) >= 3:
            marker_groups["part1"] = sorted_clusters[0][1]  # PHẦN I
            marker_groups["part2"] = sorted_clusters[1][1]  # PHẦN II
            marker_groups["part3"] = sorted_clusters[2][1]  # PHẦN III
        elif len(sorted_clusters) == 2:
            marker_groups["part1"] = sorted_clusters[0][1]
            marker_groups["part2"] = sorted_clusters[1][1]
            # Tạo part3 giả từ markers cuối
            marker_groups["part3"] = sorted_clusters[1][1][-2:] if len(sorted_clusters[1][1]) > 2 else []
        elif len(sorted_clusters) == 1:
            # Chia markers thành 3 phần theo tỷ lệ
            all_markers = sorted_clusters[0][1]
            total = len(all_markers)
            if total >= 6:
                marker_groups["part1"] = all_markers[:total//3]
                marker_groups["part2"] = all_markers[total//3:2*total//3]
                marker_groups["part3"] = all_markers[2*total//3:]
            else:
                marker_groups["part1"] = all_markers
        
        # Debug visualization
        self.visualize_clusters(sorted_markers, clusters, marker_groups)
        
        return marker_groups

    def crop_blocks_from_groups(self, bottom_region: np.ndarray, marker_groups: Dict[str, List[Dict]]) -> Dict[str, List[np.ndarray]]:
        """
        Bước 5: Tính bounding box và crop ảnh cho mỗi nhóm
        """
        block_images = {"part1": [], "part2": [], "part3": []}
        height, width = bottom_region.shape[:2]

        # PHẦN I: 4 block dọc
        if marker_groups.get("part1"):
            blocks = self.divide_part1_blocks(bottom_region, marker_groups["part1"])
            block_images["part1"] = blocks
        else:
            # Fallback cho PHẦN I: 60% đầu chia 4 cột
            part1_height = int(height * 0.6)
            part1_region = bottom_region[0:part1_height, :]
            blocks = self.divide_region_into_columns(part1_region, 4, "part1_fallback")
            block_images["part1"] = blocks

        # PHẦN II: 8 block nhỏ
        if marker_groups.get("part2"):
            blocks = self.divide_part2_blocks(bottom_region, marker_groups["part2"])
            block_images["part2"] = blocks
        else:
            # Fallback cho PHẦN II: 25% giữa chia 8 cột
            part2_start = int(height * 0.6)
            part2_height = int(height * 0.25)
            part2_region = bottom_region[part2_start:part2_start + part2_height, :]
            blocks = self.divide_region_into_columns(part2_region, 8, "part2_fallback")
            block_images["part2"] = blocks

        # PHẦN III: 6 block số
        if marker_groups.get("part3"):
            blocks = self.divide_part3_blocks(bottom_region, marker_groups["part3"])
            block_images["part3"] = blocks
        else:
            # Fallback cho PHẦN III: 15% cuối chia 6 cột
            part3_start = int(height * 0.85)
            part3_region = bottom_region[part3_start:, :]
            blocks = self.divide_region_into_columns(part3_region, 6, "part3_fallback")
            block_images["part3"] = blocks

        return block_images

    def divide_region_into_columns(self, region: np.ndarray, num_columns: int, prefix: str) -> List[np.ndarray]:
        """
        Chia một vùng thành số cột nhất định
        """
        height, width = region.shape[:2]
        col_width = width // num_columns
        blocks = []

        for i in range(num_columns):
            start_x = i * col_width
            end_x = (i + 1) * col_width if i < num_columns - 1 else width

            block = region[:, start_x:end_x]
            blocks.append(block)

            # Save debug image
            self.save_debug_image(block, f"06_{prefix}_col_{i+1}", f"{prefix} Column {i+1}")

        return blocks

    def divide_part1_blocks(self, region: np.ndarray, markers: List[Dict]) -> List[np.ndarray]:
        """Chia PHẦN I thành 4 block dọc"""
        if not markers:
            return []
        
        # Tính bounding box tổng thể của PHẦN I
        all_x = [m['bbox'][0] for m in markers] + [m['bbox'][0] + m['bbox'][2] for m in markers]
        all_y = [m['bbox'][1] for m in markers] + [m['bbox'][1] + m['bbox'][3] for m in markers]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Mở rộng bounding box
        padding = 20
        min_x = max(0, min_x - padding)
        max_x = min(region.shape[1], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(region.shape[0], max_y + padding)
        
        # Crop toàn bộ PHẦN I
        part1_region = region[min_y:max_y, min_x:max_x]
        self.save_debug_image(part1_region, "06_part1_full_region", "Part I full region")
        
        # Chia thành 4 block theo x
        width = part1_region.shape[1]
        block_width = width // 4
        
        blocks = []
        for i in range(4):
            start_x = i * block_width
            end_x = (i + 1) * block_width if i < 3 else width
            
            block = part1_region[:, start_x:end_x]
            blocks.append(block)
            
            # Save debug image
            self.save_debug_image(block, f"07_part1_block_{i+1}", f"Part I Block {i+1}")
        
        return blocks

    def divide_part2_blocks(self, region: np.ndarray, markers: List[Dict]) -> List[np.ndarray]:
        """Chia PHẦN II thành 8 block nhỏ"""
        if not markers:
            return []
        
        # Tương tự PHẦN I nhưng chia thành 8 block
        all_x = [m['bbox'][0] for m in markers] + [m['bbox'][0] + m['bbox'][2] for m in markers]
        all_y = [m['bbox'][1] for m in markers] + [m['bbox'][1] + m['bbox'][3] for m in markers]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        padding = 15
        min_x = max(0, min_x - padding)
        max_x = min(region.shape[1], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(region.shape[0], max_y + padding)
        
        part2_region = region[min_y:max_y, min_x:max_x]
        self.save_debug_image(part2_region, "08_part2_full_region", "Part II full region")
        
        # Chia thành 8 block
        width = part2_region.shape[1]
        block_width = width // 8
        
        blocks = []
        for i in range(8):
            start_x = i * block_width
            end_x = (i + 1) * block_width if i < 7 else width
            
            block = part2_region[:, start_x:end_x]
            blocks.append(block)
            
            self.save_debug_image(block, f"09_part2_block_{i+1}", f"Part II Block {i+1}")
        
        return blocks

    def divide_part3_blocks(self, region: np.ndarray, markers: List[Dict]) -> List[np.ndarray]:
        """Chia PHẦN III thành 6 block số"""
        if not markers:
            return []
        
        # Tương tự nhưng chia thành 6 block
        all_x = [m['bbox'][0] for m in markers] + [m['bbox'][0] + m['bbox'][2] for m in markers]
        all_y = [m['bbox'][1] for m in markers] + [m['bbox'][1] + m['bbox'][3] for m in markers]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        padding = 15
        min_x = max(0, min_x - padding)
        max_x = min(region.shape[1], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(region.shape[0], max_y + padding)
        
        part3_region = region[min_y:max_y, min_x:max_x]
        self.save_debug_image(part3_region, "10_part3_full_region", "Part III full region")
        
        # Chia thành 6 block
        width = part3_region.shape[1]
        block_width = width // 6
        
        blocks = []
        for i in range(6):
            start_x = i * block_width
            end_x = (i + 1) * block_width if i < 5 else width
            
            block = part3_region[:, start_x:end_x]
            blocks.append(block)
            
            self.save_debug_image(block, f"11_part3_block_{i+1}", f"Part III Block {i+1}")
        
        return blocks

    def format_result(self, block_images: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """
        Bước 6: Format kết quả theo yêu cầu
        """
        result = {
            "part1_blocks": block_images.get("part1", []),
            "part2_blocks": block_images.get("part2", []),
            "part3_blocks": block_images.get("part3", []),
            "summary": {
                "part1_count": len(block_images.get("part1", [])),
                "part2_count": len(block_images.get("part2", [])),
                "part3_count": len(block_images.get("part3", [])),
                "total_blocks": sum(len(blocks) for blocks in block_images.values())
            }
        }
        
        # Save summary
        summary_file = self.debug_dir / "block_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "part1_blocks": result["summary"]["part1_count"],
                "part2_blocks": result["summary"]["part2_count"], 
                "part3_blocks": result["summary"]["part3_count"],
                "total_blocks": result["summary"]["total_blocks"]
            }, f, indent=2, ensure_ascii=False)
        
        return result

    def visualize_clusters(self, markers: List[Dict], clusters: np.ndarray, marker_groups: Dict):
        """Visualize marker clusters"""
        if not markers:
            return
        
        # Tạo ảnh với màu khác nhau cho mỗi cluster
        max_x = max(m['center'][0] for m in markers) + 100
        max_y = max(m['center'][1] for m in markers) + 100
        
        cluster_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (marker, cluster_id) in enumerate(zip(markers, clusters)):
            center = marker['center']
            color = colors[cluster_id % len(colors)] if cluster_id != -1 else (128, 128, 128)
            
            cv2.circle(cluster_image, center, 8, color, -1)
            cv2.putText(cluster_image, f"C{cluster_id}", 
                       (center[0] - 15, center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.save_debug_image(cluster_image, "05b_marker_clusters", "Marker clusters visualization")

    def visualize_markers_and_blocks(self, bottom_region: np.ndarray, markers: List[Dict], marker_groups: Dict):
        """
        Bước 7: Vẽ marker + bounding box để xác minh
        """
        visualization = bottom_region.copy()
        
        # Vẽ tất cả markers
        for marker in markers:
            center = marker['center']
            bbox = marker['bbox']
            
            # Vẽ marker
            cv2.circle(visualization, center, 5, (0, 255, 0), -1)
            cv2.rectangle(visualization, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
        
        # Vẽ bounding box cho mỗi phần
        colors = {"part1": (255, 0, 0), "part2": (0, 255, 0), "part3": (0, 0, 255)}
        
        for part_name, part_markers in marker_groups.items():
            if not part_markers:
                continue
                
            # Tính bounding box của phần
            all_x = [m['bbox'][0] for m in part_markers] + [m['bbox'][0] + m['bbox'][2] for m in part_markers]
            all_y = [m['bbox'][1] for m in part_markers] + [m['bbox'][1] + m['bbox'][3] for m in part_markers]
            
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            # Vẽ bounding box
            color = colors.get(part_name, (255, 255, 255))
            cv2.rectangle(visualization, (min_x - 20, min_y - 20), 
                         (max_x + 20, max_y + 20), color, 3)
            
            # Thêm label
            cv2.putText(visualization, part_name.upper(), 
                       (min_x - 20, min_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        self.save_debug_image(visualization, "12_final_visualization", "Final markers and blocks visualization")

    def fallback_ratio_division(self, bottom_region: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Fallback method: Chia block theo tỷ lệ khi không phát hiện được marker
        """
        logger.info("Using fallback ratio-based division")

        height, width = bottom_region.shape[:2]
        block_images = {"part1": [], "part2": [], "part3": []}

        # PHẦN I: 60% đầu, chia thành 4 cột
        part1_height = int(height * 0.6)
        part1_region = bottom_region[0:part1_height, :]

        col_width = width // 4
        for i in range(4):
            start_x = i * col_width
            end_x = (i + 1) * col_width if i < 3 else width
            block = part1_region[:, start_x:end_x]
            block_images["part1"].append(block)

            # Save debug
            self.save_debug_image(block, f"13_fallback_part1_block_{i+1}", f"Fallback Part I Block {i+1}")

        # PHẦN II: 25% giữa, chia thành 8 block
        part2_start = part1_height
        part2_height = int(height * 0.25)
        part2_region = bottom_region[part2_start:part2_start + part2_height, :]

        block_width = width // 8
        for i in range(8):
            start_x = i * block_width
            end_x = (i + 1) * block_width if i < 7 else width
            block = part2_region[:, start_x:end_x]
            block_images["part2"].append(block)

            # Save debug
            self.save_debug_image(block, f"14_fallback_part2_block_{i+1}", f"Fallback Part II Block {i+1}")

        # PHẦN III: 15% cuối, chia thành 6 block
        part3_start = part2_start + part2_height
        part3_region = bottom_region[part3_start:, :]

        block_width = width // 6
        for i in range(6):
            start_x = i * block_width
            end_x = (i + 1) * block_width if i < 5 else width
            block = part3_region[:, start_x:end_x]
            block_images["part3"].append(block)

            # Save debug
            self.save_debug_image(block, f"15_fallback_part3_block_{i+1}", f"Fallback Part III Block {i+1}")

        # Save fallback visualization
        fallback_viz = bottom_region.copy()

        # Vẽ các đường chia
        cv2.line(fallback_viz, (0, part1_height), (width, part1_height), (255, 0, 0), 3)
        cv2.line(fallback_viz, (0, part2_start + part2_height), (width, part2_start + part2_height), (0, 255, 0), 3)

        # Vẽ chia cột cho từng phần
        for i in range(1, 4):
            x = i * (width // 4)
            cv2.line(fallback_viz, (x, 0), (x, part1_height), (255, 0, 0), 2)

        for i in range(1, 8):
            x = i * (width // 8)
            cv2.line(fallback_viz, (x, part2_start), (x, part2_start + part2_height), (0, 255, 0), 2)

        for i in range(1, 6):
            x = i * (width // 6)
            cv2.line(fallback_viz, (x, part3_start), (x, height), (0, 0, 255), 2)

        # Thêm labels
        cv2.putText(fallback_viz, "PART I (4 blocks)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(fallback_viz, "PART II (8 blocks)", (10, part2_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(fallback_viz, "PART III (6 blocks)", (10, part3_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.save_debug_image(fallback_viz, "16_fallback_visualization", "Fallback ratio-based division")

        return block_images


# Global instance
marker_block_detector = MarkerBlockDetector()
