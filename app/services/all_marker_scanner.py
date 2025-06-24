"""
All Marker Scanner Service
Quét và đánh dấu tất cả các marker (lớn và nhỏ) trên một ảnh tổng hợp

Features:
- Phát hiện marker vuông lớn (large markers)
- Phát hiện marker vuông nhỏ (small markers)  
- Đánh dấu và phân biệt 2 loại marker bằng màu sắc
- Tạo ảnh tổng hợp với tất cả marker được label
- Thống kê số lượng và vị trí marker
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.cluster import DBSCAN
import json

logger = logging.getLogger(__name__)


class AllMarkerScanner:
    """
    Service quét và đánh dấu tất cả marker trong ảnh OMR
    """

    def __init__(self):
        self.debug_dir = Path("data/grading/all_markers_debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Xóa debug images cũ
        for file in self.debug_dir.glob("*.jpg"):
            file.unlink()
        
        # Màu sắc cho các loại marker
        self.colors = {
            'large_marker': (0, 0, 255),      # Đỏ cho marker lớn
            'small_marker': (0, 255, 0),      # Xanh lá cho marker nhỏ
            'text_large': (255, 255, 255),    # Trắng cho text marker lớn
            'text_small': (0, 0, 0),          # Đen cho text marker nhỏ
            'bbox_large': (255, 0, 0),        # Xanh dương cho bbox marker lớn
            'bbox_small': (255, 255, 0)       # Vàng cho bbox marker nhỏ
        }

    def save_debug_image(self, image: np.ndarray, step_name: str, description: str = ""):
        """Lưu ảnh debug với tên mô tả"""
        try:
            filename = f"{step_name}.jpg"
            filepath = self.debug_dir / filename
            cv2.imwrite(str(filepath), image)
            logger.info(f"All markers debug image saved: {filename} - {description}")
        except Exception as e:
            logger.error(f"Error saving all markers debug image {step_name}: {e}")

    def scan_all_markers(self, image_input) -> Dict[str, Any]:
        """
        Quét tất cả marker trong ảnh và tạo ảnh đánh dấu tổng hợp

        Args:
            image_input: Đường dẫn đến ảnh (str) hoặc numpy array

        Returns:
            Dict chứa thông tin tất cả marker và đường dẫn ảnh kết quả
        """
        try:
            # Load image - handle both path and numpy array
            if isinstance(image_input, str):
                logger.info(f"Loading image from path: {image_input}")
                original_image = cv2.imread(image_input)
                if original_image is None:
                    raise ValueError(f"Cannot load image: {image_input}")
            elif isinstance(image_input, np.ndarray):
                logger.info("Using provided numpy array image")
                original_image = image_input.copy()
            else:
                raise ValueError(f"Invalid image input type: {type(image_input)}")

            logger.info(f"Image dimensions: {original_image.shape}")

            # Lưu ảnh gốc
            self.save_debug_image(original_image, "01_original_image", "Original input image")
            
            # Bước 1: Phát hiện marker lớn
            large_markers = self.detect_large_markers(original_image)
            
            # Bước 2: Phát hiện marker nhỏ
            small_markers = self.detect_small_markers(original_image)
            
            # Bước 3: Tạo ảnh tổng hợp với tất cả marker
            all_markers_image = self.create_all_markers_visualization(
                original_image, large_markers, small_markers
            )
            
            # Bước 4: Tạo thống kê
            statistics = self.generate_marker_statistics(large_markers, small_markers)
            
            # Bước 5: Lưu kết quả
            result = {
                "success": True,
                "large_markers": large_markers,
                "small_markers": small_markers,
                "statistics": statistics,
                "all_markers_image_path": str(self.debug_dir / "05_all_markers_final.jpg"),
                "debug_dir": str(self.debug_dir)
            }
            
            # Lưu thống kê ra file JSON
            stats_file = self.debug_dir / "marker_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in all marker scanning: {e}")
            return {
                "success": False,
                "error": str(e),
                "large_markers": [],
                "small_markers": [],
                "statistics": {}
            }

    def detect_large_markers(self, image: np.ndarray) -> List[Dict]:
        """
        Phát hiện marker vuông lớn
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Multiple thresholds để tìm marker đen lớn
        _, binary1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        _, binary2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        _, binary3 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        # Adaptive threshold
        binary4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)

        # Combine tất cả threshold
        binary = cv2.bitwise_or(cv2.bitwise_or(binary1, binary2), cv2.bitwise_or(binary3, binary4))
        
        # Morphological operations để làm sạch
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debug: Vẽ tất cả contours để xem
        all_contours_image = image.copy()
        cv2.drawContours(all_contours_image, contours, -1, (0, 255, 255), 1)
        self.save_debug_image(all_contours_image, "02a_all_contours_large", f"All contours for large markers: {len(contours)}")

        large_markers = []
        marked_image = image.copy()

        logger.info(f"Found {len(contours)} total contours for large marker detection")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Debug: Log tất cả contours
            if i < 20:  # Chỉ log 20 contours đầu
                logger.info(f"Large contour {i}: area={area:.1f}")

            # Lọc marker lớn theo diện tích (19x19 = 361 pixels)
            # Dựa trên kết quả thực tế: area 192-218 là marker lớn
            if 180 < area < 300:  # Marker lớn 19x19 (thực tế area ~200px)
                logger.info(f"Large marker candidate {i}: area={area:.1f}")
                # Xấp xỉ contour thành đa giác
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                logger.info(f"Large marker {i}: area={area:.1f}, approx_points={len(approx)}")

                # Kiểm tra có phải hình vuông không (linh hoạt hơn cho small markers)
                if len(approx) >= 3:  # Accept triangles and above
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    logger.info(f"Large marker {i}: area={area:.1f}, aspect_ratio={aspect_ratio:.2f}")

                    # Tỉ lệ khung hình ≈ 1.0 (vuông) - linh hoạt hơn
                    if 0.7 <= aspect_ratio <= 1.4:
                        center = (x + w // 2, y + h // 2)
                        marker_info = {
                            'id': f"L{len(large_markers) + 1}",
                            'center': center,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'contour': contour.tolist(),  # Convert to list for JSON
                            'type': 'large'
                        }
                        large_markers.append(marker_info)
                        
                        # Vẽ marker lớn
                        cv2.drawContours(marked_image, [contour], -1, self.colors['large_marker'], 3)
                        cv2.rectangle(marked_image, (x, y), (x + w, y + h), self.colors['bbox_large'], 2)
                        cv2.circle(marked_image, center, 8, self.colors['large_marker'], -1)
                        cv2.putText(marked_image, marker_info['id'], 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text_large'], 2)
        
        self.save_debug_image(marked_image, "02_large_markers_detected",
                             f"Large markers detected: {len(large_markers)}")

        logger.info(f"Detected {len(large_markers)} large markers (19x19px, area 250-600)")
        if large_markers:
            for marker in large_markers:
                logger.info(f"  {marker['id']}: center{marker['center']}, area={marker['area']:.1f}, ratio={marker['aspect_ratio']:.2f}")

        return large_markers

    def detect_small_markers(self, image: np.ndarray) -> List[Dict]:
        """
        Phát hiện marker vuông nhỏ với smart algorithm
        Target: 18 small markers
        """
        # Use smart algorithm to find exactly 18 small markers
        return self.smart_detect_small_markers(image)

    def smart_detect_small_markers(self, image: np.ndarray) -> List[Dict]:
        """
        Smart algorithm để tìm chính xác 18 small markers
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Step 1: Find all potential square candidates
        candidates = self.find_square_candidates(gray)

        # Step 2: Filter by size (small markers)
        small_candidates = [c for c in candidates if c['area'] <= 150]

        # Step 3: Smart spatial filtering to get exactly 18
        final_small_markers = self.spatial_filter_small_markers(small_candidates, image.shape)

        # Step 4: Convert to expected format and create visualization
        small_markers = []
        marked_image = image.copy()

        for i, candidate in enumerate(final_small_markers):
            center = candidate['center']
            bbox = candidate['bbox']

            marker_info = {
                'id': f"S{i + 1}",
                'center': center,
                'bbox': bbox,
                'area': candidate['area'],
                'aspect_ratio': candidate['aspect_ratio'],
                'contour': candidate.get('contour', []),
                'type': 'small'
            }
            small_markers.append(marker_info)

            # Vẽ marker nhỏ
            x, y, w, h = bbox
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), self.colors['bbox_small'], 1)
            cv2.circle(marked_image, center, 4, self.colors['small_marker'], -1)
            cv2.putText(marked_image, marker_info['id'],
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_small'], 1)

        self.save_debug_image(marked_image, "03_small_markers_detected",
                             f"Smart small markers detected: {len(small_markers)}")

        logger.info(f"Smart detected {len(small_markers)} small markers (target: 18)")
        for marker in small_markers:
            logger.info(f"  {marker['id']}: center{marker['center']}, area={marker['area']:.1f}, "
                       f"ratio={marker['aspect_ratio']:.2f}")

        return small_markers

    def find_square_candidates(self, gray: np.ndarray) -> List[Dict]:
        """Tìm tất cả candidates có hình dạng vuông (KHÔNG phải tròn)"""
        candidates = []

        # Multiple threshold approaches
        thresholds = [80, 100, 120]

        for thresh in thresholds:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by area (reasonable range for markers)
                if 30 < area < 500:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # STRICT: Must be exactly 4 corners (square/rectangle)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h

                        # Square-like aspect ratio (chặt chẽ hơn)
                        if 0.8 <= aspect_ratio <= 1.25:
                            # Calculate solidity (area/convex_hull_area)
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = area / hull_area if hull_area > 0 else 0

                            # Calculate rectangularity (area vs bounding box area)
                            bbox_area = w * h
                            rectangularity = area / bbox_area if bbox_area > 0 else 0

                            # BALANCED filters to distinguish from circles:
                            # 1. Good solidity (filled shape)
                            # 2. Good rectangularity (square-like, not round)
                            # 3. Reasonable circularity (not too round)
                            if solidity > 0.75 and rectangularity > 0.65:
                                # Calculate circularity to reject round shapes
                                perimeter = cv2.arcLength(contour, True)
                                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                                # Reject if too circular (likely a filled bubble)
                                # Based on debug: squares have circularity ~0.785, circles >0.8
                                if circularity < 0.82:  # Allow squares with circularity up to 0.82
                                    center = (x + w // 2, y + h // 2)

                                    candidate = {
                                        'center': center,
                                        'area': area,
                                        'aspect_ratio': aspect_ratio,
                                        'solidity': solidity,
                                        'rectangularity': rectangularity,
                                        'circularity': circularity,
                                        'bbox': (x, y, w, h),
                                        'contour': contour,
                                        'approx_points': len(approx)
                                    }
                                    candidates.append(candidate)

        # Remove duplicates
        unique_candidates = self.remove_duplicate_candidates(candidates)
        return unique_candidates

    def spatial_filter_small_markers(self, candidates: List[Dict], image_shape: Tuple) -> List[Dict]:
        """Lọc small markers dựa trên pattern không gian để lấy đúng 18 markers"""
        if len(candidates) <= 18:
            return candidates

        # Small markers thường tạo thành grid pattern
        if len(candidates) < 5:
            return candidates

        # Extract positions
        positions = np.array([candidate['center'] for candidate in candidates])

        # Use DBSCAN to find clusters
        clustering = DBSCAN(eps=50, min_samples=2).fit(positions)
        labels = clustering.labels_

        # Score candidates based on clustering and grid-like arrangement
        scored_candidates = []

        for i, candidate in enumerate(candidates):
            score = 0

            # Prefer candidates in clusters (grid-like)
            if labels[i] != -1:  # Not noise
                cluster_size = np.sum(labels == labels[i])
                score += cluster_size * 2

            # Prefer smaller areas (more likely to be small markers)
            score += max(0, 100 - candidate['area']) / 10

            # Prefer better aspect ratios (closer to 1.0 = square)
            score += 10 - abs(candidate['aspect_ratio'] - 1.0) * 5

            # Prefer higher solidity (filled shapes)
            score += candidate['solidity'] * 5

            # IMPORTANT: Prefer higher rectangularity (square-like, not round)
            score += candidate.get('rectangularity', 0) * 10

            # IMPORTANT: Penalize high circularity (round shapes)
            circularity = candidate.get('circularity', 1.0)
            score -= circularity * 8  # Subtract points for being too circular

            # Prefer exactly 4 corners (true squares/rectangles)
            if candidate.get('approx_points', 0) == 4:
                score += 5

            scored_candidates.append((score, candidate))

        # Sort by score and take top 18
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [candidate for score, candidate in scored_candidates[:18]]

    def remove_duplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Loại bỏ candidates trùng lặp"""
        unique = []

        for candidate in candidates:
            is_duplicate = False
            center = candidate['center']

            for existing in unique:
                existing_center = existing['center']
                distance = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)

                if distance < 15:  # Within 15 pixels
                    is_duplicate = True
                    # Keep the one with better quality (higher solidity)
                    if candidate['solidity'] > existing['solidity']:
                        unique.remove(existing)
                        unique.append(candidate)
                    break

            if not is_duplicate:
                unique.append(candidate)

        return unique


    def create_all_markers_visualization(self, original_image: np.ndarray, 
                                       large_markers: List[Dict], 
                                       small_markers: List[Dict]) -> np.ndarray:
        """
        Tạo ảnh tổng hợp với tất cả marker được đánh dấu
        """
        # Tạo ảnh tổng hợp
        all_markers_image = original_image.copy()
        
        # Vẽ tất cả marker lớn
        for marker in large_markers:
            center = marker['center']
            x, y, w, h = marker['bbox']
            
            # Vẽ contour và bounding box
            cv2.rectangle(all_markers_image, (x, y), (x + w, y + h), self.colors['bbox_large'], 3)
            cv2.circle(all_markers_image, center, 10, self.colors['large_marker'], -1)
            
            # Label với background
            label = marker['id']
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(all_markers_image, (x, y - 30), (x + label_size[0] + 10, y - 5), 
                         self.colors['large_marker'], -1)
            cv2.putText(all_markers_image, label, (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text_large'], 2)
        
        # Vẽ tất cả marker nhỏ
        for marker in small_markers:
            center = marker['center']
            x, y, w, h = marker['bbox']
            
            # Vẽ contour và bounding box
            cv2.rectangle(all_markers_image, (x, y), (x + w, y + h), self.colors['bbox_small'], 2)
            cv2.circle(all_markers_image, center, 5, self.colors['small_marker'], -1)
            
            # Label với background
            label = marker['id']
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(all_markers_image, (x, y - 20), (x + label_size[0] + 5, y - 2), 
                         self.colors['small_marker'], -1)
            cv2.putText(all_markers_image, label, (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_small'], 1)
        
        # Thêm legend
        self.add_legend(all_markers_image, len(large_markers), len(small_markers))
        
        # Lưu ảnh tổng hợp
        self.save_debug_image(all_markers_image, "05_all_markers_final", 
                             f"All markers visualization: {len(large_markers)} large + {len(small_markers)} small")
        
        return all_markers_image

    def add_legend(self, image: np.ndarray, large_count: int, small_count: int):
        """
        Thêm legend vào ảnh
        """
        height, width = image.shape[:2]
        
        # Vị trí legend (góc trên phải)
        legend_x = width - 300
        legend_y = 30
        
        # Background cho legend
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (width - 10, legend_y + 120), (255, 255, 255), -1)
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (width - 10, legend_y + 120), (0, 0, 0), 2)
        
        # Title
        cv2.putText(image, "MARKER LEGEND", (legend_x, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Large markers legend
        cv2.rectangle(image, (legend_x, legend_y + 35), (legend_x + 20, legend_y + 55), 
                     self.colors['large_marker'], -1)
        cv2.putText(image, f"Large Markers: {large_count}", (legend_x + 30, legend_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Small markers legend
        cv2.rectangle(image, (legend_x, legend_y + 65), (legend_x + 20, legend_y + 85), 
                     self.colors['small_marker'], -1)
        cv2.putText(image, f"Small Markers: {small_count}", (legend_x + 30, legend_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Total
        cv2.putText(image, f"Total: {large_count + small_count}", (legend_x, legend_y + 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def generate_marker_statistics(self, large_markers: List[Dict], small_markers: List[Dict]) -> Dict:
        """
        Tạo thống kê về marker
        """
        statistics = {
            "total_markers": len(large_markers) + len(small_markers),
            "large_markers": {
                "count": len(large_markers),
                "average_area": np.mean([m['area'] for m in large_markers]) if large_markers else 0,
                "average_aspect_ratio": np.mean([m['aspect_ratio'] for m in large_markers]) if large_markers else 0,
                "positions": [{"id": m['id'], "center": m['center'], "area": m['area']} for m in large_markers]
            },
            "small_markers": {
                "count": len(small_markers),
                "average_area": np.mean([m['area'] for m in small_markers]) if small_markers else 0,
                "average_aspect_ratio": np.mean([m['aspect_ratio'] for m in small_markers]) if small_markers else 0,
                "positions": [{"id": m['id'], "center": m['center'], "area": m['area']} for m in small_markers]
            },
            "detection_summary": {
                "large_marker_range": "150-500 pixels area (19x19 ≈ 361px, thực tế ~200px)",
                "small_marker_range": "25-150 pixels area (9x9 ≈ 81px)",
                "aspect_ratio_large": "0.85-1.15",
                "aspect_ratio_small": "0.8-1.25",
                "expected_large_size": "19x19 pixels",
                "expected_small_size": "9x9 pixels",
                "note": "Thực tế marker lớn có area ~200px thay vì 361px"
            }
        }
        
        return statistics


# Global instance
all_marker_scanner = AllMarkerScanner()
