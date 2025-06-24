"""
Marker-Based Block Divider
Chia form OMR thành các blocks dựa trên markers đã phát hiện
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

logger = logging.getLogger(__name__)

class MarkerBasedBlockDivider:
    """
    Chia form OMR thành các blocks dựa trên vị trí markers
    """
    
    def __init__(self):
        self.debug_dir = Path("data/grading/block_division_debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors for visualization
        self.colors = {
            'large_marker': (0, 0, 255),      # Red
            'small_marker': (0, 255, 0),      # Green
            'block_border': (255, 0, 0),      # Blue
            'section_border': (0, 255, 255),  # Yellow
            'text': (255, 255, 255)           # White
        }
    
    def divide_form_into_blocks(self, image: np.ndarray, large_markers: List[Dict],
                               small_markers: List[Dict]) -> Dict[str, Any]:
        """
        Chia form thành các blocks dựa trên markers với enhanced contour detection
        """
        logger.info("Starting enhanced marker-based block division...")

        # Step 1: Enhanced contour detection để tìm thêm markers
        enhanced_markers = self.enhanced_contour_detection(image)

        # Step 2: Combine với markers đã có
        all_large_markers = large_markers + enhanced_markers['large_markers']
        all_small_markers = small_markers + enhanced_markers['small_markers']

        # Remove duplicates
        all_large_markers = self.remove_duplicate_markers(all_large_markers)
        all_small_markers = self.remove_duplicate_markers(all_small_markers)

        logger.info(f"Enhanced detection: {len(all_large_markers)} large + {len(all_small_markers)} small markers")

        # Step 3: Phân tích layout tổng thể
        layout_analysis = self.analyze_form_layout(image, all_large_markers, all_small_markers)

        # Step 4: Chia thành các regions chính dựa trên contours
        main_regions = self.divide_main_regions_by_contours(image, all_large_markers, all_small_markers, layout_analysis)

        # Step 5: Chia các regions thành blocks nhỏ dựa trên contour patterns
        detailed_blocks = self.divide_into_contour_based_blocks(image, all_small_markers, main_regions)

        # Step 6: Tạo visualization với enhanced contours
        visualization = self.create_enhanced_block_visualization(image, detailed_blocks, all_large_markers, all_small_markers, enhanced_markers)

        # Step 7: Tạo kết quả cuối cùng
        result = {
            'success': True,
            'layout_analysis': layout_analysis,
            'main_regions': main_regions,
            'detailed_blocks': detailed_blocks,
            'enhanced_markers': enhanced_markers,
            'visualization_path': str(self.debug_dir / "block_division_result.jpg"),
            'total_blocks': len(detailed_blocks),
            'block_summary': self.create_block_summary(detailed_blocks)
        }

        # Save visualization
        cv2.imwrite(str(self.debug_dir / "block_division_result.jpg"), visualization)

        # Save detailed results
        self.save_detailed_results(result)

        logger.info(f"Enhanced block division completed. Created {len(detailed_blocks)} blocks.")

        return result

    def save_debug_image(self, image: np.ndarray, filename: str, description: str = ""):
        """
        Lưu debug image với description
        """
        try:
            if not filename.endswith('.jpg'):
                filename += '.jpg'

            filepath = self.debug_dir / filename
            success = cv2.imwrite(str(filepath), image)

            if success:
                logger.info(f"Debug image saved: {filepath} - {description}")
            else:
                logger.error(f"Failed to save debug image: {filepath}")

        except Exception as e:
            logger.error(f"Error saving debug image {filename}: {str(e)}")

    def enhanced_contour_detection(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Enhanced contour detection giống như trong all_marker_scanner
        """
        logger.info("Running enhanced contour detection...")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Multiple thresholds để tìm marker đen (giống all_marker_scanner)
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

        # Debug: Vẽ tất cả contours
        all_contours_image = image.copy()
        cv2.drawContours(all_contours_image, contours, -1, (0, 255, 255), 1)
        self.save_debug_image(all_contours_image, "01_enhanced_all_contours", f"Enhanced contours: {len(contours)}")

        # Phân loại contours thành large và small markers
        large_markers = []
        small_markers = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Lọc theo diện tích
            if area < 25:  # Quá nhỏ
                continue

            # Xấp xỉ contour thành đa giác
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Kiểm tra có phải hình vuông không
            if len(approx) >= 3:  # Accept triangles and above
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Tỉ lệ khung hình ≈ 1.0 (vuông)
                if 0.7 <= aspect_ratio <= 1.4:
                    center = (x + w // 2, y + h // 2)

                    # Phân loại large vs small
                    if 180 < area < 300:  # Large marker (giống all_marker_scanner)
                        marker_info = {
                            'id': f"EL{len(large_markers) + 1}",
                            'center': center,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'contour': contour.tolist(),
                            'type': 'large',
                            'source': 'enhanced'
                        }
                        large_markers.append(marker_info)

                    elif 25 <= area <= 150:  # Small marker
                        # Additional checks for small markers
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0

                        bbox_area = w * h
                        rectangularity = area / bbox_area if bbox_area > 0 else 0

                        # Circularity check
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                        # Filter square-like shapes (not circles)
                        if solidity > 0.75 and rectangularity > 0.65 and circularity < 0.82:
                            marker_info = {
                                'id': f"ES{len(small_markers) + 1}",
                                'center': center,
                                'bbox': (x, y, w, h),
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'solidity': solidity,
                                'rectangularity': rectangularity,
                                'circularity': circularity,
                                'contour': contour.tolist(),
                                'type': 'small',
                                'source': 'enhanced'
                            }
                            small_markers.append(marker_info)

        # Create visualization
        enhanced_image = image.copy()

        # Draw large markers
        for marker in large_markers:
            center = marker['center']
            x, y, w, h = marker['bbox']
            cv2.rectangle(enhanced_image, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red
            cv2.circle(enhanced_image, center, 8, (0, 0, 255), -1)
            cv2.putText(enhanced_image, marker['id'], (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw small markers
        for marker in small_markers:
            center = marker['center']
            x, y, w, h = marker['bbox']
            cv2.rectangle(enhanced_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green
            cv2.circle(enhanced_image, center, 4, (0, 255, 0), -1)
            cv2.putText(enhanced_image, marker['id'], (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        self.save_debug_image(enhanced_image, "02_enhanced_markers_detected",
                             f"Enhanced markers: {len(large_markers)} large + {len(small_markers)} small")

        logger.info(f"Enhanced detection found {len(large_markers)} large + {len(small_markers)} small markers")

        return {
            'large_markers': large_markers,
            'small_markers': small_markers,
            'total_contours': len(contours)
        }

    def remove_duplicate_markers(self, markers: List[Dict]) -> List[Dict]:
        """
        Remove duplicate markers based on proximity
        """
        if not markers:
            return markers

        unique_markers = []

        for marker in markers:
            is_duplicate = False
            center = marker['center']

            for existing in unique_markers:
                existing_center = existing['center']
                distance = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)

                if distance < 20:  # Within 20 pixels
                    is_duplicate = True
                    # Keep the one with larger area (more reliable)
                    if marker['area'] > existing['area']:
                        unique_markers.remove(existing)
                        unique_markers.append(marker)
                    break

            if not is_duplicate:
                unique_markers.append(marker)

        return unique_markers

    def analyze_form_layout(self, image: np.ndarray, large_markers: List[Dict],
                           small_markers: List[Dict]) -> Dict[str, Any]:
        """
        Phân tích layout tổng thể của form
        """
        height, width = image.shape[:2]
        
        # Phân tích vị trí large markers
        large_positions = [marker['center'] for marker in large_markers]
        
        # Phân tích vị trí small markers
        small_positions = [marker['center'] for marker in small_markers]
        
        # Tìm boundaries chính
        if large_positions:
            large_x_coords = [pos[0] for pos in large_positions]
            large_y_coords = [pos[1] for pos in large_positions]
            
            main_boundaries = {
                'left': min(large_x_coords),
                'right': max(large_x_coords),
                'top': min(large_y_coords),
                'bottom': max(large_y_coords)
            }
        else:
            main_boundaries = {
                'left': 0,
                'right': width,
                'top': 0,
                'bottom': height
            }
        
        # Phân tích grid pattern của small markers
        grid_analysis = self.analyze_marker_grid(small_markers)
        
        layout = {
            'image_size': (width, height),
            'main_boundaries': main_boundaries,
            'large_marker_count': len(large_markers),
            'small_marker_count': len(small_markers),
            'grid_analysis': grid_analysis
        }
        
        return layout

    def divide_main_regions_by_contours(self, image: np.ndarray, large_markers: List[Dict],
                                       small_markers: List[Dict], layout_analysis: Dict) -> Dict[str, Dict]:
        """
        Chia thành các regions chính dựa trên contour patterns
        """
        height, width = image.shape[:2]

        # Combine all markers for analysis
        all_markers = large_markers + small_markers

        if not all_markers:
            # Fallback to simple division
            return {
                'full_form': {
                    'name': 'Full Form',
                    'bbox': (0, 0, width, height),
                    'description': 'Entire form (no markers detected)'
                }
            }

        # Sort markers by Y coordinate to find horizontal divisions
        sorted_by_y = sorted(all_markers, key=lambda m: m['center'][1])
        y_coords = [m['center'][1] for m in sorted_by_y]

        # Find natural breaks in Y coordinates
        y_breaks = self.find_natural_breaks(y_coords, min_gap=100)

        regions = {}

        if len(y_breaks) >= 2:
            # Multiple regions based on Y breaks
            for i in range(len(y_breaks)):
                if i == 0:
                    # Top region
                    y_start = 0
                    y_end = y_breaks[i] + 50
                    region_name = 'top_region'
                    description = 'Student Info and Test Code area'
                elif i == len(y_breaks) - 1:
                    # Bottom region
                    y_start = y_breaks[i-1] - 50
                    y_end = height
                    region_name = 'bottom_region'
                    description = 'Answer area (bottom section)'
                else:
                    # Middle regions
                    y_start = y_breaks[i-1] - 50
                    y_end = y_breaks[i] + 50
                    region_name = f'middle_region_{i}'
                    description = f'Answer area (section {i})'

                regions[region_name] = {
                    'name': region_name.replace('_', ' ').title(),
                    'bbox': (0, max(0, y_start), width, min(height, y_end)),
                    'description': description,
                    'markers_in_region': len([m for m in all_markers
                                            if y_start <= m['center'][1] <= y_end])
                }
        else:
            # Single region fallback
            regions['main_region'] = {
                'name': 'Main Region',
                'bbox': (0, 0, width, height),
                'description': 'Main form area',
                'markers_in_region': len(all_markers)
            }

        logger.info(f"Created {len(regions)} regions based on contour patterns")
        for name, region in regions.items():
            logger.info(f"  {name}: {region['bbox']}, markers: {region['markers_in_region']}")

        return regions

    def find_natural_breaks(self, values: List[float], min_gap: float = 50) -> List[float]:
        """
        Find natural breaks in a list of values
        """
        if len(values) < 2:
            return values

        sorted_values = sorted(values)
        breaks = [sorted_values[0]]

        for i in range(1, len(sorted_values)):
            if sorted_values[i] - sorted_values[i-1] > min_gap:
                breaks.append(sorted_values[i])

        return breaks

    def divide_into_contour_based_blocks(self, image: np.ndarray, small_markers: List[Dict],
                                        main_regions: Dict) -> List[Dict]:
        """
        Chia các regions thành blocks dựa trên contour patterns
        """
        blocks = []

        for region_name, region_info in main_regions.items():
            region_bbox = region_info['bbox']
            region_markers = self.get_markers_in_region(small_markers, region_bbox)

            if region_markers:
                # Create blocks based on marker clusters
                region_blocks = self.create_contour_based_blocks(
                    image, region_markers, region_name, region_bbox
                )
                blocks.extend(region_blocks)
            else:
                # Create single block for region without markers
                blocks.append({
                    'id': f"{region_name}_block_1",
                    'bbox': region_bbox,
                    'markers': [],
                    'type': 'region_block',
                    'region': region_name,
                    'marker_count': 0
                })

        return blocks

    def create_contour_based_blocks(self, image: np.ndarray, markers: List[Dict],
                                   region_name: str, region_bbox: Tuple) -> List[Dict]:
        """
        Tạo blocks dựa trên contour patterns và marker clustering
        """
        blocks = []

        if not markers:
            return blocks

        # Advanced clustering based on both X and Y coordinates
        marker_clusters = self.advanced_marker_clustering(markers)

        for i, cluster in enumerate(marker_clusters):
            # Calculate block bbox that encompasses the cluster
            block_bbox = self.calculate_smart_block_bbox(cluster, region_bbox, image.shape)

            # Determine block type based on cluster characteristics
            block_type = self.determine_block_type(cluster, region_name)

            blocks.append({
                'id': f"{region_name}_block_{i+1}",
                'bbox': block_bbox,
                'markers': cluster,
                'type': block_type,
                'region': region_name,
                'marker_count': len(cluster),
                'cluster_center': self.calculate_cluster_center(cluster),
                'cluster_spread': self.calculate_cluster_spread(cluster)
            })

        return blocks

    def advanced_marker_clustering(self, markers: List[Dict]) -> List[List[Dict]]:
        """
        Advanced clustering algorithm for markers
        """
        if not markers:
            return []

        if len(markers) == 1:
            return [markers]

        # Use hierarchical clustering based on distance
        positions = np.array([marker['center'] for marker in markers])

        # Calculate distance matrix
        n = len(markers)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 +
                              (positions[i][1] - positions[j][1])**2)
                distances[i][j] = distances[j][i] = dist

        # Simple clustering: group markers within threshold distance
        clusters = []
        used = set()

        for i, marker in enumerate(markers):
            if i in used:
                continue

            cluster = [marker]
            used.add(i)

            # Find nearby markers
            for j, other_marker in enumerate(markers):
                if j in used:
                    continue

                if distances[i][j] < 150:  # Threshold distance
                    cluster.append(other_marker)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def calculate_smart_block_bbox(self, cluster: List[Dict], region_bbox: Tuple,
                                  image_shape: Tuple) -> Tuple:
        """
        Calculate smart bounding box for a cluster of markers
        """
        if not cluster:
            return region_bbox

        # Get cluster bounds
        x_coords = [m['center'][0] for m in cluster]
        y_coords = [m['center'][1] for m in cluster]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Add intelligent padding based on cluster size and spread
        cluster_width = max_x - min_x
        cluster_height = max_y - min_y

        # Adaptive padding
        padding_x = max(50, cluster_width * 0.3)
        padding_y = max(50, cluster_height * 0.3)

        # Calculate final bbox
        final_x1 = max(0, min_x - padding_x)
        final_y1 = max(0, min_y - padding_y)
        final_x2 = min(image_shape[1], max_x + padding_x)
        final_y2 = min(image_shape[0], max_y + padding_y)

        return (int(final_x1), int(final_y1), int(final_x2), int(final_y2))

    def determine_block_type(self, cluster: List[Dict], region_name: str) -> str:
        """
        Determine block type based on cluster characteristics
        """
        cluster_size = len(cluster)

        if cluster_size == 1:
            return 'single_marker_block'
        elif cluster_size <= 3:
            return 'small_cluster_block'
        elif cluster_size <= 6:
            return 'medium_cluster_block'
        else:
            return 'large_cluster_block'

    def calculate_cluster_center(self, cluster: List[Dict]) -> Tuple[int, int]:
        """
        Calculate center point of a cluster
        """
        if not cluster:
            return (0, 0)

        x_coords = [m['center'][0] for m in cluster]
        y_coords = [m['center'][1] for m in cluster]

        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        return (center_x, center_y)

    def calculate_cluster_spread(self, cluster: List[Dict]) -> float:
        """
        Calculate spread (standard deviation) of cluster
        """
        if len(cluster) < 2:
            return 0.0

        center = self.calculate_cluster_center(cluster)
        distances = []

        for marker in cluster:
            dist = np.sqrt((marker['center'][0] - center[0])**2 +
                          (marker['center'][1] - center[1])**2)
            distances.append(dist)

        return float(np.std(distances))

    def analyze_marker_grid(self, small_markers: List[Dict]) -> Dict[str, Any]:
        """
        Phân tích pattern grid của small markers
        """
        if not small_markers:
            return {'rows': 0, 'cols': 0, 'pattern': 'none'}
        
        positions = [marker['center'] for marker in small_markers]
        
        # Group by Y coordinates (rows)
        y_coords = [pos[1] for pos in positions]
        y_unique = sorted(list(set([round(y/20)*20 for y in y_coords])))  # Round to nearest 20
        
        # Group by X coordinates (cols)
        x_coords = [pos[0] for pos in positions]
        x_unique = sorted(list(set([round(x/20)*20 for x in x_coords])))  # Round to nearest 20
        
        # Analyze pattern
        rows = len(y_unique)
        cols = len(x_unique)
        
        return {
            'rows': rows,
            'cols': cols,
            'y_positions': y_unique,
            'x_positions': x_unique,
            'pattern': f"{rows}x{cols}_grid"
        }
    
    def divide_main_regions(self, image: np.ndarray, large_markers: List[Dict], 
                           layout_analysis: Dict) -> Dict[str, Dict]:
        """
        Chia thành các regions chính dựa trên large markers
        """
        height, width = image.shape[:2]
        boundaries = layout_analysis['main_boundaries']
        
        # Dựa trên ảnh, có 3 regions chính theo chiều dọc
        regions = {}
        
        if large_markers:
            # Sort large markers by Y coordinate
            sorted_markers = sorted(large_markers, key=lambda m: m['center'][1])
            
            # Tìm Y coordinates để chia regions
            y_coords = [m['center'][1] for m in sorted_markers]
            
            # Chia thành 3 regions: top, middle, bottom
            if len(y_coords) >= 2:
                # Top region (student info area)
                top_y = min(y_coords)
                middle_y = np.median(y_coords)
                bottom_y = max(y_coords)
                
                regions['top_region'] = {
                    'name': 'Student Info Area',
                    'bbox': (boundaries['left'], 0, boundaries['right'], int(middle_y)),
                    'description': 'Student ID and Test Code area'
                }
                
                regions['middle_region'] = {
                    'name': 'Answer Area Part I & II',
                    'bbox': (boundaries['left'], int(middle_y), boundaries['right'], int(bottom_y)),
                    'description': 'Multiple choice and True/False questions'
                }
                
                regions['bottom_region'] = {
                    'name': 'Answer Area Part III',
                    'bbox': (boundaries['left'], int(bottom_y), boundaries['right'], height),
                    'description': 'Digit selection area'
                }
        
        return regions
    
    def divide_into_detailed_blocks(self, image: np.ndarray, small_markers: List[Dict], 
                                   main_regions: Dict) -> List[Dict]:
        """
        Chia các regions thành blocks chi tiết dựa trên small markers
        """
        blocks = []
        
        if not small_markers:
            logger.warning("No small markers found for detailed block division")
            return blocks
        
        # Group small markers by regions
        for region_name, region_info in main_regions.items():
            region_bbox = region_info['bbox']
            region_markers = self.get_markers_in_region(small_markers, region_bbox)
            
            if region_markers:
                region_blocks = self.create_blocks_from_markers(
                    image, region_markers, region_name, region_bbox
                )
                blocks.extend(region_blocks)
        
        return blocks
    
    def get_markers_in_region(self, markers: List[Dict], region_bbox: Tuple) -> List[Dict]:
        """
        Lấy các markers nằm trong region
        """
        x1, y1, x2, y2 = region_bbox
        region_markers = []
        
        for marker in markers:
            mx, my = marker['center']
            if x1 <= mx <= x2 and y1 <= my <= y2:
                region_markers.append(marker)
        
        return region_markers
    
    def create_blocks_from_markers(self, image: np.ndarray, markers: List[Dict], 
                                  region_name: str, region_bbox: Tuple) -> List[Dict]:
        """
        Tạo blocks từ markers trong một region
        """
        blocks = []
        
        if len(markers) < 2:
            # Nếu ít markers, tạo 1 block cho cả region
            blocks.append({
                'id': f"{region_name}_block_1",
                'bbox': region_bbox,
                'markers': markers,
                'type': 'single_block',
                'region': region_name
            })
            return blocks
        
        # Sort markers by position
        sorted_markers = sorted(markers, key=lambda m: (m['center'][1], m['center'][0]))
        
        # Group markers thành các clusters
        marker_clusters = self.cluster_markers_by_proximity(sorted_markers)
        
        # Tạo blocks từ clusters
        for i, cluster in enumerate(marker_clusters):
            block_bbox = self.calculate_block_bbox_from_markers(cluster, region_bbox, image.shape)
            
            blocks.append({
                'id': f"{region_name}_block_{i+1}",
                'bbox': block_bbox,
                'markers': cluster,
                'type': 'marker_based_block',
                'region': region_name,
                'marker_count': len(cluster)
            })
        
        return blocks
    
    def cluster_markers_by_proximity(self, markers: List[Dict], distance_threshold: int = 100) -> List[List[Dict]]:
        """
        Nhóm markers thành clusters dựa trên khoảng cách
        """
        if not markers:
            return []
        
        clusters = []
        used_markers = set()
        
        for i, marker in enumerate(markers):
            if i in used_markers:
                continue
            
            # Tạo cluster mới
            cluster = [marker]
            used_markers.add(i)
            
            # Tìm markers gần đó
            for j, other_marker in enumerate(markers):
                if j in used_markers:
                    continue
                
                distance = np.sqrt(
                    (marker['center'][0] - other_marker['center'][0])**2 +
                    (marker['center'][1] - other_marker['center'][1])**2
                )
                
                if distance < distance_threshold:
                    cluster.append(other_marker)
                    used_markers.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def calculate_block_bbox_from_markers(self, markers: List[Dict], region_bbox: Tuple, 
                                         image_shape: Tuple) -> Tuple:
        """
        Tính toán bbox của block dựa trên markers
        """
        if not markers:
            return region_bbox
        
        # Lấy tọa độ của tất cả markers
        x_coords = [m['center'][0] for m in markers]
        y_coords = [m['center'][1] for m in markers]
        
        # Tính bbox với padding
        padding = 50
        min_x = max(0, min(x_coords) - padding)
        max_x = min(image_shape[1], max(x_coords) + padding)
        min_y = max(0, min(y_coords) - padding)
        max_y = min(image_shape[0], max(y_coords) + padding)
        
        return (min_x, min_y, max_x, max_y)

    def create_enhanced_block_visualization(self, image: np.ndarray, blocks: List[Dict],
                                          large_markers: List[Dict], small_markers: List[Dict],
                                          enhanced_markers: Dict) -> np.ndarray:
        """
        Tạo enhanced visualization với contour-based blocks
        """
        vis_image = image.copy()

        # Vẽ enhanced contours background (faded)
        enhanced_large = enhanced_markers.get('large_markers', [])
        enhanced_small = enhanced_markers.get('small_markers', [])

        # Vẽ enhanced markers với transparency effect
        for marker in enhanced_large:
            center = marker['center']
            x, y, w, h = marker['bbox']
            # Faded red for enhanced large markers
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (100, 100, 255), 1)
            cv2.circle(vis_image, center, 6, (100, 100, 255), 1)

        for marker in enhanced_small:
            center = marker['center']
            x, y, w, h = marker['bbox']
            # Faded green for enhanced small markers
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (100, 255, 100), 1)
            cv2.circle(vis_image, center, 3, (100, 255, 100), 1)

        # Vẽ original markers (bright colors)
        for marker in large_markers:
            center = marker['center']
            cv2.circle(vis_image, center, 10, self.colors['large_marker'], -1)
            cv2.putText(vis_image, "L", (center[0]-5, center[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

        for marker in small_markers:
            center = marker['center']
            cv2.circle(vis_image, center, 5, self.colors['small_marker'], -1)
            cv2.putText(vis_image, "S", (center[0]-3, center[1]+3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)

        # Vẽ blocks với different colors based on type
        block_colors = {
            'single_marker_block': (255, 200, 0),      # Orange
            'small_cluster_block': (255, 0, 200),      # Magenta
            'medium_cluster_block': (0, 200, 255),     # Cyan
            'large_cluster_block': (200, 0, 255),      # Purple
            'region_block': (128, 128, 128)            # Gray
        }

        for i, block in enumerate(blocks):
            bbox = block['bbox']
            x1, y1, x2, y2 = bbox
            block_type = block.get('type', 'unknown')

            # Get color for block type
            color = block_colors.get(block_type, self.colors['block_border'])

            # Vẽ khung block với thickness based on marker count
            thickness = min(4, max(1, block.get('marker_count', 0)))
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # Vẽ cluster center if available
            if 'cluster_center' in block:
                center = block['cluster_center']
                cv2.circle(vis_image, center, 8, color, 2)

            # Vẽ label block với background
            label = f"B{i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(vis_image, (x1, y1-25), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(vis_image, label, (x1+5, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)

            # Vẽ marker count
            marker_count = block.get('marker_count', 0)
            if marker_count > 0:
                count_text = f"M:{marker_count}"
                cv2.putText(vis_image, count_text, (x1+5, y1+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Thêm enhanced legend
        self.add_enhanced_legend_to_visualization(vis_image, blocks, large_markers, small_markers, enhanced_markers)

        return vis_image

    def add_enhanced_legend_to_visualization(self, image: np.ndarray, blocks: List[Dict],
                                           large_markers: List[Dict], small_markers: List[Dict],
                                           enhanced_markers: Dict):
        """
        Thêm enhanced legend với thông tin chi tiết
        """
        height, width = image.shape[:2]

        # Background cho legend (larger)
        legend_x = width - 350
        legend_y = 30
        legend_height = 200

        cv2.rectangle(image, (legend_x-10, legend_y-10),
                     (width-10, legend_y+legend_height), (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x-10, legend_y-10),
                     (width-10, legend_y+legend_height), self.colors['text'], 2)

        # Title
        cv2.putText(image, "ENHANCED BLOCK DIVISION", (legend_x, legend_y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

        # Original markers
        cv2.putText(image, f"Original Large: {len(large_markers)}", (legend_x, legend_y+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['large_marker'], 1)
        cv2.putText(image, f"Original Small: {len(small_markers)}", (legend_x, legend_y+55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['small_marker'], 1)

        # Enhanced markers
        enhanced_large_count = len(enhanced_markers.get('large_markers', []))
        enhanced_small_count = len(enhanced_markers.get('small_markers', []))
        cv2.putText(image, f"Enhanced Large: {enhanced_large_count}", (legend_x, legend_y+75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        cv2.putText(image, f"Enhanced Small: {enhanced_small_count}", (legend_x, legend_y+95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        # Blocks
        cv2.putText(image, f"Total Blocks: {len(blocks)}", (legend_x, legend_y+115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

        # Block types
        block_types = {}
        for block in blocks:
            block_type = block.get('type', 'unknown')
            block_types[block_type] = block_types.get(block_type, 0) + 1

        y_offset = 135
        for block_type, count in block_types.items():
            cv2.putText(image, f"{block_type}: {count}", (legend_x, legend_y+y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            y_offset += 15

    def create_block_visualization(self, image: np.ndarray, blocks: List[Dict],
                                  large_markers: List[Dict], small_markers: List[Dict]) -> np.ndarray:
        """
        Tạo visualization cho block division
        """
        vis_image = image.copy()
        
        # Vẽ large markers
        for marker in large_markers:
            center = marker['center']
            cv2.circle(vis_image, center, 8, self.colors['large_marker'], -1)
            cv2.putText(vis_image, "L", (center[0]-5, center[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Vẽ small markers
        for marker in small_markers:
            center = marker['center']
            cv2.circle(vis_image, center, 4, self.colors['small_marker'], -1)
        
        # Vẽ blocks
        for i, block in enumerate(blocks):
            bbox = block['bbox']
            x1, y1, x2, y2 = bbox
            
            # Vẽ khung block
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), self.colors['block_border'], 2)
            
            # Vẽ label block
            label = f"Block {i+1}"
            cv2.putText(vis_image, label, (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Thêm legend
        self.add_legend_to_visualization(vis_image, len(blocks), len(large_markers), len(small_markers))
        
        return vis_image
    
    def add_legend_to_visualization(self, image: np.ndarray, block_count: int, 
                                   large_count: int, small_count: int):
        """
        Thêm legend vào visualization
        """
        height, width = image.shape[:2]
        
        # Background cho legend
        legend_x = width - 250
        legend_y = 30
        
        cv2.rectangle(image, (legend_x-10, legend_y-10), 
                     (width-10, legend_y+100), (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x-10, legend_y-10), 
                     (width-10, legend_y+100), self.colors['text'], 2)
        
        # Title
        cv2.putText(image, "BLOCK DIVISION", (legend_x, legend_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Stats
        cv2.putText(image, f"Blocks: {block_count}", (legend_x, legend_y+35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(image, f"Large Markers: {large_count}", (legend_x, legend_y+55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['large_marker'], 1)
        cv2.putText(image, f"Small Markers: {small_count}", (legend_x, legend_y+75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['small_marker'], 1)
    
    def create_block_summary(self, blocks: List[Dict]) -> Dict[str, Any]:
        """
        Tạo summary của blocks
        """
        summary = {
            'total_blocks': len(blocks),
            'blocks_by_region': {},
            'blocks_by_type': {},
            'average_block_size': 0
        }
        
        # Group by region
        for block in blocks:
            region = block.get('region', 'unknown')
            if region not in summary['blocks_by_region']:
                summary['blocks_by_region'][region] = 0
            summary['blocks_by_region'][region] += 1
        
        # Group by type
        for block in blocks:
            block_type = block.get('type', 'unknown')
            if block_type not in summary['blocks_by_type']:
                summary['blocks_by_type'][block_type] = 0
            summary['blocks_by_type'][block_type] += 1
        
        # Calculate average size
        if blocks:
            total_area = 0
            for block in blocks:
                x1, y1, x2, y2 = block['bbox']
                area = (x2 - x1) * (y2 - y1)
                total_area += area
            summary['average_block_size'] = total_area / len(blocks)
        
        return summary
    
    def save_detailed_results(self, result: Dict):
        """
        Lưu kết quả chi tiết
        """
        # Save JSON results
        json_path = self.debug_dir / "block_division_results.json"
        
        # Convert numpy types to native Python types for JSON serialization
        json_result = self.convert_for_json(result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {json_path}")
    
    def convert_for_json(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization
        """
        if isinstance(obj, dict):
            return {key: self.convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


# Global instance
marker_based_block_divider = MarkerBasedBlockDivider()
