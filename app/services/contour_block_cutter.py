"""
Contour Block Cutter
Cắt các vùng dựa trên contours được phát hiện trong enhanced detection
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

logger = logging.getLogger(__name__)

class ContourBlockCutter:
    """
    Cắt form OMR thành các blocks dựa trên contours chính xác
    """
    
    def __init__(self):
        self.debug_dir = Path("data/grading/contour_blocks_debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors for visualization
        self.colors = {
            'contour': (0, 255, 255),         # Yellow (giống 01_enhanced_all_contours.jpg)
            'block_border': (255, 0, 0),      # Blue
            'text': (255, 255, 255),          # White
            'background': (0, 0, 0)           # Black
        }
    
    def cut_contour_blocks(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Cắt các blocks dựa trên contours được phát hiện
        """
        logger.info("Starting contour-based block cutting...")
        
        # Step 1: Detect all contours (giống enhanced detection)
        contours_data = self.detect_all_contours(image)
        
        # Step 2: Filter và classify contours
        classified_contours = self.classify_contours(contours_data['contours'], image.shape)
        
        # Step 3: Cắt các vùng dựa trên contours
        cut_blocks = self.cut_blocks_from_contours(image, classified_contours)
        
        # Step 4: Tạo visualization
        visualization = self.create_cutting_visualization(image, classified_contours, cut_blocks)
        
        # Step 5: Tạo kết quả
        result = {
            'success': True,
            'total_contours': len(contours_data['contours']),
            'classified_contours': classified_contours,
            'cut_blocks': cut_blocks,
            'visualization_path': str(self.debug_dir / "contour_blocks_result.jpg"),
            'block_images_dir': str(self.debug_dir / "individual_blocks"),
            'summary': self.create_cutting_summary(cut_blocks)
        }
        
        # Save visualization
        cv2.imwrite(str(self.debug_dir / "contour_blocks_result.jpg"), visualization)
        
        # Save detailed results
        self.save_cutting_results(result)
        
        logger.info(f"Contour cutting completed. Created {len(cut_blocks)} blocks.")
        
        return result
    
    def detect_all_contours(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect all contours giống như enhanced detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Multiple thresholds (giống enhanced detection)
        _, binary1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        _, binary2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        _, binary3 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        # Adaptive threshold
        binary4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)

        # Combine tất cả threshold
        binary = cv2.bitwise_or(cv2.bitwise_or(binary1, binary2), cv2.bitwise_or(binary3, binary4))
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: Vẽ tất cả contours (giống 01_enhanced_all_contours.jpg)
        all_contours_image = image.copy()
        cv2.drawContours(all_contours_image, contours, -1, self.colors['contour'], 1)
        self.save_debug_image(all_contours_image, "01_all_contours_for_cutting", 
                             f"All contours for cutting: {len(contours)}")
        
        return {
            'contours': contours,
            'binary_image': binary,
            'cleaned_image': cleaned
        }
    
    def classify_contours(self, contours: List, image_shape: Tuple) -> Dict[str, List]:
        """
        Phân loại contours thành các loại khác nhau để cắt
        """
        height, width = image_shape[:2]
        
        classified = {
            'large_regions': [],      # Vùng lớn (có thể là sections)
            'medium_blocks': [],      # Blocks trung bình
            'small_markers': [],      # Markers nhỏ
            'noise': []              # Noise
        }
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate properties
            aspect_ratio = float(w) / h if h > 0 else 0
            area_ratio = area / (width * height) if width * height > 0 else 0
            
            contour_info = {
                'id': i,
                'contour': contour,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'area_ratio': area_ratio,
                'center': (x + w // 2, y + h // 2)
            }
            
            # Classify based on size and properties
            if area < 20:
                classified['noise'].append(contour_info)
            elif area < 200:
                classified['small_markers'].append(contour_info)
            elif area < 5000:
                classified['medium_blocks'].append(contour_info)
            else:
                classified['large_regions'].append(contour_info)
        
        logger.info(f"Classified contours: "
                   f"large={len(classified['large_regions'])}, "
                   f"medium={len(classified['medium_blocks'])}, "
                   f"small={len(classified['small_markers'])}, "
                   f"noise={len(classified['noise'])}")
        
        return classified
    
    def cut_blocks_from_contours(self, image: np.ndarray, classified_contours: Dict) -> List[Dict]:
        """
        Cắt các blocks từ contours đã phân loại
        """
        cut_blocks = []
        
        # Create directory for individual block images
        blocks_dir = self.debug_dir / "individual_blocks"
        blocks_dir.mkdir(exist_ok=True)
        
        # Process large regions first
        for region in classified_contours['large_regions']:
            block_info = self.cut_single_block(image, region, 'large_region', blocks_dir)
            if block_info:
                cut_blocks.append(block_info)
        
        # Process medium blocks
        for block in classified_contours['medium_blocks']:
            block_info = self.cut_single_block(image, block, 'medium_block', blocks_dir)
            if block_info:
                cut_blocks.append(block_info)
        
        # Process small markers (group nearby ones)
        small_groups = self.group_nearby_contours(classified_contours['small_markers'])
        for group_id, group in enumerate(small_groups):
            if len(group) > 1:
                # Create combined block for group
                combined_bbox = self.calculate_combined_bbox(group)
                block_info = self.cut_combined_block(image, group, combined_bbox, 
                                                   f'small_group_{group_id}', blocks_dir)
                if block_info:
                    cut_blocks.append(block_info)
            else:
                # Single small marker
                block_info = self.cut_single_block(image, group[0], 'small_marker', blocks_dir)
                if block_info:
                    cut_blocks.append(block_info)
        
        return cut_blocks
    
    def cut_single_block(self, image: np.ndarray, contour_info: Dict, 
                        block_type: str, output_dir: Path) -> Dict:
        """
        Cắt một block đơn lẻ
        """
        try:
            x, y, w, h = contour_info['bbox']
            
            # Add padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # Cut the block
            block_image = image[y1:y2, x1:x2]
            
            if block_image.size == 0:
                return None
            
            # Save block image
            block_filename = f"{block_type}_block_{contour_info['id']}.jpg"
            block_path = output_dir / block_filename
            cv2.imwrite(str(block_path), block_image)
            
            return {
                'id': f"{block_type}_{contour_info['id']}",
                'type': block_type,
                'original_bbox': contour_info['bbox'],
                'padded_bbox': (x1, y1, x2, y2),
                'area': contour_info['area'],
                'image_path': str(block_path),
                'image_size': (block_image.shape[1], block_image.shape[0]),
                'contour_count': 1
            }
            
        except Exception as e:
            logger.error(f"Error cutting block {contour_info['id']}: {str(e)}")
            return None
    
    def cut_combined_block(self, image: np.ndarray, contour_group: List[Dict], 
                          combined_bbox: Tuple, block_id: str, output_dir: Path) -> Dict:
        """
        Cắt block kết hợp từ nhiều contours
        """
        try:
            x1, y1, x2, y2 = combined_bbox
            
            # Cut the combined block
            block_image = image[y1:y2, x1:x2]
            
            if block_image.size == 0:
                return None
            
            # Save block image
            block_filename = f"combined_{block_id}.jpg"
            block_path = output_dir / block_filename
            cv2.imwrite(str(block_path), block_image)
            
            return {
                'id': block_id,
                'type': 'combined_block',
                'original_bbox': None,
                'padded_bbox': combined_bbox,
                'area': sum(c['area'] for c in contour_group),
                'image_path': str(block_path),
                'image_size': (block_image.shape[1], block_image.shape[0]),
                'contour_count': len(contour_group),
                'component_contours': [c['id'] for c in contour_group]
            }
            
        except Exception as e:
            logger.error(f"Error cutting combined block {block_id}: {str(e)}")
            return None
    
    def group_nearby_contours(self, contours: List[Dict], distance_threshold: int = 50) -> List[List[Dict]]:
        """
        Nhóm các contours gần nhau
        """
        if not contours:
            return []
        
        groups = []
        used = set()
        
        for i, contour in enumerate(contours):
            if i in used:
                continue
            
            group = [contour]
            used.add(i)
            center = contour['center']
            
            # Find nearby contours
            for j, other_contour in enumerate(contours):
                if j in used:
                    continue
                
                other_center = other_contour['center']
                distance = np.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
                
                if distance < distance_threshold:
                    group.append(other_contour)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def calculate_combined_bbox(self, contour_group: List[Dict]) -> Tuple[int, int, int, int]:
        """
        Tính bbox kết hợp cho một nhóm contours
        """
        if not contour_group:
            return (0, 0, 0, 0)
        
        min_x = min(c['bbox'][0] for c in contour_group)
        min_y = min(c['bbox'][1] for c in contour_group)
        max_x = max(c['bbox'][0] + c['bbox'][2] for c in contour_group)
        max_y = max(c['bbox'][1] + c['bbox'][3] for c in contour_group)
        
        # Add padding
        padding = 15
        return (max(0, min_x - padding), max(0, min_y - padding), 
                max_x + padding, max_y + padding)
    
    def create_cutting_visualization(self, image: np.ndarray, classified_contours: Dict, 
                                   cut_blocks: List[Dict]) -> np.ndarray:
        """
        Tạo visualization cho quá trình cắt blocks
        """
        vis_image = image.copy()
        
        # Vẽ tất cả contours với màu vàng (giống 01_enhanced_all_contours.jpg)
        all_contours = []
        for category in classified_contours.values():
            for contour_info in category:
                all_contours.append(contour_info['contour'])
        
        cv2.drawContours(vis_image, all_contours, -1, self.colors['contour'], 1)
        
        # Vẽ bounding boxes của các blocks đã cắt
        for i, block in enumerate(cut_blocks):
            bbox = block['padded_bbox']
            x1, y1, x2, y2 = bbox
            
            # Color based on block type
            if block['type'] == 'large_region':
                color = (0, 0, 255)  # Red
                thickness = 3
            elif block['type'] == 'medium_block':
                color = (255, 0, 0)  # Blue
                thickness = 2
            elif block['type'] == 'combined_block':
                color = (255, 0, 255)  # Magenta
                thickness = 2
            else:
                color = (0, 255, 0)  # Green
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            label = f"B{i+1}"
            cv2.putText(vis_image, label, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add legend
        self.add_cutting_legend(vis_image, classified_contours, cut_blocks)
        
        return vis_image
    
    def add_cutting_legend(self, image: np.ndarray, classified_contours: Dict, cut_blocks: List[Dict]):
        """
        Thêm legend cho cutting visualization
        """
        height, width = image.shape[:2]
        
        # Legend position
        legend_x = width - 300
        legend_y = 30
        
        # Background
        cv2.rectangle(image, (legend_x-10, legend_y-10), 
                     (width-10, legend_y+150), self.colors['background'], -1)
        cv2.rectangle(image, (legend_x-10, legend_y-10), 
                     (width-10, legend_y+150), self.colors['text'], 2)
        
        # Title
        cv2.putText(image, "CONTOUR BLOCK CUTTING", (legend_x, legend_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Stats
        cv2.putText(image, f"Total Blocks Cut: {len(cut_blocks)}", (legend_x, legend_y+35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Block types
        block_types = {}
        for block in cut_blocks:
            block_type = block['type']
            block_types[block_type] = block_types.get(block_type, 0) + 1
        
        y_offset = 55
        for block_type, count in block_types.items():
            cv2.putText(image, f"{block_type}: {count}", (legend_x, legend_y+y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            y_offset += 20
    
    def create_cutting_summary(self, cut_blocks: List[Dict]) -> Dict[str, Any]:
        """
        Tạo summary cho quá trình cắt
        """
        summary = {
            'total_blocks_cut': len(cut_blocks),
            'blocks_by_type': {},
            'average_block_size': 0,
            'largest_block': None,
            'smallest_block': None
        }
        
        if not cut_blocks:
            return summary
        
        # Count by type
        for block in cut_blocks:
            block_type = block['type']
            summary['blocks_by_type'][block_type] = summary['blocks_by_type'].get(block_type, 0) + 1
        
        # Calculate average size
        total_area = sum(block['area'] for block in cut_blocks)
        summary['average_block_size'] = total_area / len(cut_blocks)
        
        # Find largest and smallest
        summary['largest_block'] = max(cut_blocks, key=lambda b: b['area'])['id']
        summary['smallest_block'] = min(cut_blocks, key=lambda b: b['area'])['id']
        
        return summary
    
    def save_cutting_results(self, result: Dict):
        """
        Lưu kết quả cutting
        """
        json_path = self.debug_dir / "contour_cutting_results.json"
        
        # Convert for JSON serialization
        json_result = self.convert_for_json(result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Cutting results saved to: {json_path}")
    
    def save_debug_image(self, image: np.ndarray, filename: str, description: str = ""):
        """
        Lưu debug image
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
    
    def convert_for_json(self, obj):
        """
        Convert numpy types for JSON serialization
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
contour_block_cutter = ContourBlockCutter()
