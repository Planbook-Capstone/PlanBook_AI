#!/usr/bin/env python3
"""
Smart Marker Finder
Tìm kiếm thông minh 6 large markers + 18 small markers
Sử dụng pattern analysis và spatial distribution
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN

def smart_marker_search():
    """Tìm kiếm thông minh markers"""
    
    print("🧠 Smart Marker Finder")
    print("=" * 60)
    print("Target: 6 large markers + 18 small markers = 24 total")
    print("Strategy: Pattern analysis + Spatial distribution")
    print("=" * 60)
    
    # Load image
    image_path = "data/grading/sample.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"✅ Image loaded: {image.shape}")
    
    # Create debug directory
    debug_dir = Path("data/grading/smart_search")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear old files
    for file in debug_dir.glob("*.jpg"):
        file.unlink()
    
    # Step 1: Find all potential square-like objects
    print("\n🔍 Step 1: Finding potential square objects...")
    candidates = find_square_candidates(gray, image)
    print(f"   Found {len(candidates)} potential squares")
    
    # Step 2: Classify by size and quality
    print("\n📊 Step 2: Classifying by size and quality...")
    large_markers, small_markers = classify_markers(candidates)
    print(f"   Large marker candidates: {len(large_markers)}")
    print(f"   Small marker candidates: {len(small_markers)}")
    
    # Step 3: Spatial analysis to filter real markers
    print("\n🎯 Step 3: Spatial analysis...")
    final_large = spatial_filter_large_markers(large_markers, image.shape)
    final_small = spatial_filter_small_markers(small_markers, image.shape)
    
    print(f"   Final large markers: {len(final_large)}")
    print(f"   Final small markers: {len(final_small)}")
    
    # Step 4: Visualization
    print("\n🎨 Step 4: Creating visualization...")
    create_smart_visualization(image, final_large, final_small, debug_dir)
    
    print(f"\n🎯 Smart Search Results:")
    print(f"   Large markers: {len(final_large)}/6 (target)")
    print(f"   Small markers: {len(final_small)}/18 (target)")
    print(f"   Total: {len(final_large) + len(final_small)}/24 (target)")
    print(f"   Success rate: {(len(final_large) + len(final_small))/24*100:.1f}%")
    
    return final_large, final_small

def find_square_candidates(gray, original_image):
    """Tìm tất cả candidates có hình dạng vuông"""
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
                
                # Check if it's roughly square-like
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Square-like aspect ratio
                    if 0.7 <= aspect_ratio <= 1.4:
                        # Calculate solidity (area/convex_hull_area)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        # Filter by solidity (how "filled" the shape is)
                        if solidity > 0.7:
                            center = (x + w // 2, y + h // 2)
                            
                            candidate = {
                                'center': center,
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'solidity': solidity,
                                'bbox': (x, y, w, h),
                                'contour': contour,
                                'approx_points': len(approx)
                            }
                            candidates.append(candidate)
    
    # Remove duplicates
    unique_candidates = remove_duplicates(candidates)
    return unique_candidates

def classify_markers(candidates):
    """Phân loại markers theo kích thước"""
    large_markers = []
    small_markers = []
    
    for candidate in candidates:
        area = candidate['area']
        
        # Based on our previous analysis:
        # Large markers: ~187-218 area
        # Small markers: ~49 area
        
        if area > 150:  # Large marker threshold
            large_markers.append(candidate)
        else:  # Small marker
            small_markers.append(candidate)
    
    return large_markers, small_markers

def spatial_filter_large_markers(candidates, image_shape):
    """Lọc large markers dựa trên vị trí không gian"""
    if len(candidates) <= 6:
        return candidates
    
    # Large markers thường ở các góc và biên
    height, width = image_shape[:2]
    
    # Define regions where large markers are likely to be
    corner_regions = [
        (0, 0, width//3, height//3),           # Top-left
        (2*width//3, 0, width, height//3),     # Top-right
        (0, 2*height//3, width//3, height),    # Bottom-left
        (2*width//3, 2*height//3, width, height), # Bottom-right
        (width//3, 0, 2*width//3, height//4),  # Top-center
        (width//3, 3*height//4, 2*width//3, height), # Bottom-center
    ]
    
    # Score candidates based on position
    scored_candidates = []
    for candidate in candidates:
        center = candidate['center']
        score = 0
        
        # Check if in corner/edge regions
        for x1, y1, x2, y2 in corner_regions:
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                score += 10
                break
        
        # Prefer larger areas (more likely to be real markers)
        score += candidate['area'] / 50
        
        # Prefer better aspect ratios (closer to 1.0)
        score += 10 - abs(candidate['aspect_ratio'] - 1.0) * 5
        
        # Prefer higher solidity
        score += candidate['solidity'] * 5
        
        scored_candidates.append((score, candidate))
    
    # Sort by score and take top 6
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [candidate for score, candidate in scored_candidates[:6]]

def spatial_filter_small_markers(candidates, image_shape):
    """Lọc small markers dựa trên pattern không gian"""
    if len(candidates) <= 18:
        return candidates
    
    # Small markers thường tạo thành grid pattern
    # Use clustering to find grid-like arrangements
    
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
        
        # Prefer better aspect ratios
        score += 10 - abs(candidate['aspect_ratio'] - 1.0) * 3
        
        # Prefer higher solidity
        score += candidate['solidity'] * 3
        
        scored_candidates.append((score, candidate))
    
    # Sort by score and take top 18
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [candidate for score, candidate in scored_candidates[:18]]

def remove_duplicates(candidates):
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

def create_smart_visualization(image, large_markers, small_markers, debug_dir):
    """Tạo visualization thông minh"""
    
    # Create final image
    final_img = image.copy()
    
    # Draw large markers
    for i, marker in enumerate(large_markers):
        center = marker['center']
        bbox = marker['bbox']
        
        # Draw bounding box
        cv2.rectangle(final_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
        
        # Draw center
        cv2.circle(final_img, center, 8, (0, 0, 255), -1)
        
        # Label
        cv2.putText(final_img, f"L{i+1}", (center[0] - 15, center[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Area info
        cv2.putText(final_img, f"{marker['area']:.0f}", (center[0] - 15, center[1] + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw small markers
    for i, marker in enumerate(small_markers):
        center = marker['center']
        bbox = marker['bbox']
        
        # Draw bounding box
        cv2.rectangle(final_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
        
        # Draw center
        cv2.circle(final_img, center, 4, (0, 255, 0), -1)
        
        # Label
        cv2.putText(final_img, f"S{i+1}", (center[0] - 10, center[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add comprehensive legend
    legend_y = 30
    cv2.putText(final_img, f"SMART MARKER DETECTION", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    legend_y += 30
    cv2.putText(final_img, f"Large Markers: {len(large_markers)}/6 (target)", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    legend_y += 25
    cv2.putText(final_img, f"Small Markers: {len(small_markers)}/18 (target)", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    legend_y += 25
    cv2.putText(final_img, f"Total: {len(large_markers) + len(small_markers)}/24", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save final image
    cv2.imwrite(str(debug_dir / "smart_markers_final.jpg"), final_img)
    
    print(f"   Smart visualization saved: {debug_dir / 'smart_markers_final.jpg'}")

if __name__ == "__main__":
    smart_marker_search()
