#!/usr/bin/env python3
"""
Comprehensive Marker Search
Tìm kiếm toàn diện tất cả potential markers trong ảnh
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

def comprehensive_marker_analysis():
    """Phân tích toàn diện để tìm tất cả marker"""
    
    print("🔍 Comprehensive Marker Search")
    print("=" * 60)
    print("Target: 6 large markers + 18 small markers = 24 total")
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
    debug_dir = Path("data/grading/comprehensive_search")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear old files
    for file in debug_dir.glob("*.jpg"):
        file.unlink()
    
    # Save original
    cv2.imwrite(str(debug_dir / "01_original.jpg"), image)
    cv2.imwrite(str(debug_dir / "02_grayscale.jpg"), gray)
    
    # Try multiple threshold approaches
    all_candidates = []
    
    print("\n📊 Trying multiple detection approaches:")
    
    # Approach 1: Multiple fixed thresholds
    thresholds = [40, 60, 80, 100, 120, 140, 160]
    for i, thresh in enumerate(thresholds):
        candidates = find_markers_with_threshold(gray, thresh, f"fixed_{thresh}")
        all_candidates.extend(candidates)
        print(f"   Fixed threshold {thresh}: {len(candidates)} candidates")
        
        # Save debug image
        debug_img = visualize_candidates(image.copy(), candidates, (0, 255, 255))
        cv2.imwrite(str(debug_dir / f"03_thresh_{thresh}.jpg"), debug_img)
    
    # Approach 2: Adaptive thresholds
    adaptive_params = [
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 9, 2),
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 3),
        (cv2.ADAPTIVE_THRESH_MEAN_C, 9, 2),
        (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
    ]
    
    for i, (method, block_size, C) in enumerate(adaptive_params):
        candidates = find_markers_with_adaptive(gray, method, block_size, C, f"adaptive_{i}")
        all_candidates.extend(candidates)
        method_name = "GAUSSIAN" if method == cv2.ADAPTIVE_THRESH_GAUSSIAN_C else "MEAN"
        print(f"   Adaptive {method_name} {block_size},{C}: {len(candidates)} candidates")
        
        # Save debug image
        debug_img = visualize_candidates(image.copy(), candidates, (255, 0, 255))
        cv2.imwrite(str(debug_dir / f"04_adaptive_{i}.jpg"), debug_img)
    
    # Approach 3: Edge detection
    candidates = find_markers_with_edges(gray, "edges")
    all_candidates.extend(candidates)
    print(f"   Edge detection: {len(candidates)} candidates")
    
    debug_img = visualize_candidates(image.copy(), candidates, (0, 255, 0))
    cv2.imwrite(str(debug_dir / "05_edges.jpg"), debug_img)
    
    # Remove duplicates
    unique_candidates = remove_duplicate_candidates(all_candidates)
    print(f"\n📈 Total unique candidates: {len(unique_candidates)}")
    
    # Classify by size
    large_candidates = []
    small_candidates = []
    
    for candidate in unique_candidates:
        area = candidate['area']
        if area > 150:  # Large marker threshold
            large_candidates.append(candidate)
        else:
            small_candidates.append(candidate)
    
    print(f"   Large candidates (area > 150): {len(large_candidates)}")
    print(f"   Small candidates (area ≤ 150): {len(small_candidates)}")
    
    # Show area distribution
    areas = [c['area'] for c in unique_candidates]
    areas.sort()
    print(f"   Area range: {min(areas):.1f} - {max(areas):.1f}")
    print(f"   Area distribution: {areas[:20]}...")  # First 20
    
    # Create final visualization
    final_img = image.copy()
    
    # Draw large candidates in red
    for candidate in large_candidates:
        center = candidate['center']
        cv2.circle(final_img, center, 8, (0, 0, 255), -1)
        cv2.putText(final_img, f"L{candidate['area']:.0f}", 
                   (center[0] - 20, center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw small candidates in green
    for candidate in small_candidates:
        center = candidate['center']
        cv2.circle(final_img, center, 4, (0, 255, 0), -1)
        cv2.putText(final_img, f"S{candidate['area']:.0f}", 
                   (center[0] - 15, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add legend
    cv2.putText(final_img, f"Large: {len(large_candidates)} (target: 6)", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(final_img, f"Small: {len(small_candidates)} (target: 18)", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(final_img, f"Total: {len(unique_candidates)} (target: 24)", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite(str(debug_dir / "06_final_comprehensive.jpg"), final_img)
    
    print(f"\n🎯 Analysis Complete:")
    print(f"   Found: {len(large_candidates)} large + {len(small_candidates)} small = {len(unique_candidates)} total")
    print(f"   Target: 6 large + 18 small = 24 total")
    print(f"   Success rate: {len(unique_candidates)/24*100:.1f}%")
    print(f"   Debug images saved to: {debug_dir}")
    
    return unique_candidates

def find_markers_with_threshold(gray, threshold, name):
    """Tìm markers với fixed threshold"""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return find_contour_candidates(binary, name)

def find_markers_with_adaptive(gray, method, block_size, C, name):
    """Tìm markers với adaptive threshold"""
    binary = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY_INV, block_size, C)
    return find_contour_candidates(binary, name)

def find_markers_with_edges(gray, name):
    """Tìm markers với edge detection"""
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return find_contour_candidates(edges, name)

def find_contour_candidates(binary, name):
    """Tìm candidates từ binary image"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Very loose filtering
        if 10 < area < 1000:  # Very wide range
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Very loose aspect ratio
            if 0.3 <= aspect_ratio <= 3.0:
                center = (x + w // 2, y + h // 2)
                candidate = {
                    'center': center,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'bbox': (x, y, w, h),
                    'source': name
                }
                candidates.append(candidate)
    
    return candidates

def remove_duplicate_candidates(candidates):
    """Loại bỏ candidates trùng lặp"""
    unique = []
    
    for candidate in candidates:
        is_duplicate = False
        center = candidate['center']
        
        for existing in unique:
            existing_center = existing['center']
            distance = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)
            
            # If centers are within 10 pixels, consider duplicate
            if distance < 10:
                is_duplicate = True
                # Keep the one with larger area
                if candidate['area'] > existing['area']:
                    unique.remove(existing)
                    unique.append(candidate)
                break
        
        if not is_duplicate:
            unique.append(candidate)
    
    return unique

def visualize_candidates(image, candidates, color):
    """Vẽ candidates lên ảnh"""
    for candidate in candidates:
        center = candidate['center']
        cv2.circle(image, center, 3, color, -1)
    
    return image

if __name__ == "__main__":
    comprehensive_marker_analysis()
