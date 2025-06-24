#!/usr/bin/env python3
"""
Debug script để phân tích và phân loại marker đúng
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.all_marker_scanner import all_marker_scanner

def analyze_markers():
    """Phân tích chi tiết các marker được phát hiện"""
    
    print("🔍 Debug Marker Classification")
    print("=" * 60)
    
    # Load image
    image_path = "data/grading/sample.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    print(f"✅ Image loaded: {image.shape}")
    
    # Analyze both large and small marker detection
    print("\n📊 Analyzing Large Marker Detection:")
    large_markers = analyze_large_markers(image)
    
    print("\n📊 Analyzing Small Marker Detection:")
    small_markers = analyze_small_markers(image)
    
    print(f"\n📈 Summary:")
    print(f"   Large markers found: {len(large_markers)}")
    print(f"   Small markers found: {len(small_markers)}")
    
    # Analyze area distribution
    if large_markers or small_markers:
        all_areas = []
        if large_markers:
            large_areas = [m['area'] for m in large_markers]
            all_areas.extend(large_areas)
            print(f"   Large marker areas: {large_areas}")
        
        if small_markers:
            small_areas = [m['area'] for m in small_markers]
            all_areas.extend(small_areas)
            print(f"   Small marker areas: {small_areas}")
        
        print(f"   All areas: {sorted(all_areas)}")
        
        # Suggest correct classification
        print(f"\n💡 Suggested Classification:")
        print(f"   Expected large (19x19=361): areas > 150")
        print(f"   Expected small (9x9=81): areas < 150")
        
        for area in sorted(all_areas):
            if area > 150:
                print(f"   Area {area:.1f} → LARGE marker")
            else:
                print(f"   Area {area:.1f} → SMALL marker")

def analyze_large_markers(image):
    """Phân tích detection của large markers"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple thresholds
    _, binary1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    _, binary2 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_or(binary1, binary2)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"   Total contours found: {len(contours)}")
    
    large_markers = []
    area_candidates = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        area_candidates.append(area)
        
        if i < 10:  # Show first 10
            print(f"   Contour {i}: area={area:.1f}")
        
        # Current filter for large markers
        if 180 < area < 300:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                if 0.7 <= aspect_ratio <= 1.4:
                    marker_info = {
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': (x + w // 2, y + h // 2)
                    }
                    large_markers.append(marker_info)
                    print(f"   ✅ Large marker: area={area:.1f}, ratio={aspect_ratio:.2f}")
                else:
                    print(f"   ❌ Rejected (aspect): area={area:.1f}, ratio={aspect_ratio:.2f}")
            else:
                print(f"   ❌ Rejected (shape): area={area:.1f}, points={len(approx)}")
    
    # Show area distribution
    area_candidates.sort()
    print(f"   Area range: {min(area_candidates):.1f} - {max(area_candidates):.1f}")
    print(f"   Areas in range 180-300: {[a for a in area_candidates if 180 < a < 300]}")
    
    return large_markers

def analyze_small_markers(image):
    """Phân tích detection của small markers"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple approaches
    binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    _, binary2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    _, binary3 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_or(cv2.bitwise_or(binary1, binary2), binary3)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"   Total contours found: {len(contours)}")
    
    small_markers = []
    area_candidates = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        area_candidates.append(area)
        
        if i < 10:  # Show first 10
            print(f"   Contour {i}: area={area:.1f}")
        
        # Current filter for small markers
        if 25 < area < 180:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                if 0.8 <= aspect_ratio <= 1.25:
                    marker_info = {
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': (x + w // 2, y + h // 2)
                    }
                    small_markers.append(marker_info)
                    print(f"   ✅ Small marker: area={area:.1f}, ratio={aspect_ratio:.2f}")
                else:
                    print(f"   ❌ Rejected (aspect): area={area:.1f}, ratio={aspect_ratio:.2f}")
            else:
                print(f"   ❌ Rejected (shape): area={area:.1f}, points={len(approx)}")
    
    # Show area distribution
    area_candidates.sort()
    print(f"   Area range: {min(area_candidates):.1f} - {max(area_candidates):.1f}")
    print(f"   Areas in range 25-180: {[a for a in area_candidates if 25 < a < 180]}")
    
    return small_markers

if __name__ == "__main__":
    analyze_markers()
