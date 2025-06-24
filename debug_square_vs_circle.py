#!/usr/bin/env python3
"""
Debug Square vs Circle Detection
Phân tích chi tiết để phân biệt marker vuông và bubble tròn
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.all_marker_scanner import all_marker_scanner

def analyze_square_vs_circle():
    """Phân tích chi tiết square vs circle"""
    
    print("🔍 Square vs Circle Analysis")
    print("=" * 60)
    print("Goal: Distinguish square markers from round answer bubbles")
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
    debug_dir = Path("data/grading/square_vs_circle")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear old files
    for file in debug_dir.glob("*.jpg"):
        file.unlink()
    
    # Step 1: Find all contours
    print("\n🔍 Step 1: Finding all contours...")
    all_candidates = find_all_shape_candidates(gray, image, debug_dir)
    
    # Step 2: Analyze shape characteristics
    print(f"\n📊 Step 2: Analyzing {len(all_candidates)} candidates...")
    analyze_shape_characteristics(all_candidates)
    
    # Step 3: Apply strict square filters
    print(f"\n🎯 Step 3: Applying strict square filters...")
    square_candidates = apply_strict_square_filters(all_candidates)
    
    # Step 4: Create comparison visualization
    print(f"\n🎨 Step 4: Creating visualization...")
    create_comparison_visualization(image, all_candidates, square_candidates, debug_dir)
    
    print(f"\n📈 Final Results:")
    print(f"   All candidates: {len(all_candidates)}")
    print(f"   Square candidates: {len(square_candidates)}")
    print(f"   Filtered out: {len(all_candidates) - len(square_candidates)} (likely circles/bubbles)")

def find_all_shape_candidates(gray, image, debug_dir):
    """Tìm tất cả shape candidates"""
    candidates = []
    
    # Multiple thresholds
    thresholds = [80, 100, 120]
    
    for thresh in thresholds:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (small markers range)
            if 20 < area < 200:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Basic filters
                if 0.5 <= aspect_ratio <= 2.0:
                    # Calculate shape metrics
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    bbox_area = w * h
                    rectangularity = area / bbox_area if bbox_area > 0 else 0
                    
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
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
                        'approx_points': len(approx),
                        'threshold': thresh
                    }
                    candidates.append(candidate)
    
    # Remove duplicates
    unique_candidates = remove_duplicates(candidates)
    return unique_candidates

def analyze_shape_characteristics(candidates):
    """Phân tích đặc điểm hình dạng"""
    
    print(f"   Analyzing {len(candidates)} candidates...")
    
    # Group by characteristics
    squares = []
    circles = []
    others = []
    
    for candidate in candidates:
        approx_points = candidate['approx_points']
        circularity = candidate['circularity']
        rectangularity = candidate['rectangularity']
        
        # Classification logic
        if approx_points == 4 and circularity < 0.8 and rectangularity > 0.7:
            squares.append(candidate)
        elif circularity > 0.8:
            circles.append(candidate)
        else:
            others.append(candidate)
    
    print(f"   Likely squares: {len(squares)}")
    print(f"   Likely circles: {len(circles)}")
    print(f"   Others: {len(others)}")
    
    # Show examples
    if squares:
        print(f"\n   📐 Square Examples:")
        for i, sq in enumerate(squares[:5]):
            print(f"      Square {i+1}: area={sq['area']:.1f}, points={sq['approx_points']}, "
                  f"circularity={sq['circularity']:.3f}, rectangularity={sq['rectangularity']:.3f}")
    
    if circles:
        print(f"\n   ⭕ Circle Examples:")
        for i, circ in enumerate(circles[:5]):
            print(f"      Circle {i+1}: area={circ['area']:.1f}, points={circ['approx_points']}, "
                  f"circularity={circ['circularity']:.3f}, rectangularity={circ['rectangularity']:.3f}")

def apply_strict_square_filters(candidates):
    """Áp dụng filter chặt chẽ cho square"""
    
    strict_squares = []
    
    for candidate in candidates:
        # STRICT square criteria:
        # 1. Exactly 4 corners
        # 2. Low circularity (not round)
        # 3. High rectangularity (fills bounding box)
        # 4. Good aspect ratio (square-like)
        # 5. High solidity (filled)
        
        if (candidate['approx_points'] == 4 and
            candidate['circularity'] < 0.82 and
            candidate['rectangularity'] > 0.65 and
            0.8 <= candidate['aspect_ratio'] <= 1.25 and
            candidate['solidity'] > 0.75):
            
            strict_squares.append(candidate)
    
    print(f"   Strict square filter: {len(strict_squares)}/{len(candidates)} passed")
    
    return strict_squares

def create_comparison_visualization(image, all_candidates, square_candidates, debug_dir):
    """Tạo visualization so sánh"""
    
    # Image 1: All candidates
    all_img = image.copy()
    for candidate in all_candidates:
        center = candidate['center']
        cv2.circle(all_img, center, 3, (0, 255, 255), -1)  # Yellow for all
        cv2.putText(all_img, f"{candidate['area']:.0f}", 
                   (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    cv2.putText(all_img, f"All Candidates: {len(all_candidates)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(debug_dir / "01_all_candidates.jpg"), all_img)
    
    # Image 2: Square candidates only
    square_img = image.copy()
    for candidate in square_candidates:
        center = candidate['center']
        bbox = candidate['bbox']
        
        # Draw bounding box
        cv2.rectangle(square_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.circle(square_img, center, 3, (0, 255, 0), -1)  # Green for squares
        cv2.putText(square_img, f"S{candidate['area']:.0f}", 
                   (center[0] - 15, center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.putText(square_img, f"Square Markers: {len(square_candidates)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(debug_dir / "02_square_markers.jpg"), square_img)
    
    # Image 3: Rejected candidates (likely circles)
    rejected = [c for c in all_candidates if c not in square_candidates]
    rejected_img = image.copy()
    for candidate in rejected:
        center = candidate['center']
        cv2.circle(rejected_img, center, 3, (0, 0, 255), -1)  # Red for rejected
        cv2.putText(rejected_img, f"R{candidate['area']:.0f}", 
                   (center[0] - 15, center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.putText(rejected_img, f"Rejected (Circles/Others): {len(rejected)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(debug_dir / "03_rejected_candidates.jpg"), rejected_img)
    
    print(f"   Visualizations saved to: {debug_dir}")

def remove_duplicates(candidates):
    """Remove duplicate candidates"""
    unique = []
    
    for candidate in candidates:
        is_duplicate = False
        center = candidate['center']
        
        for existing in unique:
            existing_center = existing['center']
            distance = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)
            
            if distance < 15:
                is_duplicate = True
                # Keep the one with better square characteristics
                if (candidate['rectangularity'] > existing['rectangularity'] and 
                    candidate['circularity'] < existing['circularity']):
                    unique.remove(existing)
                    unique.append(candidate)
                break
        
        if not is_duplicate:
            unique.append(candidate)
    
    return unique

if __name__ == "__main__":
    analyze_square_vs_circle()
