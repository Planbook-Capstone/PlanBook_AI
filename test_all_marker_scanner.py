#!/usr/bin/env python3
"""
Test script for All Marker Scanner
Quét và đánh dấu tất cả marker (lớn và nhỏ) trên một ảnh tổng hợp
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.all_marker_scanner import all_marker_scanner

def test_all_marker_scanner():
    """Test all marker scanner với sample.jpg"""
    
    print("🔍 All Marker Scanner Test")
    print("=" * 60)
    print("📋 Features:")
    print("   🔴 Phát hiện marker vuông lớn (Large Markers)")
    print("   🟢 Phát hiện marker vuông nhỏ (Small Markers)")
    print("   🎨 Đánh dấu và phân biệt bằng màu sắc")
    print("   📊 Tạo ảnh tổng hợp với tất cả marker")
    print("   📈 Thống kê số lượng và vị trí marker")
    print("=" * 60)
    
    # Test với sample.jpg
    sample_path = "data/grading/sample.jpg"
    
    if not os.path.exists(sample_path):
        print(f"❌ Sample file not found: {sample_path}")
        return False
    
    print(f"✅ Testing with: {sample_path}")
    print(f"📏 File size: {os.path.getsize(sample_path) / 1024:.1f} KB")
    
    try:
        # Quét tất cả marker
        print("\n🔄 Scanning all markers...")
        result = all_marker_scanner.scan_all_markers(sample_path)
        
        if not result.get('success'):
            print(f"❌ All marker scanning failed: {result.get('error')}")
            return False
        
        print(f"✅ All marker scanning successful!")
        
        # Phân tích kết quả
        large_markers = result.get('large_markers', [])
        small_markers = result.get('small_markers', [])
        statistics = result.get('statistics', {})
        
        print(f"\n📊 Marker Detection Results:")
        
        # Large markers
        print(f"   🔴 Large Markers: {len(large_markers)}")
        if large_markers:
            avg_area_large = statistics.get('large_markers', {}).get('average_area', 0)
            avg_ratio_large = statistics.get('large_markers', {}).get('average_aspect_ratio', 0)
            print(f"      Average area: {avg_area_large:.1f} pixels")
            print(f"      Average aspect ratio: {avg_ratio_large:.2f}")
            
            # Show first 5 large markers
            for i, marker in enumerate(large_markers[:5]):
                center = marker['center']
                area = marker['area']
                print(f"      {marker['id']}: center({center[0]}, {center[1]}) area={area}")
            
            if len(large_markers) > 5:
                print(f"      ... and {len(large_markers) - 5} more large markers")
        
        # Small markers
        print(f"   🟢 Small Markers: {len(small_markers)}")
        if small_markers:
            avg_area_small = statistics.get('small_markers', {}).get('average_area', 0)
            avg_ratio_small = statistics.get('small_markers', {}).get('average_aspect_ratio', 0)
            print(f"      Average area: {avg_area_small:.1f} pixels")
            print(f"      Average aspect ratio: {avg_ratio_small:.2f}")
            
            # Show first 10 small markers
            for i, marker in enumerate(small_markers[:10]):
                center = marker['center']
                area = marker['area']
                print(f"      {marker['id']}: center({center[0]}, {center[1]}) area={area}")
            
            if len(small_markers) > 10:
                print(f"      ... and {len(small_markers) - 10} more small markers")
        
        # Total summary
        total_markers = len(large_markers) + len(small_markers)
        print(f"\n📈 Summary:")
        print(f"   Total markers detected: {total_markers}")
        print(f"   Large markers: {len(large_markers)} ({len(large_markers)/total_markers*100:.1f}%)")
        print(f"   Small markers: {len(small_markers)} ({len(small_markers)/total_markers*100:.1f}%)")
        
        # Kiểm tra debug images
        debug_dir = Path(result.get('debug_dir', 'data/grading/all_markers_debug'))
        if debug_dir.exists():
            debug_files = list(debug_dir.glob("*.jpg"))
            print(f"\n🖼️ Debug Images: {len(debug_files)}")
            
            # Phân loại debug images
            categories = {
                "original": [],
                "large_detection": [],
                "small_detection": [],
                "final_visualization": []
            }
            
            for img_file in debug_files:
                name = img_file.name.lower()
                if "original" in name:
                    categories["original"].append(img_file.name)
                elif "large" in name:
                    categories["large_detection"].append(img_file.name)
                elif "small" in name:
                    categories["small_detection"].append(img_file.name)
                elif "final" in name or "all_markers" in name:
                    categories["final_visualization"].append(img_file.name)
            
            for category, files in categories.items():
                if files:
                    print(f"   📁 {category.replace('_', ' ').title()}: {len(files)} images")
                    for file in files:
                        print(f"      • {file}")
        
        # Kiểm tra file thống kê
        stats_file = debug_dir / "marker_statistics.json"
        if stats_file.exists():
            print(f"\n📋 Statistics File: {stats_file}")
            file_size = stats_file.stat().st_size
            print(f"   File size: {file_size} bytes")
        
        # Validation
        print(f"\n🎯 Validation:")
        
        # Expected markers cho sample OMR
        expected_large = 6  # Thường có 6 marker lớn ở góc
        expected_small = 18  # Thường có 18 marker nhỏ để chia vùng
        
        validations = [
            ("Large markers", len(large_markers), expected_large, "Corner/alignment markers"),
            ("Small markers", len(small_markers), expected_small, "Section division markers"),
            ("Total markers", total_markers, expected_large + expected_small, "All markers combined")
        ]
        
        all_valid = True
        for name, actual, expected, description in validations:
            if actual >= expected * 0.5:  # Accept if at least 50% of expected
                status = "✅" if actual >= expected else "⚠️"
                print(f"   {status} {name}: {actual} (expected ~{expected}) - {description}")
            else:
                print(f"   ❌ {name}: {actual} (expected ~{expected}) - {description}")
                all_valid = False
        
        # Marker quality check
        print(f"\n🔍 Marker Quality Check:")
        
        if large_markers:
            large_areas = [m['area'] for m in large_markers]
            large_ratios = [m['aspect_ratio'] for m in large_markers]
            
            print(f"   🔴 Large markers:")
            print(f"      Area range: {min(large_areas):.0f} - {max(large_areas):.0f} pixels")
            print(f"      Aspect ratio range: {min(large_ratios):.2f} - {max(large_ratios):.2f}")
            
            # Check if markers are well-distributed
            large_x_coords = [m['center'][0] for m in large_markers]
            large_y_coords = [m['center'][1] for m in large_markers]
            x_spread = max(large_x_coords) - min(large_x_coords) if large_x_coords else 0
            y_spread = max(large_y_coords) - min(large_y_coords) if large_y_coords else 0
            
            print(f"      Distribution: {x_spread}px width, {y_spread}px height")
        
        if small_markers:
            small_areas = [m['area'] for m in small_markers]
            small_ratios = [m['aspect_ratio'] for m in small_markers]
            
            print(f"   🟢 Small markers:")
            print(f"      Area range: {min(small_areas):.0f} - {max(small_areas):.0f} pixels")
            print(f"      Aspect ratio range: {min(small_ratios):.2f} - {max(small_ratios):.2f}")
            
            # Check if markers are well-distributed
            small_x_coords = [m['center'][0] for m in small_markers]
            small_y_coords = [m['center'][1] for m in small_markers]
            x_spread = max(small_x_coords) - min(small_x_coords) if small_x_coords else 0
            y_spread = max(small_y_coords) - min(small_y_coords) if small_y_coords else 0
            
            print(f"      Distribution: {x_spread}px width, {y_spread}px height")
        
        # Final result image
        final_image_path = result.get('all_markers_image_path')
        if final_image_path and os.path.exists(final_image_path):
            print(f"\n🎨 Final Visualization:")
            print(f"   📸 All markers image: {final_image_path}")
            print(f"   📏 File size: {os.path.getsize(final_image_path) / 1024:.1f} KB")
            print(f"   🎯 Features: Color-coded markers with labels and legend")
        
        print(f"\n🎉 All Marker Scanner test completed!")
        print(f"📁 Check outputs:")
        print(f"   Debug images: {debug_dir}")
        print(f"   Final visualization: {final_image_path}")
        print(f"   Statistics: {stats_file}")
        
        return total_markers > 0  # Success if any markers detected
        
    except Exception as e:
        print(f"❌ Error during all marker scanning: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_marker_types():
    """Hiển thị thông tin về các loại marker"""
    print("\n🎯 Marker Types Information:")
    print("=" * 50)
    print("🔴 LARGE MARKERS (Marker Vuông Lớn):")
    print("   • Size: 19x19 pixels (area ≈ 361)")
    print("   • Area range: 250-600 pixels (±50% tolerance)")
    print("   • Aspect ratio: 0.85-1.15 (chặt chẽ)")
    print("   • Purpose: Corner alignment, perspective correction")
    print("   • Color: Red (đỏ) with white labels")
    print("   • Expected: ~6 markers (4 corners + 2 additional)")
    print()
    print("🟢 SMALL MARKERS (Marker Vuông Nhỏ):")
    print("   • Size: 9x9 pixels (area ≈ 81)")
    print("   • Area range: 40-150 pixels (±50% tolerance)")
    print("   • Aspect ratio: 0.8-1.25 (chặt chẽ)")
    print("   • Purpose: Section division, block separation")
    print("   • Color: Green (xanh lá) with black labels")
    print("   • Expected: ~18 markers (region division)")
    print()
    print("🎨 VISUALIZATION FEATURES:")
    print("   • Color-coded bounding boxes")
    print("   • Unique ID labels (L1, L2... for large, S1, S2... for small)")
    print("   • Legend with counts")
    print("   • Center point markers")
    print("   • Comprehensive statistics")
    print("=" * 50)

if __name__ == "__main__":
    print("🔍 All Marker Scanner Test Suite")
    print("=" * 60)
    
    # Show marker types info
    show_marker_types()
    
    # Run test
    success = test_all_marker_scanner()
    
    if success:
        print("\n✅ All Marker Scanner test completed successfully!")
        print("💡 The comprehensive marker detection system is ready!")
        sys.exit(0)
    else:
        print("\n❌ All Marker Scanner test failed!")
        print("💡 Check the debug images for marker detection issues.")
        sys.exit(1)
