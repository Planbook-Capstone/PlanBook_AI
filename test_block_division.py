"""
Test script cho Block Division System
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from app.services.all_marker_scanner import all_marker_scanner
from app.services.marker_based_block_divider import marker_based_block_divider

def test_block_division():
    """
    Test block division với sample.jpg
    """
    print("=== TESTING BLOCK DIVISION SYSTEM ===")
    
    # Đường dẫn ảnh test
    image_path = "data/grading/sample.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"✅ Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return False
    
    print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Step 1: Scan markers
    print("\n🔍 Step 1: Scanning markers...")
    marker_result = all_marker_scanner.scan_all_markers(image)
    
    if not marker_result['success']:
        print(f"❌ Marker scanning failed: {marker_result.get('error', 'Unknown error')}")
        return False
    
    large_markers = marker_result['large_markers']
    small_markers = marker_result['small_markers']
    
    print(f"✅ Found {len(large_markers)} large markers")
    print(f"✅ Found {len(small_markers)} small markers")
    
    # Print marker details
    print("\n📍 Large Markers:")
    for i, marker in enumerate(large_markers[:5]):  # Show first 5
        print(f"  L{i+1}: center={marker['center']}, area={marker['area']}")
    
    print("\n📍 Small Markers:")
    for i, marker in enumerate(small_markers[:10]):  # Show first 10
        print(f"  S{i+1}: center={marker['center']}, area={marker['area']}")
    
    # Step 2: Divide blocks
    print("\n🔧 Step 2: Dividing into blocks...")
    block_result = marker_based_block_divider.divide_form_into_blocks(
        image, large_markers, small_markers
    )
    
    if not block_result['success']:
        print(f"❌ Block division failed: {block_result.get('error', 'Unknown error')}")
        return False
    
    # Print results
    total_blocks = block_result['total_blocks']
    main_regions = block_result['main_regions']
    detailed_blocks = block_result['detailed_blocks']
    
    print(f"✅ Created {total_blocks} blocks")
    print(f"✅ Main regions: {list(main_regions.keys())}")
    
    # Print block details
    print("\n📦 Detailed Blocks:")
    for block in detailed_blocks:
        bbox = block['bbox']
        print(f"  {block['id']}: region={block['region']}, bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}), markers={block.get('marker_count', 0)}")
    
    # Print summary
    block_summary = block_result['block_summary']
    print(f"\n📊 Block Summary:")
    print(f"  Total blocks: {block_summary['total_blocks']}")
    print(f"  Blocks by region: {block_summary['blocks_by_region']}")
    print(f"  Blocks by type: {block_summary['blocks_by_type']}")
    print(f"  Average block size: {block_summary['average_block_size']:.0f} pixels²")
    
    # Check visualization
    viz_path = block_result.get('visualization_path')
    if viz_path and os.path.exists(viz_path):
        print(f"✅ Visualization saved: {viz_path}")
    else:
        print("⚠️ Visualization not found")
    
    # Save test results
    test_results = {
        "test_name": "Block Division Test",
        "image_path": image_path,
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "marker_detection": {
            "large_markers": len(large_markers),
            "small_markers": len(small_markers),
            "total_markers": len(large_markers) + len(small_markers)
        },
        "block_division": {
            "total_blocks": total_blocks,
            "main_regions": list(main_regions.keys()),
            "block_summary": block_summary
        },
        "success": True
    }
    
    # Save to file
    results_path = "data/grading/block_division_test_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Test results saved: {results_path}")
    
    print("\n🎉 BLOCK DIVISION TEST COMPLETED SUCCESSFULLY!")
    return True

def test_specific_regions():
    """
    Test chi tiết cho từng region
    """
    print("\n=== TESTING SPECIFIC REGIONS ===")
    
    image_path = "data/grading/sample.jpg"
    image = cv2.imread(image_path)
    
    # Get markers
    marker_result = all_marker_scanner.scan_all_markers(image)
    large_markers = marker_result['large_markers']
    small_markers = marker_result['small_markers']
    
    # Get block division
    block_result = marker_based_block_divider.divide_form_into_blocks(
        image, large_markers, small_markers
    )
    
    main_regions = block_result['main_regions']
    detailed_blocks = block_result['detailed_blocks']
    
    # Analyze each region
    for region_name, region_info in main_regions.items():
        print(f"\n🏷️ Region: {region_name}")
        print(f"   Description: {region_info['description']}")
        print(f"   BBox: {region_info['bbox']}")
        
        # Count blocks in this region
        region_blocks = [b for b in detailed_blocks if b['region'] == region_name]
        print(f"   Blocks: {len(region_blocks)}")
        
        for block in region_blocks:
            print(f"     - {block['id']}: {block['type']}, markers={block.get('marker_count', 0)}")
    
    return True

if __name__ == "__main__":
    try:
        # Test basic block division
        success = test_block_division()
        
        if success:
            # Test specific regions
            test_specific_regions()
            
            print("\n✅ ALL TESTS PASSED!")
        else:
            print("\n❌ TESTS FAILED!")
            
    except Exception as e:
        print(f"\n💥 TEST ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
