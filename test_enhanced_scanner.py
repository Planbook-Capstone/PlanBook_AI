#!/usr/bin/env python3
"""
Test Enhanced All Marker Scanner
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.all_marker_scanner import all_marker_scanner

def test_enhanced_scanner():
    """Test enhanced scanner"""
    
    print("🧠 Testing Enhanced All Marker Scanner")
    print("=" * 60)
    
    # Test với sample.jpg
    sample_path = "data/grading/sample.jpg"
    
    if not os.path.exists(sample_path):
        print(f"❌ Sample file not found: {sample_path}")
        return False
    
    print(f"✅ Testing with: {sample_path}")
    
    try:
        # Process with enhanced scanner
        print("\n🔄 Running enhanced scanner...")
        result = all_marker_scanner.scan_all_markers(sample_path)
        
        if not result.get('success'):
            print(f"❌ Enhanced scanner failed: {result.get('error')}")
            return False
        
        print(f"✅ Enhanced scanner successful!")
        
        # Analyze results
        large_markers = result.get('large_markers', [])
        small_markers = result.get('small_markers', [])
        
        print(f"\n📊 Enhanced Results:")
        print(f"   Large markers: {len(large_markers)}/6 (target)")
        print(f"   Small markers: {len(small_markers)}/18 (target)")
        print(f"   Total: {len(large_markers) + len(small_markers)}/24 (target)")
        print(f"   Success rate: {(len(large_markers) + len(small_markers))/24*100:.1f}%")
        
        # Show details
        if large_markers:
            print(f"\n🔴 Large Markers:")
            for marker in large_markers:
                print(f"   {marker['id']}: center{marker['center']}, area={marker['area']}")
        
        if small_markers:
            print(f"\n🟢 Small Markers:")
            for marker in small_markers:
                print(f"   {marker['id']}: center{marker['center']}, area={marker['area']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during enhanced scanner testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_scanner()
