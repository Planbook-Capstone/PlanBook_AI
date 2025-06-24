"""
Test script cho contour block cutting API
"""

import requests
import json

def test_contour_cutting_api():
    """
    Test contour block cutting API endpoint
    """
    print("=== TESTING CONTOUR BLOCK CUTTING API ===")
    
    url = "http://localhost:8000/api/v1/omr_debug/cut_contour_blocks"
    
    try:
        # Test without file (should use sample.jpg)
        print("✂️ Testing cut_contour_blocks without file (using sample.jpg)...")
        
        # Create empty FormData
        files = {}
        
        response = requests.post(url, files=files, timeout=60)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"📁 Filename: {result.get('filename', 'N/A')}")
            print(f"📐 Image size: {result.get('image_size', 'N/A')}")
            
            # Contour detection info
            contour_detection = result.get('contour_detection', {})
            print(f"🔍 Total contours: {contour_detection.get('total_contours', 0)}")
            
            classified = contour_detection.get('classified_contours', {})
            print(f"📦 Large regions: {classified.get('large_regions', 0)}")
            print(f"🔲 Medium blocks: {classified.get('medium_blocks', 0)}")
            print(f"🔸 Small markers: {classified.get('small_markers', 0)}")
            print(f"🗑️ Noise: {classified.get('noise', 0)}")
            
            # Block cutting info
            block_cutting = result.get('block_cutting', {})
            print(f"✂️ Total blocks cut: {block_cutting.get('total_blocks_cut', 0)}")
            print(f"📊 Average block size: {block_cutting.get('average_block_size', 0):.1f}")
            print(f"🏆 Largest block: {block_cutting.get('largest_block', 'N/A')}")
            print(f"🏅 Smallest block: {block_cutting.get('smallest_block', 'N/A')}")
            
            # Block types
            blocks_by_type = block_cutting.get('blocks_by_type', {})
            print("📋 Blocks by type:")
            for block_type, count in blocks_by_type.items():
                print(f"   {block_type}: {count}")
            
            # Output files
            output_files = result.get('output_files', {})
            print(f"🖼️ Total block images: {output_files.get('total_block_images', 0)}")
            print(f"📁 Individual blocks dir: {output_files.get('individual_blocks_dir', 'N/A')}")
            
            # Debug info
            debug_info = result.get('debug_info', {})
            print(f"🔧 Debug images: {debug_info.get('total_debug_images', 0)}")
            
            print(f"💬 Message: {result.get('message', 'N/A')}")
            
            return True
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timeout - API took too long to respond")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - Is the server running?")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        return False

def test_with_file():
    """
    Test với file upload
    """
    print("\n=== TESTING WITH FILE UPLOAD ===")
    
    url = "http://localhost:8000/api/v1/omr_debug/cut_contour_blocks"
    
    try:
        # Test with actual file
        image_path = "data/grading/sample.jpg"
        
        with open(image_path, 'rb') as f:
            files = {'file': ('sample.jpg', f, 'image/jpeg')}
            
            response = requests.post(url, files=files, timeout=60)
            
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File upload test successful!")
            print(f"📁 Filename: {result.get('filename', 'N/A')}")
            print(f"✂️ Total blocks cut: {result.get('block_cutting', {}).get('total_blocks_cut', 0)}")
            return True
        else:
            print(f"❌ File upload test failed with status {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except FileNotFoundError:
        print(f"📁 File not found: {image_path}")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Test 1: Without file (default sample.jpg)
        success1 = test_contour_cutting_api()
        
        # Test 2: With file upload
        success2 = test_with_file()
        
        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 TEST ERROR: {str(e)}")
