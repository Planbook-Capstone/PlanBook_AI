"""
Test script cho divide_blocks API
"""

import requests
import json

def test_divide_blocks_api():
    """
    Test divide_blocks API endpoint
    """
    print("=== TESTING DIVIDE BLOCKS API ===")
    
    url = "http://localhost:8000/api/v1/omr_debug/divide_blocks"
    
    try:
        # Test without file (should use sample.jpg)
        print("🔧 Testing divide_blocks without file (using sample.jpg)...")
        
        # Create empty FormData
        files = {}
        
        response = requests.post(url, files=files, timeout=60)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"📁 Filename: {result.get('filename', 'N/A')}")
            print(f"📐 Image size: {result.get('image_size', 'N/A')}")
            print(f"🔴 Large markers: {result.get('marker_detection', {}).get('large_markers_count', 0)}")
            print(f"🟢 Small markers: {result.get('marker_detection', {}).get('small_markers_count', 0)}")
            print(f"📦 Total blocks: {result.get('block_division', {}).get('total_blocks', 0)}")
            print(f"🏷️ Regions: {len(result.get('block_division', {}).get('main_regions', []))}")
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
    
    url = "http://localhost:8000/api/v1/omr_debug/divide_blocks"
    
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
            print(f"📦 Total blocks: {result.get('block_division', {}).get('total_blocks', 0)}")
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
        success1 = test_divide_blocks_api()
        
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
