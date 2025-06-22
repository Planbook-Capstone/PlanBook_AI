#!/usr/bin/env python3
"""
Script test OMR Debug API
"""

import requests
import json

def test_omr_debug_api():
    """Test OMR Debug API endpoints"""
    
    base_url = "http://127.0.0.1:8000/api/v1/omr_debug"
    
    print("🔍 Testing OMR Debug API")
    print("=" * 50)
    
    # Test 1: Process test image
    print("\n1. Testing process_test_image endpoint...")
    try:
        response = requests.post(f"{base_url}/process_test_image")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Process test image successful!")
            print(f"📝 Student ID: {result.get('student_id', 'N/A')}")
            print(f"📋 Test Code: {result.get('test_code', 'N/A')}")
            print(f"📚 Total Answers: {result.get('total_answers', 0)}")
            print(f"🖼️  Debug Files: {len(result.get('debug_files', []))}")
            
            # Show some answers
            answers = result.get('answers', {})
            if answers:
                print(f"\n📖 Sample answers (first 10):")
                for i, (q_num, answer) in enumerate(list(answers.items())[:10]):
                    print(f"   Q{q_num}: {answer}")
                if len(answers) > 10:
                    print(f"   ... and {len(answers) - 10} more")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: List debug images
    print("\n2. Testing debug_images endpoint...")
    try:
        response = requests.get(f"{base_url}/debug_images")
        
        if response.status_code == 200:
            result = response.json()
            debug_files = result.get('debug_files', [])
            print(f"✅ Found {len(debug_files)} debug images")
            
            for file_info in debug_files[:5]:  # Show first 5
                print(f"   - {file_info['filename']} ({file_info['size']} bytes)")
            
            if len(debug_files) > 5:
                print(f"   ... and {len(debug_files) - 5} more files")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 3: Get processing steps info
    print("\n3. Testing processing_steps endpoint...")
    try:
        response = requests.get(f"{base_url}/processing_steps")
        
        if response.status_code == 200:
            result = response.json()
            steps = result.get('processing_steps', [])
            print(f"✅ Found {len(steps)} processing steps")
            
            for step in steps[:3]:  # Show first 3
                print(f"   Step {step['step']}: {step['name']} - {step['description']}")
            
            if len(steps) > 3:
                print(f"   ... and {len(steps) - 3} more steps")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API Testing completed!")
    print("\n💡 You can now:")
    print("   - View debug images in browser at: http://127.0.0.1:8000/api/v1/omr_debug/debug_image/<filename>")
    print("   - Check API docs at: http://127.0.0.1:8000/api/v1/docs")
    print("   - View debug images in folder: data/grading/debug/")

if __name__ == "__main__":
    test_omr_debug_api()
