#!/usr/bin/env python3
"""
Test script để test quick-textbook-analysis API
"""

import requests
import json
import time

def test_quick_analysis():
    """Test quick analysis API"""
    
    # 1. Tạo task
    print("🚀 Creating quick analysis task...")
    
    url = "http://127.0.0.1:8000/api/v1/pdf/quick-textbook-analysis"
    
    try:
        with open("data/sáchgiaokhoa/chemistry_test.pdf", "rb") as f:
            files = {"file": ("chemistry_test.pdf", f, "application/pdf")}
            data = {"create_embeddings": "true"}
            
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Task created successfully!")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            task_id = result.get("task_id")
            if task_id:
                print(f"\n📋 Task ID: {task_id}")
                
                # 2. Kiểm tra status
                print("\n🔍 Checking task status...")
                status_url = f"http://127.0.0.1:8000/api/v1/tasks/status/{task_id}"
                
                for i in range(10):  # Check 10 times
                    time.sleep(2)
                    status_response = requests.get(status_url)
                    
                    if status_response.status_code == 200:
                        status_result = status_response.json()
                        print(f"Status: {status_result.get('status')} - Progress: {status_result.get('progress')}% - {status_result.get('message')}")
                        
                        if status_result.get('status') == 'completed':
                            print("\n🎉 Task completed!")
                            
                            # 3. Lấy kết quả
                            result_url = f"http://127.0.0.1:8000/api/v1/tasks/result/{task_id}"
                            result_response = requests.get(result_url)
                            
                            if result_response.status_code == 200:
                                final_result = result_response.json()
                                print("📊 Final result:")
                                print(json.dumps({
                                    "success": final_result.get("success"),
                                    "book_id": final_result.get("book_id"),
                                    "filename": final_result.get("filename"),
                                    "statistics": final_result.get("statistics"),
                                    "embeddings_created": final_result.get("embeddings_created"),
                                    "embeddings_info": final_result.get("embeddings_info")
                                }, indent=2, ensure_ascii=False))
                            else:
                                print(f"❌ Error getting result: {result_response.status_code}")
                                print(result_response.text)
                            break
                        elif status_result.get('status') == 'failed':
                            print(f"❌ Task failed: {status_result.get('error')}")
                            break
                    else:
                        print(f"❌ Error checking status: {status_response.status_code}")
                        break
                        
        else:
            print(f"❌ Error creating task: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print("❌ File not found: data/sáchgiaokhoa/chemistry_test.pdf")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_quick_analysis()
