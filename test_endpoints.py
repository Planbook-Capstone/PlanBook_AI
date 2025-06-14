#!/usr/bin/env python3
"""
Test script để test các endpoints mới
"""

import requests
import json
import time

def test_create_task():
    """Test tạo task mới"""
    print("🚀 Creating new task...")
    
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
            return result.get("task_id")
        else:
            print(f"❌ Error creating task: {response.status_code}")
            print(response.text)
            return None
            
    except FileNotFoundError:
        print("❌ File not found: data/sáchgiaokhoa/chemistry_test.pdf")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_get_all_tasks():
    """Test getAllTask endpoint"""
    print("\n📋 Testing getAllTask endpoint...")
    
    url = "http://127.0.0.1:8000/api/v1/tasks/getAllTask"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ getAllTask successful!")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_get_all_textbooks():
    """Test getAllTextBook endpoint"""
    print("\n📚 Testing getAllTextBook endpoint...")
    
    url = "http://127.0.0.1:8000/api/v1/pdf/getAllTextBook"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ getAllTextBook successful!")
            print(f"Found {result.get('total_textbooks', 0)} textbooks")
            
            # Show summary of each textbook
            for i, textbook in enumerate(result.get('textbooks', [])):
                print(f"\n📖 Textbook {i+1}:")
                print(f"  - Book ID: {textbook.get('book_id')}")
                print(f"  - Success: {textbook.get('success')}")
                print(f"  - Chapters: {textbook.get('statistics', {}).get('total_chapters', 0)}")
                print(f"  - Lessons: {textbook.get('statistics', {}).get('total_lessons', 0)}")
                print(f"  - Method: {textbook.get('processing_info', {}).get('processing_method')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing new endpoints...")
    
    # Test getAllTextBook
    test_get_all_textbooks()
    
    # Test getAllTask
    test_get_all_tasks()
    
    # Test creating new task
    task_id = test_create_task()
    
    if task_id:
        print(f"\n⏳ Waiting for task {task_id} to process...")
        time.sleep(3)
        
        # Test getAllTask again to see the new task
        test_get_all_tasks()
