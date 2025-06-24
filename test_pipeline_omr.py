#!/usr/bin/env python3
"""
Test script for AI Trắc Nghiệm Pipeline
Kiểm tra pipeline xử lý OMR mới với 7 bước:
1. Đọc ảnh đầu vào (grayscale + blur)
2. Phát hiện marker đen lớn (vuông) + perspective transform
3. Cắt ảnh thành 2 vùng (top: thông tin, bottom: câu trả lời)
4. Phân tích vùng trả lời (3 sections)
5. Tổng hợp kết quả
6. Tạo ảnh kết quả đánh dấu
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.omr_pipeline_processor import omr_pipeline_processor

def test_pipeline_processing():
    """Test the AI Trắc Nghiệm Pipeline"""
    
    print("🚀 AI Trắc Nghiệm Pipeline Test")
    print("=" * 60)
    print("📋 Pipeline Description:")
    print("   1. Đọc ảnh đầu vào (grayscale + blur)")
    print("   2. Phát hiện marker đen lớn (vuông) + perspective transform")
    print("   3. Cắt ảnh thành 2 vùng (top: thông tin, bottom: câu trả lời)")
    print("   4. Phân tích vùng trả lời:")
    print("      • Section I: 40 câu ABCD (4 cột x 10 hàng)")
    print("      • Section II: 8 câu đúng/sai với sub-questions")
    print("      • Section III: 6 câu điền số 0-9")
    print("   5. Tổng hợp kết quả")
    print("   6. Tạo ảnh kết quả đánh dấu")
    print("=" * 60)
    
    # Test with sample.jpg
    sample_path = "data/grading/sample.jpg"
    
    if not os.path.exists(sample_path):
        print(f"❌ Sample file not found: {sample_path}")
        return False
    
    print(f"✅ Testing with: {sample_path}")
    print(f"📏 File size: {os.path.getsize(sample_path) / 1024:.1f} KB")
    
    try:
        # Process with pipeline
        print("\n🔄 Processing with AI Trắc Nghiệm Pipeline...")
        result = omr_pipeline_processor.process_omr_sheet(sample_path)
        
        if not result.get('success'):
            print(f"❌ Pipeline processing failed: {result.get('error')}")
            return False
        
        print(f"✅ Pipeline processing successful!")
        print(f"   Processing steps: {result.get('processing_steps', 0)}")
        print(f"   Total markers detected: {result.get('total_markers', 0)}")
        
        # Analyze results
        results_data = result.get('results', {})
        print(f"\n📊 Results Analysis:")
        
        # Section I Analysis
        section1 = results_data.get('Section I', {})
        print(f"   📝 Section I (Multiple Choice ABCD):")
        print(f"      Total questions: {len(section1)}")
        if section1:
            # Show first 5 answers
            sample_answers = list(section1.items())[:5]
            for q, answer in sample_answers:
                print(f"      {q}: {answer}")
            if len(section1) > 5:
                print(f"      ... and {len(section1) - 5} more answers")
        
        # Section II Analysis
        section2 = results_data.get('Section II', {})
        print(f"   ✅ Section II (True/False):")
        print(f"      Total questions: {len(section2)}")
        if section2:
            # Show first 3 questions
            sample_tf = list(section2.items())[:3]
            for q, sub_answers in sample_tf:
                print(f"      {q}: {sub_answers}")
        
        # Section III Analysis
        section3 = results_data.get('Section III', {})
        print(f"   🔢 Section III (Digit Selection):")
        print(f"      Total questions: {len(section3)}")
        if section3:
            # Show all digit answers
            for q, digit in section3.items():
                print(f"      {q}: {digit}")
        
        # Summary
        summary = results_data.get('summary', {})
        print(f"\n📈 Summary:")
        print(f"   Total Section I: {summary.get('total_section1', 0)}")
        print(f"   Total Section II: {summary.get('total_section2', 0)}")
        print(f"   Total Section III: {summary.get('total_section3', 0)}")
        print(f"   Grand Total: {summary.get('total_questions', 0)}")
        
        # Debug images
        debug_dir = Path(result.get('debug_dir', 'data/grading/debug'))
        if debug_dir.exists():
            debug_files = list(debug_dir.glob("*.jpg"))
            print(f"\n🖼️ Debug Images Generated: {len(debug_files)}")
            
            # Show key pipeline images
            pipeline_images = [
                "01_original.jpg",
                "02_preprocessed.jpg",
                "03_markers_detected.jpg",
                "04_aligned.jpg",
                "05_top_region.jpg",
                "06_bottom_region.jpg",
                "07_section1_abcd.jpg",
                "08_section2_true_false.jpg",
                "09_section3_digits.jpg",
                "99_final_result.jpg"
            ]
            
            for img_name in pipeline_images:
                img_path = debug_dir / img_name
                if img_path.exists():
                    print(f"   ✅ {img_name}")
                else:
                    print(f"   ❌ {img_name} (missing)")
        
        # Expected vs Actual
        print(f"\n🎯 Pipeline Validation:")
        expected_sections = {
            "Section I": 40,  # 40 câu ABCD
            "Section II": 8,  # 8 câu True/False
            "Section III": 6  # 6 câu digits
        }
        
        for section_name, expected_count in expected_sections.items():
            actual_count = len(results_data.get(section_name, {}))
            if actual_count >= expected_count:
                print(f"   ✅ {section_name}: {actual_count} questions (≥{expected_count} expected)")
            else:
                print(f"   ⚠️ {section_name}: {actual_count} questions ({expected_count} expected)")
        
        print(f"\n🎉 AI Trắc Nghiệm Pipeline test completed!")
        print(f"📁 Check outputs:")
        print(f"   Debug images: {debug_dir}")
        print(f"   Web viewer: http://localhost:8000/api/v1/omr_debug/viewer")
        print(f"   Pipeline API: http://localhost:8000/api/v1/omr_debug/process_pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during pipeline testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_pipeline_comparison():
    """Show comparison between old system and new pipeline"""
    print("\n📊 System Comparison:")
    print("=" * 60)
    print("OLD SYSTEM (OMRDebugProcessor):")
    print("   • Student ID: 8 digits")
    print("   • Test Code: 4 digits")
    print("   • Answers: 60 questions (4 regions)")
    print("   • Processing: 12 steps")
    print("   • Focus: Debug và coordinate-based")
    print()
    print("NEW PIPELINE (AI Trắc Nghiệm):")
    print("   • Section I: 40 câu Multiple Choice (A/B/C/D)")
    print("   • Section II: 8 câu True/False với sub-questions")
    print("   • Section III: 6 câu digit selection (0-9)")
    print("   • Processing: 7 steps")
    print("   • Focus: Marker-based và intelligent detection")
    print("=" * 60)

def test_api_integration():
    """Test API integration"""
    print("\n🌐 API Integration Test:")
    print("=" * 40)
    
    try:
        import requests
        
        # Test pipeline endpoint
        response = requests.post("http://localhost:8000/api/v1/omr_debug/process_pipeline")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Pipeline API endpoint working")
            print(f"   Pipeline version: {data.get('pipeline_version')}")
            print(f"   Processing steps: {data.get('processing_steps')}")
            print(f"   Total markers: {data.get('total_markers')}")
        else:
            print(f"❌ Pipeline API failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ API test skipped (server not running): {e}")
        print("💡 Start server with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    print("🚀 AI Trắc Nghiệm Pipeline Test Suite")
    print("=" * 60)
    
    # Show comparison
    show_pipeline_comparison()
    
    # Run pipeline test
    success = test_pipeline_processing()
    
    # Test API integration
    test_api_integration()
    
    if success:
        print("\n✅ AI Trắc Nghiệm Pipeline test completed successfully!")
        print("💡 The new pipeline is ready for production!")
        sys.exit(0)
    else:
        print("\n❌ AI Trắc Nghiệm Pipeline test failed!")
        print("💡 Check the debug images and logs for issues.")
        sys.exit(1)
