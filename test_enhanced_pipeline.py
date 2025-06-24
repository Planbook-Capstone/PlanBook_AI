#!/usr/bin/env python3
"""
Test script for Enhanced AI Trắc Nghiệm Pipeline with Marker Detection
Kiểm tra pipeline nâng cao với:
- Marker detection cho top region (student code, test code)
- Marker detection cho bottom region (section divisions)
- Enhanced bubble detection với contour analysis
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.omr_pipeline_processor import omr_pipeline_processor

def test_enhanced_pipeline():
    """Test the Enhanced AI Trắc Nghiệm Pipeline with marker detection"""
    
    print("🚀 Enhanced AI Trắc Nghiệm Pipeline Test")
    print("=" * 70)
    print("📋 Enhanced Features:")
    print("   🎯 Marker-based region division")
    print("   🔍 Small marker detection in top/bottom regions")
    print("   📊 Enhanced bubble detection with contour analysis")
    print("   🎨 Comprehensive debug visualization")
    print("   📐 Precise section cutting based on markers")
    print("=" * 70)
    
    # Test with sample.jpg
    sample_path = "data/grading/sample.jpg"
    
    if not os.path.exists(sample_path):
        print(f"❌ Sample file not found: {sample_path}")
        return False
    
    print(f"✅ Testing with: {sample_path}")
    print(f"📏 File size: {os.path.getsize(sample_path) / 1024:.1f} KB")
    
    try:
        # Process with enhanced pipeline
        print("\n🔄 Processing with Enhanced AI Pipeline...")
        result = omr_pipeline_processor.process_omr_sheet(sample_path)
        
        if not result.get('success'):
            print(f"❌ Enhanced pipeline processing failed: {result.get('error')}")
            return False
        
        print(f"✅ Enhanced pipeline processing successful!")
        print(f"   Processing steps: {result.get('processing_steps', 0)}")
        print(f"   Total markers detected: {result.get('total_markers', 0)}")
        
        # Analyze enhanced results
        results_data = result.get('results', {})
        print(f"\n📊 Enhanced Results Analysis:")
        
        # Student ID and Test Code
        student_id = results_data.get('student_id', 'Not detected')
        test_code = results_data.get('test_code', 'Not detected')
        print(f"   👤 Student ID: {student_id} (8 digits expected)")
        print(f"   📝 Test Code: {test_code} (4 digits expected)")
        
        # Section Analysis
        section1 = results_data.get('Section I', {})
        section2 = results_data.get('Section II', {})
        section3 = results_data.get('Section III', {})
        
        print(f"   📝 Section I (Multiple Choice ABCD):")
        print(f"      Total questions: {len(section1)}")
        if section1:
            # Analyze answer distribution
            answer_dist = {}
            for answer in section1.values():
                answer_dist[answer] = answer_dist.get(answer, 0) + 1
            print(f"      Answer distribution: {answer_dist}")
            
            # Show sample answers
            sample_answers = list(section1.items())[:5]
            for q, answer in sample_answers:
                print(f"      {q}: {answer}")
            if len(section1) > 5:
                print(f"      ... and {len(section1) - 5} more answers")
        
        print(f"   ✅ Section II (True/False):")
        print(f"      Total questions: {len(section2)}")
        if section2:
            # Show sample T/F answers
            sample_tf = list(section2.items())[:3]
            for q, sub_answers in sample_tf:
                print(f"      {q}: {sub_answers}")
        
        print(f"   🔢 Section III (Digit Selection):")
        print(f"      Total questions: {len(section3)}")
        if section3:
            # Show all digit answers
            for q, digit in section3.items():
                print(f"      {q}: {digit}")
        
        # Enhanced Debug Analysis
        debug_dir = Path(result.get('debug_dir', 'data/grading/debug'))
        if debug_dir.exists():
            debug_files = list(debug_dir.glob("*.jpg"))
            print(f"\n🖼️ Enhanced Debug Images: {len(debug_files)}")
            
            # Categorize debug images
            categories = {
                'marker_detection': [],
                'region_division': [],
                'section_processing': [],
                'bubble_detection': [],
                'final_results': []
            }
            
            for img_file in debug_files:
                name = img_file.name.lower()
                if 'marker' in name:
                    categories['marker_detection'].append(img_file.name)
                elif 'region' in name or 'top' in name or 'bottom' in name:
                    categories['region_division'].append(img_file.name)
                elif 'section' in name or 'col' in name or 'block' in name:
                    categories['section_processing'].append(img_file.name)
                elif 'detection' in name or 'binary' in name or 'analysis' in name:
                    categories['bubble_detection'].append(img_file.name)
                elif 'final' in name or 'result' in name:
                    categories['final_results'].append(img_file.name)
            
            for category, files in categories.items():
                if files:
                    print(f"   📁 {category.replace('_', ' ').title()}: {len(files)} images")
                    for file in files[:3]:  # Show first 3
                        print(f"      • {file}")
                    if len(files) > 3:
                        print(f"      ... and {len(files) - 3} more")
        
        # Enhanced Validation
        print(f"\n🎯 Enhanced Pipeline Validation:")
        
        # Check Student ID format
        if student_id and student_id != 'Not detected':
            if len(student_id) == 8 and student_id.isdigit():
                print(f"   ✅ Student ID format: Valid 8-digit number")
            else:
                print(f"   ⚠️ Student ID format: {len(student_id)} characters (8 expected)")
        else:
            print(f"   ❌ Student ID: Not detected")
        
        # Check Test Code format
        if test_code and test_code != 'Not detected':
            if len(test_code) == 4 and test_code.isdigit():
                print(f"   ✅ Test Code format: Valid 4-digit number")
            else:
                print(f"   ⚠️ Test Code format: {len(test_code)} characters (4 expected)")
        else:
            print(f"   ❌ Test Code: Not detected")
        
        # Check section completeness
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
        
        # Enhanced Features Summary
        print(f"\n🌟 Enhanced Features Performance:")
        
        # Count marker-based images
        marker_images = len([f for f in debug_files if 'marker' in f.name.lower()])
        region_images = len([f for f in debug_files if any(x in f.name.lower() for x in ['region', 'col', 'block'])])
        detection_images = len([f for f in debug_files if 'detection' in f.name.lower()])
        
        print(f"   🎯 Marker Detection: {marker_images} debug images")
        print(f"   📐 Region Division: {region_images} debug images")
        print(f"   🔍 Enhanced Detection: {detection_images} debug images")
        
        print(f"\n🎉 Enhanced AI Pipeline test completed!")
        print(f"📁 Check enhanced outputs:")
        print(f"   Debug images: {debug_dir}")
        print(f"   Web viewer: http://localhost:8002/api/v1/omr_viewer/viewer")
        print(f"   Enhanced API: http://localhost:8002/api/v1/omr_debug/process_pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during enhanced pipeline testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_enhancement_features():
    """Show what's new in the enhanced pipeline"""
    print("\n🌟 Enhanced Pipeline Features:")
    print("=" * 50)
    print("NEW MARKER-BASED PROCESSING:")
    print("   🎯 Small marker detection in top/bottom regions")
    print("   📐 Precise region cutting based on marker positions")
    print("   🔍 Enhanced bubble detection with contour analysis")
    print("   📊 Combined scoring: pixel ratio + contour detection")
    print()
    print("TOP REGION ENHANCEMENTS:")
    print("   👤 Student ID: 8-digit extraction with marker guidance")
    print("   📝 Test Code: 4-digit extraction with marker guidance")
    print("   🎨 Column-by-column debug visualization")
    print()
    print("BOTTOM REGION ENHANCEMENTS:")
    print("   📝 Section I: 4-column division based on markers")
    print("   ✅ Section II: 8-block division based on markers")
    print("   🔢 Section III: 6-column division based on markers")
    print("   🎯 Fallback to ratio-based division if insufficient markers")
    print("=" * 50)

if __name__ == "__main__":
    print("🚀 Enhanced AI Trắc Nghiệm Pipeline Test Suite")
    print("=" * 70)
    
    # Show enhancement features
    show_enhancement_features()
    
    # Run enhanced pipeline test
    success = test_enhanced_pipeline()
    
    if success:
        print("\n✅ Enhanced AI Pipeline test completed successfully!")
        print("💡 The enhanced marker-based pipeline is ready!")
        sys.exit(0)
    else:
        print("\n❌ Enhanced AI Pipeline test failed!")
        print("💡 Check the debug images for marker detection issues.")
        sys.exit(1)
