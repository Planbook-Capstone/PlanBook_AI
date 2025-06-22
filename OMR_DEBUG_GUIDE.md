# 🔍 Hướng dẫn sử dụng OMR Debug System

## Tổng quan

Hệ thống OMR Debug được thiết kế để xử lý và chấm điểm tự động phiếu trắc nghiệm Việt Nam với debug chi tiết từng bước. Hệ thống tạo ra 20 debug images để bạn có thể kiểm tra và điều chỉnh từng bước xử lý.

## 🚀 Khởi động hệ thống

```bash
# Khởi động server
fastapi dev app/main.py

# Server sẽ chạy tại: http://127.0.0.1:8000
```

## 📱 Giao diện Web

Truy cập: **http://127.0.0.1:8000/api/v1/omr_debug/viewer**

### Các chức năng chính:

1. **🚀 Xử lý ảnh test**: Xử lý ảnh mẫu `data/grading/test_images/1.jpeg`
2. **🔄 Làm mới**: Tải lại trang để xem debug images mới
3. **🗑️ Xóa debug images**: Xóa tất cả debug images hiện tại

## 🔧 API Endpoints

### 1. Xử lý ảnh test
```http
POST /api/v1/omr_debug/process_test_image
```

**Response:**
```json
{
  "success": true,
  "student_id": "00000000",
  "test_code": "000", 
  "total_answers": 60,
  "answers": {"1": "A", "2": "B", ...},
  "debug_files": ["01_original.jpg", ...],
  "message": "Processed successfully. 20 debug images created."
}
```

### 2. Lấy danh sách debug images
```http
GET /api/v1/omr_debug/debug_images
```

### 3. Xem debug image
```http
GET /api/v1/omr_debug/debug_image/{filename}
```

### 4. Xóa debug images
```http
DELETE /api/v1/omr_debug/clear_debug
```

### 5. Thông tin các bước xử lý
```http
GET /api/v1/omr_debug/processing_steps
```

## 📊 Debug Images được tạo

| Tên file | Mô tả |
|----------|-------|
| `01_original.jpg` | Ảnh gốc đầu vào |
| `02_preprocessed.jpg` | Ảnh sau tiền xử lý (grayscale, denoised, enhanced, binary) |
| `03_corners_detected.jpg` | Phát hiện 4 góc markers (hình vuông đen) |
| `04_aligned.jpg` | Ảnh đã căn chỉnh bằng perspective transform |
| `05_region_student_id.jpg` | Vùng Student ID (8 cột số) |
| `06_region_test_code.jpg` | Vùng Test Code (3 cột số) |
| `07_region_answers_01_15.jpg` | Câu 01-15 |
| `08_region_answers_16_30.jpg` | Câu 16-30 |
| `09_region_answers_31_45.jpg` | Câu 31-45 |
| `10_region_answers_46_60.jpg` | Câu 46-60 |
| `11_region_answers_full.jpg` | Tất cả câu trả lời |
| `12_student_id_binary.jpg` | Student ID binary |
| `13_test_code_binary.jpg` | Test Code binary |
| `14_student_id_grid.jpg` | Grid Student ID với bubbles được đánh dấu |
| `14_test_code_grid.jpg` | Grid Test Code với bubbles được đánh dấu |
| `15_answers_01_15_answers.jpg` | Phát hiện câu trả lời 01-15 |
| `15_answers_16_30_answers.jpg` | Phát hiện câu trả lời 16-30 |
| `15_answers_31_45_answers.jpg` | Phát hiện câu trả lời 31-45 |
| `15_answers_46_60_answers.jpg` | Phát hiện câu trả lời 46-60 |
| `99_final_result.jpg` | Kết quả cuối cùng với thông tin tổng hợp |

## 🎯 Layout phiếu trắc nghiệm

### Vị trí các vùng (sau khi align về 1086x1536):

1. **Student ID**: x:630-810, y:180-600 (8 cột số)
2. **Test Code**: x:830-920, y:180-380 (3 cột số)  
3. **Answers 01-15**: x:50-300, y:700-1100
4. **Answers 16-30**: x:320-570, y:700-1100
5. **Answers 31-45**: x:590-840, y:700-1100
6. **Answers 46-60**: x:860-1036, y:700-1100

### Quy trình xử lý:

1. **Tiền xử lý**: Grayscale → Denoised → Enhanced → Binary
2. **Phát hiện markers**: Tìm 4 góc hình vuông đen
3. **Căn chỉnh**: Perspective transform về kích thước chuẩn
4. **Trích xuất vùng**: Cắt các vùng ROI theo tọa độ cố định
5. **Phát hiện bubbles**: Phân tích grid và tính tỷ lệ tô (>40%)
6. **Kết quả**: Trả về Student ID, Test Code và 60 câu trả lời

## 🛠️ Điều chỉnh tham số

### Trong file `app/services/omr_debug_processor.py`:

```python
# Ngưỡng phát hiện bubble đã tô
filled_ratio > 0.4  # 40% pixel trắng

# Kích thước marker
500 < area < 5000  # Diện tích marker

# Tỷ lệ khung hình marker  
0.7 <= aspect_ratio <= 1.3  # Gần vuông
```

## 📝 Test Script

```bash
# Test bằng script Python
python test_omr_debug.py

# Test API
python test_omr_api.py
```

## 🔍 Kiểm tra kết quả

1. **Xem debug images**: Mở folder `data/grading/debug/`
2. **Kiểm tra alignment**: Xem `04_aligned.jpg` 
3. **Kiểm tra ROI**: Xem các file `05-11_region_*.jpg`
4. **Kiểm tra detection**: Xem các file `14-15_*.jpg`

## ⚠️ Lưu ý

- Ảnh test phải có 4 góc markers (hình vuông đen)
- Layout phiếu phải tuân theo chuẩn Việt Nam
- Chất lượng ảnh ảnh hưởng đến độ chính xác
- Debug images sẽ bị ghi đè mỗi lần xử lý mới

## 🎯 Kết quả mong đợi

- **Student ID**: 8 chữ số
- **Test Code**: 3 chữ số  
- **Answers**: 60 câu (A, B, C, D)
- **Debug images**: 20 files
- **Thời gian xử lý**: < 5 giây

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra log server
2. Xem debug images để phát hiện lỗi
3. Điều chỉnh tọa độ ROI nếu cần
4. Kiểm tra chất lượng ảnh đầu vào
