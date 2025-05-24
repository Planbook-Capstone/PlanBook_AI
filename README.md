# PlanBook AI Service

Microservice AI cho dự án PlanBook, cung cấp các chức năng AI như tạo kế hoạch bài giảng, tạo slide tự động, và tạo đề thi.

## Tính năng

- **Tạo kế hoạch bài giảng**: Tạo kế hoạch bài giảng chi tiết dựa trên môn học, cấp độ, và chủ đề.
- **Tạo slide tự động**: Chuyển đổi nội dung thành bài thuyết trình slide.
- **Tạo đề thi**: Tạo đề thi với nhiều loại câu hỏi và mức độ khó khác nhau.

## Yêu cầu

- Python 3.9+
- FastAPI
- Docker (tùy chọn)

## Cài đặt

### Sử dụng Python

1. Clone repository:

```bash
git clone <repository-url>
cd PlanBook_AI
```

2. Tạo và kích hoạt môi trường ảo:

```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

3. Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

4. Cấu hình biến môi trường:

   - Tạo file `.env` từ `.env.example`
   - Thêm API key Gemini của bạn

5. Chạy ứng dụng:

```bash
python main.py
```

### Sử dụng Docker

1. Build và chạy container:

```bash
docker-compose up -d
```

## API Endpoints

### Tạo kế hoạch bài giảng

```
POST /api/v1/ai/generate-lesson-plan
```

### Tạo slide

```
POST /api/v1/ai/create-slides
```

### Tạo đề thi

```
POST /api/v1/ai/generate-test
```

## Tài liệu API

Truy cập tài liệu API tại:

```
http://localhost:8000/api/v1/docs
```

## Phát triển

### Cấu trúc dự án

```
PlanBook_AI/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   └── ai_endpoints.py
│   │   └── api.py
│   ├── core/
│   │   └── config.py
│   ├── models/
│   ├── schemas/
│   │   └── ai_schemas.py
│   ├── services/
│   │   └── ai_service.py
│   └── utils/
├── tests/
├── .env
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── main.py
├── README.md
└── requirements.txt
```
