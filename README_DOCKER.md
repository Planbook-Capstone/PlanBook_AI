# 🐳 Docker Setup cho PlanBook AI

## Cách sử dụng

### 1. Chuẩn bị
```bash
# Sao chép file environment
cp .env.docker .env

# Chỉnh sửa file .env và thay đổi GEMINI_API_KEY
# GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Khởi chạy
```bash
# Khởi động tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng services
docker-compose down
```

### 3. Truy cập
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/v1/docs
- **MongoDB**: localhost:27017
- **Redis**: localhost:6379
- **Qdrant**: http://localhost:6333

## Services bao gồm:
- **planbook-api**: FastAPI application
- **mongodb**: Database chính
- **redis**: Cache
- **qdrant**: Vector database

## Lệnh hữu ích:
```bash
# Rebuild khi có thay đổi code
docker-compose up -d --build

# Xem trạng thái
docker-compose ps

# Xóa tất cả (bao gồm data)
docker-compose down -v
```
