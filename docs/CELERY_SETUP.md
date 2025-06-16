# PlanBookAI Celery Configuration Guide

## Tổng quan

PlanBookAI sử dụng Celery để xử lý các tác vụ nặng trong background như:
- Phân tích PDF sách giáo khoa
- Tạo embeddings cho RAG search
- Xử lý CV/Resume
- Các tác vụ AI khác

## Kiến trúc

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│   Redis Broker  │───▶│ Celery Workers  │
│                 │    │                 │    │                 │
│ - API Endpoints │    │ - Task Queue    │    │ - PDF Queue     │
│ - Task Creation │    │ - Result Store  │    │ - Embeddings Q  │
│ - Status Check  │    │                 │    │ - CV Queue      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         │              ┌─────────────────┐             │
         └─────────────▶│   MongoDB       │◀────────────┘
                        │                 │
                        │ - Task Status   │
                        │ - Task Results  │
                        │ - Metadata      │
                        └─────────────────┘
```

## Cài đặt và Chạy

### 1. Cài đặt Dependencies

```bash
# Redis
sudo apt-get install redis-server
# hoặc
brew install redis

# MongoDB
sudo apt-get install mongodb
# hoặc
brew install mongodb-community

# Python packages (đã có trong requirements.txt)
pip install celery redis pymongo
```

### 2. Chạy Services

#### Option 1: Automatic (Recommended)
```bash
# Start tất cả services
python scripts/start_planbook_celery.py --mode local

# Chỉ start workers (nếu đã có Redis/MongoDB)
python scripts/start_planbook_celery.py --mode worker-only

# Start với Docker
python scripts/start_planbook_celery.py --mode docker
```

#### Option 2: Manual
```bash
# 1. Start Redis
redis-server

# 2. Start MongoDB
mongod

# 3. Start Celery Workers
python scripts/start_celery_worker.py --queue pdf_queue --concurrency 2
python scripts/start_celery_worker.py --queue embeddings_queue --concurrency 1
python scripts/start_celery_worker.py --queue cv_queue --concurrency 1

# 4. Start FastAPI
fastapi dev app/main.py
```

### 3. Docker Compose
```bash
# Start tất cả với Docker
docker-compose -f docker-compose.celery.yml up --build

# Stop
docker-compose -f docker-compose.celery.yml down
```

## Quản lý và Monitoring

### 1. Health Check
```bash
# Check Celery health
curl http://localhost:8000/api/v1/celery/health

# Worker stats
curl http://localhost:8000/api/v1/celery/workers

# Queue info
curl http://localhost:8000/api/v1/celery/queues
```

### 2. Management Commands
```bash
# Check worker status
python scripts/celery_management.py status

# Inspect active tasks
python scripts/celery_management.py inspect active

# Test Celery
python scripts/celery_management.py test

# Monitor real-time
python scripts/celery_management.py monitor

# Purge pending tasks (NGUY HIỂM!)
python scripts/celery_management.py purge
```

### 3. Flower Monitoring (Optional)
```bash
# Install flower
pip install flower

# Start flower
celery -A app.core.celery_app:celery_app flower

# Access: http://localhost:5555
```

## Configuration

### Queues và Routing

- **pdf_queue**: Xử lý PDF (2 workers)
- **embeddings_queue**: Tạo embeddings (1 worker)
- **cv_queue**: Xử lý CV (1 worker)
- **default**: Tasks khác

### Task Types

- `quick_analysis`: Phân tích nhanh PDF
- `process_textbook`: Xử lý sách giáo khoa với metadata
- `process_textbook_auto`: Tự động phân tích metadata
- `process_cv`: Xử lý CV/Resume
- `create_embeddings`: Tạo embeddings
- `update_embeddings`: Cập nhật embeddings

## API Usage

### 1. Tạo Task
```python
# Quick analysis
files = {"file": ("textbook.pdf", pdf_content, "application/pdf")}
response = requests.post(
    "http://localhost:8000/api/v1/pdf/quick-textbook-analysis",
    files=files,
    data={"create_embeddings": "true"}
)
task_id = response.json()["task_id"]
```

### 2. Check Status
```python
response = requests.get(f"http://localhost:8000/api/v1/tasks/{task_id}/status")
status = response.json()
```

### 3. Get Result
```python
response = requests.get(f"http://localhost:8000/api/v1/tasks/{task_id}/result")
result = response.json()
```

## Troubleshooting

### 1. Worker không start
```bash
# Check Redis connection
redis-cli ping

# Check MongoDB connection
mongo --eval "db.adminCommand('ping')"

# Check Python path
python -c "from app.core.celery_app import celery_app; print('OK')"
```

### 2. Tasks bị stuck
```bash
# Check active tasks
python scripts/celery_management.py inspect active

# Purge pending tasks
python scripts/celery_management.py purge
```

### 3. Memory issues
```bash
# Restart workers
docker-compose -f docker-compose.celery.yml restart celery_worker_pdf

# Check memory usage
docker stats
```

## Best Practices

1. **Monitoring**: Luôn monitor worker health và task status
2. **Scaling**: Tăng số workers dựa trên load
3. **Error Handling**: Tasks có retry logic tự động
4. **Cleanup**: Định kỳ purge completed tasks
5. **Logging**: Check logs để debug issues

## Environment Variables

```bash
# .env file
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
MONGODB_URL=mongodb://localhost:27017/planbook_db
```

## Performance Tuning

- **PDF Queue**: 2 workers (CPU intensive)
- **Embeddings Queue**: 1 worker (Memory intensive)
- **CV Queue**: 1 worker (Balanced)
- **Prefork Pool**: Tốt cho CPU-bound tasks
- **Task Time Limits**: 30-40 phút cho PDF lớn
