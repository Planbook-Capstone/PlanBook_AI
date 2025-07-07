"""
Demo endpoints để minh họa cách sử dụng authentication
"""

from fastapi import APIRouter, Depends
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from app.core.auth_middleware import require_auth, optional_auth, get_auth_info
from app.models.auth_models import TokenVerificationResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/protected-endpoint")
async def protected_endpoint(client_id: str = Depends(require_auth)):
    """
    Endpoint yêu cầu authentication bắt buộc
    
    Chỉ client có token hợp lệ mới có thể truy cập.
    """
    logger.info(f"Protected endpoint accessed by client: {client_id}")
    
    return {
        "message": "Bạn đã truy cập thành công endpoint được bảo vệ!",
        "client_id": client_id,
        "timestamp": datetime.now(),
        "endpoint_type": "protected"
    }


@router.get("/optional-auth-endpoint")
async def optional_auth_endpoint(client_id: Optional[str] = Depends(optional_auth)):
    """
    Endpoint với authentication tùy chọn
    
    Có thể truy cập với hoặc không có token.
    Nội dung trả về sẽ khác nhau tùy theo trạng thái authentication.
    """
    if client_id:
        logger.info(f"Optional auth endpoint accessed by authenticated client: {client_id}")
        return {
            "message": "Chào mừng client đã xác thực!",
            "client_id": client_id,
            "authenticated": True,
            "timestamp": datetime.now(),
            "special_data": "Dữ liệu đặc biệt chỉ dành cho client đã xác thực"
        }
    else:
        logger.info("Optional auth endpoint accessed by anonymous user")
        return {
            "message": "Chào mừng người dùng ẩn danh!",
            "authenticated": False,
            "timestamp": datetime.now(),
            "public_data": "Dữ liệu công khai cho tất cả người dùng"
        }


@router.get("/detailed-auth-info")
async def detailed_auth_info(auth_info: TokenVerificationResponse = Depends(get_auth_info)):
    """
    Endpoint trả về thông tin chi tiết về authentication
    
    Hiển thị đầy đủ thông tin về token và client.
    """
    logger.info(f"Detailed auth info requested by client: {auth_info.client_id}")
    
    return {
        "message": "Thông tin xác thực chi tiết",
        "auth_details": {
            "client_id": auth_info.client_id,
            "token_valid": auth_info.valid,
            "expires_at": auth_info.expires_at,
            "verification_message": auth_info.message
        },
        "timestamp": datetime.now(),
        "endpoint_type": "detailed_auth"
    }


@router.post("/api-tool-demo")
async def api_tool_demo(
    data: Dict[str, Any],
    client_id: str = Depends(require_auth)
):
    """
    Demo endpoint mô phỏng tool API cho bên thứ 3
    
    Đây là ví dụ về cách các bên khác có thể sử dụng API tools của hệ thống.
    """
    logger.info(f"API tool demo called by client: {client_id} with data: {data}")
    
    # Mô phỏng xử lý dữ liệu từ bên thứ 3
    processed_data = {
        "original_data": data,
        "processed_by": "PlanBook AI System",
        "client_id": client_id,
        "processing_timestamp": datetime.now(),
        "status": "success"
    }
    
    # Mô phỏng một số logic xử lý
    if "query" in data:
        processed_data["query_result"] = f"Đã xử lý query: {data['query']}"
    
    if "action" in data:
        processed_data["action_result"] = f"Đã thực hiện action: {data['action']}"
    
    return {
        "message": "Tool API đã xử lý thành công",
        "result": processed_data,
        "usage_info": {
            "client_authenticated": True,
            "api_version": "v1",
            "rate_limit_remaining": "999/1000",  # Mô phỏng rate limiting
            "next_reset": "2024-01-01T00:00:00Z"
        }
    }


@router.get("/health-protected")
async def health_check_protected(client_id: str = Depends(require_auth)):
    """
    Health check endpoint yêu cầu authentication
    
    Cho phép client kiểm tra kết nối và trạng thái authentication.
    """
    logger.info(f"Protected health check by client: {client_id}")
    
    return {
        "status": "healthy",
        "message": "Hệ thống hoạt động bình thường",
        "client_id": client_id,
        "authenticated": True,
        "timestamp": datetime.now(),
        "services": {
            "authentication": "active",
            "database": "connected",
            "api": "operational"
        }
    }


@router.get("/usage-examples")
async def get_usage_examples():
    """
    Endpoint công khai cung cấp hướng dẫn sử dụng authentication
    
    Không yêu cầu authentication để dễ dàng truy cập hướng dẫn.
    """
    return {
        "message": "Hướng dẫn sử dụng Authentication API",
        "workflow": {
            "step_1": {
                "action": "Đăng ký client",
                "endpoint": "POST /auth/register-client",
                "description": "Tạo ClientID và ClientSecret mới"
            },
            "step_2": {
                "action": "Tạo token",
                "endpoint": "POST /auth/token",
                "description": "Sử dụng ClientID/ClientSecret để tạo access token"
            },
            "step_3": {
                "action": "Sử dụng API",
                "endpoint": "Các endpoint khác",
                "description": "Gửi token trong header Authorization: Bearer <token>"
            }
        },
        "example_usage": {
            "register_client": {
                "method": "POST",
                "url": "/auth/register-client",
                "body": {
                    "client_name": "My Application",
                    "description": "Ứng dụng của tôi",
                    "contact_email": "contact@example.com"
                }
            },
            "get_token": {
                "method": "POST",
                "url": "/auth/token",
                "body": {
                    "client_id": "client_xxx",
                    "client_secret": "secret_xxx"
                }
            },
            "use_api": {
                "method": "GET",
                "url": "/demo/protected-endpoint",
                "headers": {
                    "Authorization": "Bearer <your_token_here>"
                }
            }
        },
        "available_demo_endpoints": [
            "GET /demo/protected-endpoint - Endpoint yêu cầu auth",
            "GET /demo/optional-auth-endpoint - Endpoint auth tùy chọn",
            "GET /demo/detailed-auth-info - Thông tin auth chi tiết",
            "POST /demo/api-tool-demo - Demo tool API",
            "GET /demo/health-protected - Health check có auth",
            "GET /demo/usage-examples - Hướng dẫn này (không cần auth)"
        ]
    }
