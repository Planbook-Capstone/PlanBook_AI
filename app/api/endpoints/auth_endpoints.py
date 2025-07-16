"""
API endpoints cho hệ thống xác thực
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging

from app.models.auth_models import (
    TokenRequest,
    TokenResponse,
    TokenVerificationRequest,
    TokenVerificationResponse,
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    ClientInfo,
    AuthError
)
from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/register-client", response_model=ClientRegistrationResponse)
async def register_client(request: ClientRegistrationRequest):
    """
    Đăng ký client mới và nhận ClientID/ClientSecret
    
    **Lưu ý quan trọng**: Client Secret chỉ hiển thị 1 lần duy nhất.
    Vui lòng lưu trữ an toàn để sử dụng cho việc tạo token.
    """
    try:
        result = await auth_service.register_client(
            client_name=request.client_name,
            description=request.description,
            contact_email=request.contact_email
        )
        
        logger.info(f"Client registered successfully: {result.client_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to register client: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Không thể đăng ký client: {str(e)}"
        )


@router.post("/token", response_model=TokenResponse)
async def generate_token(request: TokenRequest):
    """
    Tạo access token từ ClientID và ClientSecret
    
    Token này sẽ được sử dụng để xác thực các API calls khác.
    """
    try:
        result = await auth_service.generate_access_token(
            client_id=request.client_id,
            client_secret=request.client_secret
        )
        
        logger.info(f"Token generated for client: {request.client_id}")
        return result
        
    except ValueError as e:
        logger.warning(f"Invalid credentials for client: {request.client_id}")
        raise HTTPException(
            status_code=401,
            detail="Client credentials không hợp lệ"
        )
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tạo token: {str(e)}"
        )


@router.post("/verify-token", response_model=TokenVerificationResponse)
async def verify_token(request: TokenVerificationRequest):
    """
    Xác thực token
    
    Kiểm tra xem token có hợp lệ và còn hiệu lực không.
    """
    try:
        result = await auth_service.verify_token(request.token)
        
        if result.valid:
            logger.info(f"Token verified successfully for client: {result.client_id}")
        else:
            logger.warning(f"Token verification failed: {result.message}")
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to verify token: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Không thể xác thực token: {str(e)}"
        )


@router.get("/client-info", response_model=ClientInfo)
async def get_client_info(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Lấy thông tin client từ token
    
    Yêu cầu Bearer token trong header Authorization.
    """
    try:
        # Verify token trước
        verification = await auth_service.verify_token(credentials.credentials)
        
        if not verification.valid:
            raise HTTPException(
                status_code=401,
                detail=f"Token không hợp lệ: {verification.message}"
            )
        
        # Lấy thông tin client
        client_info = await auth_service.get_client_info(verification.client_id)
        
        if not client_info:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy thông tin client"
            )
        
        logger.info(f"Client info retrieved for: {verification.client_id}")
        return client_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get client info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Không thể lấy thông tin client: {str(e)}"
        )


@router.post("/revoke-token")
async def revoke_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Thu hồi token hiện tại
    
    Token sẽ không thể sử dụng được nữa sau khi thu hồi.
    """
    try:
        # Verify token trước
        verification = await auth_service.verify_token(credentials.credentials)
        
        if not verification.valid:
            raise HTTPException(
                status_code=401,
                detail=f"Token không hợp lệ: {verification.message}"
            )
        
        # Thu hồi token
        success = await auth_service.revoke_token(credentials.credentials)
        
        if success:
            logger.info(f"Token revoked for client: {verification.client_id}")
            return {"message": "Token đã được thu hồi thành công"}
        else:
            raise HTTPException(
                status_code=400,
                detail="Không thể thu hồi token"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke token: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Không thể thu hồi token: {str(e)}"
        )


@router.get("/health")
async def auth_health_check():
    """
    Kiểm tra trạng thái hệ thống xác thực
    """
    try:
        # Test kết nối database
        await auth_service.initialize()
        
        # Cleanup expired tokens
        cleaned_count = await auth_service.cleanup_expired_tokens()
        
        return {
            "status": "healthy",
            "service": "Authentication Service",
            "database": "connected",
            "expired_tokens_cleaned": cleaned_count,
            "endpoints": [
                "POST /register-client - Đăng ký client mới",
                "POST /token - Tạo access token",
                "POST /verify-token - Xác thực token",
                "GET /client-info - Lấy thông tin client",
                "POST /revoke-token - Thu hồi token",
                "GET /health - Kiểm tra trạng thái"
            ]
        }
        
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
