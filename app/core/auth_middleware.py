"""
Middleware và dependencies cho xác thực API
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging

from app.services.auth_service import get_auth_service
from app.models.auth_models import TokenVerificationResponse

logger = logging.getLogger(__name__)
security = HTTPBearer()


class AuthMiddleware:
    """Middleware xử lý xác thực API"""
    
    @staticmethod
    async def verify_api_token(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> str:
        """
        Dependency để xác thực token và trả về client_id
        
        Sử dụng trong các endpoint cần xác thực:
        @router.get("/protected-endpoint")
        async def protected_endpoint(client_id: str = Depends(AuthMiddleware.verify_api_token)):
            # Logic xử lý với client_id đã được xác thực
            pass
        """
        try:
            auth_service = get_auth_service()
            await auth_service.initialize()
            verification = await auth_service.verify_token(credentials.credentials)
            
            if not verification.valid:
                logger.warning(f"Token verification failed: {verification.message}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Token không hợp lệ: {verification.message}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            logger.debug(f"Token verified successfully for client: {verification.client_id}")
            return verification.client_id
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Lỗi xác thực token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    async def verify_api_token_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[str]:
        """
        Dependency để xác thực token tùy chọn
        
        Trả về client_id nếu token hợp lệ, None nếu không có token hoặc token không hợp lệ.
        Không raise exception, phù hợp cho các endpoint có thể hoạt động với hoặc không có xác thực.
        """
        if not credentials:
            return None
            
        try:
            verification = await auth_service.verify_token(credentials.credentials)
            
            if verification.valid:
                logger.debug(f"Optional token verified for client: {verification.client_id}")
                return verification.client_id
            else:
                logger.debug(f"Optional token verification failed: {verification.message}")
                return None
                
        except Exception as e:
            logger.debug(f"Optional token verification error: {e}")
            return None
    
    @staticmethod
    async def get_token_info(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> TokenVerificationResponse:
        """
        Dependency để lấy thông tin chi tiết về token
        
        Trả về đầy đủ thông tin verification thay vì chỉ client_id
        """
        try:
            verification = await auth_service.verify_token(credentials.credentials)
            
            if not verification.valid:
                logger.warning(f"Token verification failed: {verification.message}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Token không hợp lệ: {verification.message}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            logger.debug(f"Token info retrieved for client: {verification.client_id}")
            return verification
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token info retrieval error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Lỗi lấy thông tin token",
                headers={"WWW-Authenticate": "Bearer"},
            )


# Convenience functions để sử dụng trực tiếp
async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Function tiện lợi để require authentication
    Trả về client_id
    """
    return await AuthMiddleware.verify_api_token(credentials)


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[str]:
    """
    Function tiện lợi cho optional authentication
    Trả về client_id hoặc None
    """
    return await AuthMiddleware.verify_api_token_optional(credentials)


async def get_auth_info(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenVerificationResponse:
    """
    Function tiện lợi để lấy thông tin authentication đầy đủ
    """
    return await AuthMiddleware.get_token_info(credentials)


# Decorator để bảo vệ endpoint (sử dụng cho các function thông thường)
def require_api_auth(func):
    """
    Decorator để bảo vệ endpoint với authentication
    
    Sử dụng:
    @require_api_auth
    async def my_endpoint(client_id: str, other_params...):
        # client_id sẽ được inject tự động
        pass
    """
    async def wrapper(*args, **kwargs):
        # Lấy credentials từ request context (cần implement thêm nếu cần)
        # Hiện tại khuyến khích sử dụng Depends() thay vì decorator
        pass
    return wrapper
