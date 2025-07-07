"""
Service xử lý xác thực API với ClientID/ClientSecret và token
"""

import secrets
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
import logging

from app.core.config import settings
from app.models.auth_models import (
    ClientCredentials,
    TokenResponse,
    TokenVerificationResponse,
    ClientRegistrationResponse,
    ClientInfo,
    AuthError
)

logger = logging.getLogger(__name__)


class AuthService:
    """Service xử lý xác thực API"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.clients_collection = None
        self.tokens_collection = None
        self._initialized = False
    
    async def initialize(self):
        """Khởi tạo kết nối MongoDB"""
        if self._initialized:
            return
            
        try:
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.db = self.client[settings.MONGODB_DATABASE]
            self.clients_collection = self.db["api_clients"]
            self.tokens_collection = self.db["api_tokens"]
            
            # Tạo index cho hiệu suất
            await self.clients_collection.create_index("client_id", unique=True)
            await self.tokens_collection.create_index("token_hash", unique=True)
            await self.tokens_collection.create_index("expires_at")
            
            self._initialized = True
            logger.info("AuthService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AuthService: {e}")
            raise
    
    def _generate_client_id(self) -> str:
        """Tạo Client ID ngẫu nhiên"""
        return f"client_{secrets.token_urlsafe(16)}"
    
    def _generate_client_secret(self) -> str:
        """Tạo Client Secret ngẫu nhiên"""
        return secrets.token_urlsafe(32)
    
    def _hash_secret(self, secret: str) -> str:
        """Hash client secret để lưu trữ an toàn"""
        return hashlib.sha256(secret.encode()).hexdigest()
    
    def _generate_token(self, client_id: str) -> tuple[str, datetime]:
        """Tạo JWT token cho client"""
        expires_at = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        payload = {
            "client_id": client_id,
            "exp": expires_at,
            "iat": datetime.utcnow(),
            "type": "api_access"
        }
        
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return token, expires_at
    
    def _hash_token(self, token: str) -> str:
        """Hash token để lưu trữ"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def register_client(
        self, 
        client_name: str, 
        description: Optional[str] = None,
        contact_email: Optional[str] = None
    ) -> ClientRegistrationResponse:
        """Đăng ký client mới và tạo credentials"""
        await self.initialize()
        
        try:
            client_id = self._generate_client_id()
            client_secret = self._generate_client_secret()
            client_secret_hash = self._hash_secret(client_secret)
            
            client_data = {
                "client_id": client_id,
                "client_secret_hash": client_secret_hash,
                "client_name": client_name,
                "description": description,
                "contact_email": contact_email,
                "created_at": datetime.utcnow(),
                "is_active": True,
                "last_used": None
            }
            
            await self.clients_collection.insert_one(client_data)
            
            logger.info(f"New client registered: {client_id}")
            
            return ClientRegistrationResponse(
                client_id=client_id,
                client_secret=client_secret,  # Chỉ trả về 1 lần duy nhất
                client_name=client_name,
                created_at=client_data["created_at"],
                message="Client đã được đăng ký thành công. Vui lòng lưu Client Secret vì sẽ không hiển thị lại."
            )
            
        except Exception as e:
            logger.error(f"Failed to register client: {e}")
            raise
    
    async def verify_client_credentials(self, client_id: str, client_secret: str) -> bool:
        """Xác thực client credentials"""
        await self.initialize()
        
        try:
            client = await self.clients_collection.find_one({
                "client_id": client_id,
                "is_active": True
            })
            
            if not client:
                return False
            
            client_secret_hash = self._hash_secret(client_secret)
            return client["client_secret_hash"] == client_secret_hash
            
        except Exception as e:
            logger.error(f"Failed to verify client credentials: {e}")
            return False
    
    async def generate_access_token(self, client_id: str, client_secret: str) -> TokenResponse:
        """Tạo access token cho client"""
        await self.initialize()
        
        # Xác thực credentials
        if not await self.verify_client_credentials(client_id, client_secret):
            raise ValueError("Invalid client credentials")
        
        try:
            # Tạo token
            token, expires_at = self._generate_token(client_id)
            token_hash = self._hash_token(token)
            
            # Lưu token vào database
            token_data = {
                "token_hash": token_hash,
                "client_id": client_id,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at,
                "is_active": True
            }
            
            await self.tokens_collection.insert_one(token_data)
            
            # Cập nhật last_used cho client
            await self.clients_collection.update_one(
                {"client_id": client_id},
                {"$set": {"last_used": datetime.utcnow()}}
            )
            
            logger.info(f"Access token generated for client: {client_id}")
            
            return TokenResponse(
                access_token=token,
                token_type="Bearer",
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                created_at=token_data["created_at"]
            )
            
        except Exception as e:
            logger.error(f"Failed to generate access token: {e}")
            raise
    
    async def verify_token(self, token: str) -> TokenVerificationResponse:
        """Xác thực token"""
        await self.initialize()
        
        try:
            # Decode JWT token
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            client_id = payload.get("client_id")
            
            if not client_id:
                return TokenVerificationResponse(
                    valid=False,
                    message="Invalid token format"
                )
            
            # Kiểm tra token trong database
            token_hash = self._hash_token(token)
            token_record = await self.tokens_collection.find_one({
                "token_hash": token_hash,
                "is_active": True
            })
            
            if not token_record:
                return TokenVerificationResponse(
                    valid=False,
                    message="Token not found or inactive"
                )
            
            # Kiểm tra client còn active không
            client = await self.clients_collection.find_one({
                "client_id": client_id,
                "is_active": True
            })
            
            if not client:
                return TokenVerificationResponse(
                    valid=False,
                    message="Client not found or inactive"
                )
            
            return TokenVerificationResponse(
                valid=True,
                client_id=client_id,
                expires_at=token_record["expires_at"],
                message="Token is valid"
            )
            
        except jwt.ExpiredSignatureError:
            return TokenVerificationResponse(
                valid=False,
                message="Token has expired"
            )
        except jwt.InvalidTokenError:
            return TokenVerificationResponse(
                valid=False,
                message="Invalid token"
            )
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return TokenVerificationResponse(
                valid=False,
                message="Token verification failed"
            )
    
    async def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """Lấy thông tin client"""
        await self.initialize()
        
        try:
            client = await self.clients_collection.find_one({"client_id": client_id})
            
            if not client:
                return None
            
            return ClientInfo(
                client_id=client["client_id"],
                client_name=client["client_name"],
                description=client.get("description"),
                contact_email=client.get("contact_email"),
                created_at=client["created_at"],
                is_active=client["is_active"],
                last_used=client.get("last_used")
            )
            
        except Exception as e:
            logger.error(f"Failed to get client info: {e}")
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """Thu hồi token"""
        await self.initialize()
        
        try:
            token_hash = self._hash_token(token)
            result = await self.tokens_collection.update_one(
                {"token_hash": token_hash},
                {"$set": {"is_active": False}}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    async def cleanup_expired_tokens(self):
        """Dọn dẹp các token đã hết hạn"""
        await self.initialize()
        
        try:
            result = await self.tokens_collection.delete_many({
                "expires_at": {"$lt": datetime.utcnow()}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} expired tokens")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0


# Singleton instance
auth_service = AuthService()
