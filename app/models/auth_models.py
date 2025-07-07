"""
Models cho hệ thống xác thực API
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ClientCredentials(BaseModel):
    """Model cho thông tin client credentials"""
    client_id: str = Field(..., description="Client ID")
    client_secret: str = Field(..., description="Client Secret")


class TokenRequest(BaseModel):
    """Model cho request tạo token"""
    client_id: str = Field(..., description="Client ID")
    client_secret: str = Field(..., description="Client Secret")


class TokenResponse(BaseModel):
    """Model cho response token"""
    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    created_at: datetime = Field(..., description="Token creation time")


class TokenVerificationRequest(BaseModel):
    """Model cho request verify token"""
    token: str = Field(..., description="Token cần verify")


class TokenVerificationResponse(BaseModel):
    """Model cho response verify token"""
    valid: bool = Field(..., description="Token có hợp lệ không")
    client_id: Optional[str] = Field(None, description="Client ID nếu token hợp lệ")
    expires_at: Optional[datetime] = Field(None, description="Thời gian hết hạn token")
    message: Optional[str] = Field(None, description="Thông báo lỗi nếu có")


class ClientRegistrationRequest(BaseModel):
    """Model cho request đăng ký client mới"""
    client_name: str = Field(..., description="Tên client")
    description: Optional[str] = Field(None, description="Mô tả client")
    contact_email: Optional[str] = Field(None, description="Email liên hệ")


class ClientRegistrationResponse(BaseModel):
    """Model cho response đăng ký client"""
    client_id: str = Field(..., description="Client ID được tạo")
    client_secret: str = Field(..., description="Client Secret được tạo")
    client_name: str = Field(..., description="Tên client")
    created_at: datetime = Field(..., description="Thời gian tạo")
    message: str = Field(..., description="Thông báo thành công")


class ClientInfo(BaseModel):
    """Model cho thông tin client"""
    client_id: str = Field(..., description="Client ID")
    client_name: str = Field(..., description="Tên client")
    description: Optional[str] = Field(None, description="Mô tả client")
    contact_email: Optional[str] = Field(None, description="Email liên hệ")
    created_at: datetime = Field(..., description="Thời gian tạo")
    is_active: bool = Field(default=True, description="Trạng thái hoạt động")
    last_used: Optional[datetime] = Field(None, description="Lần sử dụng cuối")


class AuthError(BaseModel):
    """Model cho lỗi xác thực"""
    error: str = Field(..., description="Loại lỗi")
    error_description: str = Field(..., description="Mô tả lỗi")
    timestamp: datetime = Field(default_factory=datetime.now, description="Thời gian lỗi")
