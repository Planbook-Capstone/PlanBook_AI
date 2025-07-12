"""
LLM Endpoints - Endpoint đơn giản để xử lý text với LLM
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter()


class LLMRequest(BaseModel):
    """Request model cho LLM endpoint"""
    text: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 4096
    custom_prompt: Optional[str] = None


class LLMResponse(BaseModel):
    """Response model cho LLM endpoint"""
    success: bool
    response: str
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


@router.post("/generate", response_model=LLMResponse)
async def generate_llm_response(request: LLMRequest):
    """
    Endpoint LLM đơn giản nhận vào string và trả về response
    
    Args:
        request: LLMRequest chứa text cần xử lý
        
    Returns:
        LLMResponse chứa kết quả từ LLM
    """
    try:
        llm_service = get_llm_service()

        # Kiểm tra LLM service có sẵn không
        if not llm_service.is_available():
            raise HTTPException(
                status_code=503, 
                detail="LLM service không khả dụng. Vui lòng kiểm tra cấu hình API key."
            )
        
        # Tạo prompt - sử dụng custom_prompt nếu có, không thì dùng text trực tiếp
        if request.custom_prompt:
            prompt = f"{request.custom_prompt}\n\nText cần xử lý:\n{request.text}"
        else:
            prompt = request.text
        
        # Gọi LLM service
        logger.info(f"Processing LLM request with {len(request.text)} characters")
        result = await llm_service._generate_content(
            prompt=prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if result["success"]:
            return LLMResponse(
                success=True,
                response=result["text"],
                error=None,
                usage=result.get("usage")
            )
        else:
            return LLMResponse(
                success=False,
                response="",
                error=result["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in LLM generate endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi xử lý LLM: {str(e)}"
        )


