"""
RAG Endpoints
Endpoints cho RAG (Retrieval-Augmented Generation) queries
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.rag_service import get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter()


class RAGQueryRequest(BaseModel):
    """Request model cho RAG query"""
    query: str
    book_id: Optional[str] = None
    lesson_id: Optional[str] = None
    limit: int = 5
    semantic_tags: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000


class RAGQueryResponse(BaseModel):
    """Response model cho RAG query"""
    success: bool
    query: str
    answer: str
    sources: list
    context_used: str
    book_id: Optional[str] = None
    lesson_id: Optional[str] = None
    total_sources: int
    processing_time: Optional[float] = None
    error: Optional[str] = None


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Xử lý RAG query với semantic search
    
    Args:
        request: RAG query request với các tham số
        
    Returns:
        RAGQueryResponse: Kết quả RAG với answer và sources
    """
    try:
        logger.info(f"🔍 RAG Query: {request.query[:100]}...")
        
        # Lấy RAG service
        rag_service = get_rag_service()
        
        # Xử lý RAG query
        result = await rag_service.process_rag_query(
            query=request.query,
            book_id=request.book_id,
            lesson_id=request.lesson_id,
            limit=request.limit,
            semantic_tags=request.semantic_tags,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"RAG processing failed: {result.get('error', 'Unknown error')}"
            )
        
        return RAGQueryResponse(
            success=True,
            query=request.query,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            context_used=result.get("context_used", ""),
            book_id=request.book_id,
            lesson_id=request.lesson_id,
            total_sources=len(result.get("sources", [])),
            processing_time=result.get("processing_time")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in RAG query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/query", response_model=RAGQueryResponse)
async def rag_query_get(
    query: str = Query(..., description="Câu hỏi cần trả lời"),
    book_id: Optional[str] = Query(None, description="ID sách cụ thể"),
    lesson_id: Optional[str] = Query(None, description="ID bài học cụ thể"),
    limit: int = Query(5, description="Số lượng sources tối đa", ge=1, le=20),
    semantic_tags: Optional[str] = Query(None, description="Lọc theo semantic tags (phân cách bằng dấu phẩy)"),
    temperature: float = Query(0.3, description="Temperature cho LLM", ge=0.0, le=1.0),
    max_tokens: int = Query(2000, description="Số token tối đa cho response", ge=100, le=4000)
):
    """
    RAG query endpoint (GET method)
    
    Tìm kiếm semantic và tạo câu trả lời dựa trên nội dung sách giáo khoa
    """
    request = RAGQueryRequest(
        query=query,
        book_id=book_id,
        lesson_id=lesson_id,
        limit=limit,
        semantic_tags=semantic_tags,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return await rag_query(request)
