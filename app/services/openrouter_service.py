"""
OpenRouter Service để gọi LLM thông qua OpenRouter API
"""
import logging
import json
import requests
import threading
import time
from typing import Dict, Any, Optional, List
from app.core.config import settings

logger = logging.getLogger(__name__)

class OpenRouterService:
    """
    Service sử dụng OpenRouter API để gọi các LLM models
    """

    def __init__(self):
        """Initialize OpenRouter service"""
        # Chỉ set flag, không khởi tạo service ngay
        self.api_key = None
        self.base_url = None
        self.model = None
        self.site_url = None
        self.site_name = None
        self.available = False
        self._service_initialized = False

    def _ensure_service_initialized(self):
        """Ensure OpenRouter service is initialized"""
        if not self._service_initialized:
            logger.info("🔄 OpenRouterService: First-time initialization triggered")
            self._init_service()
            self._service_initialized = True
            logger.info("✅ OpenRouterService: Initialization completed")

    def _init_service(self):
        """Initialize OpenRouter service"""
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL
        self.model = settings.OPENROUTER_MODEL
        self.site_url = settings.OPENROUTER_SITE_URL
        self.site_name = settings.OPENROUTER_SITE_NAME
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Kiểm tra xem OpenRouter service có sẵn không"""
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set. OpenRouter features will be disabled.")
            return False
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """Tạo headers cho OpenRouter API request"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        
        if self.site_name:
            headers["X-Title"] = self.site_name
            
        return headers
    
    async def generate_content(self, prompt: str, temperature: float = 0.1, max_tokens: int = 4096) -> Dict[str, Any]:
        """
        Gọi OpenRouter API để generate content

        Args:
            prompt: Text prompt để gửi tới model
            temperature: Temperature cho response (0.0 - 1.0)
            max_tokens: Số token tối đa cho response

        Returns:
            Dict chứa response từ OpenRouter API
        """
        self._ensure_service_initialized()
        start_time = time.time()

        try:
            if not self.available:
                return {
                    "success": False,
                    "error": "OpenRouter service not available. Please set OPENROUTER_API_KEY.",
                    "text": ""
                }

            # Prepare request data
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Log request details
            logger.info("=" * 80)
            logger.info("🚀 OPENROUTER LLM REQUEST")
            logger.info("=" * 80)
            logger.info(f"📍 URL: {self.base_url}/chat/completions")
            logger.info(f"🤖 Model: {self.model}")
            logger.info(f"🌡️ Temperature: {temperature}")
            logger.info(f"📏 Max Tokens: {max_tokens}")
            logger.info(f"📝 Prompt Length: {len(prompt)} characters")
            logger.info("📋 Request Headers:")
            headers = self._get_headers()
            for key, value in headers.items():
                if key == "Authorization":
                    logger.info(f"   {key}: Bearer ***{value[-10:]}")  # Chỉ hiện 10 ký tự cuối của API key
                else:
                    logger.info(f"   {key}: {value}")

            logger.info("📄 Request Payload:")
            logger.info(f"   Prompt (truncated): {prompt}...")

            # Make API request
            logger.info("⏳ Sending request to OpenRouter API...")

            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=60
            )
            
            # Calculate response time
            end_time = time.time()
            response_time = end_time - start_time

            # Log response details
            logger.info("=" * 80)
            logger.info("📥 OPENROUTER LLM RESPONSE")
            logger.info("=" * 80)
            logger.info(f"⏱️ Response Time: {response_time:.2f} seconds")
            logger.info(f"📊 Status Code: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()

                # Log full response data
                logger.info("📄 Full Response Data:")
                logger.info(json.dumps(response_data, ensure_ascii=False, indent=2))

                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0]["message"]["content"]

                    # Log extracted content
                    logger.info("✅ SUCCESS - Content extracted:")
                    if len(content) > 1000:
                        logger.info(f"📝 Content (truncated): {content[:1000]}...")
                    else:
                        logger.info(f"📝 Content: {content}")

                    # Log usage statistics if available
                    if "usage" in response_data:
                        usage = response_data["usage"]
                        logger.info("📈 Token Usage:")
                        logger.info(f"   Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}")
                        logger.info(f"   Completion Tokens: {usage.get('completion_tokens', 'N/A')}")
                        logger.info(f"   Total Tokens: {usage.get('total_tokens', 'N/A')}")

                    logger.info("=" * 80)

                    return {
                        "success": True,
                        "text": content,
                        "error": None,
                        "usage": response_data.get("usage", {}),
                        "response_time": response_time
                    }
                else:
                    error_msg = "No content in OpenRouter response"
                    logger.error(f"❌ ERROR: {error_msg}")
                    logger.error(f"📄 Response Data: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                    logger.info("=" * 80)
                    return {
                        "success": False,
                        "error": error_msg,
                        "text": "",
                        "response_time": response_time
                    }
            else:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                logger.error(f"❌ HTTP ERROR: {error_msg}")
                logger.error(f"📄 Error Response: {response.text}")
                logger.info("=" * 80)
                return {
                    "success": False,
                    "error": error_msg,
                    "text": "",
                    "response_time": response_time
                }
                
        except requests.exceptions.Timeout:
            end_time = time.time()
            response_time = end_time - start_time
            error_msg = "OpenRouter API request timeout"
            logger.error("=" * 80)
            logger.error("⏰ TIMEOUT ERROR")
            logger.error("=" * 80)
            logger.error(f"❌ Error: {error_msg}")
            logger.error(f"⏱️ Time elapsed: {response_time:.2f} seconds")
            logger.error("=" * 80)
            return {
                "success": False,
                "error": error_msg,
                "text": "",
                "response_time": response_time
            }
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            response_time = end_time - start_time
            error_msg = f"OpenRouter API request failed: {str(e)}"
            logger.error("=" * 80)
            logger.error("🌐 REQUEST ERROR")
            logger.error("=" * 80)
            logger.error(f"❌ Error: {error_msg}")
            logger.error(f"⏱️ Time elapsed: {response_time:.2f} seconds")
            logger.error(f"🔍 Exception type: {type(e).__name__}")
            logger.error("=" * 80)
            return {
                "success": False,
                "error": error_msg,
                "text": "",
                "response_time": response_time
            }
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            error_msg = f"Unexpected error in OpenRouter service: {str(e)}"
            logger.error("=" * 80)
            logger.error("💥 UNEXPECTED ERROR")
            logger.error("=" * 80)
            logger.error(f"❌ Error: {error_msg}")
            logger.error(f"⏱️ Time elapsed: {response_time:.2f} seconds")
            logger.error(f"🔍 Exception type: {type(e).__name__}")
            logger.error("=" * 80)
            return {
                "success": False,
                "error": error_msg,
                "text": "",
                "response_time": response_time
            }
    
    def is_available(self) -> bool:
        """Kiểm tra xem service có sẵn không"""
        self._ensure_service_initialized()
        return self.available
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test kết nối với OpenRouter API

        Returns:
            Dict chứa kết quả test
        """
        try:
            logger.info("🧪 Testing OpenRouter API connection...")
            test_prompt = "Viết một câu chào đơn giản bằng tiếng Việt."
            result = await self.generate_content(test_prompt, temperature=0.1, max_tokens=100)

            if result["success"]:
                logger.info("✅ OpenRouter API connection test successful!")
                return {
                    "success": True,
                    "message": "OpenRouter API connection successful",
                    "model": self.model,
                    "response_preview": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"],
                    "response_time": result.get("response_time", 0)
                }
            else:
                logger.error("❌ OpenRouter API connection test failed!")
                return {
                    "success": False,
                    "message": "OpenRouter API connection failed",
                    "error": result["error"]
                }

        except Exception as e:
            logger.error(f"💥 OpenRouter API test exception: {str(e)}")
            return {
                "success": False,
                "message": "OpenRouter API test failed",
                "error": str(e)
            }


# Factory function để tạo OpenRouterService instance
def get_openrouter_service() -> OpenRouterService:
    """
    Tạo OpenRouterService instance mới

    Returns:
        OpenRouterService: Fresh instance
    """
    return OpenRouterService()
