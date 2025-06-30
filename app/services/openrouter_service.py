"""
OpenRouter Service để gọi LLM thông qua OpenRouter API
"""
import logging
import json
import requests
from typing import Dict, Any, Optional, List
from app.core.config import settings

logger = logging.getLogger(__name__)

class OpenRouterService:
    """
    Service sử dụng OpenRouter API để gọi các LLM models
    """
    
    def __init__(self):
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
            
            # Make API request
            logger.info(f"Calling OpenRouter API with model: {self.model}")
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                data=json.dumps(data),
                timeout=60
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0]["message"]["content"]
                    
                    return {
                        "success": True,
                        "text": content,
                        "error": None,
                        "usage": response_data.get("usage", {})
                    }
                else:
                    return {
                        "success": False,
                        "error": "No content in OpenRouter response",
                        "text": ""
                    }
            else:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "text": ""
                }
                
        except requests.exceptions.Timeout:
            error_msg = "OpenRouter API request timeout"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "text": ""
            }
        except requests.exceptions.RequestException as e:
            error_msg = f"OpenRouter API request failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "text": ""
            }
        except Exception as e:
            error_msg = f"Unexpected error in OpenRouter service: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "text": ""
            }
    
    def is_available(self) -> bool:
        """Kiểm tra xem service có sẵn không"""
        return self.available
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test kết nối với OpenRouter API
        
        Returns:
            Dict chứa kết quả test
        """
        try:
            test_prompt = "Viết một câu chào đơn giản bằng tiếng Việt."
            result = await self.generate_content(test_prompt, temperature=0.1, max_tokens=100)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "OpenRouter API connection successful",
                    "model": self.model,
                    "response_preview": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                }
            else:
                return {
                    "success": False,
                    "message": "OpenRouter API connection failed",
                    "error": result["error"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": "OpenRouter API test failed",
                "error": str(e)
            }


# Tạo instance global
openrouter_service = OpenRouterService()
