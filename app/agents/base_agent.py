from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_random_exponential

import google.generativeai as genai
from app.core.config import settings

class BaseAgent(ABC):
    """
    Base class cho tất cả AI Agents trong PlanBookAI system.
    
    Chứa các chức năng chung:
    - Gemini API integration
    - Logging và error handling
    - Memory management
    - Retry logic
    - Response formatting
    """
    
    def __init__(self, agent_name: str, model_name: str = "gemini-1.5-flash"):
        self.agent_name = agent_name
        self.model_name = model_name
        self.logger = self._setup_logger()
        
        # Initialize Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)
        
        # Agent memory để lưu context
        self.memory: List[Dict[str, Any]] = []
        self.max_memory_size = 10
        
        self.logger.info(f"Initialized {agent_name} with model {model_name}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger cho agent"""
        logger = logging.getLogger(f"agent.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def _call_gemini_api(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Gọi Gemini API với retry logic và error handling
        """
        try:
            if not settings.GEMINI_API_KEY:
                raise ValueError("Gemini API key not configured")
            
            # Tạo system prompt cho agent
            system_prompt = self._get_system_prompt()
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            self.logger.info(f"Calling Gemini API - Temperature: {temperature}")
            self.logger.debug(f"Prompt length: {len(full_prompt)} characters")
            
            # Cấu hình generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature
            )
            if max_tokens:
                generation_config.max_output_tokens = max_tokens
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            if not hasattr(response, 'text') or not response.text:
                raise ValueError("Empty response from Gemini API")
            
            # Lưu vào memory
            self._add_to_memory("api_call", {
                "prompt_length": len(full_prompt),
                "response_length": len(response.text),
                "temperature": temperature,
                "timestamp": datetime.now().isoformat()
            })
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {str(e)}")
            raise ValueError(f"Failed to call Gemini API: {str(e)}") from e
    
    def _add_to_memory(self, event_type: str, data: Dict[str, Any]):
        """Thêm event vào memory của agent"""
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        self.memory.append(memory_entry)
        
        # Giới hạn memory size
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    def _get_memory_context(self, event_types: Optional[List[str]] = None) -> str:
        """Lấy context từ memory để đưa vào prompt"""
        if not self.memory:
            return ""
        
        relevant_memories = self.memory
        if event_types:
            relevant_memories = [
                m for m in self.memory 
                if m["event_type"] in event_types
            ]
        
        if not relevant_memories:
            return ""
        
        context = "Previous context:\n"
        for memory in relevant_memories[-3:]:  # Lấy 3 memories gần nhất
            context += f"- {memory['event_type']}: {memory['data']}\n"
        
        return context
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response với fallback handling"""
        try:
            # Thử parse trực tiếp
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Thử tìm JSON block trong response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # Fallback: return as text content
            self.logger.warning("Could not parse JSON response, returning as text")
            return {"content": response, "format": "text"}
    
    def _validate_input(self, **kwargs) -> bool:
        """Validate input parameters - override trong subclass"""
        return True
    
    def _format_response(self, result: Any, message: str = "Success") -> Dict[str, Any]:
        """Format response theo chuẩn API"""
        return {
            "agent": self.agent_name,
            "result": result,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_name
        }
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Mỗi agent phải implement system prompt riêng
        """
        pass
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Main processing method - mỗi agent phải implement
        """
        pass
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Thông tin về agent"""
        return {
            "name": self.agent_name,
            "model": self.model_name,
            "memory_size": len(self.memory),
            "max_memory_size": self.max_memory_size,
            "status": "active"
        }
    
    def clear_memory(self):
        """Xóa memory của agent"""
        self.memory.clear()
        self.logger.info("Agent memory cleared")
