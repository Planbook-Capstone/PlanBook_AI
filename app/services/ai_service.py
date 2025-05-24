from typing import List, Dict, Any, Optional
import os
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential

import google.generativeai as genai
from app.core.config import settings

class AIService:
    def __init__(self):
        api_key = settings.GEMINI_API_KEY
        print(f"Initializing AIService with API key: {api_key[:5]}...{api_key[-5:]}")
        genai.configure(api_key=api_key)

        # Sử dụng mô hình gemini-1.5-flash thay vì gemini-pro
        # gemini-1.5-flash có giới hạn quota cao hơn và ít tốn kém hơn
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def _call_gemini_api(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Call Gemini API with retry logic
        """
        try:
            # Kiểm tra API key
            if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your_gemini_api_key_here":
                raise ValueError("Gemini API key is not configured. Please update your .env file with a valid API key.")

            system_prompt = "You are an educational AI assistant helping teachers create high-quality educational content."
            full_prompt = f"{system_prompt}\n\n{prompt}"

            # Thêm thông tin debug
            print(f"Calling Gemini API with temperature: {temperature}")
            print(f"Prompt length: {len(full_prompt)} characters")

            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )

            if not hasattr(response, 'text') or not response.text:
                raise ValueError("Empty response received from Gemini API")

            return response.text
        except Exception as e:
            error_message = f"Error calling Gemini API: {str(e)}"
            print(error_message)
            # Thêm thông tin chi tiết về lỗi
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise ValueError(error_message) from e

    def generate_lesson_plan(
        self,
        subject: str,
        grade_level: str,
        topic: str,
        duration: int,
        objectives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a lesson plan based on the provided parameters
        """
        objectives_text = ""
        if objectives:
            objectives_text = "Learning objectives:\n" + "\n".join([f"- {obj}" for obj in objectives])

        prompt = f"""
        Create a detailed lesson plan for a {duration}-minute {subject} class for {grade_level} students on the topic of "{topic}".

        {objectives_text}

        The lesson plan should include:
        1. Learning objectives
        2. Required materials
        3. Introduction/warm-up activity (5-10 minutes)
        4. Main content and activities
        5. Assessment methods
        6. Conclusion/wrap-up
        7. Homework or extension activities

        Format the response as a JSON object with these sections as keys.
        """

        response = self._call_gemini_api(prompt)

        # Try to parse the response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If the response is not valid JSON, return it as a text field
            return {"content": response}

    def create_slides(
        self,
        content: str,
        style: str = "Professional",
        num_slides: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Create presentation slides based on the provided content
        """
        prompt = f"""
        Create a {style} presentation with {num_slides} slides based on the following content:

        {content}

        For each slide, provide:
        1. A title
        2. Bullet points or content
        3. Any notes for the presenter

        Format the response as a JSON array of slide objects, where each slide has "title", "content", and "notes" fields.
        """

        response = self._call_gemini_api(prompt)

        # Try to parse the response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If the response is not valid JSON, create a simple slide structure
            return [{"title": "Generated Content", "content": response, "notes": ""}]

    def generate_test(
        self,
        subject: str,
        topic: str,
        difficulty: str = "Medium",
        num_questions: int = 10,
        question_types: List[str] = ["multiple_choice"]
    ) -> Dict[str, Any]:
        """
        Generate a test paper based on the provided parameters
        """
        question_types_text = ", ".join(question_types)

        prompt = f"""
        Create a {difficulty} difficulty {subject} test on the topic of "{topic}" with {num_questions} questions.

        Include the following question types: {question_types_text}

        For multiple-choice questions, provide 4 options and indicate the correct answer.
        For short-answer questions, provide the expected answer.
        For essay questions, provide evaluation criteria.

        Format the response as a JSON object with the following structure:
        {{
            "title": "Test title",
            "subject": "{subject}",
            "topic": "{topic}",
            "difficulty": "{difficulty}",
            "questions": [
                {{
                    "type": "multiple_choice",
                    "question": "Question text",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option A"
                }},
                {{
                    "type": "short_answer",
                    "question": "Question text",
                    "expected_answer": "Expected answer"
                }}
            ]
        }}
        """

        response = self._call_gemini_api(prompt)

        # Try to parse the response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If the response is not valid JSON, return it as a text field
            return {"content": response}
