from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# Lesson Plan Schemas
class LessonPlanRequest(BaseModel):
    subject: str = Field(..., description="Subject of the lesson (e.g., Math, Science)")
    grade_level: str = Field(..., description="Grade level (e.g., Grade 5, High School)")
    topic: str = Field(..., description="Specific topic for the lesson")
    duration: int = Field(..., description="Duration of the lesson in minutes")
    objectives: Optional[List[str]] = Field(None, description="Learning objectives for the lesson")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "subject": "Mathematics",
                "grade_level": "Grade 8",
                "topic": "Introduction to Quadratic Equations",
                "duration": 45,
                "objectives": [
                    "Understand the standard form of quadratic equations",
                    "Identify coefficients and constants in quadratic expressions",
                    "Solve simple quadratic equations by factoring"
                ]
            }
        }
    )

class LessonPlanResponse(BaseModel):
    lesson_plan: Dict[str, Any] = Field(..., description="Generated lesson plan")
    message: str = Field(..., description="Status message")

# Slide Creation Schemas
class SlideCreationRequest(BaseModel):
    content: str = Field(..., description="Content to be converted into slides")
    style: Optional[str] = Field("Professional", description="Style of the slides (e.g., Professional, Creative)")
    num_slides: Optional[int] = Field(10, description="Number of slides to generate")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Introduction to Python Programming: Python is a high-level, interpreted programming language...",
                "style": "Educational",
                "num_slides": 8
            }
        }
    )

class SlideCreationResponse(BaseModel):
    slides: List[Dict[str, Any]] = Field(..., description="Generated slides content")
    message: str = Field(..., description="Status message")

# Test Generation Schemas
class TestGenerationRequest(BaseModel):
    subject: str = Field(..., description="Subject of the test (e.g., Math, Science)")
    topic: str = Field(..., description="Specific topic for the test")
    difficulty: str = Field("Medium", description="Difficulty level (Easy, Medium, Hard)")
    num_questions: int = Field(10, description="Number of questions to generate")
    question_types: List[str] = Field(["multiple_choice"], description="Types of questions (multiple_choice, short_answer, essay)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "subject": "Physics",
                "topic": "Newton's Laws of Motion",
                "difficulty": "Medium",
                "num_questions": 15,
                "question_types": ["multiple_choice", "short_answer"]
            }
        }
    )

class TestGenerationResponse(BaseModel):
    test_paper: Dict[str, Any] = Field(..., description="Generated test paper")
    message: str = Field(..., description="Status message")
