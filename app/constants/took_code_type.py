"""
Tool Code Enum Constants
Định nghĩa các loại tool code cho message handling
"""

from enum import Enum


class ToolCodeEnum(str, Enum):
    """Enum cho các loại tool code"""

    LESSON_PLAN = "LESSON_PLAN"
    SLIDE_GENERATOR = "SLIDE_GENERATOR"
    EXAM_CREATOR = "EXAM_CREATOR"