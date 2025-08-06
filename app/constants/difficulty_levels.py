"""
Constants for difficulty levels used in exam and question processing
"""

from enum import Enum


class DifficultyLevel(Enum):
    """Enum cho các mức độ khó của câu hỏi"""
    KNOWLEDGE = "KNOWLEDGE"
    COMPREHENSION = "COMPREHENSION"
    APPLICATION = "APPLICATION"
