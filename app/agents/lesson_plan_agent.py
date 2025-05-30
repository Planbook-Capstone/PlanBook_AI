from typing import Dict, Any, List, Optional
import time
from app.agents.base_agent import BaseAgent
from app.services.rag_service import rag_service
from app.services.docx_service import docx_service
from app.database.connection import get_database_sync, LESSON_PLANS_COLLECTION
from app.database.models import GeneratedLessonPlan
from bson import ObjectId

class ChemistryLessonAgent(BaseAgent):
    """
    Agent chuyên sinh giáo án tự động cho giáo viên

    Kế thừa từ BaseAgent:
    - Gemini API calls
    - Memory management
    - Error handling
    - Logging

    Chức năng riêng:
    - Tạo giáo án theo chuẩn Bộ GD&ĐT
    - Tích hợp RAG cho nội dung sách giáo khoa
    - Tùy chỉnh theo môn học và cấp độ
    """

    def __init__(self):
        super().__init__(
            agent_name="ChemistryLessonAgent",
            model_name="gemini-1.5-flash"
        )

        # Chuyên biệt cho môn Hóa học
        self.subject = "Hóa học"
        self.grade_levels = ["10", "11", "12"]
        self.supported_subjects = ["Hóa học", "Chemistry"]  # Add supported subjects

    def _get_system_prompt(self) -> str:
        """System prompt chuyên biệt cho lesson plan generation"""
        return """
        Bạn là một AI Assistant chuyên gia về giáo dục, có nhiều năm kinh nghiệm trong việc thiết kế giáo án cho giáo viên cấp THCS và THPT tại Việt Nam.

        Nhiệm vụ của bạn:
        1. Tạo giáo án chi tiết theo chuẩn Bộ Giáo dục và Đào tạo Việt Nam
        2. Đảm bảo giáo án phù hợp với độ tuổi và trình độ học sinh
        3. Tích hợp các phương pháp giảng dạy hiện đại và tương tác
        4. Bao gồm đầy đủ các thành phần: mục tiêu, nội dung, phương pháp, đánh giá

        Luôn trả về kết quả dưới dạng JSON với cấu trúc rõ ràng và chi tiết.
        """

    def _validate_input(self, **kwargs) -> bool:
        """Validate input cho lesson plan generation"""
        required_fields = ["topic", "grade", "duration"]

        for field in required_fields:
            if field not in kwargs or not kwargs[field]:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Validate grade level (using 'grade' instead of 'grade_level')
        if str(kwargs["grade"]) not in self.grade_levels:
            self.logger.error(f"Grade level {kwargs['grade']} not in supported list: {self.grade_levels}")
            return False

        # Validate duration
        duration = kwargs.get("duration", 45)
        if not isinstance(duration, int) or duration <= 0:
            self.logger.error("Duration must be a positive integer")
            return False

        # Validate topic is not empty or placeholder
        topic = kwargs.get("topic", "").strip()
        if not topic or topic.lower() in ["string", "test", ""]:
            self.logger.error(f"Invalid topic: '{topic}'. Topic must be a real chemistry subject.")
            return False

        return True

    async def process(self, **kwargs) -> Dict[str, Any]:
        """
        Main method để tạo giáo án với RAG integration

        Args:
            topic: Chủ đề bài học
            grade: Lớp (10, 11, 12)
            duration: Thời lượng (phút)
            objectives: Mục tiêu học tập (optional)
            teaching_method: Phương pháp giảng dạy (optional)
            include_experiments: Có thí nghiệm không (optional)
        """
        start_time = time.time()

        try:
            # Validate input
            if not self._validate_input(**kwargs):
                raise ValueError("Invalid input parameters")

            # Extract parameters
            topic = kwargs["topic"]
            grade = kwargs["grade"]
            duration = kwargs.get("duration", 45)
            objectives = kwargs.get("objectives", [])
            teaching_method = kwargs.get("teaching_method", "Tương tác")
            include_experiments = kwargs.get("include_experiments", True)

            # Log processing start
            self.logger.info(f"Generating chemistry lesson plan: Grade {grade} - {topic}")

            # Add to memory
            self._add_to_memory("lesson_plan_request", {
                "topic": topic,
                "grade": grade,
                "duration": duration
            })

            # 1. RAG: Tìm kiếm nội dung liên quan
            self.logger.info("Searching for relevant content using RAG...")
            rag_results = rag_service.search_relevant_content(
                query=topic,
                grade=grade,
                content_types=["lesson", "chapter"],
                top_k=5
            )

            # 2. Build context từ RAG results
            rag_context = rag_results.get("context", "")
            relevant_lessons = []

            for result in rag_results.get("results", []):
                if "lesson_info" in result:
                    relevant_lessons.append(result["lesson_info"])

            # 3. Build enhanced prompt với RAG context
            prompt = self._build_enhanced_prompt(
                topic=topic,
                grade=grade,
                duration=duration,
                objectives=objectives,
                teaching_method=teaching_method,
                include_experiments=include_experiments,
                rag_context=rag_context,
                relevant_lessons=relevant_lessons
            )

            # 4. Call Gemini API
            self.logger.info("Generating lesson plan with Gemini...")
            response = self._call_gemini_api(prompt, temperature=0.7)

            # 5. Parse response
            lesson_plan_data = self._parse_json_response(response)
            
            # 5.5. Validate and clean lesson plan data
            lesson_plan_data = self._validate_and_clean_lesson_plan_data(lesson_plan_data)

            # 6. Enrich với metadata
            lesson_plan_data.update({
                "title": f"Giáo án Hóa học lớp {grade}",
                "subject": self.subject,
                "grade": grade,
                "topic": topic,
                "duration": duration,
                "source_lessons": [result.get("content_id") for result in rag_results.get("results", [])],
                "generated_by": self.agent_name,
                "prompt_used": prompt[:500] + "..." if len(prompt) > 500 else prompt
            })

            # 7. Validate and clean lesson plan data structure
            lesson_plan_data = self._validate_and_clean_lesson_plan_data(lesson_plan_data)

            # 8. Tạo file DOCX và lưu vào GridFS
            self.logger.info("Creating DOCX file and storing in GridFS...")
            docx_file_id = await docx_service.create_lesson_plan_docx(lesson_plan_data)
            lesson_plan_data["docx_file_id"] = docx_file_id

            # 9. Lưu vào database
            lesson_plan_record = GeneratedLessonPlan(**lesson_plan_data)
            db = get_database_sync()
            collection = db[LESSON_PLANS_COLLECTION]
            
            # Serialize model but exclude _id if it's None to let MongoDB generate it
            record_dict = lesson_plan_record.model_dump(by_alias=True, exclude_none=True)
            if "_id" in record_dict and record_dict["_id"] is None:
                del record_dict["_id"]
                
            result = collection.insert_one(record_dict)
            lesson_plan_id = str(result.inserted_id)

            # 10. Add to memory
            generation_time = time.time() - start_time
            self._add_to_memory("lesson_plan_generated", {
                "topic": topic,
                "grade": grade,
                "lesson_plan_id": lesson_plan_id,
                "docx_file_id": docx_file_id,
                "generation_time": generation_time,
                "rag_results_count": len(rag_results.get("results", [])),
                "success": True
            })

            self.logger.info(f"Successfully generated lesson plan: {lesson_plan_id} in {generation_time:.2f}s")

            return self._format_response(
                result={
                    "lesson_plan_id": lesson_plan_id,
                    "docx_file_id": docx_file_id,
                    "lesson_plan_data": lesson_plan_data,
                    "generation_time": generation_time,
                    "rag_sources": len(rag_results.get("results", []))
                },
                message=f"Giáo án Hóa học lớp {grade} - {topic} đã được tạo và lưu vào GridFS thành công"
            )

        except Exception as e:
            self.logger.error(f"Failed to generate lesson plan: {str(e)}")

            # Add error to memory
            self._add_to_memory("lesson_plan_error", {
                "error": str(e),
                "parameters": kwargs,
                "generation_time": time.time() - start_time
            })

            raise ValueError(f"Không thể tạo giáo án: {str(e)}") from e

    def _build_enhanced_prompt(
        self,
        topic: str,
        grade: str,
        duration: int,
        objectives: List[str],
        teaching_method: str,
        include_experiments: bool,
        rag_context: str,
        relevant_lessons: List[Dict[str, Any]]
    ) -> str:
        """Build enhanced prompt với RAG context"""

        objectives_text = ""
        if objectives:
            objectives_text = "Mục tiêu cụ thể:\n" + "\n".join([f"- {obj}" for obj in objectives])

        experiments_text = ""
        if include_experiments:
            experiments_text = "Lưu ý: Cần tích hợp thí nghiệm minh họa phù hợp."

        relevant_info = ""
        if relevant_lessons:
            relevant_info = "Thông tin từ sách giáo khoa liên quan:\n"
            for lesson in relevant_lessons[:3]:  # Lấy 3 lessons đầu
                relevant_info += f"- {lesson.get('title', '')}\n"
                if lesson.get('key_concepts'):
                    relevant_info += f"  Khái niệm: {', '.join(lesson['key_concepts'][:3])}\n"
                if lesson.get('formulas'):
                    relevant_info += f"  Công thức: {', '.join(lesson['formulas'][:3])}\n"

        prompt = f"""
        Bạn là chuyên gia giáo dục Hóa học với 20 năm kinh nghiệm giảng dạy cấp THPT tại Việt Nam.

        Nhiệm vụ: Tạo giáo án chi tiết cho bài học {duration} phút về chủ đề "{topic}" cho học sinh lớp {grade}.

        {objectives_text}

        Phương pháp giảng dạy: {teaching_method}
        {experiments_text}

        {relevant_info}

        NGUYÊN LIỆU TỪ SÁCH GIÁO KHOA:
        {rag_context}

        YÊU CẦU GIÁO ÁN:

        1. **Thông tin chung**:
           - title: Tên bài học
           - subject: "Hóa học"
           - grade: "{grade}"
           - duration: {duration}
           - lesson_type: "Lý thuyết" hoặc "Thực hành"

        2. **objectives** (mảng các mục tiêu):
           - Kiến thức: Học sinh biết được...
           - Kỹ năng: Học sinh làm được...
           - Thái độ: Học sinh có thái độ...

        3. **materials** (mảng dụng cụ cần thiết):
           - Dụng cụ thí nghiệm
           - Hóa chất
           - Thiết bị hỗ trợ

        4. **activities** (mảng các hoạt động):
           - name: Tên hoạt động
           - duration: Thời gian
           - content: Nội dung chi tiết
           - teacher_activities: Hoạt động của GV
           - student_activities: Hoạt động của HS

        5. **assessment** (đối tượng đánh giá):
           - methods: Phương pháp đánh giá
           - criteria: Tiêu chí đánh giá
           - questions: Câu hỏi kiểm tra

        6. **homework** (mảng bài tập về nhà):
           - Bài tập củng cố
           - Chuẩn bị bài mới

        LƯU Ý QUAN TRỌNG:
        - Sử dụng nội dung từ sách giáo khoa đã cung cấp
        - Đảm bảo tính khoa học và chính xác
        - Phù hợp với chương trình Bộ GD&ĐT
        - Tích hợp thí nghiệm thực tế
        - Kết nối với thực tiễn cuộc sống

        BẮT BUỘC TRẠNG THÁI JSON OUTPUT:
        Trả về CHÍNH XÁC theo cấu trúc JSON sau (không được thiếu hoặc None):
        {{
            "title": "Cấu tạo nguyên tử",
            "subject": "Hóa học",
            "grade": "{grade}",
            "duration": {duration},
            "lesson_type": "Lý thuyết",
            "objectives": [
                "Mục tiêu kiến thức 1",
                "Mục tiêu kỹ năng 2",
                "Mục tiêu thái độ 3"
            ],
            "materials": [
                "Dụng cụ 1",
                "Hóa chất 2",
                "Thiết bị 3"
            ],
            "activities": [
                {{
                    "name": "Hoạt động 1",
                    "duration": 10,
                    "content": "Nội dung chi tiết",
                    "teacher_activities": "Hoạt động GV",
                    "student_activities": "Hoạt động HS"
                }}
            ],
            "assessment": {{
                "methods": ["Quan sát", "Hỏi đáp"],
                "criteria": ["Tiêu chí 1", "Tiêu chí 2"],
                "questions": ["Câu hỏi 1", "Câu hỏi 2"]
            }},
            "homework": [
                "Bài tập 1",
                "Chuẩn bị bài mới"
            ]
        }}

        TUYỆT ĐỐI KHÔNG được trả về null, None, hoặc undefined cho bất kỳ mảng nào.
        """

        return prompt

    def get_supported_grades(self) -> List[str]:
        """Trả về danh sách lớp được hỗ trợ"""
        return self.grade_levels

    def _validate_and_clean_lesson_plan_data(self, lesson_plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean lesson plan data structure to ensure all required fields exist"""
        
        # Ensure all required fields exist with proper types
        required_arrays = ["objectives", "materials", "activities", "homework"]
        
        for field in required_arrays:
            if field not in lesson_plan_data or lesson_plan_data[field] is None:
                self.logger.warning(f"Field '{field}' is missing or None, setting to empty array")
                lesson_plan_data[field] = []
            elif not isinstance(lesson_plan_data[field], list):
                self.logger.warning(f"Field '{field}' is not a list, converting to array")
                if isinstance(lesson_plan_data[field], str):
                    lesson_plan_data[field] = [lesson_plan_data[field]]
                else:
                    lesson_plan_data[field] = []
        
        # Ensure assessment is an object with required structure
        if "assessment" not in lesson_plan_data or lesson_plan_data["assessment"] is None:
            lesson_plan_data["assessment"] = {
                "methods": ["Quan sát", "Hỏi đáp"],
                "criteria": ["Hiểu bài", "Tham gia tích cực"],
                "questions": ["Câu hỏi kiểm tra nhanh"]
            }
        elif isinstance(lesson_plan_data["assessment"], dict):
            # Ensure assessment sub-fields are arrays
            for sub_field in ["methods", "criteria", "questions"]:
                if sub_field not in lesson_plan_data["assessment"] or lesson_plan_data["assessment"][sub_field] is None:
                    lesson_plan_data["assessment"][sub_field] = []
                elif not isinstance(lesson_plan_data["assessment"][sub_field], list):
                    lesson_plan_data["assessment"][sub_field] = []
        
        # Ensure activities is properly structured
        if lesson_plan_data["activities"]:
            cleaned_activities = []
            for i, activity in enumerate(lesson_plan_data["activities"]):
                if isinstance(activity, dict):
                    cleaned_activity = {
                        "name": activity.get("name", f"Hoạt động {i+1}"),
                        "duration": activity.get("duration", 10),
                        "content": activity.get("content", "Nội dung hoạt động"),
                        "teacher_activities": activity.get("teacher_activities", "Hướng dẫn học sinh"),
                        "student_activities": activity.get("student_activities", "Tham gia hoạt động")
                    }
                    cleaned_activities.append(cleaned_activity)
                else:
                    # Convert string to activity object
                    cleaned_activities.append({
                        "name": f"Hoạt động {i+1}",
                        "duration": 15,
                        "content": str(activity),
                        "teacher_activities": "Hướng dẫn học sinh",
                        "student_activities": "Tham gia hoạt động"
                    })
            lesson_plan_data["activities"] = cleaned_activities
        
        # Ensure basic fields exist
        lesson_plan_data.setdefault("title", f"Bài học {lesson_plan_data.get('topic', 'Chưa xác định')}")
        lesson_plan_data.setdefault("subject", "Hóa học")
        lesson_plan_data.setdefault("lesson_type", "Lý thuyết")
        
        self.logger.info("Lesson plan data validated and cleaned successfully")
        return lesson_plan_data
