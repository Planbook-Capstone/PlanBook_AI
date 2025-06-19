import pandas as pd
from fastapi import UploadFile
from typing import Dict
import io
import logging

logger = logging.getLogger(__name__)

class ExcelProcessor:
    """Xử lý file Excel chứa đáp án"""

    async def load_answer_keys(self, excel_file: UploadFile) -> Dict[str, Dict[int, str]]:
        """
        Đọc đáp án từ file Excel
        Format: TestCode | question_1 | question_2 | ... | question_n
        """
        try:
            # Đọc nội dung file
            content = await excel_file.read()

            # Đọc file Excel vào DataFrame
            df = pd.read_excel(io.BytesIO(content))

            # Chuẩn hóa tên cột (xoá BOM và khoảng trắng)
            df.columns = [str(col).strip().replace('\ufeff', '') for col in df.columns]
            logger.info(f"Excel Columns: {df.columns.tolist()}")

            # Kiểm tra xem có cột TestCode không
            if "TestCode" not in df.columns:
                raise ValueError("Excel file is missing required column: 'TestCode'")

            answer_keys = {}

            for _, row in df.iterrows():
                test_code = str(row["TestCode"]).strip()
                answers = {}

                # Duyệt qua các cột có prefix là 'question_'
                for col in df.columns:
                    if col.startswith("question_"):
                        try:
                            question_num = int(col.split("_")[1])
                            answer = str(row[col]).strip().upper()
                            if answer in ["A", "B", "C", "D"]:
                                answers[question_num] = answer
                        except (ValueError, AttributeError):
                            continue  # Bỏ qua nếu giá trị không hợp lệ

                answer_keys[test_code] = answers
                logger.info(f"Loaded {len(answers)} answers for test code {test_code}")

            return answer_keys

        except Exception as e:
            logger.error(f"❌ Error loading answer keys: {str(e)}")
            raise Exception(f"Failed to load answer keys from Excel file: {str(e)}")
