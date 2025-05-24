import os
from dotenv import load_dotenv
import sys

# Hiển thị thư mục hiện tại
print(f"Current directory: {os.getcwd()}")

# Kiểm tra xem file .env có tồn tại không
env_path = os.path.join(os.getcwd(), '.env')
print(f"Checking for .env file at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")

# Nếu file tồn tại, hiển thị nội dung
if os.path.exists(env_path):
    print("\nFile content:")
    with open(env_path, 'r') as f:
        print(f.read())

# Tải biến môi trường
print("\nLoading environment variables...")
load_dotenv()

# Kiểm tra biến môi trường
gemini_api_key = os.getenv("GEMINI_API_KEY")
print(f"GEMINI_API_KEY: {gemini_api_key[:5]}...{gemini_api_key[-5:] if gemini_api_key else ''}")

# Kiểm tra các biến môi trường khác
print(f"API_PREFIX: {os.getenv('API_PREFIX')}")
print(f"DEBUG: {os.getenv('DEBUG')}")
print(f"PROJECT_NAME: {os.getenv('PROJECT_NAME')}")

# Kiểm tra biến môi trường hệ thống
print("\nChecking system environment variables...")
for key, value in os.environ.items():
    if "GEMINI" in key:
        print(f"{key}: {value[:5]}...{value[-5:]}")
