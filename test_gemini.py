import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key[:5]}...{api_key[-5:]}")

# Configure the API
genai.configure(api_key=api_key)

# List available models
print("Available models:")
for model in genai.list_models():
    print(f"- {model.name}")

# Test the API
try:
    # Try with gemini-1.5-pro
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content("Hello, please respond with 'OK'.")
    print(f"Response: {response.text}")
    print("API test successful!")
except Exception as e:
    print(f"Error with gemini-1.5-pro: {str(e)}")
    try:
        # Try with gemini-1.0-pro
        model = genai.GenerativeModel('gemini-1.0-pro')
        response = model.generate_content("Hello, please respond with 'OK'.")
        print(f"Response: {response.text}")
        print("API test successful with gemini-1.0-pro!")
    except Exception as e2:
        print(f"Error with gemini-1.0-pro: {str(e2)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
