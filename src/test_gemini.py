import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("gemini-1.5-flash")  # safe and fast model
    response = model.generate_content("Hello Gemini, say 'Agent working!'")
    print("✅ Gemini response:", response.text.strip())
except Exception as e:
    print("❌ Gemini API call failed:", e)
