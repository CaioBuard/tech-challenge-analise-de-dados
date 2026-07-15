import os
from dotenv import load_dotenv

load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com"
MEDICAL_DOMAIN = "medical-v1"
LANGUAGE_CODE = "en"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite"
