import os
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = "AIzaSyAI5xtW-81f8A34q15YkQVczmOAIz3F8DU"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Checking available models for your API Key...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f" - {m.name}")