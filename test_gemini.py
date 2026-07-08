from google import genai
import os

try:
    print("Testing genai Client...")
    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Hello, say "GenAI Activated!"',
    )
    print("Success text:", response.text.strip())
except Exception as e:
    import traceback
    traceback.print_exc()
