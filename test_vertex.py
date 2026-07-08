from google import genai
try:
    print("Testing genai Client with VertexAI...")
    client = genai.Client(vertexai=True)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Hello, say "Vertex AI Activated!"',
    )
    print("Success text:", response.text.strip())
except Exception as e:
    import traceback
    traceback.print_exc()
