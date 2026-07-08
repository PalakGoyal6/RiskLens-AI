import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

openai_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")
model_name = os.environ.get("LLM_MODEL")

print(f"Key: {openai_key[:10]}...")
print(f"Base URL: {base_url}")
print(f"Model: {model_name}")

headers = {
    "Authorization": f"Bearer {openai_key}",
    "Content-Type": "application/json"
}

# Define tools
tools_spec = [
    {
        "type": "function",
        "function": {
            "name": "get_applicant_data",
            "description": "Get the full profile and latest risk prediction for an applicant",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "The applicant ID"
                    }
                },
                "required": ["applicant_id"]
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are an assistant. The current applicant is ID 100002."},
    {"role": "user", "content": "why did this applicant get flagged?"}
]

req_payload = {
    "model": model_name,
    "messages": messages,
    "tools": tools_spec,
    "tool_choice": "auto"
}

response = requests.post(f"{base_url}/chat/completions", headers=headers, json=req_payload)
print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(json.dumps(response.json(), indent=2))
