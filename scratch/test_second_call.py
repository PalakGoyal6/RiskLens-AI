import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

openai_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")
model_name = os.environ.get("LLM_MODEL")

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

# First turn payload
req_payload = {
    "model": model_name,
    "messages": messages,
    "tools": tools_spec,
    "tool_choice": "auto"
}

response = requests.post(f"{base_url}/chat/completions", headers=headers, json=req_payload)
res_json = response.json()
choice_message = res_json["choices"][0]["message"]

print("FIRST RESPONSE:")
print(json.dumps(choice_message, indent=2))

# Execute mock tool
tool_call = choice_message["tool_calls"][0]
tool_result = {
    "applicant_id": 100002,
    "name": "John Doe",
    "actual_repayment_status": "Paid",
    "predicted_default_probability": 0.28,
    "risk_classification": "High Risk",
    "top_shap_risk_drivers": [
        {"feature": "EXT_SOURCE_2", "display_name": "Credit Bureau Score A", "shap_value": 0.12, "feature_value": 0.15},
        {"feature": "BUREAU_DEBT_CREDIT_RATIO", "display_name": "Debt-to-Credit Ratio", "shap_value": 0.08, "feature_value": 0.75}
    ]
}

# Append assistant message
assistant_msg = {
    "role": "assistant",
    "content": choice_message.get("content")
}
if "tool_calls" in choice_message:
    # Clean up the tool call for the request
    tool_calls_clean = []
    for tc in choice_message["tool_calls"]:
        tool_calls_clean.append({
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"]
            }
        })
    assistant_msg["tool_calls"] = tool_calls_clean

messages.append(assistant_msg)

# Append tool message
tool_msg = {
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "name": tool_call["function"]["name"],
    "content": json.dumps(tool_result)
}
messages.append(tool_msg)

print("\nCONSTRUCTED MESSAGES FOR SECOND TURN:")
print(json.dumps(messages, indent=2))

second_payload = {
    "model": model_name,
    "messages": messages,
    "tools": tools_spec
}

second_response = requests.post(f"{base_url}/chat/completions", headers=headers, json=second_payload)
print(f"\nSecond Status Code: {second_response.status_code}")
print("Second Response JSON:")
print(json.dumps(second_response.json(), indent=2))
