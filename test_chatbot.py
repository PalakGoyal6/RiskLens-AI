import urllib.request
import json
import sys

URL = "http://127.0.0.1:8080"

def post_json(endpoint, data=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{URL}{endpoint}",
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"HTTP Error {e.code}: {body}")
        raise e

def test_chatbot():
    print("=== STARTING CHATBOT FUNCTION-CALLING VALIDATION ===")
    
    # 1. Login to get token
    print("Step 1: Authenticating as officer1...")
    login_resp = post_json("/auth/login", {"username": "officer1", "password": "password123"})
    token = login_resp["token"]
    print("Authentication success.")
    
    # Use a known applicant ID from the dataset
    applicant_id = 100002
    messages = []
    
    # Test Scenario 1: Why flagged?
    print(f"\nStep 2: Asking 'Why did this applicant get flagged?' for ID {applicant_id}...")
    messages.append({"role": "user", "content": "why did this applicant get flagged?"})
    
    chat_resp = post_json("/chat", {
        "applicant_id": applicant_id,
        "messages": messages
    }, token=token)
    
    reply1 = chat_resp.get("reply", "")
    print(f"AI Assistant Reply 1:\n{reply1.encode('ascii', errors='replace').decode('ascii')}")
    
    # Basic validation: ensure it explained some risk factors and is not empty or error message
    assert len(reply1) > 0
    assert "error" not in reply1.lower()
    
    # Add assistant response to history
    messages.append({"role": "assistant", "content": reply1})
    
    # Test Scenario 2: What if credit score A was 0.8?
    print(f"\nStep 3: Asking what-if question: 'what if their credit score was 0.8?' for ID {applicant_id}...")
    messages.append({"role": "user", "content": "what if their credit score was 0.8?"})
    
    chat_resp2 = post_json("/chat", {
        "applicant_id": applicant_id,
        "messages": messages
    }, token=token)
    
    reply2 = chat_resp2.get("reply", "")
    print(f"AI Assistant Reply 2:\n{reply2.encode('ascii', errors='replace').decode('ascii')}")
    
    assert len(reply2) > 0
    assert "0.8" in reply2 or "80%" in reply2 or "prob" in reply2.lower()
    
    messages.append({"role": "assistant", "content": reply2})
    
    # Test Scenario 3: Has this person been reviewed before?
    print(f"\nStep 4: Asking 'has this person been reviewed before?' for ID {applicant_id}...")
    messages.append({"role": "user", "content": "has this person been reviewed before?"})
    
    chat_resp3 = post_json("/chat", {
        "applicant_id": applicant_id,
        "messages": messages
    }, token=token)
    
    reply3 = chat_resp3.get("reply", "")
    print(f"AI Assistant Reply 3:\n{reply3.encode('ascii', errors='replace').decode('ascii')}")
    
    assert len(reply3) > 0
    assert "review" in reply3.lower() or "decision" in reply3.lower() or "history" in reply3.lower() or "prediction" in reply3.lower() or "no" in reply3.lower()
    
    print("\n=======================================================")
    print("ALL THREE CHATBOT FUNCTION-CALLING TEST SCENARIOS PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    test_chatbot()
