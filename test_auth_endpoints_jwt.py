import urllib.request
import urllib.parse
import json
import sys

URL = "http://127.0.0.1:8000"

def get_json(endpoint, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{URL}{endpoint}", headers=headers, method="GET")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def post_json(endpoint, data, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def test_v2_endpoints():
    print("Testing RiskLens AI JWT & Auditing Endpoints...")
    
    # 1. Login with JWT
    print("\n1. Testing Login POST /auth/login...")
    login_payload = {"username": "officer1", "password": "password123"}
    resp = post_json("/auth/login", login_payload)
    token = resp["access_token"]
    user_id = resp["user"]["id"]
    print("Success: Retrieved Access Token:", token[:25] + "...")
    assert "token_type" in resp and resp["token_type"] == "bearer"
    
    # 2. Search applicants with faked names
    print("\n2. Testing Search GET /applicants/search?query=...")
    search_resp = get_json("/applicants/search?query=1000", token=token)
    print(f"Success: Matches count: {len(search_resp)}")
    print(f"First match: ID: {search_resp[0]['id']}, Name: {search_resp[0]['name']}")
    assert "name" in search_resp[0]
    match_id = search_resp[0]["id"]
    
    # 3. Lookup full profile
    print(f"\n3. Testing Lookup GET /applicants/{{id}} for ID {match_id}...")
    profile = get_json(f"/applicants/{match_id}", token=token)
    print(f"Success: Name: {profile['name']}, Target: {profile['target']}")
    assert profile["name"] is not None
    
    # 4. Predict endpoint (logs prediction)
    print(f"\n4. Testing Predict GET /applicants/{{id}}/predict for ID {match_id}...")
    prediction = get_json(f"/applicants/{match_id}/predict", token=token)
    pred_id = prediction["prediction_id"]
    print(f"Success: Prob: {prediction['probability']:.4f}, Prediction ID: {pred_id}")
    assert prediction["probability"] >= 0.0
    
    # 5. Record Underwriting Decision
    print("\n5. Testing Decision POST /decisions...")
    decision_payload = {
        "prediction_id": pred_id,
        "decision": "approved",
        "notes": "E2E automated validation test notes."
    }
    decision_resp = post_json("/decisions", decision_payload, token=token)
    print(f"Success: Decision ID: {decision_resp['decision_id']}")
    assert decision_resp["success"] is True
    
    # 6. Check history audit trail
    print(f"\n6. Testing Audit Trail GET /applicants/{{id}}/history for ID {match_id}...")
    history = get_json(f"/applicants/{match_id}/history", token=token)
    print(f"Success: History records count: {len(history)}")
    print(f"Most recent decision: {history[0]['decision']} by {history[0]['officer_name']}")
    assert history[0]["decision"] == "approved"
    
    # 7. Check officer activity oversight
    print(f"\n7. Testing Officer Activity GET /officers/{{id}}/activity for officer ID {user_id}...")
    activity = get_json(f"/officers/{user_id}/activity", token=token)
    print(f"Success: Activity entries: {len(activity)}")
    print(f"Most recent activity: Applicant: {activity[0]['applicant_name']}, Decision: {activity[0]['decision']}")
    assert len(activity) > 0
    assert activity[0]["decision"] == "approved"
    
    print("\n=======================================================")
    print("ALL SPECIFIED ENDPOINTS PASSED E2E INTEGRATION SUCCESS!")
    print("=======================================================")

if __name__ == "__main__":
    test_v2_endpoints()
