import urllib.request
import json
import sys

URL = "http://127.0.0.1:8000"

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
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def get_json(endpoint, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{URL}{endpoint}",
        headers=headers,
        method="GET"
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def test_full_flow():
    print("Starting End-to-End Simulation...")
    
    # 1. Login
    print("Step 1: Logging in as officer1...")
    login_resp = post_json("/api/auth/login", {"username": "officer1", "password": "password123"})
    token = login_resp["token"]
    print(f"Logged in successfully. User: {login_resp['user']['username']}, Role: {login_resp['user']['role']}")
    
    # 2. Search lookup prefix for ID 10000
    print("Step 2: Searching prefix '10000'...")
    search_resp = get_json("/api/applicants/search?q=10000", token=token)
    print(f"Search matches: {search_resp}")
    assert len(search_resp) > 0
    match_id = search_resp[0]["id"]
    
    # 3. Lookup details
    print(f"Step 3: Looking up features for match ID {match_id}...")
    lookup_resp = get_json(f"/api/applicants/lookup/{match_id}", token=token)
    print(f"Target: {lookup_resp['target']}, Features count: {len(lookup_resp['features'])}")
    assert len(lookup_resp['features']) == 190
    
    # 4. Score match ID
    print(f"Step 4: Scoring applicant {match_id} (fetching details)...")
    score_resp = get_json(f"/api/applicant/{match_id}", token=token)
    prediction_id = score_resp["prediction_id"]
    probability = score_resp["probability"]
    print(f"AI Probability: {probability:.4f}, Prediction ID: {prediction_id}")
    
    # 5. Overriding model decision with a decline
    print(f"Step 5: Submitting manual override to decline applicant {match_id}...")
    decision_resp = post_json(f"/api/applicant/{match_id}/decision", {
        "prediction_id": prediction_id,
        "decision": "declined",
        "notes": "Manual decline due to negative subjective indicators."
    }, token=token)
    print(f"Recorded decision success: {decision_resp['success']}, Decision ID: {decision_resp['decision_id']}")
    
    # 6. Check decision history
    print(f"Step 6: Fetching decision history for applicant {match_id}...")
    history_resp = get_json(f"/api/applicant/{match_id}/history", token=token)
    print(f"History entry: {history_resp[0]}")
    assert history_resp[0]["decision"] == "declined"
    assert history_resp[0]["notes"] == "Manual decline due to negative subjective indicators."
    
    # 7. Check performance override metrics
    print("Step 7: Checking override stats in Model Performance...")
    stats_resp = get_json("/api/decisions/stats", token=token)
    print(f"Stats: {stats_resp}")
    assert stats_resp["total_decisions"] >= 1
    
    print("\n=====================================================")
    print("END-TO-END FLOW SIMULATION COMPLETED SUCCESSFULLY!")
    print("=====================================================")

if __name__ == "__main__":
    test_full_flow()
