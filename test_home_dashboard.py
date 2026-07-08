import urllib.request
import json
import sys
import uuid

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

def run_verification():
    print("====================================================")
    print("VERIFYING DASHBOARD HOME & SIGNUP END-TO-END FLOW")
    print("====================================================")

    # 1. Sign Up a new user (simulate not logging in automatically)
    unique_email = f"officer-{uuid.uuid4().hex[:6]}@recurclub.com"
    print(f"\nStep 1: Signing up new user '{unique_email}'...")
    signup_payload = {
        "fullname": "Test Officer",
        "email": unique_email,
        "password": "strongPassword123",
        "role": "loan_officer"
    }
    signup_resp = post_json("/api/auth/signup", signup_payload)
    print(f"Signup response: {signup_resp}")
    assert signup_resp["success"] is True

    # 2. Login as the newly created user
    print(f"\nStep 2: Logging in as '{unique_email}'...")
    login_resp = post_json("/api/auth/login", {"username": signup_payload["email"], "password": signup_payload["password"]})
    token = login_resp["token"]
    print("Login successful. Received JWT token.")

    # 3. Fetch Dashboard Home (should be empty/initial state for new user)
    print("\nStep 3: Fetching Dashboard Home state...")
    home_data = get_json("/api/dashboard/home", token=token)
    print("Home Stats:", home_data["stats"])
    print(f"Pending cases count: {len(home_data['pending_cases'])}")
    print(f"Recent activity count: {len(home_data['recent_activity'])}")
    assert len(home_data["pending_cases"]) == 5
    initial_decisions = home_data["stats"]["total_decisions"]

    # 4. Make an underwriting decision on one of the pending cases
    pending_case = home_data["pending_cases"][0]
    app_id = pending_case["id"]
    print(f"\nStep 4: Submitting underwriting decision on pending applicant {app_id}...")
    
    # Run prediction first to generate a prediction_id
    pred_data = get_json(f"/api/applicant/{app_id}", token=token)
    pred_id = pred_data["prediction_id"]
    print(f"Generated prediction ID: {pred_id}")

    # Submit decision override
    decision_payload = {
        "prediction_id": pred_id,
        "decision": "approved",
        "notes": f"Verified E2E audit trail for applicant {pending_case['name']}."
    }
    decision_resp = post_json(f"/api/applicant/{app_id}/decision", decision_payload, token=token)
    print(f"Decision response: {decision_resp}")
    assert decision_resp["success"] is True

    # 5. Fetch Dashboard Home again and verify updates
    print("\nStep 5: Fetching Dashboard Home after decision...")
    updated_home_data = get_json("/api/dashboard/home", token=token)
    print("Updated Stats:", updated_home_data["stats"])
    print(f"New Decisions Count: {updated_home_data['stats']['total_decisions']}")
    print(f"New Pending Count: {updated_home_data['stats']['total_pending']}")
    
    # Confirm stats changed
    assert updated_home_data["stats"]["total_decisions"] == initial_decisions + 1
    
    # Verify recent activity item contains our decision
    recent_act = updated_home_data["recent_activity"][0]
    print(f"Most recent activity in feed: {recent_act}")
    assert recent_act["applicant_id"] == app_id
    assert recent_act["decision"] == "approved"
    assert recent_act["notes"] == decision_payload["notes"]

    print("\n====================================================")
    print("DASHBOARD HOME & SIGNUP VERIFICATION SUCCESSFUL!")
    print("====================================================")

if __name__ == "__main__":
    run_verification()
