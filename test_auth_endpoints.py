import urllib.request
import json
import sys
import time

URL = "http://127.0.0.1:8000"

def req_json(endpoint, method="GET", data=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    
    req = urllib.request.Request(
        f"{URL}{endpoint}",
        data=req_data,
        headers=headers,
        method=method
    )
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
        except Exception:
            err_body = str(e)
        return e.code, err_body

def run_tests():
    print("=====================================================")
    print("Testing Authentication and Underwriting Override APIs")
    print("=====================================================")
    
    # Test 1: Invalid Login
    print("\n[1/9] Testing invalid login...")
    status, res = req_json("/api/auth/login", "POST", {"username": "officer1", "password": "wrongpassword"})
    print(f"Status: {status}, Response: {res}")
    assert status == 401
    assert "Invalid" in res.get("detail", "")
    print("Success!")

    # Test 2: Valid Login
    print("\n[2/9] Testing valid login...")
    status, res = req_json("/api/auth/login", "POST", {"username": "officer1", "password": "password123"})
    print(f"Status: {status}, User: {res.get('user')}")
    assert status == 200
    token = res.get("token")
    assert token is not None
    print("Success!")

    # Test 3: Get profile details (/api/auth/me)
    print("\n[3/9] Testing /api/auth/me...")
    status, res = req_json("/api/auth/me", "GET", token=token)
    print(f"Status: {status}, User: {res}")
    assert status == 200
    assert res.get("username") == "officer1"
    print("Success!")

    # Test 4: Applicant lookup from database (SK_ID_CURR: 100028 or similar from X_test index)
    # Let's see: we know ID 100088 is a valid ID because it's printed in index.js presets or we can search.
    # Let's try 100002 first, which is the very first one in standard Home Credit application_test/train datasets.
    print("\n[4/9] Testing applicant lookup /api/applicants/lookup/100002...")
    status, res = req_json("/api/applicants/lookup/100002", "GET", token=token)
    print(f"Status: {status}, Target: {res.get('target')}, Features length: {len(res.get('features', {})) if res.get('features') else 0}")
    # Wait, is 100002 in X_test? If it's not, we might get 404, which is fine, let's verify if we get a 200 or 404.
    # If it's 404, we will try to find a valid ID by fetching the /api/applicants list.
    if status == 404:
        print("100002 not found, querying /api/applicants to find a valid ID...")
        # /api/applicants returns cohort of 1000
        _, applicants = req_json("/api/applicants")
        valid_id = applicants[0]["id"]
        print(f"Found valid applicant ID: {valid_id}")
        status, res = req_json(f"/api/applicants/lookup/{valid_id}", "GET", token=token)
        print(f"Status: {status}, Features length: {len(res.get('features', {}))}")
    else:
        valid_id = 100002
        
    assert status == 200
    assert len(res["features"]) > 0
    print("Success!")

    # Test 5: Prediction Logging & Returning Prediction ID
    print(f"\n[5/9] Testing prediction logging on GET /api/applicant/{valid_id}...")
    status, res = req_json(f"/api/applicant/{valid_id}", "GET", token=token)
    print(f"Status: {status}, Probability: {res.get('probability'):.4f}, Prediction ID: {res.get('prediction_id')}")
    assert status == 200
    prediction_id = res.get("prediction_id")
    assert prediction_id is not None
    print("Success!")

    # Test 6: Record Underwriting Override/Decision
    print(f"\n[6/9] Recording manual decision on prediction {prediction_id}...")
    decision_payload = {
        "prediction_id": prediction_id,
        "decision": "approved",
        "notes": "Applicant looks highly stable despite minor credit flags."
    }
    status, res = req_json(f"/api/applicant/{valid_id}/decision", "POST", decision_payload, token=token)
    print(f"Status: {status}, Response: {res}")
    assert status == 200
    assert res.get("success") is True
    print("Success!")

    # Test 7: Verify Decision History Timeline
    print(f"\n[7/9] Verifying decision history timeline for applicant {valid_id}...")
    status, res = req_json(f"/api/applicant/{valid_id}/history", "GET", token=token)
    print(f"Status: {status}, History item count: {len(res)}")
    assert status == 200
    assert len(res) > 0
    latest_item = res[0]
    assert latest_item.get("prediction_id") == prediction_id
    assert latest_item.get("decision") == "approved"
    assert latest_item.get("notes") == "Applicant looks highly stable despite minor credit flags."
    print("Success!")

    # Test 8: Get Overrides Stats
    print("\n[8/9] Testing decisions/stats endpoint...")
    status, res = req_json("/api/decisions/stats", "GET", token=token)
    print(f"Status: {status}, Response: {res}")
    assert status == 200
    assert res.get("total_decisions") >= 1
    print("Success!")

    # Test 9: Logout
    print("\n[9/9] Testing logout...")
    status, res = req_json("/api/auth/logout", "POST", token=token)
    print(f"Status: {status}, Response: {res}")
    assert status == 200
    
    # Token should now be invalid
    status, res = req_json("/api/auth/me", "GET", token=token)
    print(f"Me after logout status: {status} (Expected: 401)")
    assert status == 401
    
    print("\n=====================================================")
    print("ALL API SECURITY & OVERRIDE TESTS PASSED SUCCESSFULLY!")
    print("=====================================================")

if __name__ == "__main__":
    run_tests()
