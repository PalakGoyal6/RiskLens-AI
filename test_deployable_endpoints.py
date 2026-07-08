import urllib.request
import json
import pickle
import pandas as pd
import sys
import os

URL = "http://127.0.0.1:8000"

def post_json(endpoint, data):
    req = urllib.request.Request(
        f"{URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def post_multipart(endpoint, filename, file_content):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    # Construct multipart request payload manually
    parts = []
    parts.append(f"--{boundary}".encode("utf-8"))
    parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode("utf-8"))
    parts.append(b"Content-Type: text/csv\r\n")
    parts.append(file_content)
    parts.append(f"--{boundary}--".encode("utf-8"))
    
    body = b"\r\n".join(parts) + b"\r\n"
    
    req = urllib.request.Request(
        f"{URL}{endpoint}",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body))
        }
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def test_endpoints():
    print("=========================================")
    print("Testing Deployable FastAPI API Endpoints")
    print("=========================================")
    
    # 1. Load an actual row from X_test to use for validation
    with open("models/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    
    # Take first row as dict
    row_dict = X_test.iloc[0].to_dict()
    # Convert numpy types to python float/int for JSON serialization
    clean_dict = {}
    for k, v in row_dict.items():
        if pd.notna(v):
            clean_dict[k] = float(v) if not isinstance(v, (int, float)) else v
        else:
            clean_dict[k] = None

    # Test 1: Predict Endpoint
    print("\n[1/3] Testing POST /predict...")
    payload = {"features": clean_dict}
    try:
        resp = post_json("/predict", payload)
        print("Success! Response:")
        print(resp)
        assert "probability" in resp
        assert "prediction" in resp
        assert "decision" in resp
    except Exception as e:
        print(f"FAILED /predict: {e}")
        sys.exit(1)

    # Test 2: Explain Endpoint
    print("\n[2/3] Testing POST /explain...")
    try:
        resp = post_json("/explain", payload)
        print("Success! Explanation returned successfully.")
        print(f"Probability: {resp['probability']:.4f}")
        print(f"Base Value: {resp['base_value']:.4f}")
        print(f"Top 3 driving features:")
        for exp in resp["explanation"][:3]:
            print(f"  - {exp['feature']}: SHAP={exp['shap_value']:.4f}, Val={exp['feature_value']}")
        assert "probability" in resp
        assert "base_value" in resp
        assert "explanation" in resp
    except Exception as e:
        print(f"FAILED /explain: {e}")
        sys.exit(1)

    # Test 3: Batch Endpoint
    print("\n[3/3] Testing POST /batch...")
    # Generate a temporary CSV with 3 rows of test data
    test_csv = "temp_batch_test.csv"
    try:
        batch_df = X_test.iloc[0:3].copy()
        # Add ID column
        batch_df["sk_id_curr"] = [100001, 100002, 100003]
        batch_df.to_csv(test_csv, index=False)
        
        with open(test_csv, "rb") as f:
            file_content = f.read()
            
        resp = post_multipart("/batch", test_csv, file_content)
        print("Success! Batch predictions response:")
        for res in resp:
            print(f"  ID: {res['id']} -> Prob: {res['probability']:.4f}, Decision: {res['decision']}")
            
        assert len(resp) == 3
        assert "probability" in resp[0]
        assert "decision" in resp[0]
    except Exception as e:
        print(f"FAILED /batch: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(test_csv):
            os.remove(test_csv)
            
    print("\n=========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=========================================")

if __name__ == "__main__":
    test_endpoints()
