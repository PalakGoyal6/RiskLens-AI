import os
import requests

url_base = "https://palak-project-resource.services.ai.azure.com/api/projects/Palak-Project/openai/v1/responses"
key = os.getenv("AZURE_AI_KEY", "your-azure-key-placeholder")
payload = {
    "model": "gpt-5-mini",
    "messages": [{"role": "user", "content": "Hello, say 'Azure Connection Success!'"}]
}

# Let's test different headers and path variations!
tests = [
    # Test 1: Direct POST to URL with Bearer Authorization
    {
        "url": url_base,
        "headers": {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    },
    # Test 2: POST to /chat/completions with Bearer Authorization
    {
        "url": f"{url_base}/chat/completions",
        "headers": {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    },
    # Test 3: Direct POST to URL with api-key header (common in Azure)
    {
        "url": url_base,
        "headers": {"api-key": key, "Content-Type": "application/json"}
    },
    # Test 4: POST to /chat/completions with api-key header
    {
        "url": f"{url_base}/chat/completions",
        "headers": {"api-key": key, "Content-Type": "application/json"}
    }
]

for idx, t in enumerate(tests, 1):
    print(f"\n--- Running Test {idx} ---")
    print("URL:", t["url"])
    print("Headers keys:", list(t["headers"].keys()))
    try:
        res = requests.post(t["url"], headers=t["headers"], json=payload, timeout=10)
        print("Status:", res.status_code)
        try:
            print("Response:", res.json())
        except:
            print("Response text:", res.text[:200])
    except Exception as e:
        print("Error:", e)
