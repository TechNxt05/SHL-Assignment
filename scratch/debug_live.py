import httpx
import json

url = "https://shl-assignment-ev9j.onrender.com/chat"
payload = {
    "messages": [
        {"role": "user", "content": "I need to hire a Java developer and an Accountant. What tests do you have?"}
    ]
}

try:
    print("Sending request...")
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Content: {response.text}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
