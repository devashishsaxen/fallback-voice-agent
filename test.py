import os

import requests


api_key = os.getenv("XAI_API_KEY")
if not api_key:
    raise SystemExit("Set XAI_API_KEY before running this script.")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

payload = {
    "messages": [
        {"role": "system", "content": "You are a test assistant."},
        {
            "role": "user",
            "content": "Testing. Just say hi and hello world and nothing else.",
        },
    ],
    "model": "grok-4-latest",
    "stream": False,
    "temperature": 0,
}

response = requests.post(
    "https://api.x.ai/v1/chat/completions",
    headers=headers,
    json=payload,
    timeout=30,
)
response.raise_for_status()
data = response.json()
print(data["choices"][0]["message"]["content"])
