import os, requests
from dotenv import load_dotenv
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_BASE_URL").rstrip("/")
api_key  = os.getenv("AZURE_OPENAI_API_KEY")

# Common deployment names teams use
candidates = [
    "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo",
    "gpt-35-turbo", "gpt-35-turbo-16k",
    "gpt4o", "gpt4", "gpt-4.1",
    "veserve-gpt4o", "veserve-gpt4", "chat", "default"
]

for name in candidates:
    url = f"{endpoint}/openai/deployments/{name}/chat/completions?api-version=2024-08-01-preview"
    r = requests.post(url, headers={"api-key": api_key, "Content-Type": "application/json"},
                      json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    status = "OK" if r.status_code == 200 else f"FAIL {r.status_code}"
    print(f"{name:30s} -> {status}")
