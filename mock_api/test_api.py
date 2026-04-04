import json
import os

import requests

BASE = os.environ.get("GIGSHIELD_MOCK_API_BASE", "http://127.0.0.1:8000").rstrip("/")

def test(endpoint):
    print(f"\nTesting {endpoint}")
    res = requests.get(BASE + endpoint)
    print("Status:", res.status_code)
    print(json.dumps(res.json(), indent=2))

def run_tests():
    test("/api/weather?city=Chennai&scenario=flood")
    test("/api/weather?city=Mumbai")
    test("/api/news?city=Delhi&scenario=strike")
    test("/api/news?city=Bengaluru")
    test("/api/telecom?city=Hyderabad")
    test("/api/fuel?city=Pune")
    test("/api/platform")

    test("/api/news?city=Delhi&scenario=curfew")
    test("/api/news?city=Mumbai&scenario=protest")
    test("/api/news?limit=10")  # mixed multi-city

    print("\nTesting POST /api/claims/rollout")
    r = requests.post(
        f"{BASE}/api/claims/rollout",
        json={"worker_id": "w-demo", "decision": "rollout_test"},
        timeout=30,
    )
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    run_tests()