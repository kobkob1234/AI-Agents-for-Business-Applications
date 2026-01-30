
import pytest
import httpx
import os
import json

# Target URL provided by user
BASE_URL = os.environ.get("ASI_API_URL", "https://asi-agent-s5nr.onrender.com")

def test_remote_happy_path_e2e():
    """
    Live Remote Integration Test:
    Runs a full E2E query against the DEPLOYED agent.
    """
    print(f"\n---------------------------------------------------")
    print(f" टारGETING REMOTE URL: {BASE_URL}")
    print(f"---------------------------------------------------\n")

    input_prompt = "Analyze the incident involving the Falcon 7X and B777 at ZZZ."
    payload = {"prompt": input_prompt}
    
    # Increase timeout to 300s to handle Cold Starts + ReAct Loop
    timeout = httpx.Timeout(300.0, connect=60.0)
    
    try:
        with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
            # 1. Execute
            response = client.post("/api/execute", json=payload)
            
            # 2. Extract Data
            print(f"Status Code: {response.status_code}")
            try:
                data = response.json()
                print(f"Response Body Preview: {str(data)[:200]}...")
            except json.JSONDecodeError:
                print(f"Raw Response: {response.text[:200]}")
                data = {}

            # 3. Assertions
            if response.status_code != 200:
                pytest.fail(f"Remote API failed with {response.status_code}: {response.text}")
                
            assert data.get("status") == "ok"
            assert "steps" in data
            assert len(data["steps"]) > 0
            
            # Verify we got a report
            report = data.get("response")
            assert report is not None
            assert len(report) > 50 

    except httpx.RequestError as exc:
        pytest.fail(f"An error occurred while requesting {exc.request.url!r}: {exc}")
