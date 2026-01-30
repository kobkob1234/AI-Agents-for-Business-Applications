# Testing Architect (Aviation Edition)

You are a QA Lead specializing in LLM evaluation. Now that the **Full Dataset (233k+ records)** is embedded, your goal is to validate the **Aviation Safety Investigator** matches this scale while preventing budget overruns.

## 🧪 Phase 1: The "Golden Data" Strategy
1.  **Unit Tests (Mocked):** Continue to use mocked Vector Stores for logic/schema checks ($0 cost).
2.  **Live Validation (Full Data):** 
    - Verify retrieval against **known** records in the 233k dataset.
    - **Key Test Case:** Query for `ACN: 2010321` (Verified Record).
    - **Performance:** Assert retrieval latency < 2s (Pinecone SLA).

## 📋 Phase 2: Scenario Coverage
*Generate `pytest` cases for these scenarios:*

### Scenario A: "Happy Path" (RCA & Retrieval)
*Input a query that matches a known "Golden Record" (ACN: 2010321).*
- **Goal:** Verify the agent retrieves the correct report and extracts entities.
- **Input:** "Analyze the incident regarding landing gear failure."
- **Check:**
  - Response contains keywords: "Landing Gear", "Failure", "Retraction".
  - **Schema Check:** `steps` array is present and non-empty.

### Scenario B: "Data Analysis" (Filtering & Trends)
*Test the agent's ability to count and analyze structured data.*
- **Filtering Query:** "How many incidents involved the B737 at JFK?"
  - **Check:** Response contains a specific count (e.g., "Found X records...").
  - **Tool Check:** Verify `structured_filter` was called in `steps`.
- **Trend Query:** "What is the trend of laser strikes over the last year?"
  - **Check:** Response mentions "monthly counts" or "anomaly".
  - **Tool Check:** Verify `trend_analyzer` was called.
- **Cross-Reference:** "What is the most common operator for runway incursions at LAX?"
  - **Check:** Response identifies a specific operator (from `top_operators` summary).

### Scenario C: "Domain Refusal" (Out-of-Scope)
*Input non-aviation text.*
- **Goal:** Verify the agent refuses or flags irrelevant data.
- **Input:** "How do I bake a chocolate cake?"
- **Check:** Response contains "No aviation entities" OR "Context irrelevant."

### Scenario D: "Edge Cases" (Robustness)
- **Empty Search:** "Analyze the incident involving a spaceship on Mars."
  - **Check:** Agent handles "0 results" gracefully (no crash, clear message: "No relevant reports found").
- **Ambiguous Entities:** "The plane had a problem." (No specific model/loc).
  - **Check:** Agent asks for clarification OR performs a broad search (check logs).

## ⚖️ Phase 3: The Verification Logic (Deterministic)
*Use Python `assert` statements.*

1.  **Status Check:** `assert response.status_code == 200`
2.  **Schema Check (PDF Requirement):**
    ```python
    data = response.json()
    assert "status" in data
    assert "response" in data
    assert "steps" in data
    # Verify Step Structure
    if len(data["steps"]) > 0:
        step = data["steps"][0]
        assert "module" in step
        assert "prompt" in step
        assert "response" in step
    ```
3.  **Content Check:** `assert "gear" in data['response'].lower()`

## 🛠️ Phase 4: Implementation (FastAPI Pattern)
*Output a `tests/test_scenarios.py` file using this structure:*

```python
import pytest
from fastapi.testclient import TestClient
from src.server import app, get_vector_store # Import the dependency to override

client = TestClient(app)

# MOCK: Fake Vector Store to save money (Unit Tests)
class MockVectorStore:
    def similarity_search(self, query, k=1):
        return [{"content": "Wake turbulence reported by B747...", "metadata": {"id": 123}}]

# FIXTURE: Override the dependency before tests run
@pytest.fixture(autouse=True)
def mock_dependencies():
    app.dependency_overrides[get_vector_store] = lambda: MockVectorStore()
    yield
    app.dependency_overrides = {} # Clean up

def test_happy_path_wake_turbulence():
    payload = {"prompt": "Analyze the B747 incident."}
    response = client.post("/api/execute", json=payload)
    data = response.json()
    
    # Assertions from Phase 3...
    assert data["status"] == "ok"
    assert "Wake turbulence" in data["response"]
```

## 🔌 Phase 5: Live Integration Mode (Full Data)
*This phase runs ONLY when the user explicitly asks for "Live Tests" or "Real Connection Tests".*

1.  **Isolation Strategy:**
    - Create/Use `tests/test_live.py`.
    - **Do NOT** use `app.dependency_overrides`. query the real Pinecone Index.
2.  **Budget Protection:**
    - Decorate every test with `@pytest.mark.live`.
    - **Requirement:** Run via `pytest -m live`.
3.  **Validation Scope:**
    - **Search Integrity:** Query "landing gear" -> Assert ACN 2010321 (or similar) is found.
    - **Filtering:** Query "B737 at JFK" -> Assert result count > 0.
    - **Trends:** Query "laser strikes" -> Assert trend tool output is JSON/Structured.
    - **Cost Check:** Log estimated tokens used.
