import pytest
from pydantic import ValidationError
from src.server import ExecuteResponse, Step

def test_execute_response_strictness():
    """
    Verify that ExecuteResponse enforces the strict schema required by the PDF.
    Required keys: status, error, response, steps.
    """
    # Valid Case
    valid_data = {
        "status": "ok",
        "error": None,
        "response": "Test response",
        "steps": [
            {"module": "Test", "prompt": {"input": "p"}, "response": {"output": "r"}}
        ]
    }
    model = ExecuteResponse(**valid_data)
    assert model.status == "ok"
    assert len(model.steps) == 1

def test_steps_structure():
    """
    Verify that 'steps' must be a list of objects, not strings.
    """
    # Invalid Case: Steps as strings
    invalid_data = {
        "status": "ok",
        "steps": ["Step 1", "Step 2"] # Should fail
    }
    
    # We expect pydantic to raise an error or try to parse (and fail if strict)
    # Since Step is a model, passing a string might fail validation
    with pytest.raises(ValidationError):
        ExecuteResponse(**invalid_data)

def test_step_required_fields():
    """
    Verify Step model requires module, prompt, response.
    """
    with pytest.raises(ValidationError):
        Step(module="Missing fields")
