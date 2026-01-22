from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
import json
import re

class SimpleMockLLM(LLM):
    """
    A simple rule-based mock LLM that generates responses based on input keywords.
    This ensures the agent can be tested with ANY input, not just the hardcoded example.
    """

    @property
    def _llm_type(self) -> str:
        return "simple_mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if "Extract key entities" in prompt:
            return self._mock_entity_extraction(prompt)

        if "Root Cause Analysis (RCA) report" in prompt:
            return self._mock_report_generation(prompt)

        # ReAct decision mock
        if "ReAct loop" in prompt or ("Available tools" in prompt and "decision" in prompt):
            step_match = re.search(r"Step:\s*(\d+)\s*/\s*(\d+)", prompt)
            if step_match:
                step = int(step_match.group(1))
                max_steps = int(step_match.group(2))
                if step >= max_steps - 1:
                    return json.dumps({
                        "decision": "final",
                        "reasoning_summary": "Sufficient evidence gathered to produce the report.",
                        "action": "",
                        "action_input": {},
                        "final": "Synthesize a full RCA report from findings."
                    })

            return json.dumps({
                "decision": "act",
                "reasoning_summary": "Need corroborating historical evidence before final conclusions.",
                "action": "semantic_search",
                "action_input": {"query": "aviation incident", "k": 5},
                "final": ""
            })

        return "Mock response"

    def _mock_entity_extraction(self, prompt: str) -> str:
        entities = {}

        aircraft_match = re.search(r'\b([AB]\d{3}[a-zA-Z0-9\-\s]*)\b', prompt)
        if aircraft_match:
            entities["Aircraft Model"] = aircraft_match.group(1).strip()
        else:
            entities["Aircraft Model"] = "Unknown Aircraft"

        loc_match = re.search(r'\b([A-Z]{3})\b', prompt)
        excluded = ["THE", "AND", "FOR", "MAX", "NEO", "JET", "AIR", "FAA", "ATC", "VFR", "IFR", "GPS", "ILS", "RCA"]
        if loc_match and loc_match.group(1) not in excluded:
            entities["Location"] = loc_match.group(1)
        else:
            entities["Location"] = "Unknown Location"

        keywords = re.findall(r'\b[A-Z][a-z]+\b', prompt)
        common = ["The", "And", "For", "With", "From", "This", "Report", "Input", "Entities", "Findings", "You", "Are", "Extract"]
        keywords = [k for k in keywords if k not in common]

        entities["Event Type"] = keywords[0] if keywords else "Incident"
        entities["Keywords"] = keywords

        return json.dumps(entities)

    def _mock_report_generation(self, prompt: str) -> str:
        return """**Autonomous Safety Investigation Report (MOCK)**

**Executive Summary**
This is a generated report based on the provided input. The agent identified key entities and attempted to find corroborating evidence.

**Findings**
- **Analysis**: The system processed the input using rule-based mock logic (due to missing API Key).
- **Data**: Semantic search was simulated.

**Conclusion**
This confirms the architecture handles the input genericly. The specific details would be populated by the LLM in a production environment.
"""
