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
        # Check if we are doing Entity Extraction (JSON output expected)
        if "Extract key entities" in prompt:
            return self._mock_entity_extraction(prompt)
        
        # Check if we are doing Synthesis (Report output expected)
        if "Root Cause Analysis (RCA) report" in prompt:
            return self._mock_report_generation(prompt)
            
        return "Mock response"

    def _mock_entity_extraction(self, prompt: str) -> str:
        # Extract the user input part (usually at the end)
        # We'll just look for obvious patterns in the whole prompt for simplicity
        
        entities = {}
        
        # Heuristic: Look for "B7..." or "A3.." specific aircraft
        aircraft_match = re.search(r'\b([AB]\d{3}[a-zA-Z0-9\-\s]*)\b', prompt)
        if aircraft_match:
            entities["Aircraft Model"] = aircraft_match.group(1).strip()
        else:
            entities["Aircraft Model"] = "Unknown Aircraft"
            
        # Heuristic: Look for 3-letter airport codes
        loc_match = re.search(r'\b([A-Z]{3})\b', prompt)
        if loc_match and loc_match.group(1) not in ["THE", "AND", "FOR"]:
            entities["Location"] = loc_match.group(1)
        else:
            entities["Location"] = "Unknown Location"
            
        # Heuristic: Find something that looks like an event or keyword
        # Just grab capitalized words as keywords
        keywords = re.findall(r'\b[A-Z][a-z]+\b', prompt)
        # Filter out common words
        common = ["The", "And", "For", "With", "From", "This", "Report", "Input", "Entities", "Findings", "You", "Are", "Extract"]
        keywords = [k for k in keywords if k not in common]
        
        entities["Event Type"] = keywords[0] if keywords else "Incident"
        entities["Keywords"] = keywords
        
        return json.dumps(entities)

    def _mock_report_generation(self, prompt: str) -> str:
        # Parse out the inputs that were injected into the prompt
        # Prompt usually contains "Input Report: ... Entities: ... Findings: ..."
        
        # We'll just generate a template response that reflects the content
        
        return """**Autonomous Safety Investigation Report (MOCK)**

**Executive Summary**
This is a generated report based on the provided input. The agent identified key entities and attempted to find corroborating evidence.

**Findings**
- **Analysis**: The system processed the input using rule-based mock logic (due to missing API Key).
- **Data**: Semantic search was simulated.

**Conclusion**
This confirms the architecture handles the input genericly. The specific details would be populated by the LLM in a production environment.
"""
