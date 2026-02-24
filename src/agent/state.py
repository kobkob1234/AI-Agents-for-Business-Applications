from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage

class StepLog(TypedDict):
    module: str
    prompt: Dict[str, Any]
    response: Dict[str, Any]

class ReActStep(TypedDict):
    step: int
    reasoning_summary: str
    action: str
    action_input: Dict[str, Any]
    observation: Any

class AgentState(TypedDict):
    input_report: str
    input_report_trimmed: str
    extracted_entities: Dict[str, Any]
    evidence_risks: List[str]
    evidence_map: Dict[str, Any]
    messages: Annotated[List[BaseMessage], operator.add]
    findings: List[str]
    final_report: str
    steps_trace: Annotated[List[StepLog], operator.add]

    # ReAct-specific fields
    react_steps: Annotated[List[ReActStep], operator.add]
    react_decision: Dict[str, Any]
    tool_history: List[str]
    df_context: Any
    step_count: int
    max_steps: int
    last_observation: Any

    # Optimization guardrails
    last_action_fingerprint: str
    repeated_action_count: int
    duplicate_guard_triggered: bool
    fast_track_ready: bool
    fast_track_reason: str
