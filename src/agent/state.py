from typing import TypedDict, Annotated, List, Union, Dict, Any
import operator
from langchain_core.messages import BaseMessage

class StepLog(TypedDict):
    module: str
    prompt: Any
    response: Any

class AgentState(TypedDict):
    input_report: str
    extracted_entities: dict
    plan: List[str]
    messages: Annotated[List[BaseMessage], operator.add]
    findings: List[str]
    final_report: str
    # Add steps_trace to track API requirements
    steps_trace: Annotated[List[StepLog], operator.add]
