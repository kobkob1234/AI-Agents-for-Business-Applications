from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.agent.state import AgentState
import os

class Synthesizer:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.llmod.ai/v1")
        model_name = "RPRTHPB-gpt-5-mini"

        if api_key:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=1,
                api_key=api_key,
                base_url=base_url
            )
        else:
            from src.agent.mock_llm import SimpleMockLLM
            self.llm = SimpleMockLLM()

    def generate_report(self, state: AgentState):
        print("--- Generating Report ---")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an ASI (Autonomous Safety Investigator). Generate a comprehensive Root Cause Analysis (RCA) report.

Structure your report with these sections in markdown format:
## Executive Summary
Brief overview of the incident and primary findings.

## Historical Corroboration
Analysis of similar past incidents from the ASRS database.

## Trend Analysis
Statistical patterns and anomaly detection results.

## Cross-Reference Findings
Correlations with operators, maintenance, and organizational factors.

## Root Cause Assessment
Primary and contributing causes identified.

## Recommendations
Actionable safety recommendations based on the analysis.

Be concise but thorough. Focus on safety implications."""),
            ("user", "Input Report: {input_report}\n\nExtracted Entities: {entities}\n\nReAct Trace (reasoning summaries + observations): {react_trace}\n\nTool Findings: {findings}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            report = chain.invoke({
                "input_report": state["input_report"],
                "entities": state["extracted_entities"],
                "react_trace": state.get("react_steps", []),
                "findings": state["findings"]
            })
        except Exception as e:
            report = f"Failed to generate report: {e}"

        try:
            system_msg = prompt.messages[0].prompt.template
            user_msg = f"Input Report: {state['input_report']}\n\nExtracted Entities: {state['extracted_entities']}\n\nReAct Trace (reasoning summaries + observations): {state.get('react_steps', [])}\n\nTool Findings: {state['findings']}"
            full_prompt = f"System: {system_msg}\n\nUser: {user_msg}"
        except (AttributeError, IndexError):
            full_prompt = "Prompt formatting failed."

        step_log = {
            "module": "Report Generation",
            "prompt": full_prompt,
            "response": report
        }

        return {"final_report": report, "steps_trace": [step_log]}
