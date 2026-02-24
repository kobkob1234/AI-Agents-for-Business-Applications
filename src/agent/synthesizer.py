from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.agent.state import AgentState
import os

class Synthesizer:
    def __init__(self):
        app_env = os.environ.get("APP_ENV", "").strip().lower()
        strict_flag = os.environ.get("REQUIRE_STRICT_STACK", "").strip().lower()
        strict_mode = app_env in {"prod", "production"} or strict_flag in {"1", "true", "yes"}

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.llmod.ai/v1")
        model_name = "RPRTHPB-gpt-5-mini"

        if strict_mode and "llmod.ai" not in base_url:
            raise RuntimeError("Strict mode: OPENAI_BASE_URL must point to LLMod.ai")

        if api_key:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=1,
                api_key=api_key,
                base_url=base_url
            )
        else:
            if strict_mode:
                raise RuntimeError("Strict mode: OPENAI_API_KEY is required")
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

Be concise but thorough. Focus on safety implications.
Use ONLY the evidence categories and ACNs provided. Do not introduce risks that are not supported by the evidence map.
If evidence is insufficient, say so."""),
            ("user", "Input Report: {input_report}\n\nExtracted Entities: {entities}\n\nEvidence Categories: {evidence_risks}\n\nEvidence Map (risk -> ACNs): {evidence_map}\n\nReAct Trace (reasoning summaries + observations): {react_trace}\n\nTool Findings: {findings}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        report_input = state.get("input_report_trimmed") or state["input_report"]
        entities = state.get("extracted_entities", {})
        entities_safe = {k: v for k, v in entities.items() if k != "Keywords"}
        try:
            report = chain.invoke({
                "input_report": report_input,
                "entities": entities_safe,
                "evidence_risks": state.get("evidence_risks", []),
                "evidence_map": state.get("evidence_map", {}),
                "react_trace": state.get("react_steps", []),
                "findings": state["findings"]
            })
        except Exception as e:
            report = f"Failed to generate report: {e}"

        try:
            system_msg = prompt.messages[0].prompt.template
            user_msg = f"Input Report: {report_input}\n\nExtracted Entities: {entities_safe}\n\nEvidence Categories: {state.get('evidence_risks', [])}\n\nEvidence Map (risk -> ACNs): {state.get('evidence_map', {})}\n\nReAct Trace (reasoning summaries + observations): {state.get('react_steps', [])}\n\nTool Findings: {state['findings']}"
            prompt_payload = {
                "system": system_msg,
                "user": user_msg
            }
        except (AttributeError, IndexError):
            prompt_payload = {"error": "Prompt formatting failed."}

        step_log = {
            "module": "SYNTHESIZER",
            "prompt": prompt_payload,
            "response": {"report": report}
        }

        return {"final_report": report, "steps_trace": [step_log]}
