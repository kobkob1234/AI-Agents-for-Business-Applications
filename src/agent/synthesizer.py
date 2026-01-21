from langchain_openai import ChatOpenAI
from langchain_community.llms import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.agent.state import AgentState
import os

class Synthesizer:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.llmod.ai/v1")
        # Requirement: Use specific model
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
            ("user", "Input Report: {input_report}\n\nExtracted Entities: {entities}\n\nTool Findings: {findings}")
        ])
        
        # Consistent chain for both
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            # If chain is LLM (Mock), invoke with prompt string or simple input
            if isinstance(chain, FakeListLLM):
                report = chain.invoke("dummy prompt")
            else:
                report = chain.invoke({
                    "input_report": state['input_report'],
                    "entities": state['extracted_entities'],
                    "findings": state['findings']
                })
        except Exception as e:
            report = f"Failed to generate report: {e}"
            
        step_log = {
            "module": "Report Generation",
            "prompt": {
                "input_report": state['input_report'],  # Full input for complete trace
                "entities": state['extracted_entities'],
                "findings": state['findings']
            },
            "response": report  # Full response per spec requirement
        }
        
        return {"final_report": report, "steps_trace": [step_log]}
