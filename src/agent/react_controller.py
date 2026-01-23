from typing import Dict, Any, List
import os
import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from src.agent.state import AgentState
from src.agent.mock_llm import SimpleMockLLM

class ReActController:
    def __init__(self, tools_map: Dict[str, Any]):
        self.tools = tools_map
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.llmod.ai/v1")
        model_name = "RPRTHPB-gpt-5-mini"

        if api_key:
            print(f"Using LLMod.ai Model: {model_name}")
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=1,
                api_key=api_key,
                base_url=base_url
            )
        else:
            print("WARNING: OPENAI_API_KEY not found. Using SimpleMockLLM (Generic).")
            self.llm = SimpleMockLLM()

        try:
            self.max_steps = int(os.environ.get("MAX_REACT_STEPS", 6))
        except (ValueError, TypeError):
            self.max_steps = 6
        self.decision_parser = JsonOutputParser()
        self.decision_prompt = self._build_decision_prompt()
        self.deep_analysis_prompt = self._build_deep_analysis_prompt()
        self.tool_specs = self._build_tool_specs()
        self.tool_names = [tool["name"] for tool in self.tool_specs]

    def _build_tool_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "semantic_search",
                "description": "Retrieve similar historical narratives from the vector store.",
                "input_schema": {
                    "query": "string",
                    "k": "integer (optional, default 5)"
                }
            },
            {
                "name": "structured_filter",
                "description": "Filter structured ASRS records by aircraft model, location, and date range.",
                "input_schema": {
                    "make_model": "string (optional)",
                    "location": "string (optional)",
                    "date_start": "YYYY-MM-DD (optional)",
                    "date_end": "YYYY-MM-DD (optional)"
                }
            },
            {
                "name": "trend_analyzer",
                "description": "Detect report frequency anomalies over time for the filtered subset.",
                "input_schema": {
                    "use_filtered": "boolean (optional, default true)",
                    "time_col": "string (optional, default Event_Date)",
                    "metric_col": "string (optional, default ACN)"
                }
            },
            {
                "name": "deep_analysis",
                "description": "LLM-based causal analysis that synthesizes current evidence.",
                "input_schema": {
                    "focus": "string (optional)"
                }
            },
            {
                "name": "final",
                "description": "Stop tool use and hand off to report synthesis.",
                "input_schema": {}
            }
        ]

    def _build_decision_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are an autonomous safety investigator using a ReAct loop. "
             "Decide the next action based on the report, entities, and observations. "
             "Return ONLY a JSON object.\n"
             "{format_instructions}\n"
             "Rules:\n"
             "- decision must be 'act' or 'final'.\n"
             "- reasoning_summary must be 1-2 sentences, no chain-of-thought.\n"
             "- When decision=='act', set action to a tool name and provide action_input.\n"
             "- When decision=='final', set action to '' and provide a short final directive.\n"
            ),
            ("user",
             "Input report:\n{input_report}\n\n"
             "Extracted entities:\n{entities}\n\n"
             "Findings so far:\n{findings}\n\n"
             "Last observation:\n{last_observation}\n\n"
             "Tool history:\n{tool_history}\n\n"
             "Available tools:\n{tool_specs}\n\n"
             "Step: {step_count}/{max_steps}"
            )
        ])

    def _build_deep_analysis_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are an aviation safety analyst. Provide a short causal analysis (3-5 sentences). "
             "Use evidence from findings and observations. Do not reveal chain-of-thought."
            ),
            ("user",
             "Input report:\n{input_report}\n\n"
             "Extracted entities:\n{entities}\n\n"
             "Findings:\n{findings}\n\n"
             "Last observation:\n{last_observation}\n\n"
             "Focus:\n{focus}"
            )
        ])

    def extract_entities(self, state: AgentState):
        print("--- Entity Extraction ---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an aviation safety expert. Extract key entities from the safety report.

Return your response as a JSON object with the following fields:
{{
  "Aircraft Model": "the aircraft type/model",
  "Location": "airport code or location name",
  "Event Type": "type of anomaly or incident",
  "Flight Phase": "takeoff, cruise, descent, landing, etc.",
  "Keywords": ["list", "of", "relevant", "safety", "keywords"]
}}

Only return the JSON object, no additional text."""),
            ("user", "{report}")
        ])

        chain = prompt | self.llm

        try:
            res = chain.invoke({"report": state["input_report"]})
            content = res if isinstance(res, str) else res.content
            if isinstance(content, str):
                try:
                    entities = json.loads(content)
                except json.JSONDecodeError:
                    entities = {"RawExtraction": content}
            else:
                entities = content
        except Exception as e:
            print(f"Entity extraction failed: {e}")
            entities = {"Aircraft Model": "Unknown", "Location": "Unknown"}

        try:
            system_msg = prompt.messages[0].prompt.template
            user_msg = state["input_report"]
            full_prompt = f"System: {system_msg}\n\nUser: {user_msg}"
        except (AttributeError, IndexError):
            full_prompt = f"Report: {state['input_report']}"

        step_log = {
            "module": "Entity Extraction",
            "prompt": full_prompt,
            "response": entities
        }

        return {"extracted_entities": entities, "steps_trace": [step_log]}

    def decide_next(self, state: AgentState):
        print("--- ReAct Decision ---")
        prompt_vars = {
            "format_instructions": self.decision_parser.get_format_instructions(),
            "input_report": state.get("input_report", ""),
            "entities": state.get("extracted_entities", {}),
            "findings": state.get("findings", []),
            "last_observation": state.get("last_observation"),
            "tool_history": state.get("tool_history", []),
            "tool_specs": self.tool_specs,
            "step_count": state.get("step_count", 0),
            "max_steps": state.get("max_steps", self.max_steps)
        }

        chain = self.decision_prompt | self.llm | self.decision_parser

        try:
            decision = chain.invoke(prompt_vars)
        except Exception as e:
            print(f"Decision parse failed: {e}")
            decision = self._fallback_decision(state)

        decision = self._normalize_decision(decision, state)

        step_log = {
            "module": "ReAct Decision",
            "prompt": prompt_vars,
            "response": decision
        }

        return {"react_decision": decision, "steps_trace": [step_log]}

    def route_decision(self, state: AgentState):
        decision = state.get("react_decision", {})
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", self.max_steps)

        if step_count >= max_steps:
            return "final"

        if decision.get("decision") == "act":
            return "act"

        return "final"

    def execute_action(self, state: AgentState):
        print("--- ReAct Action ---")
        decision = state.get("react_decision", {})
        action = decision.get("action", "")
        action_input = decision.get("action_input", {})
        if not isinstance(action_input, dict):
            action_input = {}

        findings = list(state.get("findings", []))
        tool_history = list(state.get("tool_history", []))
        df_context = state.get("df_context")
        observation = None
        response_payload = None

        if action == "semantic_search":
            query = action_input.get("query") or self._build_semantic_query(
                state.get("extracted_entities", {}), state.get("input_report", "")
            )
            try:
                k = int(action_input.get("k", 5))
            except (ValueError, TypeError):
                k = 5
            results = self.tools["semantic_search"].search(query, k=k)
            observation = self._summarize_semantic_results(results)
            response_payload = {"query": query, "results": observation}
            findings.append(f"Semantic Search: {len(results)} similar reports found.")

        elif action == "structured_filter":
            filters = self._build_filters(action_input, state.get("extracted_entities", {}))
            if filters:
                df_res = self.tools["filtering"].filter_data(filters)
                df_context = df_res
                msg_suffix = ""
                observation = self._summarize_df(df_res)
            else:
                # No filters: keep FULL context for downstream tools (e.g. Trend Analyzer)
                # but only show a sample in the observation to the LLM.
                df_res = self.tools["filtering"].df
                df_context = df_res
                sample_df = df_res.head(100)
                summary = self._summarize_df(sample_df)
                # Overwrite count to be explicit about the difference
                observation = {
                    "count": f"{len(df_res)} (full dataset)",
                    "sample_preview_count": 100,
                    "columns": summary["columns"],
                    "top_operators": summary.get("top_operators", {})
                }
                msg_suffix = " (No filters provided. Context set to full dataset. Showing top 100 sample in observation.)"
            
            response_payload = {"filters": filters, "summary": observation}
            findings.append(f"Structured Filter: {len(df_res)} records available{msg_suffix}.")

        elif action == "trend_analyzer":
            use_filtered = action_input.get("use_filtered", True)
            time_col = action_input.get("time_col", "Event_Date")
            metric_col = action_input.get("metric_col", "ACN")
            df_to_use = df_context if (use_filtered and df_context is not None) else self.tools["trend_analyzer"].df
            observation = self.tools["trend_analyzer"].detect_anomalies(df_to_use, time_col=time_col, metric_col=metric_col)
            response_payload = {"trend": observation}
            findings.append(f"Trend Analysis: {observation}")

        elif action == "deep_analysis":
            focus = action_input.get("focus", "")
            observation = self._deep_analysis(state, focus=focus)
            response_payload = {"analysis": observation}
            findings.append(f"Deep Analysis: {observation}")

        else:
            observation = f"Unknown action: {action}"
            response_payload = {"error": observation}
            findings.append(observation)

        tool_history.append(action)
        step_count = state.get("step_count", 0) + 1

        react_step = {
            "step": step_count,
            "reasoning_summary": decision.get("reasoning_summary", ""),
            "action": action,
            "action_input": action_input,
            "observation": observation
        }

        # Use Title Case for module name to match Architecture Diagram (e.g. "Semantic Search")
        module_name = action.replace("_", " ").title() if action else "Observation"
        
        step_log = {
            "module": module_name,
            "prompt": action_input,
            "response": response_payload
        }

        return {
            "findings": findings,
            "react_steps": [react_step],
            "steps_trace": [step_log],
            "tool_history": tool_history,
            "df_context": df_context,
            "step_count": step_count,
            "last_observation": observation
        }

    def _deep_analysis(self, state: AgentState, focus: str = "") -> str:
        chain = self.deep_analysis_prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({
                "input_report": state.get("input_report", ""),
                "entities": state.get("extracted_entities", {}),
                "findings": state.get("findings", []),
                "last_observation": state.get("last_observation", ""),
                "focus": focus
            })
        except Exception as e:
            return f"Deep analysis failed: {e}"

    def _build_semantic_query(self, entities: Dict[str, Any], input_report: str) -> str:
        def is_valid(val: Any) -> bool:
            if val is None:
                return False
            val_str = str(val).strip()
            return val_str and val_str.lower() not in ["unknown", "unknown aircraft", "unknown location", ""]

        query_parts = []
        for key in ["Aircraft Model", "Event Type", "Location", "Flight Phase"]:
            val = entities.get(key, "")
            if is_valid(val):
                query_parts.append(str(val))

        return " ".join(query_parts) if query_parts else input_report[:200]

    def _build_filters(self, action_input: Dict[str, Any], entities: Dict[str, Any]) -> Dict[str, Any]:
        def is_valid_filter(val: Any) -> bool:
            """Check if a value is a valid (non-placeholder) filter value."""
            if val is None:
                return False
            val_str = str(val).strip().lower()
            placeholders = ["unknown", "unknown aircraft", "unknown location", "n/a", "none", ""]
            return val_str not in placeholders
        
        filters = {}
        make_model = action_input.get("make_model") or entities.get("Aircraft Model")
        location = action_input.get("location") or entities.get("Location")
        date_start = action_input.get("date_start")
        date_end = action_input.get("date_end")

        if is_valid_filter(make_model):
            filters["Make_Model"] = make_model
        if is_valid_filter(location):
            filters["Airport"] = location
        if date_start:
            filters["Date_Start"] = date_start
        if date_end:
            filters["Date_End"] = date_end

        return filters

    def _summarize_semantic_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary = []
        for r in results:
            summary.append({
                "ACN": r.get("ACN"),
                "score": r.get("score"),
                "snippet": (r.get("content") or "")[:200],
                "metadata": r.get("metadata", {})
            })
        return summary

    def _summarize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        summary = {
            "count": int(len(df)),
            "columns": list(df.columns)[:10]
        }
        if len(df) > 0 and "Operator" in df.columns:
            summary["top_operators"] = df["Operator"].value_counts().head(3).to_dict()
        return summary

    def _fallback_decision(self, state: AgentState) -> Dict[str, Any]:
        history = set(state.get("tool_history", []))
        if "semantic_search" not in history:
            return {
                "decision": "act",
                "reasoning_summary": "Need similar historical cases before drawing conclusions.",
                "action": "semantic_search",
                "action_input": {"query": self._build_semantic_query(state.get("extracted_entities", {}), state.get("input_report", "")), "k": 5},
                "final": ""
            }
        if "structured_filter" not in history:
            return {
                "decision": "act",
                "reasoning_summary": "Structured filtering can surface correlations by model and location.",
                "action": "structured_filter",
                "action_input": {},
                "final": ""
            }
        if "trend_analyzer" not in history:
            return {
                "decision": "act",
                "reasoning_summary": "Trend analysis checks for abnormal reporting spikes.",
                "action": "trend_analyzer",
                "action_input": {"use_filtered": True},
                "final": ""
            }
        if "deep_analysis" not in history:
            return {
                "decision": "act",
                "reasoning_summary": "A concise causal analysis ties observations to likely root causes.",
                "action": "deep_analysis",
                "action_input": {},
                "final": ""
            }

        return {
            "decision": "final",
            "reasoning_summary": "Enough evidence collected to synthesize the report.",
            "action": "",
            "action_input": {},
            "final": "Synthesize a full RCA report using all findings and observations."
        }

    def _normalize_decision(self, decision: Any, state: AgentState) -> Dict[str, Any]:
        if not isinstance(decision, dict):
            decision = {}

        decision.setdefault("decision", "act")
        decision.setdefault("reasoning_summary", "")
        decision.setdefault("action", "")
        decision.setdefault("action_input", {})
        decision.setdefault("final", "")

        if decision["decision"] == "final":
            decision["action"] = ""
            decision["action_input"] = {}
            if not decision["final"]:
                decision["final"] = "Synthesize the final RCA report from findings."
            return decision

        if decision["decision"] == "act":
            if decision["action"] == "final":
                 decision["decision"] = "final"
                 decision["action"] = ""
                 return self._normalize_decision(decision, state)
            
            if decision["action"] not in self.tool_names:
                return self._fallback_decision(state)

        return decision
