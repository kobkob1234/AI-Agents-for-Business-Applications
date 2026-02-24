from typing import Dict, Any, List, Tuple
import os
import json
import hashlib
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from src.agent.state import AgentState
from src.agent.mock_llm import SimpleMockLLM

class ReActController:
    def __init__(self, tools_map: Dict[str, Any]):
        self.tools = tools_map

        app_env = os.environ.get("APP_ENV", "").strip().lower()
        strict_flag = os.environ.get("REQUIRE_STRICT_STACK", "").strip().lower()
        self.strict_mode = app_env in {"prod", "production"} or strict_flag in {"1", "true", "yes"}

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.llmod.ai/v1")
        model_name = "RPRTHPB-gpt-5-mini"

        if self.strict_mode and "llmod.ai" not in base_url:
            raise RuntimeError("Strict mode: OPENAI_BASE_URL must point to LLMod.ai")

        if api_key:
            print(f"Using LLMod.ai Model: {model_name}")
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=1,
                api_key=api_key,
                base_url=base_url
            )
        else:
            if self.strict_mode:
                raise RuntimeError("Strict mode: OPENAI_API_KEY is required")
            print("WARNING: OPENAI_API_KEY not found. Using SimpleMockLLM (Generic).")
            self.llm = SimpleMockLLM()

        try:
            self.max_steps = int(os.environ.get("MAX_REACT_STEPS", 6))
        except (ValueError, TypeError):
            self.max_steps = 6

        try:
            self.max_report_chars = int(os.environ.get("MAX_REPORT_CHARS", 3000))
        except (ValueError, TypeError):
            self.max_report_chars = 3000

        self.decision_parser = JsonOutputParser()
        self.decision_prompt = self._build_decision_prompt()
        self.deep_analysis_prompt = self._build_deep_analysis_prompt()
        self.tool_specs = self._build_tool_specs()
        self.tool_names = [tool["name"] for tool in self.tool_specs]

        self.arch_module_name = {
            "semantic_search": "SEMANTIC SEARCH",
            "structured_filter": "STRUCTURED FILTER",
            "trend_analyzer": "TREND ANALYZER",
            "deep_analysis": "DEEP ANALYSIS"
        }

        try:
            self.duplicate_repeat_threshold = int(os.environ.get("DUPLICATE_ACTION_REPEAT_THRESHOLD", 1))
        except (ValueError, TypeError):
            self.duplicate_repeat_threshold = 1

        try:
            self.fast_track_min_cases = int(os.environ.get("FAST_TRACK_MIN_CASES", 5))
        except (ValueError, TypeError):
            self.fast_track_min_cases = 5

        try:
            self.fast_track_min_top_category = int(os.environ.get("FAST_TRACK_MIN_TOP_CATEGORY", 4))
        except (ValueError, TypeError):
            self.fast_track_min_top_category = 4

        try:
            self.fast_track_min_entity_fields = int(os.environ.get("FAST_TRACK_MIN_ENTITY_FIELDS", 2))
        except (ValueError, TypeError):
            self.fast_track_min_entity_fields = 2

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
             "Use ONLY the evidence categories and ACNs provided. "
             "Do not introduce risks that are not supported by the evidence map. "
             "If evidence is insufficient, say so. Do not reveal chain-of-thought."
            ),
            ("user",
             "Input report:\n{input_report}\n\n"
             "Extracted entities:\n{entities}\n\n"
             "Findings:\n{findings}\n\n"
             "Last observation:\n{last_observation}\n\n"
             "Evidence categories:\n{evidence_risks}\n\n"
             "Evidence map (risk -> ACNs):\n{evidence_map}\n\n"
             "Focus:\n{focus}"
            )
        ])

    def _truncate_report(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.max_report_chars:
            return text
        # Keep head + tail for context
        head = text[: int(self.max_report_chars * 0.7)]
        tail = text[-int(self.max_report_chars * 0.3):]
        return f"{head}\n...\n{tail}"

    def _canonical_action_input(self, action_input: Dict[str, Any]) -> str:
        try:
            return json.dumps(action_input, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return str(action_input)

    def _action_fingerprint(self, action: str, action_input: Dict[str, Any]) -> str:
        if not action:
            return ""
        payload = f"{action}|{self._canonical_action_input(action_input)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _is_known_entity_value(self, value: Any) -> bool:
        if value is None:
            return False
        v = str(value).strip().lower()
        return v not in {"", "unknown", "unknown aircraft", "unknown location", "n/a", "none"}

    def _evaluate_fast_track(
        self,
        results: List[Dict[str, Any]],
        evidence_map: Dict[str, List[str]],
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        unique_acns = set()
        for r in results:
            acn = r.get("ACN") or r.get("metadata", {}).get("ACN")
            if acn:
                unique_acns.add(str(acn))

        category_sizes = {
            category: len(acns)
            for category, acns in evidence_map.items()
            if isinstance(acns, list)
        }

        if category_sizes:
            top_category = max(category_sizes, key=category_sizes.get)
            top_category_count = category_sizes[top_category]
        else:
            top_category = ""
            top_category_count = 0

        known_entity_fields = sum(
            1 for k in ["Aircraft Model", "Location", "Event Type"]
            if self._is_known_entity_value(entities.get(k))
        )

        has_repeated_cases = len(unique_acns) >= self.fast_track_min_cases
        has_stable_evidence = top_category_count >= self.fast_track_min_top_category
        has_entity_support = known_entity_fields >= self.fast_track_min_entity_fields

        ready = has_repeated_cases and has_stable_evidence and has_entity_support

        reason = (
            f"cases={len(unique_acns)} (min {self.fast_track_min_cases}), "
            f"top_category={top_category or 'none'}:{top_category_count} (min {self.fast_track_min_top_category}), "
            f"known_entity_fields={known_entity_fields} (min {self.fast_track_min_entity_fields})"
        )

        return {
            "ready": ready,
            "reason": reason,
            "metrics": {
                "unique_cases": len(unique_acns),
                "top_category": top_category,
                "top_category_count": top_category_count,
                "known_entity_fields": known_entity_fields
            }
        }

    def _extract_evidence_map(self, results: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[str]]]:
        categories = {
            "Adverse Weather / Crosswinds / Visibility": [
                "crosswind", "tailwind", "wind", "windshear", "microburst", "turbulence",
                "visibility", "ceiling", "imc", "wx", "weather", "gust", "storm", "thunder"
            ],
            "Runway Configuration / ATC Flow": [
                "runway", "configuration", "flow", "arrival", "departure", "tower", "tracon",
                "atc", "sequence", "vector", "capacity"
            ],
            "Wake Turbulence": [
                "wake turbulence", "wake vortex", "wake"
            ],
            "Airspace Congestion / VFR Conflicts": [
                "vfr", "class b", "airspace", "conflict", "tcas", "ra", "traffic"
            ],
            "Runway Contamination / Braking Action": [
                "contamination", "braking action", "wet runway", "snow", "ice", "slush", "fair braking"
            ]
        }

        evidence: Dict[str, set] = {k: set() for k in categories}
        for r in results:
            text = (r.get("content") or "").lower()
            acn = r.get("ACN") or r.get("metadata", {}).get("ACN")
            if not text or not acn:
                continue
            for cat, keys in categories.items():
                if any(k in text for k in keys):
                    evidence[cat].add(str(acn))

        evidence_map = {k: sorted(list(v)) for k, v in evidence.items() if v}
        evidence_risks = list(evidence_map.keys())
        return evidence_risks, evidence_map

    def extract_entities(self, state: AgentState):
        print("--- Entity Extraction ---")
        trimmed_report = self._truncate_report(state.get("input_report", ""))
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
            res = chain.invoke({"report": trimmed_report})
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
            user_msg = trimmed_report
            prompt_payload = {
                "system": system_msg,
                "user": user_msg
            }
        except (AttributeError, IndexError):
            prompt_payload = {"report": trimmed_report}

        if not isinstance(entities, dict):
            entities = {"raw_response": str(entities)}

        step_log = {
            "module": "ENTITY EXTRACTION",
            "prompt": prompt_payload,
            "response": entities
        }

        return {
            "extracted_entities": entities,
            "input_report_trimmed": trimmed_report,
            "steps_trace": [step_log]
        }

    def decide_next(self, state: AgentState):
        print("--- ReAct Decision ---")
        report_text = state.get("input_report_trimmed") or state.get("input_report", "")
        entities = state.get("extracted_entities", {})
        entities_safe = {k: v for k, v in entities.items() if k != "Keywords"}
        
        # Guard 1: strict duplicate blocking (same action + same effective input)
        if state.get("duplicate_guard_triggered", False):
            decision = {
                "decision": "final",
                "reasoning_summary": "Duplicate action/input detected; forcing finalize to avoid loop.",
                "action": "",
                "action_input": {},
                "final": "Proceed to synthesis with current evidence."
            }
            step_log = {
                "module": "REACT DECIDER",
                "prompt": {
                    "guard": "duplicate_action",
                    "last_action_fingerprint": state.get("last_action_fingerprint", ""),
                    "repeated_action_count": state.get("repeated_action_count", 0)
                },
                "response": decision
            }
            return {"react_decision": decision, "steps_trace": [step_log]}

        # Guard 2: fast-track finalize for high-confidence obvious incidents
        if state.get("fast_track_ready", False):
            decision = {
                "decision": "final",
                "reasoning_summary": "High-confidence corroboration reached; fast-tracking to synthesis.",
                "action": "",
                "action_input": {},
                "final": "Generate final RCA report directly from current evidence."
            }
            step_log = {
                "module": "REACT DECIDER",
                "prompt": {
                    "guard": "fast_track_finalize",
                    "reason": state.get("fast_track_reason", ""),
                    "tool_history": state.get("tool_history", [])
                },
                "response": decision
            }
            return {"react_decision": decision, "steps_trace": [step_log]}

        prompt_vars = {
            "format_instructions": self.decision_parser.get_format_instructions(),
            "input_report": report_text,
            "entities": entities_safe,
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
            "module": "REACT DECIDER",
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
        evidence_risks = list(state.get("evidence_risks", []))
        evidence_map = dict(state.get("evidence_map", {}))

        fast_track_ready = bool(state.get("fast_track_ready", False))
        fast_track_reason = str(state.get("fast_track_reason", ""))

        observation = None
        response_payload: Dict[str, Any] = {}
        step_log = None
        prompt_payload: Dict[str, Any] = {
            "action": action,
            "action_input": action_input
        }
        fingerprint_input: Dict[str, Any] = dict(action_input)

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
            evidence_risks, evidence_map = self._extract_evidence_map(results)

            fast_track_eval = self._evaluate_fast_track(
                results=results,
                evidence_map=evidence_map,
                entities=state.get("extracted_entities", {})
            )
            fast_track_ready = fast_track_eval["ready"]
            fast_track_reason = fast_track_eval["reason"]

            prompt_payload.update({"query": query, "k": k})
            fingerprint_input = {"query": query, "k": k}
            response_payload = {
                "query": query,
                "result_count": len(results),
                "results": observation,
                "evidence_risks": evidence_risks,
                "evidence_map": evidence_map,
                "fast_track": fast_track_eval
            }
            findings.append(f"Semantic Search: {len(results)} similar reports found.")
            if fast_track_ready:
                findings.append(f"Fast-track condition met: {fast_track_reason}")

            step_log = {
                "module": self.arch_module_name["semantic_search"],
                "prompt": prompt_payload,
                "response": response_payload
            }

        elif action == "structured_filter":
            filters = self._build_filters(action_input, state.get("extracted_entities", {}))

            if filters:
                df_res = self.tools["filtering"].filter_data(filters)
                df_context = df_res
                observation = self._summarize_df(df_res)
                msg_suffix = ""
                used_full_dataset = False
            else:
                df_res = self.tools["filtering"].df
                df_context = df_res
                sample_df = df_res.head(100)
                summary = self._summarize_df(sample_df)
                observation = {
                    "count": f"{len(df_res)} (full dataset)",
                    "sample_preview_count": 100,
                    "columns": summary["columns"],
                    "top_operators": summary.get("top_operators", {})
                }
                msg_suffix = " (No filters provided. Context set to full dataset. Showing top 100 sample in observation.)"
                used_full_dataset = True

            prompt_payload.update({"filters": filters, "used_full_dataset": used_full_dataset})
            fingerprint_input = {"filters": filters, "used_full_dataset": used_full_dataset}
            response_payload = {"filters_applied": filters, "summary": observation}
            findings.append(f"Structured Filter: {len(df_res)} records available{msg_suffix}.")
            step_log = {
                "module": self.arch_module_name["structured_filter"],
                "prompt": prompt_payload,
                "response": response_payload
            }

        elif action == "trend_analyzer":
            use_filtered = action_input.get("use_filtered", True)
            time_col = action_input.get("time_col", "Event_Date")
            metric_col = action_input.get("metric_col", "ACN")
            df_to_use = df_context if (use_filtered and df_context is not None) else self.tools["trend_analyzer"].df

            observation = self.tools["trend_analyzer"].detect_anomalies(
                df_to_use,
                time_col=time_col,
                metric_col=metric_col
            )

            prompt_payload.update({
                "use_filtered": use_filtered,
                "time_col": time_col,
                "metric_col": metric_col,
                "rows_used": int(len(df_to_use))
            })
            fingerprint_input = {
                "use_filtered": use_filtered,
                "time_col": time_col,
                "metric_col": metric_col,
                "rows_used": int(len(df_to_use))
            }
            response_payload = {"trend": observation}
            findings.append(f"Trend Analysis: {observation}")
            step_log = {
                "module": self.arch_module_name["trend_analyzer"],
                "prompt": prompt_payload,
                "response": response_payload
            }

        elif action == "deep_analysis":
            focus = action_input.get("focus", "")
            result = self._deep_analysis(state, focus=focus)
            observation = result.get("observation", "")
            actual_prompt = result.get("actual_prompt", "")

            prompt_payload.update({"focus": focus, "llm_prompt": actual_prompt})
            fingerprint_input = {"focus": focus}
            response_payload = {"analysis": observation}
            findings.append(f"Deep Analysis: {observation}")
            step_log = {
                "module": self.arch_module_name["deep_analysis"],
                "prompt": prompt_payload,
                "response": response_payload
            }

        else:
            observation = f"Unknown action: {action}"
            response_payload = {"error": observation}
            findings.append(observation)
            step_log = {
                "module": "UNKNOWN ACTION",
                "prompt": prompt_payload,
                "response": response_payload
            }

        tool_history.append(action)
        step_count = state.get("step_count", 0) + 1

        new_fingerprint = self._action_fingerprint(action, fingerprint_input)
        prev_fingerprint = state.get("last_action_fingerprint", "")

        if new_fingerprint and new_fingerprint == prev_fingerprint:
            repeated_action_count = int(state.get("repeated_action_count", 0)) + 1
        else:
            repeated_action_count = 0

        duplicate_guard_triggered = bool(new_fingerprint) and (
            repeated_action_count >= self.duplicate_repeat_threshold
        )

        if duplicate_guard_triggered:
            findings.append(
                f"Duplicate guard triggered: repeated action '{action}' with unchanged effective input."
            )

        react_step = {
            "step": step_count,
            "reasoning_summary": decision.get("reasoning_summary", ""),
            "action": action,
            "action_input": action_input,
            "observation": observation
        }

        return {
            "findings": findings,
            "react_steps": [react_step],
            "steps_trace": [step_log] if step_log else [],
            "tool_history": tool_history,
            "df_context": df_context,
            "step_count": step_count,
            "last_observation": observation,
            "evidence_risks": evidence_risks,
            "evidence_map": evidence_map,
            "last_action_fingerprint": new_fingerprint,
            "repeated_action_count": repeated_action_count,
            "duplicate_guard_triggered": duplicate_guard_triggered,
            "fast_track_ready": fast_track_ready,
            "fast_track_reason": fast_track_reason
        }

    def _deep_analysis(self, state: AgentState, focus: str = "") -> Dict[str, str]:
        chain = self.deep_analysis_prompt | self.llm | StrOutputParser()
        
        # Prepare inputs to capture the prompt
        entities = state.get("extracted_entities", {})
        entities_safe = {k: v for k, v in entities.items() if k != "Keywords"}
        inputs = {
            "input_report": state.get("input_report_trimmed") or state.get("input_report", ""),
            "entities": entities_safe,
            "findings": state.get("findings", []),
            "last_observation": state.get("last_observation", ""),
            "evidence_risks": state.get("evidence_risks", []),
            "evidence_map": state.get("evidence_map", {}),
            "focus": focus
        }
        
        try:
            # Generate the actual prompt string for logging traceability
            formatted_prompt = self.deep_analysis_prompt.format_prompt(**inputs).to_string()
            
            # Invoke LLM
            observation = chain.invoke(inputs)
            
            return {
                "observation": observation,
                "actual_prompt": formatted_prompt
            }
        except Exception as e:
            return {
                "observation": f"Deep analysis failed: {e}",
                "actual_prompt": "Prompt generation failed"
            }

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
        if len(df) > 0:
            operator_col = None
            for candidate in ["Operator", "Aircraft Operator", "Aircraft Operator.1"]:
                if candidate in df.columns:
                    operator_col = candidate
                    break
            if operator_col:
                summary["top_operators"] = df[operator_col].value_counts().head(3).to_dict()
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
