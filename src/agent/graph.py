from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.react_controller import ReActController
from src.agent.synthesizer import Synthesizer
from src.tools.semantic_search import SemanticSearch
from src.tools.filtering import StructuredFilter
from src.tools.trend_analyzer import TrendAnalyzer
from src.utils.data_loader import load_data, preprocess_data

class ASIAgent:
    def __init__(self):
        print("Initializing ASI Agent...")
        self.df = load_data()
        self.df = preprocess_data(self.df)

        self.semantic_search = SemanticSearch()
        self.structured_filter = StructuredFilter(self.df)
        self.trend_analyzer = TrendAnalyzer(self.df)

        tools_map = {
            "semantic_search": self.semantic_search,
            "filtering": self.structured_filter,
            "trend_analyzer": self.trend_analyzer
        }

        self.react = ReActController(tools_map)
        self.synthesizer = Synthesizer()
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("extractor", self.react.extract_entities)
        workflow.add_node("react_decider", self.react.decide_next)
        workflow.add_node("react_act", self.react.execute_action)
        workflow.add_node("synthesizer", self.synthesizer.generate_report)

        workflow.set_entry_point("extractor")
        workflow.add_edge("extractor", "react_decider")

        workflow.add_conditional_edges(
            "react_decider",
            self.react.route_decision,
            {
                "act": "react_act",
                "final": "synthesizer"
            }
        )

        workflow.add_edge("react_act", "react_decider")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def run(self, input_report: str):
        initial_state = {
            "input_report": input_report,
            "input_report_trimmed": "",
            "messages": [],
            "findings": [],
            "extracted_entities": {},
            "evidence_risks": [],
            "evidence_map": {},
            "steps_trace": [],
            "react_steps": [],
            "react_decision": {},
            "tool_history": [],
            "df_context": None,
            "step_count": 0,
            "max_steps": self.react.max_steps,
            "last_observation": None
        }
        return self.workflow.invoke(initial_state)

    def run_streaming(self, input_report: str):
        """
        Generator that yields SSE events for each step of the ReAct loop.
        Used for real-time chain-of-thought streaming.
        """
        initial_state = {
            "input_report": input_report,
            "input_report_trimmed": "",
            "messages": [],
            "findings": [],
            "extracted_entities": {},
            "evidence_risks": [],
            "evidence_map": {},
            "steps_trace": [],
            "react_steps": [],
            "react_decision": {},
            "tool_history": [],
            "df_context": None,
            "step_count": 0,
            "max_steps": self.react.max_steps,
            "last_observation": None
        }
        
        final_state = None
        all_steps = []
        last_entities = {}
        
        # Stream through graph execution
        for event in self.workflow.stream(initial_state):
            # event is a dict with node_name -> output
            for node_name, node_output in event.items():
                # Create SSE event based on node type
                if node_name == "extractor":
                    entities = node_output.get("extracted_entities", {})
                    last_entities = entities
                    # Add to full trace
                    new_trace_logs = node_output.get("steps_trace", [])
                    all_steps.extend(new_trace_logs)
                    
                    yield {
                        "type": "step",
                        "step": "ENTITY EXTRACTION",
                        "status": "complete",
                        "detail": f"Extracted: {', '.join(str(v) for v in entities.values() if v and str(v).lower() not in ['unknown', 'unknown aircraft', 'unknown location'])[:100]}"
                    }
                
                elif node_name == "react_decider":
                    decision = node_output.get("react_decision", {})
                    action = decision.get("action", "")
                    reasoning = decision.get("reasoning_summary", "")[:100]
                    
                    # Add to full trace
                    new_trace_logs = node_output.get("steps_trace", [])
                    all_steps.extend(new_trace_logs)
                    
                    if decision.get("decision") == "final":
                        yield {
                            "type": "step",
                            "step": "REACT DECIDER",
                            "status": "complete",
                            "detail": f"Decision: Finalize report"
                        }
                    else:
                        yield {
                            "type": "step",
                            "step": "REACT DECIDER",
                            "status": "complete",
                            "detail": f"Next action: {action}"
                        }
                
                elif node_name == "react_act":
                    react_steps = node_output.get("react_steps", [])
                    if react_steps:
                        last_step = react_steps[-1]
                        action = last_step.get("action", "")
                        observation = last_step.get("observation", "")

                        new_trace_logs = node_output.get("steps_trace", [])
                        all_steps.extend(new_trace_logs)

                        module_name_map = {
                            "semantic_search": "SEMANTIC SEARCH",
                            "filtering": "STRUCTURED FILTER",
                            "trend_analyzer": "TREND ANALYZER",
                            "deep_analysis": "DEEP ANALYSIS" # Assuming this might be added later
                        }
                        module_label = module_name_map.get(action, f"TOOL: {action}")

                        if isinstance(observation, dict):
                            obs_preview = str(observation)[:100]
                        elif isinstance(observation, list):
                            obs_preview = f"{len(observation)} items"
                        else:
                            obs_preview = str(observation)[:100]

                        yield {
                            "type": "step",
                            "step": module_label,
                            "status": "complete",
                            "detail": obs_preview
                        }
                
                elif node_name == "synthesizer":
                    report = node_output.get("final_report", "")
                    
                    # Add final step log
                    new_trace_logs = node_output.get("steps_trace", [])
                    all_steps.extend(new_trace_logs)
                    
                    yield {
                        "type": "step",
                        "step": "SYNTHESIZER",
                        "status": "complete",
                        "detail": "Generating final RCA report..."
                    }
                    # Also yield the final results with FULL trace
                    yield {
                        "type": "result",
                        "response": report,
                        "steps": all_steps,
                        "entities": last_entities
                    }
                
                # Update final_state
                if final_state is None:
                    final_state = node_output
                else:
                    final_state.update(node_output)

        
        # If we didn't get a result event, yield final state
        if final_state and "final_report" in final_state:
            pass  # Already yielded in synthesizer node
