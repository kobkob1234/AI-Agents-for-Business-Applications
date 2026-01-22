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
            "messages": [],
            "findings": [],
            "extracted_entities": {},
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
