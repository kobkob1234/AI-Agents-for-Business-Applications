from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.planner import Planner
from src.agent.synthesizer import Synthesizer
from src.tools.semantic_search import SemanticSearch
from src.tools.filtering import StructuredFilter
from src.tools.trend_analyzer import TrendAnalyzer
from src.utils.data_loader import load_data, preprocess_data

class ASIAgent:
    def __init__(self):
        print("Initializing ASI Agent...")
        # Load Data once
        self.df = load_data()
        self.df = preprocess_data(self.df)
        
        # Init Tools
        self.semantic_search = SemanticSearch()
        # Ensure semantic search has data? 
        # Ideally, we should check if index exists. If not, ingest.
        # For this prototype, we'll assume it's pre-ingested or we ingest a subset on demand?
        # Let's simple check: if collection empty, ingest subsample.
        # But determining if empty is tricky without access.
        # We will assume ingestion is detailed elsewhere or done via separate script.
        
        self.structured_filter = StructuredFilter(self.df)
        self.trend_analyzer = TrendAnalyzer(self.df)
        
        tools_map = {
            "semantic_search": self.semantic_search,
            "filtering": self.structured_filter,
            "trend_analyzer": self.trend_analyzer
        }
        
        self.planner = Planner(tools_map)
        self.synthesizer = Synthesizer()
        self.workflow = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Define Nodes
        workflow.add_node("extractor", self.planner.extract_entities)
        workflow.add_node("planner", self.planner.plan_step)
        workflow.add_node("tool_executor", self.planner.execute_tools)
        workflow.add_node("synthesizer", self.synthesizer.generate_report)
        
        # Define Edges
        workflow.set_entry_point("extractor")
        workflow.add_edge("extractor", "planner")
        
        # Conditional edge from planner
        workflow.add_conditional_edges(
            "planner",
            self.planner.should_continue,
            {
                "continue": "tool_executor",
                "end": "synthesizer"
            }
        )
        
        workflow.add_edge("tool_executor", "planner")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
        
    def run(self, input_report: str):
        initial_state = {
            "input_report": input_report,
            "messages": [],
            "findings": [],
            "extracted_entities": {},
            "plan": [],
            "steps_trace": []  # Initialize steps trace for API requirement
        }
        return self.workflow.invoke(initial_state)
