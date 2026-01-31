from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Generator
import re
import os
import json
import uvicorn
from src.agent.graph import ASIAgent

# --- Pydantic Models ---
class ExecuteInput(BaseModel):
    prompt: str

class Step(BaseModel):
    module: str
    prompt: Any
    response: Any

class ExecuteResponse(BaseModel):
    status: str
    error: Optional[str] = None
    response: Optional[str] = None
    steps: List[Step] = Field(default_factory=list)

# --- App Init ---
app = FastAPI()

# Add CORS middleware for web deployment
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for course project
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (Frontend)
# We ensure the directory exists first
if not os.path.exists("src/static"):
    os.makedirs("src/static")
app.mount("/static", StaticFiles(directory="src/static", html=True), name="static")

# Init Agent Global
# We initialize it once on startup
agent = None

AVIATION_KEYWORDS = {
    "aviation", "aircraft", "airplane", "flight", "pilot", "crew", "cockpit",
    "runway", "takeoff", "landing", "approach", "departure", "descent", "climb",
    "atc", "tower", "airspace", "turbulence", "wake", "asrs", "faa"
}

def is_aviation_prompt(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if any(word in lowered for word in AVIATION_KEYWORDS):
        return True
    # Aircraft pattern: e.g., B737, A320, B777
    if re.search(r"\b[ab]\d{3,4}\b", lowered):
        return True
    return False

@app.on_event("startup")
async def startup_event():
    global agent
    try:
        agent = ASIAgent()
        print("ASI Agent Initialized.")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")

# --- Endpoints ---

@app.get("/")
async def root():
    return FileResponse("src/static/index.html")

@app.get("/api/team_info")
async def team_info():
    """
    Returns student details.
    """
    return {
        "group_batch_order_number": "1_11",
        "team_name": "ASI Team",
        "students": [
            { "name": "Kobi Amit", "email": "kobiamit@campus.technion.ac.il" },
            { "name": "Yaniv Steiner", "email": "Yaniv11410@campus.technion.ac.il" },
            { "name": "Itav Dan", "email": "itav.dan@campus.technion.ac.il" },
            { "name": "Yuval Komar", "email": "Yuval.komar@campus.technion.ac.il" }
        ]
    }

@app.get("/api/agent_info")
async def agent_info():
    return {
        "description": "Autonomous Safety Investigator (ASI) - An AI agent that analyzes aviation safety reports using a ReAct-based architecture to identify root causes and systemic issues.",
        "purpose": "To autonomously investigate aviation incidents by extracting entities, searching historical data, cross-referencing patterns, and generating comprehensive Root Cause Analysis (RCA) reports.",
        "prompt_template": {
            "template": "Analyze the following aviation safety report and identify potential root causes: {report}"
        },
        "prompt_examples": [
            {
                "prompt": "Location: SAN. Airplane: B737 MAX 8. Event: Descent. Narrative: Experiencing unstable approach and high sink rate due to wake turbulence from preceding A321.",
                "full_response": "## Executive Summary\nThe reported incident involved a B737 MAX 8 experiencing unstable approach conditions at SAN during descent phase. The primary contributing factor appears to be wake turbulence from a preceding A321 aircraft.\n\n## Historical Corroboration\nSemantic search of the ASRS database identified 5 similar cases involving wake turbulence encounters during approach phase.\n\n## Trend Analysis\nAnalysis indicates stable reporting patterns for wake turbulence incidents, with mean monthly reports around 12-15 for B737 aircraft.\n\n## Cross-Reference Findings\nFiltered data shows 23 similar reports for B737 aircraft at SAN. Most common operator: Southwest Airlines.\n\n## Root Cause Assessment\nInsufficient separation from preceding heavier aircraft during approach phase.\n\n## Recommendations\n1. Review wake turbulence separation standards\n2. Enhance pilot awareness training for wake vortex encounters",
                "steps": [
                    { "module": "ENTITY EXTRACTION", "prompt": "Extract key entities from the report...", "response": "Captured: B737 MAX 8, SAN, Descent" },
                    { "module": "REACT DECIDER", "prompt": "Decide next action based on findings...", "response": "Action: deep_analysis" },
                    { "module": "DEEP ANALYSIS", "prompt": "LLM causal analysis prompt...", "response": "Confirmed recurrent issue" },
                    { "module": "SYNTHESIZER", "prompt": "Synthesize findings...", "response": "Generated RCA Report" }
                ]
            }
        ]
    }

@app.get("/api/model_architecture")
async def model_architecture():
    # Return a static image.
    # User must ensure 'architecture.png' exists in src/static or generate it.
    # For now, we return a 404 or a placeholder if file missing.
    image_path = "src/static/architecture.png"
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    else:
        # Create a dummy image or error
        return JSONResponse(status_code=404, content={"error": "Architecture image not found. Please place 'architecture.png' in src/static/."})

@app.post("/api/execute", response_model=ExecuteResponse)
def execute(input_data: ExecuteInput):
    global agent
    import time
    from src.utils.supabase_manager import supabase_manager

    start_time = time.time()
    if not agent:
        execution_time_ms = int((time.time() - start_time) * 1000)
        # Log error to Supabase (if available)
        supabase_manager.log_execution(
            prompt=input_data.prompt,
            entities={},
            steps=[],
            final_report=None,
            execution_time_ms=execution_time_ms,
            status="error"
        )
        return ExecuteResponse(
            status="error",
            error="Agent not initialized",
            response=None,
            steps=[]
        )

    if not is_aviation_prompt(input_data.prompt):
        execution_time_ms = int((time.time() - start_time) * 1000)
        refusal = "This request is not related to aviation safety; I cannot comply or assist."
        supabase_manager.log_execution(
            prompt=input_data.prompt,
            entities={},
            steps=[],
            final_report=refusal,
            execution_time_ms=execution_time_ms,
            status="completed"
        )
        return ExecuteResponse(
            status="ok",
            error=None,
            response=refusal,
            steps=[]
        )
    
    try:
        # Run agent
        # agent.run returns the final STATE
        final_state = agent.run(input_data.prompt)
        
        # Extract response
        report = final_state.get("final_report", "No report generated.")
        
        # Extract steps
        # steps_trace was added to AgentState in state.py
        steps = final_state.get("steps_trace", [])
        entities = final_state.get("extracted_entities", {})
        
        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Log to Supabase (primary database requirement)
        supabase_manager.log_execution(
            prompt=input_data.prompt,
            entities=entities,
            steps=steps,
            final_report=report,
            execution_time_ms=execution_time_ms,
            status="completed"
        )
        
        return ExecuteResponse(
            status="ok",
            response=report,
            steps=steps
        )
        
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        print(f"Execution error: {e}")
        
        # Log error to Supabase
        supabase_manager.log_execution(
            prompt=input_data.prompt,
            entities={},
            steps=[],
            final_report=None,
            execution_time_ms=execution_time_ms,
            status="error"
        )
        
        # Return structured error response instead of raising HTTPException
        return ExecuteResponse(
            status="error",
            error=str(e),
            response=None,
            steps=[]
        )

@app.post("/api/execute_stream")
async def execute_stream(input_data: ExecuteInput):
    """
    SSE streaming endpoint for real-time chain-of-thought updates.
    Yields events as the agent processes each ReAct step.
    """
    def stream_generator() -> Generator[str, None, None]:
        global agent
        if not agent:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent not initialized'})}\n\n"
            return
        
        import time
        from src.utils.supabase_manager import supabase_manager
        
        start_time = time.time()

        if not is_aviation_prompt(input_data.prompt):
            refusal = "This request is not related to aviation safety; I cannot comply or assist."
            execution_time_ms = int((time.time() - start_time) * 1000)
            supabase_manager.log_execution(
                prompt=input_data.prompt,
                entities={},
                steps=[],
                final_report=refusal,
                execution_time_ms=execution_time_ms,
                status="completed"
            )
            yield f"data: {json.dumps({'type': 'result', 'response': refusal, 'steps': [], 'entities': {}})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            return
        
        final_payload = None
        try:
            # Stream agent execution
            for event in agent.run_streaming(input_data.prompt):
                if event.get("type") == "result":
                    final_payload = event
                yield f"data: {json.dumps(event)}\n\n"
            
            # Get final state for logging
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Log final result to Supabase (if available)
            if final_payload:
                supabase_manager.log_execution(
                    prompt=input_data.prompt,
                    entities=final_payload.get("entities", {}),
                    steps=final_payload.get("steps", []),
                    final_report=final_payload.get("response"),
                    execution_time_ms=execution_time_ms,
                    status="completed"
                )

            # Signal completion
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            print(f"Streaming error: {e}")
            supabase_manager.log_execution(
                prompt=input_data.prompt,
                entities={},
                steps=[],
                final_report=None,
                execution_time_ms=execution_time_ms,
                status="error"
            )
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

if __name__ == "__main__":
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
