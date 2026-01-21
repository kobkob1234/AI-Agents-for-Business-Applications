from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os
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
    steps: List[Step] = []

# --- App Init ---
app = FastAPI()

# Mount Static Files (Frontend)
# We ensure the directory exists first
if not os.path.exists("src/static"):
    os.makedirs("src/static")
app.mount("/static", StaticFiles(directory="src/static", html=True), name="static")

# Init Agent Global
# We initialize it once on startup
agent = None

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
        "group_batch_order_number": "BATCH_ORDER_PLACEHOLDER",
        "team_name": "ASI Team",
        "students": [
            { "name": "Student A", "email": "student.a@example.com" },
            { "name": "Student B", "email": "student.b@example.com" },
            { "name": "Student C", "email": "student.c@example.com" }
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
                    "Entity Extraction: Extract aircraft model, location, event type from report",
                    "Semantic Search: Query vector database for similar historical incidents",
                    "Structured Filter: Apply database filters for aircraft type and airport",
                    "Cross-Referencing: Analyze operator patterns and correlations",
                    "Trend Analyzer: Detect anomalies in incident reporting patterns",
                    "Deep Analysis: Compare with safety manuals and past reports",
                    "Report Generation: Synthesize findings into comprehensive RCA report"
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
async def execute(input_data: ExecuteInput):
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    import time
    from src.utils.supabase_manager import supabase_manager
    
    start_time = time.time()
    
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
        
        return ExecuteResponse(
            status="error",
            error=str(e),
            response=None,
            steps=[]
        )

if __name__ == "__main__":
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
