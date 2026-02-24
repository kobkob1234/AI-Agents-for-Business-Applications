from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Generator
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

def _is_strict_mode() -> bool:
    app_env = os.getenv("APP_ENV", "").strip().lower()
    strict_flag = os.getenv("REQUIRE_STRICT_STACK", "").strip().lower()
    return app_env in {"prod", "production"} or strict_flag in {"1", "true", "yes"}

def _error_payload(message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "error": message,
        "response": None,
        "steps": []
    }

def _format_validation_error(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request payload"
    first = errors[0]
    loc = ".".join(str(x) for x in first.get("loc", []))
    msg = first.get("msg", "Invalid request payload")
    return f"{loc}: {msg}" if loc else msg

def _is_execute_request(request: Request) -> bool:
    return request.url.path == "/api/execute"

def _validate_required_integrations() -> None:
    if not _is_strict_mode():
        return

    required_env = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "SUPABASE_URL", "SUPABASE_KEY", "PINECONE_API_KEY"]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Strict mode: missing required env vars: {', '.join(missing)}")

    if "llmod.ai" not in os.getenv("OPENAI_BASE_URL", ""):
        raise RuntimeError("Strict mode: OPENAI_BASE_URL must point to LLMod.ai")

    from src.utils.supabase_manager import supabase_manager
    if not supabase_manager.is_connected():
        raise RuntimeError("Strict mode: Supabase must be connected")

    from src.tools.vector_store_manager import HAS_PINECONE
    if not HAS_PINECONE:
        raise RuntimeError("Strict mode: Pinecone SDK is required")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if _is_execute_request(request):
        return JSONResponse(
            status_code=200,
            content=_error_payload(f"Invalid request payload: {_format_validation_error(exc)}")
        )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if _is_execute_request(request):
        return JSONResponse(
            status_code=200,
            content=_error_payload(str(exc.detail))
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    if _is_execute_request(request):
        return JSONResponse(
            status_code=200,
            content=_error_payload(f"Unexpected server error: {str(exc)}")
        )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.on_event("startup")
async def startup_event():
    global agent
    try:
        _validate_required_integrations()
        agent = ASIAgent()
        print("ASI Agent Initialized.")
    except Exception as e:
        agent = None
        print(f"Failed to initialize agent: {e}")

@app.get("/api/runtime_diagnostics")
async def runtime_diagnostics():
    from src.utils.supabase_manager import supabase_manager
    from src.tools.vector_store_manager import HAS_PINECONE

    base_url = os.getenv("OPENAI_BASE_URL", "")
    llm_provider = "llmod.ai" if "llmod.ai" in base_url else "non-llmod"

    vector_backend = None
    if agent is not None:
        try:
            vector_backend = type(agent.semantic_search.vectorstore).__name__
        except Exception:
            vector_backend = "unknown"

    return {
        "app_env": os.getenv("APP_ENV", "development"),
        "strict_mode": _is_strict_mode(),
        "llm": {
            "provider_detected": llm_provider,
            "base_url": base_url,
            "api_key_present": bool(os.getenv("OPENAI_API_KEY"))
        },
        "supabase": {
            "url_present": bool(os.getenv("SUPABASE_URL")),
            "key_present": bool(os.getenv("SUPABASE_KEY")),
            "connected": supabase_manager.is_connected()
        },
        "pinecone": {
            "api_key_present": bool(os.getenv("PINECONE_API_KEY")),
            "sdk_available": HAS_PINECONE
        },
        "runtime": {
            "agent_initialized": agent is not None,
            "vector_backend": vector_backend
        }
    }

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
                "full_response": "## Executive Summary\n...\n## Recommendations\n1. ...\n2. ...",
                "steps": [
                    {
                        "module": "ENTITY EXTRACTION",
                        "prompt": {"task": "extract_entities"},
                        "response": {"Aircraft Model": "B737 MAX 8", "Location": "SAN", "Event Type": "Wake Turbulence", "Flight Phase": "Descent"}
                    },
                    {
                        "module": "REACT DECIDER",
                        "prompt": {"task": "decide_next_action"},
                        "response": {"decision": "act", "action": "semantic_search"}
                    },
                    {
                        "module": "SEMANTIC SEARCH",
                        "prompt": {"query": "B737 MAX 8 wake turbulence SAN descent", "k": 5},
                        "response": {"result_count": 5}
                    },
                    {
                        "module": "REACT DECIDER",
                        "prompt": {"task": "decide_next_action"},
                        "response": {"decision": "act", "action": "structured_filter"}
                    },
                    {
                        "module": "STRUCTURED FILTER",
                        "prompt": {"Make_Model": "B737 MAX 8", "Airport": "SAN"},
                        "response": {"count": 23}
                    },
                    {
                        "module": "REACT DECIDER",
                        "prompt": {"task": "decide_next_action"},
                        "response": {"decision": "act", "action": "trend_analyzer"}
                    },
                    {
                        "module": "TREND ANALYZER",
                        "prompt": {"use_filtered": True, "time_col": "Event_Date", "metric_col": "ACN"},
                        "response": {"status": "analyzed"}
                    },
                    {
                        "module": "REACT DECIDER",
                        "prompt": {"task": "decide_next_action"},
                        "response": {"decision": "act", "action": "deep_analysis"}
                    },
                    {
                        "module": "DEEP ANALYSIS",
                        "prompt": {"focus": "Root causes from collected evidence"},
                        "response": {"analysis": "Likely wake encounter with contributing sequencing factors."}
                    },
                    {
                        "module": "REACT DECIDER",
                        "prompt": {"task": "decide_next_action"},
                        "response": {"decision": "final"}
                    },
                    {
                        "module": "SYNTHESIZER",
                        "prompt": {"task": "compose_final_rca"},
                        "response": {"report": "Final RCA report..."}
                    }
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
