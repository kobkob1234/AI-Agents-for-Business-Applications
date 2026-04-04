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
    prompt: Dict[str, Any]
    response: Dict[str, Any]

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
is_agent_initializing = False

def init_agent_sync():
    global agent, is_agent_initializing
    try:
        print("Starting background ASI Agent initialization...")
        agent_instance = ASIAgent()
        agent = agent_instance
        print("ASI Agent Initialized successfully.")
    except Exception as e:
        agent = None
        print(f"Failed to initialize agent in background: {e}")
    finally:
        is_agent_initializing = False

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

def _ensure_step_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, (list, tuple, set)):
        return {"items": list(value)}
    return {"value": value}

def _normalize_steps(steps: Any) -> List[Dict[str, Any]]:
    if not isinstance(steps, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            normalized.append({
                "module": f"STEP_{i + 1}",
                "prompt": {"value": "Invalid step shape"},
                "response": {"value": step}
            })
            continue

        module = step.get("module", f"STEP_{i + 1}")
        if not isinstance(module, str):
            module = str(module)

        normalized.append({
            "module": module,
            "prompt": _ensure_step_object(step.get("prompt")),
            "response": _ensure_step_object(step.get("response"))
        })

    return normalized

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
    global is_agent_initializing
    try:
        _validate_required_integrations()
        print("Validations passed. Spawning background agent initialization...")
        is_agent_initializing = True
        import asyncio
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, init_agent_sync)
    except Exception as e:
        is_agent_initializing = False
        print(f"Failed to validate integrations: {e}")

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
                "prompt": "Location: SAN. Airplane: B737 MAX 8. Event: Descent.\n\nNarrative: During approach to runway 27, we experienced unstable approach conditions with high sink rate. The preceding aircraft was an A321. We encountered significant wake turbulence approximately 3 miles from the runway threshold. Aircraft required go-around due to inability to stabilize before decision height.",
                "full_response": """## Executive Summary
During an approach to SAN Runway 27 in a B737 MAX 8 the flight crew encountered significant wake turbulence from a preceding A321 at roughly 3 nm from the threshold. The wake-induced high sink rate led to an unstable approach and a go-around. Primary finding: wake turbulence encounter combined with close arrival sequencing/ATC flow produced an unstabilized final segment requiring a missed approach.

Supporting evidence (ASRS ACNs): wake-turbulence and unstable-approach events appear in the ASRS records identified for similar scenarios (notably ACNs 2294909, 2176230, 2037409, 1659088, 1527855, 1684068, 1854226, 1924227, 861861). Note: a structured ASRS filter for B737 MAX 8 at SAN returned 0 records, so direct historical records for this exact make-model/location are not available in the structured query - corroboration relies on semantically similar reports (ACNs above).

## Historical Corroboration
Semantically similar ASRS narratives show consistent patterns:
- Wake-turbulence encounters on final leading to unstable approaches and go-arounds: ACNs 2176230, 1924227, 1527855, 2294909, 861861.
- Close ATC sequencing or inadequate spacing cited as causal/enabling factors: ACNs 1684068, 1659088, 1854226, 1527855, 2037409.
- Several reports involve approaches to Runway 27 or similar SAN procedures: ACNs 2037409, 2294909.

Relevant ACNs by observed signatures:
- Wake Turbulence (directly relevant): 1527855, 1659088, 1684068, 1854226, 1924227, 2037409, 2176230, 2294909, 861861.
- Examples with unstable approach/go-around after wake encounter: 2176230, 1924227, 1527855, 2294909.

## Trend Analysis
- Dataset size / detection: semantic search returned 9 distinct ACNs that match the wake-turbulence -> unstable approach -> go-around pattern.
- Category co-occurrence:
  - All nine ACNs appear in the Wake Turbulence category.
  - Many of these ACNs also appear under Runway Configuration / ATC Flow and Adverse Weather / Crosswinds / Visibility categories, indicating frequent multi-factor contributions (see Evidence Map lists).
- Anomaly: structured filtering for this exact make/model/location (B737 MAX 8 at SAN) yielded zero structured records - suggesting either under-reporting or that most comparable records are semantically similar but not indexed under that exact structured query.
- Summary pattern: wake turbulence on final, often when following larger or similar-category aircraft and during close sequencing, is a recurring precursor to unstable approaches and go-arounds in the identified ACNs (1527855, 1659088, 1684068, 1854226, 1924227, 2037409, 2176230, 2294909, 861861).

## Cross-Reference Findings
(Using only the provided evidence categories and ACNs)

- Wake Turbulence <-> Runway Configuration / ATC Flow:
  - Overlap ACNs: 1527855, 1659088, 1684068, 1854226, 1924227, 2037409, 2176230, 2294909. These reports indicate ATC sequencing/spacing and runway/flow choices commonly co-occur with wake encounters.
- Wake Turbulence <-> Adverse Weather / Crosswinds / Visibility:
  - Overlap ACNs: 1527855, 1659088, 1684068, 1854226, 1924227, 2037409, 2176230, 2294909, 861861. Many events flagged both wake turbulence and weather/crosswind considerations.
- Wake Turbulence <-> Airspace Congestion / VFR Conflicts:
  - Overlap ACNs: 1527855, 1659088, 1684068, 1854226, 2037409, 2176230, 2294909, 861861. Congestion/sequencing pressures are recurring themes.
- Runway Contamination / Braking Action:
  - Fewer overlaps (1527855, 1659088, 2037409, 2294909); not a primary driver in the present narrative but appears in some historical records.

Limitations: Evidence provided does not contain operator, maintenance, or organizational-specific identifiers beyond the ACNs and category mappings; therefore there is insufficient evidence to assert maintenance or operator procedural failures. Cross-references are limited to category-ACN intersections only.

## Root Cause Assessment
Primary Root Cause
- Wake turbulence encounter from preceding aircraft (A321) on short/close final producing an abrupt sink-rate and roll/instability that made the approach unstabilized (supported by Wake Turbulence ACNs: 1527855, 1659088, 1684068, 1854226, 1924227, 2037409, 2176230, 2294909, 861861).

Contributing Causes
- ATC sequencing / runway configuration (inadequate spacing or sequencing that placed the B737 MAX 8 on final within hazardous wake vortex proximity of the A321): supported by overlapping ACNs in Runway Configuration / ATC Flow (1527855, 1659088, 1684068, 1854226, 1924227, 2037409, 2176230, 2294909).
- Airspace congestion / operational pressure (traffic density, VFR conflicts or flow compression increasing the likelihood of close-follow situations): supported by ACNs listed under Airspace Congestion / VFR Conflicts (1527855, 1659088, 1684068, 1854226, 2037409, 2176230, 2294909, 861861).
- Environmental factors (wind/crosswind/visibility can exacerbate wake effects and recovery workload): supported by ACNs in Adverse Weather / Crosswinds / Visibility (1527855, 1659088, 1684068, 1854226, 1924227, 2037409, 2176230, 2294909, 861861).
- Insufficient model-and-location specific historical data: Structured query returned zero B737 MAX 8 @ SAN records (limitation), reducing ability to determine if MAX-8 responses differ materially from other types in SAN approaches.

## Recommendations
(Each recommendation is tied to evidence categories/ACNs that support it)

1. Reinforce ATC wake-separation and sequencing practices for final approach to Runway 27 (Runway Configuration / ATC Flow; Wake Turbulence).
   - Action: Review local arrival/sequencing logic and apply increased spacing or vectoring for aircraft following A321/large types where wake vortices are likely (supported ACNs: 1527855, 1684068, 1854226, 2037409, 2294909).
2. Issue explicit wake-turbulence advisories to crews when an A321 or heavier traffic is on short final (Wake Turbulence; Airspace Congestion).
   - Action: ATC phraseology protocols to include short-final wake advisories and suggested spacing when preceding traffic is known to generate significant wake (ACNs: 2176230, 2294909, 861861).
3. Emphasize stabilized-approach criteria and decision-height adherence in operator SOPs and simulator training (Wake Turbulence; Adverse Weather).
   - Action: Recurrent training scenarios on low-altitude wake encounters and mandatory go-around decision timelines; reinforce that a go-around is the appropriate response to approach destabilization (ACNs: 1527855, 2176230, 1924227).
4. Increase crew briefing emphasis on anticipated wake turbulence when following similar or larger aircraft on final (Airspace Congestion; Wake Turbulence).
   - Action: Include explicit brief items: preceding-type, expected vortex behavior, go-around criteria and PF/PM roles during wake encounter recovery (ACNs: 1659088, 1684068, 2037409).
5. Monitor and, if indicated, adjust final approach tracks or intercept points to mitigate vortex encounter in known problem sectors (Runway Configuration / ATC Flow; Wake Turbulence).
   - Action: Conduct local arrival-path risk assessment for Runway 27 using recorded incidents (ACNs: 2037409, 2294909).
6. Improve reporting granularity and local capture of make-model/location incidents (data collection) to close the evidence gap for B737 MAX 8 at SAN.
   - Action: Encourage structured ASRS reporting fields to capture exact make/model and airport to allow trend detection and targeted mitigations (limitation noted: structured query returned 0 records for B737 MAX 8 at SAN).
7. Where environmental factors are present (crosswinds/turbulence), consider conservative minima or sequencing adjustments (Adverse Weather / Crosswinds / Visibility).
   - Action: Issue operational guidance to increase spacing or alter approach configurations under adverse weather flags (ACNs: 1527855, 1659088, 2037409).

Limitations and further work
- The structured ASRS filter returned no direct B737 MAX 8 at SAN records; conclusions rely on semantically similar ACNs. There is insufficient evidence to attribute operator, maintenance, or organizational causal factors beyond the categories and ACNs provided. Further targeted data collection at SAN (flight data, ATC tapes, precise spacing records) and structured ASRS entries for B737 MAX 8 events would improve root-cause specificity.

Prepared from provided evidence categories and ACNs only: Wake Turbulence; Runway Configuration / ATC Flow; Adverse Weather / Crosswinds / Visibility; Airspace Congestion / VFR Conflicts; Runway Contamination / Braking Action (as applicable) and ACNs: 1527855, 1659088, 1684068, 1854226, 1924227, 2037409, 2176230, 2294909, 861861.""",
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
        error_msg = "Agent is currently initializing in the background. Please try again in a few moments." if is_agent_initializing else "Agent not initialized"
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
            error=error_msg,
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
        steps = _normalize_steps(final_state.get("steps_trace", []))
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
        global agent, is_agent_initializing
        if not agent:
            error_msg = "Agent is currently initializing in the background. Please try again in a few moments." if is_agent_initializing else "Agent not initialized"
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
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
                    steps=_normalize_steps(final_payload.get("steps", [])),
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
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=True)
