# Aviation Safety Agent (ASI Agent) ✈️

An intelligent autonomous agent designed to investigate aviation safety reports (NASA ASRS). Built with **LangGraph** and **LLMod.ai**, it performs deep root cause analysis using a ReAct-based architecture.

![Architecture](src/static/architecture.png)

## 🌟 Features

*   **Autonomous Investigation**: A 5-stage pipeline (Entity Extraction → Search → Cross-Ref → Analysis → Report).
*   **ReAct Architecture**: Uses dynamic planning and tool execution.
*   **Semantic Search**: Retrieves similar historical incidents using **Pinecone** / **ChromaDB**.
*   **Trend Analysis**: Detects anomaly patterns in incident frequency over time.
*   **Execution Tracing**: Full visibility into the agent's "thought process" (steps, prompts, responses).
*   **Premium UI**: Dark glassmorphism interface with architecture diagrams and real-time streaming.
*   **Cloud Logging**: All executions are logged to **Supabase** for audit and playback.

## 🛠️ Tech Stack

*   **Core**: Python 3.11, [LangChain](https://langchain.com), [LangGraph](https://langchain-ai.github.io/langgraph/)
*   **API**: [FastAPI](https://fastapi.tiangolo.com)
*   **Frontend**: HTML5, Vanilla JS, CSS3 (Glassmorphism)
*   **LLM Provider**: [LLMod.ai](https://llmod.ai) (`gpt-5-mini` & `text-embedding-3-small`)
*   **Databases**:
    *   **Supabase**: Primary database (Execution Logs)
    *   **Pinecone**: Production Vector Store
    *   **ChromaDB**: Local Development Vector Store

## 🚀 Quick Start (Local)

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/kobkob1234/AI-Agents-for-Business-Applications.git
    cd AI-Agents-for-Business-Applications
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file (or use system env vars):
    ```env
    OPENAI_API_KEY=sk-FA8TYuSqxj15-1EpOYzbKA  # LLMod.ai Key
    OPENAI_BASE_URL=https://api.llmod.ai/v1
    SUPABASE_URL=https://lwnzbsulpbesggkhttnv.supabase.co
    SUPABASE_KEY=your_supabase_anon_key
    PINECONE_API_KEY=your_pinecone_key
    ```

4.  **Run the Server**:
    ```bash
    uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
    ```
    Open [http://localhost:8000](http://localhost:8000) in your browser.

## ☁️ Deployment (Render)

This project is configured for **Render**.

*   **Build Command**: `pip install -r requirements.txt`
*   **Start Command**: `uvicorn src.server:app --host 0.0.0.0 --port $PORT`
*   **Python Version**: `3.11.0` (set via `PYTHON_VERSION` env var)

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/team_info` | Returns team name & student details. |
| `GET` | `/api/agent_info` | Returns prompt templates & examples. |
| `GET` | `/api/model_architecture` | Returns the architecture diagram image. |
| `POST` | `/api/execute` | Runs the agent. accepts `{"prompt": "..."}`. |

## ✅ Submission Checklist & GitHub Requirements

To satisfy the course requirements, the GitHub repo includes:

### 1. Codebase Structure
*   [x] `requirements.txt` - Complete dependency list.
*   [x] `README.md` - Documentation (You are reading it!).
*   [x] `src/` - Source code only (Clean of temp files).
    *   `src/agent/` - Agent logic (Graph, Planner, Synthesizer).
    *   `src/tools/` - Tools (Search, Trends, Filter).
    *   `src/utils/` - Data loading & DB managers.
    *   `src/static/` - Frontend assets & images.
*   [x] `.gitignore` - Properly setup to exclude `data/`, secrets, and venv.

### 2. Required Functionality
*   [x] **Supabase Integration**: Logs executions to `agent_executions` table.
*   [x] **Pinecone Integration**: Vector search ready.
*   [x] **LLM Enforced**: Uses `RPRTHPB-gpt-5-mini`.
*   [x] **Endpoints**: All 4 required endpoints implemented.

### 3. Final Verification for Student
Before submitting, ensure:
- [ ] **`/api/team_info` Updated**: Did you put your real names in `src/server.py`?
- [ ] **Pinecone Data**: Did you ingest the data into your Pinecone index?
- [ ] **Render URL**: Is the site live and accessible?

---

**Developed for AI Agents for Business Applications Course (Winter 2026)**
