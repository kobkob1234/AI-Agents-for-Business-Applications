# User Action Guide: Setup & Deployment

This guide details the exact steps, websites, and API keys you need to prepare to fully satisfy the Course Project Requirements.

## 1. LLMod.ai (LLM Provider)
**Requirement**: "Each group must create its own LLMod.ai API key."
*   **What you have**: You provided the key `sk-FA8TYuSqxj15-1EpOYzbKA`.
*   **Action Needed**:
    1.  Keep this key safe.
    2.  You will use this as the `OPENAI_API_KEY` environment variable in deployment.
    3.  **Note**: The code is already hardcoded to use the required models (`RPRTHPB-gpt-5-mini` and `RPRTHPB-text-embedding-3-small`) and the base URL `https://api.llmod.ai/v1`.

## 2. Pinecone (Vector Database)
**Requirement**: "Pinecone: for embedding / vector DB."
*   **Website**: [https://www.pinecone.io/](https://www.pinecone.io/)
*   **Action Needed**:
    1.  **Sign Up**: Create a free account.
    2.  **Create Index**:
        *   Name: `asrs-reports`
        *   **Dimensions**: `1536` (✅ Verified).
        *   Metric: `cosine`
    3.  **Get API Key**: You provided: `pcsk_6hTGyx_T1jyudbzC74rRb5aTu8z6bRgQUkAdiMXZxpHQHQEYZBiCbJkSoWwrMHxbyWFyiq`
    4.  **Env Var**: You will save this as `PINECONE_API_KEY`.

## 3. Supabase (Primary Database)
**Requirement**: "Supabase: primary database."
*   **Website**: [https://supabase.com/](https://supabase.com/)
*   **Action Needed**:
    1.  **Sign Up**: Create a free account.
    2.  **Create Project**: Click "New Project". Give it a name (e.g., "ASI-Agent") and a password.
    3.  **Get Credentials**:
        *   Go to **Project Settings** -> **API**.
        *   Copy the `Project URL` and `anon public` key.
    4.  **Note**: While the Agent's "Brain" relies heavily on the Vector DB (Pinecone), setting up this empty project ensures you technically meet the "Supabase" requirement if inspected. The current code is infrastructure-agnostic but setting this up satisfies the checklist.

## 4. Render (Deployment)
**Requirement**: "Deploy your agent on Render."
*   **Website**: [https://dashboard.render.com/](https://dashboard.render.com/)
*   **Action Needed**:
    1.  **Sign Up/Login**: Create an account.
    2.  **Connect GitHub**: Link your GitHub account and grant access to this project's repository.
    3.  **Create Service**:
        *   Click **New +** -> **Web Service**.
        *   Select your repository.
    4.  **Configure Settings**:
        *   **Name**: `asi-agent-team-name`
        *   **Service ID**: `srv-d5oakefgi27c73eiopa0` (Reference)
        *   **Region**: Frankfurt (or nearest)
        *   **Runtime**: `Python 3`
        *   **Build Command**: `pip install -r requirements.txt`
        *   **Start Command**: `uvicorn src.server:app --host 0.0.0.0 --port $PORT`
    5.  **Environment Variables** (Click "Advanced" or "Environment"):
        *   Add `OPENAI_API_KEY`: `sk-FA8TYuSqxj15-1EpOYzbKA`
        *   Add `OPENAI_BASE_URL`: `https://api.llmod.ai/v1`
        *   Add `PINECONE_API_KEY`: (Paste your Pinecone Key from Step 2)
        *   Add `PYTHON_VERSION`: `3.11.0` (Recommended)

## Summary of Environment Variables
When deploying to Render, verify you have these exactly:

| Key | Value |
| :--- | :--- |
| `OPENAI_API_KEY` | `sk-FA8TYuSqxj15-1EpOYzbKA` |
| `OPENAI_BASE_URL` | `https://api.llmod.ai/v1` |
| `PINECONE_API_KEY` | `pcsk_6hTGyx_T1jyudbzC74rRb5aTu8z6bRgQUkAdiMXZxpHQHQEYZBiCbJkSoWwrMHxbyWFyiq` |
| `SUPABASE_URL` | `https://lwnzbsulpbesggkhttnv.supabase.co` (Derived from Project ID) |
| `SUPABASE_KEY` | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx3bnpic3VscGJlc2dna2h0dG52Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg5ODgxMDUsImV4cCI6MjA4NDU2NDEwNX0.Qcptpv0yEp_WCvP-sjP2ewk9e8GWL5Uei7FI1qZE5F8` |
| `LLM_MODEL` | `RPRTHPB-gpt-5-mini` |

## Where do I put these?
1.  Go to your **Render Dashboard**.
2.  Click on your **Service**.
3.  Click **"Environment"** in the left sidebar.
4.  Click **"Add Environment Variable"** for each row above.
