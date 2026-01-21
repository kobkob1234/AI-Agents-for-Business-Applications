# 🚀 Deployment Guide - What's Left To Do

This guide covers **all remaining steps** to complete and submit your Aviation Safety Agent project.

---

## ✅ Quick Checklist

| Step | Status | Action Required |
|------|--------|-----------------|
| 1. Update Team Info | ⏳ TODO | Add real student data in `server.py` |
| 2. Create Supabase Table | ⏳ TODO | Run SQL in Supabase dashboard |
| 3. Push to GitHub | ⏳ TODO | Commit and push all changes |
| 4. Deploy to Render | ⏳ TODO | Create web service |
| 5. Test Live URL | ⏳ TODO | Verify all endpoints work |
| 6. Submit | ⏳ TODO | Submit Render URL + GitHub URL |

---

## Step 1: Update Team Info (5 min)

Edit `src/server.py` line 58-66 with your real data:

```python
@app.get("/api/team_info")
async def team_info():
    return {
        "group_batch_order_number": "2_3",  # From presentation list: batch_order
        "team_name": "Your Team Name",
        "students": [
            { "name": "Your Name", "email": "your.email@campus.technion.ac.il" },
            { "name": "Partner Name", "email": "partner@campus.technion.ac.il" },
            { "name": "Third Member", "email": "third@campus.technion.ac.il" }
        ]
    }
```

---

## Step 2: Create Supabase Table (5 min)

The agent logs all executions to Supabase. You need to create the table:

1. **Go to**: https://supabase.com/dashboard
2. **Select your project**: `lwnzbsulpbesggkhttnv`
3. **Click**: SQL Editor (left sidebar)
4. **Paste and Run**:

```sql
CREATE TABLE agent_executions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    prompt TEXT NOT NULL,
    entities JSONB,
    steps JSONB,
    final_report TEXT,
    status TEXT DEFAULT 'completed',
    execution_time_ms INTEGER
);
```

5. **Click**: Run (or Ctrl+Enter)
6. **Verify**: Table appears under "Table Editor"

---

## Step 3: Push to GitHub (5 min)

```bash
cd /path/to/project

# Add all changes
git add .

# Commit with message
git commit -m "Final implementation - ready for deployment"

# Push to GitHub
git push origin main
```

---

## Step 4: Deploy to Render (10 min)

### 4.1 Create Account
1. Go to https://render.com
2. Sign up with GitHub (recommended)

### 4.2 Create Web Service
1. Click **New +** → **Web Service**
2. Connect your GitHub repository
3. Select your repo

### 4.3 Configure Service

| Setting | Value |
|---------|-------|
| **Name** | `asi-agent` (or your team name) |
| **Region** | Frankfurt (EU) or nearest |
| **Branch** | `main` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn src.server:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

### 4.4 Add Environment Variables

Click **Environment** → **Add Environment Variable** for each:

| Key | Value |
|-----|-------|
| `OPENAI_API_KEY` | `sk-FA8TYuSqxj15-1EpOYzbKA` |
| `OPENAI_BASE_URL` | `https://api.llmod.ai/v1` |
| `PINECONE_API_KEY` | `pcsk_6hTGyx_T1jyudbzC74rRb5aTu8z6bRgQUkAdiMXZxpHQHQEYZBiCbJkSoWwrMHxbyWFyiq` |
| `SUPABASE_URL` | `https://lwnzbsulpbesggkhttnv.supabase.co` |
| `SUPABASE_KEY` | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (full key) |
| `PYTHON_VERSION` | `3.11.0` |

### 4.5 Deploy
1. Click **Create Web Service**
2. Wait for build (5-10 min)
3. Your URL: `https://asi-agent.onrender.com`

---

## Step 5: Test Live URL (5 min)

Once deployed, test all endpoints:

```bash
# Test homepage
curl https://YOUR-APP.onrender.com/

# Test team info
curl https://YOUR-APP.onrender.com/api/team_info

# Test agent info
curl https://YOUR-APP.onrender.com/api/agent_info

# Test architecture (should return PNG)
curl -I https://YOUR-APP.onrender.com/api/model_architecture

# Test execution
curl -X POST https://YOUR-APP.onrender.com/api/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Location: LAX. Airplane: B737. Event: Landing gear issue."}'
```

---

## Step 6: Submit

Submit in the required format:

```
Render URL: https://asi-agent.onrender.com
GitHub Repo URL: https://github.com/yourusername/your-repo
```

---

## ⚠️ Important Notes

### Render Free Tier
- **Spin-down**: App sleeps after 15 min inactivity
- **Cold start**: First request may take 30-60 seconds
- **Keep active**: Render will shut down after 90 days of inactivity

### Data Size
- The 238K CSV records are included in the repo
- Render free tier has 512MB RAM - should be sufficient
- If build fails, consider reducing data or upgrading tier

### Supabase
- Free tier: 500MB database, 1GB bandwidth
- Executions are logged automatically
- Check logs at: Supabase Dashboard → Table Editor → agent_executions

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check `requirements.txt` has all dependencies |
| App crashes | View logs in Render dashboard |
| Supabase 404 | Create the table (Step 2) |
| Slow response | Free tier cold start - wait 30 sec |
| Memory error | Data too large - contact instructor |

---

## 📝 Summary

| Service | URL | Purpose |
|---------|-----|---------|
| **Render** | render.com | Hosting (required) |
| **Supabase** | supabase.com | Primary database (required) |
| **Pinecone** | pinecone.io | Vector DB for search |
| **LLMod.ai** | api.llmod.ai | LLM provider |

**Deadline**: 1/3/2026
