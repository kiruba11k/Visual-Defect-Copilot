# Render + Neon deployment notes

This patch adds a React/Vite frontend, Groq report generation, CORS configuration,
and a lazy Neon PostgreSQL connectivity check.

## Local setup
1. Install backend dependencies: `python -m pip install -e ".[dev,semantic]"`.
2. Copy `.env.example` to `.env` and set GROQ_API_KEY and DATABASE_URL.
3. Install frontend dependencies: `cd frontend && npm install`.
4. Set `VITE_API_URL=http://localhost:8000` in `frontend/.env`.
5. Run API: `uvicorn app.api.main:app --reload`.
6. Run UI: `cd frontend && npm run dev`.

## Neon
Use Neon's connection string. For SQLAlchemy with psycopg 3, use the
`postgresql+psycopg://` URL scheme. The `/db/health` endpoint checks connectivity.
This initial patch does not yet create application tables or migrations; that is
the next database implementation stage.

## Render
Connect the GitHub repository containing this project and create a Blueprint from
`render.yaml`. Set GROQ_API_KEY and DATABASE_URL in the backend service, and set
VITE_API_URL in the frontend service to the deployed backend URL.

## Free-tier constraints
The Render free web service has limited RAM/CPU, sleeps when idle, and has no
persistent disk. This codebase imports PyTorch/torchvision during API startup.
The current Dockerfile also installs SentenceTransformers and FAISS. These
dependencies may exceed free-instance memory. `ENABLE_SEMANTIC_RAG=false` disables
dense retrieval but does not remove the PyTorch import. Do not treat the free
deployment as validated until cold start, memory, and real inference tests pass.
A lightweight ONNX inference path or external inference worker may be needed.

This patch does not yet implement Supabase/object storage for uploaded documents
and FAISS artifacts. Do not rely on Render's local filesystem for persistent data.
