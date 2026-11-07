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

Render's local filesystem is an instance cache, not durable storage. Supabase
Storage is used to restore the reference bank and reference images after instance
restarts. Other uploaded documents and knowledge artifacts still need a separate
persistence strategy before they can be considered durable.


## Supabase reference assets

The backend restores the private reference bank from Supabase Storage on the
first investigation request after each instance start. The downloaded bank is
verified against the SHA-256 checksum in
`{category}/reference_bank_manifest.json`; reference images are cached under
`ARTIFACT_DIR/reference_images/{category}`. The existing local cache is reused
when its bank checksum matches.

Set these variables on the **backend only** (never expose the service-role key
to the Vite frontend):

- `SUPABASE_URL`: project URL, for example `https://your-project.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY`: server-side service-role secret
- `SUPABASE_STORAGE_BUCKET`: private bucket name; defaults to
  `defect-reference-data`
- `SUPABASE_REFERENCE_CATEGORY`: uploaded category; defaults to `bottle`

For local development, put the same variables in the backend `.env` file.
Do not commit `.env` or service-role credentials.

The initial integration currently supports the uploaded `bottle` reference
bank. Additional categories need their own uploaded manifests, metadata and
image objects, plus corresponding validation.
