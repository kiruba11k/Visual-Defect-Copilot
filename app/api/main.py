from pathlib import Path
from tempfile import NamedTemporaryFile
import re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.services import (
    investigate,
    index_category,
    ingest_knowledge_file,
    build_knowledge_index,
    knowledge_status,
)
from app.config import settings

app = FastAPI(
    title="Visual Manufacturing Defect Investigation Copilot",
    version="0.2.0",
    description=(
        "Research prototype combining visual anomaly detection with hybrid "
        "BM25+dense technical-document retrieval and LangGraph investigation."
    ),
)


_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/db/health")
def database_health():
    try:
        from app.database import check_database
        return {"status": "ok" if check_database() else "error"}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="Database connection failed.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/knowledge/status")
def get_knowledge_status():
    return knowledge_status()


@app.post("/knowledge/ingest")
async def ingest_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(
            status_code=415,
            detail="Supported document formats: TXT, Markdown, PDF, DOCX.",
        )
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "document").name)
    upload_dir = settings.data_dir / "knowledge" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / safe_name
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum document size is 15 MB.")
    destination.write_bytes(content)
    try:
        result = ingest_knowledge_file(destination)
        return {
    "status": "ingested",
    **result,
    "knowledge": knowledge_status(),
}
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/knowledge/index")
def index_knowledge():
    try:
        return build_knowledge_index(force=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/index/{category}")
def index(category: str):
    try:
        result = index_category(settings.data_dir / "mvtec", category)
        return {"status": "indexed", **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/investigate")
async def inspect_image(
    category: str = Form(...),
    query: str = Form(default=""),
    file: UploadFile = File(...),
):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image.")
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        return investigate(tmp_path, category, query=query or None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/artifacts/heatmap")
def heatmap():
    path = settings.artifact_dir / "last_heatmap.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No heatmap has been generated.")
    return FileResponse(path, media_type="image/png")
