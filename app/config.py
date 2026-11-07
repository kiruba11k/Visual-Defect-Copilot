from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_seconds: int = 45
    database_url: str | None = None

    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_storage_bucket: str = "defect-reference-data"
    supabase_reference_category: str = "bottle"

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    device: str = "auto"
    image_size: int = 256
    top_k: int = 5
    retrieval_top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_semantic_rag: bool = True
    cors_origins: str = "http://localhost:5173"


settings = Settings()
settings.artifact_dir.mkdir(parents=True, exist_ok=True)
