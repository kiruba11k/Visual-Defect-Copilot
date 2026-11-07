from pathlib import Path
import cv2

from app.config import settings
from app.vision.anomaly import FeatureExtractor, ReferenceBank
from app.retrieval.knowledge import KnowledgeBase
from app.workflow.investigation import build_graph

extractor = FeatureExtractor(settings.device, settings.image_size)
bank = ReferenceBank(settings.artifact_dir)
knowledge = KnowledgeBase(
    settings.data_dir / "knowledge" / "incidents.json",
    artifact_dir=settings.artifact_dir,
    embedding_model=settings.embedding_model,
    enable_semantic=settings.enable_semantic_rag,
)
workflow = build_graph(knowledge)


def index_category(dataset_dir: Path, category: str):
    normal_dir = Path(dataset_dir) / category / "train" / "good"
    images = sorted(
        list(normal_dir.glob("*.png"))
        + list(normal_dir.glob("*.jpg"))
        + list(normal_dir.glob("*.jpeg"))
    )
    if not images:
        raise FileNotFoundError(f"No train/good images found in {normal_dir}")
    return bank.build(images, extractor, category)


def ingest_knowledge_file(file_path: Path):
    return knowledge.ingest_file(file_path)


def build_knowledge_index(force=False):
    return knowledge.build_dense_index(force=force)


def investigate(image_path: Path, category: str, query: str | None = None):
    result = bank.inspect(image_path, extractor)
    heatmap_path = settings.artifact_dir / "last_heatmap.png"
    heat = (result["heatmap"] * 255).astype("uint8")
    cv2.imwrite(str(heatmap_path), cv2.applyColorMap(heat, cv2.COLORMAP_JET))
    similar = bank.similar_images(image_path, extractor, settings.top_k)
    state = workflow.invoke({
        "image_name": image_path.name,
        "category": category,
        "query": query or "",
        "vision": {k: v for k, v in result.items() if k not in {"heatmap", "anomaly_map"}},
        "similar_cases": similar,
    })
    return {
        "category": category,
        "anomaly_score": result["anomaly_score"],
        "heatmap_path": str(heatmap_path),
        "similar_cases": similar,
        "retrieved_documents": state.get("documents", []),
        "report": state["report"],
        "llm_provider": state.get("llm_provider", "unknown"),
        "llm_error": state.get("llm_error"),
        "knowledge_status": knowledge.status(),
        "warning": (
            "Research prototype. Anomaly score is not a calibrated probability "
            "or certified quality decision."
        ),
    }
