
from pathlib import Path
from functools import lru_cache
import threading

import cv2
import numpy as np

from app.config import settings


# Components are initialized only when required.
# This prevents loading PyTorch and embedding models during API startup.

_component_lock = threading.RLock()
_inference_lock = threading.Lock()

_extractor = None
_bank = None
_knowledge = None
_workflow = None
_assets_ready = False


def get_extractor():
    global _extractor

    with _component_lock:
        if _extractor is None:
            from app.vision.anomaly import FeatureExtractor

            _extractor = FeatureExtractor(
                settings.device,
                settings.image_size,
            )

    return _extractor


def get_bank():
    global _bank

    with _component_lock:
        if _bank is None:
            from app.vision.anomaly import ReferenceBank

            _bank = ReferenceBank(settings.artifact_dir)

    return _bank


def get_knowledge():
    global _knowledge

    with _component_lock:
        if _knowledge is None:
            from app.retrieval.knowledge import KnowledgeBase

            _knowledge = KnowledgeBase(
                settings.data_dir / "knowledge" / "incidents.json",
                artifact_dir=settings.artifact_dir,
                embedding_model=settings.embedding_model,
                enable_semantic=settings.enable_semantic_rag,
            )

    return _knowledge


def get_workflow():
    global _workflow

    with _component_lock:
        if _workflow is None:
            from app.workflow.investigation import build_graph

            _workflow = build_graph(get_knowledge())

    return _workflow


def knowledge_status():
    return get_knowledge().status()


def index_category(dataset_dir: Path, category: str):
    from app.vision.anomaly import FeatureExtractor

    normal_dir = Path(dataset_dir) / category / "train" / "good"

    images = sorted(
        list(normal_dir.glob("*.png"))
        + list(normal_dir.glob("*.jpg"))
        + list(normal_dir.glob("*.jpeg"))
    )

    if not images:
        raise FileNotFoundError(
            f"No train/good images found in {normal_dir}"
        )

    extractor = get_extractor()
    bank = get_bank()

    with _inference_lock:
        return bank.build(images, extractor, category)


def ingest_knowledge_file(file_path: Path):
    return get_knowledge().ingest_file(file_path)


def build_knowledge_index(force=False):
    return get_knowledge().build_dense_index(force=force)


def ensure_reference_assets(category: str):
    global _assets_ready

    configured_category = settings.supabase_reference_category.strip()
    if category != configured_category:
        raise ValueError(
            f"Reference bank is configured for '{configured_category}', not '{category}'."
        )

    with _component_lock:
        if not _assets_ready:
            from app.storage.supabase_storage import ensure_reference_assets as restore_assets

            restore_assets()
            _assets_ready = True


def investigate(
    image_path: Path,
    category: str,
    query: str | None = None,
):
    ensure_reference_assets(category)
    extractor = get_extractor()
    bank = get_bank()
    knowledge = get_knowledge()
    workflow = get_workflow()

    # Avoid simultaneous heavy inference on the free instance.
    if not _inference_lock.acquire(blocking=False):
        raise RuntimeError(
            "Another investigation is running. Please retry shortly."
        )

    try:
        result = bank.inspect(image_path, extractor)

        settings.artifact_dir.mkdir(parents=True, exist_ok=True)

        heatmap_path = settings.artifact_dir / "last_heatmap.png"
        heat = (result["heatmap"] * 255).astype("uint8")

        cv2.imwrite(
            str(heatmap_path),
            cv2.applyColorMap(heat, cv2.COLORMAP_JET),
        )

        similar = bank.similar_images(
            image_path,
            extractor,
            settings.top_k,
        )

        state = workflow.invoke({
            "image_name": image_path.name,
            "category": category,
            "query": query or "",
            "vision": {
                k: v
                for k, v in result.items()
                if k not in {"heatmap", "anomaly_map"}
            },
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
                "Research prototype. Anomaly score is not a calibrated "
                "probability or certified quality decision."
            ),
        }

    finally:
        _inference_lock.release()
