"""Hybrid technical-document retrieval: BM25 + dense embeddings + RRF fusion.

Dense retrieval uses SentenceTransformers and FAISS when available. If FAISS is
not installed, exact cosine search with NumPy is used. BM25 remains available
without the optional semantic dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import re
import hashlib

import numpy as np
from rank_bm25 import BM25Okapi

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported document type: {suffix}")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list[str]:
    """Character-window chunker with word-boundary adjustment."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class KnowledgeBase:
    """Persistent local hybrid index with transparent source metadata."""
    def __init__(
        self,
        path="data/knowledge/incidents.json",
        artifact_dir="artifacts",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        enable_semantic=True,
    ):
        self.path = Path(path)
        self.artifact_dir = Path(artifact_dir)
        self.documents_path = self.artifact_dir / "knowledge_chunks.json"
        self.embeddings_path = self.artifact_dir / "knowledge_embeddings.npy"
        self.faiss_path = self.artifact_dir / "knowledge.faiss"
        self.embedding_model_name = embedding_model
        self.enable_semantic = enable_semantic
        self.documents: list[dict] = []
        self.bm25 = None
        self._model = None
        self._embeddings = None
        self._faiss_index = None
        self.semantic_status = "not_initialized"
        self._load_documents()

    @staticmethod
    def _text(doc: dict) -> str:
        return " ".join(str(doc.get(k, "")) for k in
                        ("title", "component", "symptom", "cause", "action",
                         "notes", "category", "text", "source"))

    def _load_documents(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.documents_path.exists():
            self.documents = json.loads(self.documents_path.read_text(encoding="utf-8"))
        elif self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            # Legacy incident records become searchable knowledge documents.
            self.documents = [
                {
                    "id": str(d.get("id", f"LEGACY-{i+1:04d}")),
                    "source": str(d.get("source", "sample_incidents.json")),
                    "title": d.get("title", "Untitled incident"),
                    "text": self._text(d),
                    "metadata": {k: v for k, v in d.items() if k not in {"text"}},
                }
                for i, d in enumerate(raw)
            ]
            self._persist_documents()
        self._rebuild_bm25()

    def _persist_documents(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path.write_text(
            json.dumps(self.documents, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _rebuild_bm25(self):
        corpus = [tokenize(self._text(d)) for d in self.documents]
        self.bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None

    def ingest_file(self, file_path, chunk_size=900, overlap=140) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        text = _read_document(path)
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            raise ValueError(f"No extractable text found in {path.name}")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        # Replace previous chunks from the same source to make ingestion repeatable.
        self.documents = [d for d in self.documents if d.get("source_hash") != source_hash and d.get("source") != path.name]
        for idx, chunk in enumerate(chunks, start=1):
            self.documents.append({
                "id": f"{source_hash}-CH{idx:04d}",
                "source": path.name,
                "source_hash": source_hash,
                "chunk": idx,
                "title": path.stem,
                "text": chunk,
                "metadata": {"file_type": path.suffix.lower()},
            })
        self._persist_documents()
        self._rebuild_bm25()
        # Any new document invalidates the dense index.
        self._embeddings = None
        self._faiss_index = None
        self.embeddings_path.unlink(missing_ok=True)
        self.faiss_path.unlink(missing_ok=True)
        return {"source": path.name, "chunks_created": len(chunks), "total_chunks": len(self.documents)}

    def ingest_directory(self, directory) -> dict:
        directory = Path(directory)
        files = [p for p in sorted(directory.rglob("*"))
                 if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
        results = [self.ingest_file(p) for p in files]
        return {"files_ingested": len(results), "chunks_created": sum(r["chunks_created"] for r in results),
                "documents": results}

    def _load_embedding_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                self.semantic_status = "unavailable: install the semantic extra"
                raise RuntimeError(
                    "Semantic retrieval requires sentence-transformers. "
                    "Install with: pip install -e \".[semantic]\""
                ) from exc
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def build_dense_index(self, force=False) -> dict:
        if not self.documents:
            self.semantic_status = "empty"
            return {"semantic": False, "reason": "No knowledge documents have been ingested."}
        if self.embeddings_path.exists() and not force:
            self._embeddings = np.load(self.embeddings_path)
        else:
            model = self._load_embedding_model()
            texts = [self._text(d) for d in self.documents]
            vectors = model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=True
            ).astype(np.float32)
            self._embeddings = vectors
            np.save(self.embeddings_path, vectors)
        self._faiss_index = None
        try:
            import faiss
            index = faiss.IndexFlatIP(self._embeddings.shape[1])
            index.add(self._embeddings)
            self._faiss_index = index
            faiss.write_index(index, str(self.faiss_path))
            backend = "faiss"
        except ImportError:
            backend = "numpy_exact_cosine"
        self.semantic_status = f"ready:{backend}"
        return {"semantic": True, "backend": backend, "vectors": int(len(self._embeddings))}

    def _ensure_dense_index(self):
        if self._embeddings is None and self.embeddings_path.exists():
            self._embeddings = np.load(self.embeddings_path)
        if self._embeddings is None:
            return self.build_dense_index()
        if self._faiss_index is None:
            try:
                import faiss
                if self.faiss_path.exists():
                    self._faiss_index = faiss.read_index(str(self.faiss_path))
                else:
                    index = faiss.IndexFlatIP(self._embeddings.shape[1])
                    index.add(self._embeddings)
                    self._faiss_index = index
            except ImportError:
                pass
        return {"semantic": True}

    def search(self, query: str, top_k=5, semantic=True, rrf_k=60) -> list[dict]:
        if not self.documents or top_k <= 0:
            return []
        # BM25 ranked list
        lexical_ranked = []
        if self.bm25:
            scores = self.bm25.get_scores(tokenize(query))
            order = np.argsort(scores)[::-1]
            lexical_ranked = [(int(i), float(scores[i])) for i in order if scores[i] > 0][:max(top_k * 4, top_k)]

        dense_ranked = []
        if semantic and self.enable_semantic:
            try:
                self._ensure_dense_index()
                if self._embeddings is not None:
                    vector = self._load_embedding_model().encode(
                        [query], normalize_embeddings=True, convert_to_numpy=True
                    ).astype(np.float32)
                    if self._faiss_index is not None:
                        sims, ids = self._faiss_index.search(vector, min(len(self.documents), top_k * 4))
                        dense_ranked = [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]
                    else:
                        sims = self._embeddings @ vector[0]
                        ids = np.argsort(sims)[::-1][:max(top_k * 4, top_k)]
                        dense_ranked = [(int(i), float(sims[i])) for i in ids]
            except (RuntimeError, ImportError):
                # Lexical retrieval remains available; caller can inspect status.
                pass

        # Reciprocal Rank Fusion: avoids mixing incompatible BM25/cosine scales.
        fused = {}
        lexical_scores = dict(lexical_ranked)
        dense_scores = dict(dense_ranked)
        for rank, (idx, _) in enumerate(lexical_ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (idx, _) in enumerate(dense_ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = []
        for idx in ranked:
            doc = self.documents[idx]
            results.append({
                "id": doc.get("id"),
                "source": doc.get("source"),
                "title": doc.get("title"),
                "text": doc.get("text", self._text(doc)),
                "metadata": doc.get("metadata", {}),
                "rrf_score": float(fused[idx]),
                "bm25_score": lexical_scores.get(idx),
                "semantic_score": dense_scores.get(idx),
            })
        return results

    def status(self) -> dict:
        return {
            "documents": len(self.documents),
            "bm25_ready": self.bm25 is not None,
            "semantic_status": self.semantic_status,
            "embedding_model": self.embedding_model_name,
        }
