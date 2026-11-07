"""Supabase Storage bootstrap for the visual reference bank.

The bucket is private. Only the backend service-role credential may access it.
Objects are downloaded to the instance cache; no credentials or signed URLs are
exposed to the frontend.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.config import settings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 520, 521, 522, 524}


class SupabaseStorageError(RuntimeError):
    """Raised when reference assets cannot be restored from Supabase."""


def _credentials() -> tuple[str, str, str]:
    url = (settings.supabase_url or "").strip().rstrip("/")
    key = (settings.supabase_service_role_key or "").strip()
    bucket = (settings.supabase_storage_bucket or "").strip()

    if not url or not key:
        raise SupabaseStorageError(
            "Supabase Storage is not configured. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY on the backend."
        )
    if not bucket:
        raise SupabaseStorageError("SUPABASE_STORAGE_BUCKET must not be empty.")
    return url, key, bucket


def _request_bytes(object_path: str, *, attempts: int = 5) -> bytes:
    url, key, bucket = _credentials()
    clean_path = str(PurePosixPath(object_path))
    if clean_path.startswith("/") or ".." in PurePosixPath(clean_path).parts:
        raise SupabaseStorageError("Invalid Supabase object path.")

    encoded_path = quote(f"{bucket}/{clean_path}", safe="/")
    endpoint = f"{url}/storage/v1/object/{encoded_path}"
    request = Request(
        endpoint,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/octet-stream",
        },
        method="GET",
    )

    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in _RETRYABLE_STATUS or attempt == attempts - 1:
                raise SupabaseStorageError(
                    f"Supabase download failed for '{clean_path}' "
                    f"(HTTP {exc.code})."
                ) from exc
            logger.warning(
                "Transient Supabase HTTP %s downloading %s; retry %s/%s",
                exc.code, clean_path, attempt + 1, attempts,
            )
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == attempts - 1:
                raise SupabaseStorageError(
                    f"Supabase download failed for '{clean_path}' after "
                    f"{attempts} attempts."
                ) from exc
            logger.warning(
                "Transient Supabase network error downloading %s; retry %s/%s: %s",
                clean_path, attempt + 1, attempts, exc,
            )
        time.sleep(min(2 ** attempt, 12))

    raise SupabaseStorageError(f"Unable to download '{clean_path}'.")


def _download_to_path(object_path: str, destination: Path) -> bytes:
    payload = _request_bytes(object_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _manifest_chunk_paths(manifest: dict, category: str) -> list[str]:
    chunks = manifest.get("chunks") or manifest.get("parts")
    if not isinstance(chunks, list) or not chunks:
        raise SupabaseStorageError(
            "Reference-bank manifest must contain a non-empty 'chunks' list."
        )

    paths: list[str] = []
    for item in chunks:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = (
                item.get("path")
                or item.get("name")
                or item.get("object")
                or item.get("file")
            )
        else:
            name = None

        if not name or not isinstance(name, str):
            raise SupabaseStorageError("Invalid chunk entry in bank manifest.")
        name = name.lstrip("/")
        if name.startswith(f"{category}/"):
            paths.append(name)
        elif name.startswith("bank_parts/"):
            paths.append(f"{category}/{name}")
        else:
            paths.append(f"{category}/bank_parts/{PurePosixPath(name).name}")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_reference_assets() -> dict:
    """Restore the configured category's bank and images into local cache.

    Safe to call repeatedly. Existing bank data is reused only when its SHA-256
    matches the private bucket manifest. Images are downloaded when absent.
    """
    _, _, _ = _credentials()
    category = settings.supabase_reference_category.strip()
    if not category or "/" in category or category in {".", ".."}:
        raise SupabaseStorageError("Invalid SUPABASE_REFERENCE_CATEGORY.")

    artifact_dir = settings.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    bank_path = artifact_dir / "reference_bank.npz"
    metadata_path = artifact_dir / "reference_metadata.json"
    image_dir = artifact_dir / "reference_images" / category

    manifest_object = f"{category}/reference_bank_manifest.json"
    manifest = json.loads(_request_bytes(manifest_object).decode("utf-8"))
    expected_sha = str(manifest.get("sha256") or manifest.get("bank_sha256") or "").lower()
    if len(expected_sha) != 64:
        raise SupabaseStorageError(
            "Bank manifest is missing a valid SHA-256 checksum."
        )

    expected_size = manifest.get("bank_size") or manifest.get("size_bytes") or manifest.get("size")
    if bank_path.exists() and _sha256(bank_path) == expected_sha:
        logger.info("Using verified reference bank cache at %s", bank_path)
    else:
        chunks = _manifest_chunk_paths(manifest, category)
        temporary_bank = bank_path.with_name(bank_path.name + ".download")
        digest = hashlib.sha256()
        total_bytes = 0
        try:
            with temporary_bank.open("wb") as output:
                for chunk_path in chunks:
                    chunk = _request_bytes(chunk_path)
                    output.write(chunk)
                    digest.update(chunk)
                    total_bytes += len(chunk)
            if expected_size is not None and total_bytes != int(expected_size):
                raise SupabaseStorageError(
                    f"Reference bank size mismatch: expected {expected_size}, "
                    f"received {total_bytes} bytes."
                )
            if digest.hexdigest() != expected_sha:
                raise SupabaseStorageError(
                    "Reference bank SHA-256 verification failed."
                )
            os.replace(temporary_bank, bank_path)
        finally:
            temporary_bank.unlink(missing_ok=True)

    raw_metadata = _request_bytes(f"{category}/reference_metadata.json")
    try:
        metadata = json.loads(raw_metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupabaseStorageError("Reference metadata is not valid JSON.") from exc

    relative_paths = metadata.get("paths")
    if not isinstance(relative_paths, list) or not relative_paths:
        raise SupabaseStorageError("Reference metadata contains no image paths.")

    local_paths: list[str] = []
    for relative in relative_paths:
        if not isinstance(relative, str):
            raise SupabaseStorageError("Invalid image path in reference metadata.")
        normalized = PurePosixPath(relative.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise SupabaseStorageError("Unsafe image path in reference metadata.")
        object_path = f"{category}/{normalized.as_posix()}"
        filename = normalized.name
        if not filename:
            raise SupabaseStorageError("Empty image filename in metadata.")

        local_path = (image_dir / filename).resolve()
        if image_dir.resolve() not in local_path.parents:
            raise SupabaseStorageError("Image path escaped the local cache.")
        if not local_path.exists() or local_path.stat().st_size == 0:
            _download_to_path(object_path, local_path)
        local_paths.append(str(local_path))

    if len(local_paths) != len(set(local_paths)):
        raise SupabaseStorageError("Duplicate image paths found in metadata.")

    local_metadata = {
        **metadata,
        "category": metadata.get("category") or category,
        "paths": local_paths,
    }
    metadata_path.write_text(
        json.dumps(local_metadata, indent=2),
        encoding="utf-8",
    )

    if len(local_paths) != 209:
        logger.warning(
            "Expected 209 reference images for %s; metadata contains %s.",
            category, len(local_paths),
        )

    return {
        "category": category,
        "reference_images": len(local_paths),
        "bank_bytes": bank_path.stat().st_size,
        "bank_sha256": expected_sha,
        "cache_dir": str(artifact_dir),
    }
