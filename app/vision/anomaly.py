from pathlib import Path
import json
import numpy as np
import cv2
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

class FeatureExtractor:
    """Extract spatial features from ResNet18 layer2 (not a trained defect classifier)."""
    def __init__(self, device="auto", image_size=256):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.image_size = image_size
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        self.backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2
        ).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)

    def _tensor(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image).astype("float32") / 255.0
        x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(self.device)
        return (x - self.mean) / self.std

    @torch.inference_mode()
    def feature_map(self, image):
        fmap = self.backbone(self._tensor(image))
        return fmap.squeeze(0).permute(1,2,0).cpu().numpy().astype("float32")

    def image_embedding(self, image):
        fmap = self.feature_map(image)
        emb = fmap.mean(axis=(0,1))
        norm = np.linalg.norm(emb) + 1e-10
        return emb / norm

class ReferenceBank:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.bank_path = self.artifact_dir / "reference_bank.npz"
        self.meta_path = self.artifact_dir / "reference_metadata.json"
        self.features = None
        self.paths = []
        self.category = None

    def build(self, image_paths, extractor: FeatureExtractor, category: str):
        maps, paths = [], []
        for path in image_paths:
            fmap = extractor.feature_map(path)
            # Spatial grid features are used as normal reference patches.
            maps.append(fmap.reshape(-1, fmap.shape[-1]))
            paths.append(str(path))
        if not maps:
            raise ValueError("No normal reference images found.")
        self.features = np.concatenate(maps, axis=0)
        self.paths = paths
        self.category = category
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.bank_path, features=self.features)
        self.meta_path.write_text(json.dumps({"paths": paths, "category": category}, indent=2))
        return {"reference_images": len(paths), "reference_patches": len(self.features)}

    def load(self):
        if not self.bank_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Reference bank missing. Run the index command first.")
        self.features = np.load(self.bank_path)["features"]
        meta = json.loads(self.meta_path.read_text())
        self.paths, self.category = meta["paths"], meta["category"]
        return self

    
    def inspect(self, image, extractor: FeatureExtractor):
        if self.features is None:
            self.load()

        fmap = extractor.feature_map(image)
        h, w, c = fmap.shape

        query = fmap.reshape(-1, c).astype(np.float32)
        ref = self.features.astype(np.float32)

        # Precompute squared norms of reference features.
        ref_norms = np.sum(ref * ref, axis=1)

        distances = []

        query_batch_size = 128
        reference_batch_size = 4096

        for q_start in range(0, len(query), query_batch_size):
            q = query[q_start:q_start + query_batch_size]

            q_norms = np.sum(q * q, axis=1)
            min_distances = np.full(len(q), np.inf, dtype=np.float32)

            # Compare against reference patches in smaller batches.
            for r_start in range(0, len(ref), reference_batch_size):
                r = ref[r_start:r_start + reference_batch_size]
                r_norm = ref_norms[r_start:r_start + reference_batch_size]

                # Squared Euclidean distance:
                # ||q-r||² = ||q||² + ||r||² - 2(q.r)
                d2 = (
                    q_norms[:, None]
                    + r_norm[None, :]
                    - 2.0 * (q @ r.T)
                )

                d2 = np.maximum(d2, 0.0)

                batch_min = d2.min(axis=1)
                min_distances = np.minimum(min_distances, batch_min)

            distances.extend(np.sqrt(min_distances).tolist())

        raw = np.asarray(distances, dtype=np.float32).reshape(h, w)

        
    # Raw anomaly map resized to original image dimensions.
        raw_map = cv2.resize(
            raw,
            (self._image_width(image), self._image_height(image)),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize only for visualization, not quantitative evaluation.
        lo, hi = np.percentile(raw, [2, 98])
        heat = np.clip(
            (raw - lo) / max(float(hi - lo), 1e-8),
            0,
            1
        )

        heat = cv2.resize(
            heat,
            (self._image_width(image), self._image_height(image)),
            interpolation=cv2.INTER_LINEAR
        )

        score = float(np.percentile(raw, 95))

        return {
            "anomaly_score": score,
            "heatmap": heat,
            "anomaly_map": raw_map,
            "feature_map_shape": [h, w, c]
        }

    @staticmethod
    def _image_width(image):
        return Image.open(image).width if isinstance(image, (str, Path)) else image.shape[1]
    @staticmethod
    def _image_height(image):
        return Image.open(image).height if isinstance(image, (str, Path)) else image.shape[0]

    def similar_images(self, image, extractor: FeatureExtractor, top_k=5):
        if self.features is None:
            self.load()
        q = extractor.image_embedding(image)
        scored = []
        for path in self.paths:
            emb = extractor.image_embedding(path)
            scored.append((float(np.dot(q, emb)), path))
        return [{"similarity": s, "image_path": p} for s,p in sorted(scored, reverse=True)[:top_k]]
