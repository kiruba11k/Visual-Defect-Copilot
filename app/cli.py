import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from PIL import Image
from app.services import (index_category, investigate, bank, extractor, knowledge, ingest_knowledge_file, build_knowledge_index)
from app.config import settings

import cv2

def evaluate(dataset, category):
    base = Path(dataset) / category
    index_category(dataset, category)

    scores, labels = [], []
    pixel_score_arrays = []
    pixel_label_arrays = []

    missing_masks = []
    defect_images_with_masks = 0
    normal_images = 0

    # Fixed resolution for memory-efficient pixel evaluation.
    eval_size = (256, 256)

    test_images = sorted((base / "test").glob("*/*"))

    for image in test_images:
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        defect = image.parent.name != "good"

        out = bank.inspect(image, extractor)

        scores.append(out["anomaly_score"])
        labels.append(int(defect))

        anomaly_map = out["anomaly_map"]

        if defect:
            mask_path = (
                base / "ground_truth"
                / image.parent.name
                / f"{image.stem}_mask.png"
            )

            if not mask_path.exists():
                missing_masks.append(str(mask_path))
                continue

            mask = np.asarray(
                Image.open(mask_path).convert("L")
            )

            mask = cv2.resize(
                mask,
                eval_size,
                interpolation=cv2.INTER_NEAREST
            ) > 0

            defect_images_with_masks += 1

        else:
            normal_images += 1
            mask = np.zeros(
                (eval_size[1], eval_size[0]),
                dtype=bool
            )

        # Resize raw anomaly scores for memory-efficient evaluation.
        small_map = cv2.resize(
            anomaly_map.astype(np.float32),
            eval_size,
            interpolation=cv2.INTER_LINEAR
        )

        pixel_score_arrays.append(small_map.ravel())
        pixel_label_arrays.append(mask.ravel())

    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} defect masks are missing. "
            f"Example: {missing_masks[0]}"
        )

    result = {
        "category": category,
        "images_evaluated": len(scores),
        "normal_images": normal_images,
        "defective_images_with_masks": defect_images_with_masks,
        "missing_masks": len(missing_masks),
        "pixel_evaluation_resolution": "256x256",
    }

    result["image_auroc"] = (
        float(roc_auc_score(labels, scores))
        if len(set(labels)) > 1 else None
    )

    if pixel_score_arrays:
        all_pixel_scores = np.concatenate(pixel_score_arrays)
        all_pixel_labels = np.concatenate(pixel_label_arrays)

        result["pixel_auroc"] = (
            float(roc_auc_score(all_pixel_labels, all_pixel_scores))
            if len(np.unique(all_pixel_labels)) > 1
            else None
        )
    else:
        result["pixel_auroc"] = None

    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("index"); p.add_argument("--dataset", default="data/mvtec"); p.add_argument("--category", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--image", required=True); p.add_argument("--category", required=True); p.add_argument("--query", default=None, help="Optional defect symptom or investigation question for RAG")
    p = sub.add_parser("evaluate"); p.add_argument("--dataset", default="data/mvtec"); p.add_argument("--category", required=True)
    p = sub.add_parser("ingest"); p.add_argument("--file", required=True, help="TXT, MD, PDF or DOCX knowledge source")
    p = sub.add_parser("knowledge-index"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("knowledge-search"); p.add_argument("--query", required=True); p.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "index":
        print(json.dumps(index_category(Path(args.dataset), args.category), indent=2))
    elif args.command == "inspect":
        print(json.dumps(investigate(Path(args.image), args.category, query=args.query), indent=2))
    elif args.command == "ingest":
        print(json.dumps(ingest_knowledge_file(Path(args.file)), indent=2))
    elif args.command == "knowledge-index":
        print(json.dumps(build_knowledge_index(force=args.force), indent=2))
    elif args.command == "knowledge-search":
        print(json.dumps(knowledge.search(args.query, top_k=args.top_k), indent=2))
    else:
        print(json.dumps(evaluate(args.dataset, args.category), indent=2))

if __name__ == "__main__":
    main()
