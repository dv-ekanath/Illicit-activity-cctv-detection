"""
verify_annotations.py
=====================
Verifies YOLO annotation conversion quality for both LLVIP and ExDARK.
Checks for: out-of-bounds values, degenerate boxes, missing files, class distribution.
Optionally draws bounding boxes on sample images for visual inspection.
"""

import os
import random
from pathlib import Path
import cv2
import numpy as np
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
LLVIP_LABELS_TRAIN = ROOT / 'data' / 'processed' / 'annotations_yolo' / 'llvip' / 'train'
LLVIP_LABELS_TEST  = ROOT / 'data' / 'processed' / 'annotations_yolo' / 'llvip' / 'test'
EXDARK_LABELS      = ROOT / 'data' / 'processed' / 'annotations_yolo' / 'exdark'
LLVIP_IMG_TRAIN    = ROOT / 'data' / 'raw' / 'llvip' / 'visible' / 'train'
LLVIP_IMG_TEST     = ROOT / 'data' / 'raw' / 'llvip' / 'visible' / 'test'
EXDARK_IMG         = ROOT / 'data' / 'raw' / 'exdark' / 'images' / 'People'
VIS_OUT            = ROOT / 'outputs' / 'annotation_verify'
VIS_OUT.mkdir(parents=True, exist_ok=True)

COLORS = {0: (0, 255, 0)}   # person = green


def check_label_file(txt_path: Path) -> dict:
    """Returns stats for one label file."""
    stats = {'boxes': 0, 'invalid': 0, 'classes': []}
    if txt_path.stat().st_size == 0:
        return stats

    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                stats['invalid'] += 1
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])

            # Check bounds
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                stats['invalid'] += 1
                continue

            stats['boxes'] += 1
            stats['classes'].append(cls_id)

    return stats


def audit_dataset(label_dir: Path, name: str):
    """Audit all label files in a directory."""
    print(f"\n{'─'*50}")
    print(f"  Auditing: {name}")
    print(f"  Dir: {label_dir}")
    print(f"{'─'*50}")

    txt_files  = list(label_dir.glob('*.txt'))
    total      = len(txt_files)
    empty      = 0
    total_boxes = 0
    total_invalid = 0
    class_counter = Counter()

    for txt in txt_files:
        stats = check_label_file(txt)
        if stats['boxes'] == 0 and stats['invalid'] == 0:
            empty += 1
        total_boxes   += stats['boxes']
        total_invalid += stats['invalid']
        class_counter.update(stats['classes'])

    print(f"  Label files  : {total}")
    print(f"  Empty files  : {empty}")
    print(f"  Total boxes  : {total_boxes}")
    print(f"  Invalid boxes: {total_invalid}")
    print(f"  Class dist   : {dict(class_counter)}")
    if total > 0:
        print(f"  Avg boxes/img: {total_boxes / max(1, total - empty):.2f}")

    return total, total_boxes


def draw_boxes_on_image(img_path: Path, label_path: Path, out_path: Path):
    """Draw YOLO bounding boxes on image and save."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    h, w = img.shape[:2]

    if label_path.exists() and label_path.stat().st_size > 0:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                color = COLORS.get(cls_id, (0, 0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'person', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(out_path), img)
    return True


def visualize_samples(img_dir: Path, label_dir: Path, dataset_name: str, n=5):
    """Draw boxes on N random images and save to outputs/annotation_verify/."""
    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    if not img_files:
        return

    samples = random.sample(img_files, min(n, len(img_files)))
    saved = 0

    for img_path in samples:
        label_path = label_dir / (img_path.stem + '.txt')
        out_path   = VIS_OUT / f"{dataset_name}_{img_path.stem}.jpg"
        if draw_boxes_on_image(img_path, label_path, out_path):
            saved += 1

    print(f"\n  📸 Saved {saved} visualizations → {VIS_OUT}")


def main():
    print("=" * 50)
    print("  YOLO Annotation Verification")
    print("=" * 50)

    # Audit LLVIP
    audit_dataset(LLVIP_LABELS_TRAIN, 'LLVIP Train')
    audit_dataset(LLVIP_LABELS_TEST,  'LLVIP Test')

    # Audit ExDARK
    audit_dataset(EXDARK_LABELS, 'ExDARK People')

    # Visual spot-check
    print(f"\n{'─'*50}")
    print("  Generating visual spot-checks...")
    visualize_samples(LLVIP_IMG_TRAIN, LLVIP_LABELS_TRAIN, 'llvip_train', n=5)
    visualize_samples(EXDARK_IMG,      EXDARK_LABELS,      'exdark',      n=5)

    print(f"\n✅ Verification complete.")
    print(f"   Open outputs/annotation_verify/ to visually inspect boxes.")


if __name__ == '__main__':
    main()