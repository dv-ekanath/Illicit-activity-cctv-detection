"""
convert_llvip_to_yolo.py
========================
Converts LLVIP Pascal VOC XML annotations → YOLO format (.txt)

Input:
  data/raw/llvip/Annotations/*.xml
  Format: <xmin> <ymin> <xmax> <ymax> in absolute pixels

Output:
  data/processed/annotations_yolo/llvip/train/*.txt
  data/processed/annotations_yolo/llvip/test/*.txt
  Format: class_id cx cy w h (normalized 0-1)

Class mapping:
  person → 0

Also copies image paths into train.txt / val.txt for YOLO dataset config.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
ANN_DIR     = ROOT / 'data' / 'raw'       / 'llvip' / 'Annotations'
IMG_TRAIN   = ROOT / 'data' / 'raw'       / 'llvip' / 'visible' / 'train'
IMG_TEST    = ROOT / 'data' / 'raw'       / 'llvip' / 'visible' / 'test'
OUT_TRAIN   = ROOT / 'data' / 'processed' / 'annotations_yolo' / 'llvip' / 'train'
OUT_TEST    = ROOT / 'data' / 'processed' / 'annotations_yolo' / 'llvip' / 'test'

OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_TEST.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {'person': 0}


def parse_xml(xml_path: Path):
    """
    Parse VOC XML → list of (class_id, cx, cy, w, h) normalized.
    LLVIP BBox format: xmin ymin xmax ymax (absolute pixels).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size  = root.find('size')
    img_w = float(size.find('width').text)
    img_h = float(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        if name not in CLASS_MAP:
            continue

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Clamp to image bounds
        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))

        # Skip degenerate boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        boxes.append((CLASS_MAP[name], cx, cy, bw, bh))

    return boxes


def convert_split(img_dir: Path, out_dir: Path, split_name: str):
    img_paths = sorted(img_dir.glob('*.jpg'))
    converted, skipped, empty = 0, 0, 0

    for img_path in tqdm(img_paths, desc=f'LLVIP {split_name}'):
        xml_path = ANN_DIR / (img_path.stem + '.xml')

        if not xml_path.exists():
            skipped += 1
            continue

        boxes = parse_xml(xml_path)
        out_txt = out_dir / (img_path.stem + '.txt')

        if not boxes:
            empty += 1
            # Write empty file — YOLO needs label file to exist
            out_txt.write_text('')
            continue

        with open(out_txt, 'w') as f:
            for cls_id, cx, cy, bw, bh in boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        converted += 1

    print(f"  {split_name}: {converted} converted | {empty} empty | {skipped} skipped")
    return converted


def main():
    print("Converting LLVIP XML → YOLO format...")
    print(f"  Annotations dir : {ANN_DIR}")
    print(f"  Output train    : {OUT_TRAIN}")
    print(f"  Output test     : {OUT_TEST}\n")

    n_train = convert_split(IMG_TRAIN, OUT_TRAIN, 'train')
    n_test  = convert_split(IMG_TEST,  OUT_TEST,  'test')

    print(f"\n✅ LLVIP conversion complete")
    print(f"   Train labels : {n_train}")
    print(f"   Test  labels : {n_test}")
    print(f"   Total        : {n_train + n_test}")

    # Verify a sample
    sample = next(OUT_TRAIN.glob('*.txt'), None)
    if sample:
        print(f"\nSample label ({sample.name}):")
        print(sample.read_text().strip())


if __name__ == '__main__':
    main()