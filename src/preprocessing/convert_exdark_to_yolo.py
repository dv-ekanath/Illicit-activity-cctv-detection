"""
convert_exdark_to_yolo.py
=========================
Converts ExDARK People bbGt annotations → YOLO format (.txt)

Input:
  data/raw/exdark/Groundtruth/People/*.txt
  bbGt format: People x y w h 0 0 0 0 0 0 0
  (x, y = top-left corner, w, h = width/height — absolute pixels)

Output:
  data/processed/annotations_yolo/exdark/*.txt
  Format: class_id cx cy w h (normalized 0-1)

Class mapping:
  People → 0  (same as LLVIP person class — unified)

Note: Image dimensions are read from the actual image file
      since bbGt format does not include image size.
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
GT_DIR      = ROOT / 'data' / 'raw'       / 'exdark' / 'Groundtruth' / 'People'
IMG_DIR     = ROOT / 'data' / 'raw'       / 'exdark' / 'images'      / 'People'
OUT_DIR     = ROOT / 'data' / 'processed' / 'annotations_yolo'       / 'exdark'

OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_IMG_EXT = {'.jpg', '.jpeg', '.png'}


def get_image_size(img_stem: str) -> tuple[int, int] | None:
    """Find image file (any extension, case-insensitive) and return (width, height)."""
    # Direct lookup first
    for ext in VALID_IMG_EXT:
        img_path = IMG_DIR / (img_stem + ext)
        if img_path.exists():
            with Image.open(img_path) as img:
                return img.size
    # Fallback: case-insensitive scan
    for f in IMG_DIR.iterdir():
        if f.stem.lower() == img_stem.lower() and f.suffix.lower() in VALID_IMG_EXT:
            with Image.open(f) as img:
                return img.size
    return None


def parse_bbgt(txt_path: Path, img_w: int, img_h: int) -> list:
    """
    Parse ExDARK bbGt annotation file.
    Format per line: People x y w h 0 0 0 0 0 0 0
    x, y = top-left pixel, w, h = box size in pixels.
    Returns list of (class_id, cx, cy, bw, bh) normalized.
    """
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):  # skip header/comments
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            label = parts[0].lower()
            if label != 'people':
                continue

            try:
                x  = float(parts[1])   # top-left x
                y  = float(parts[2])   # top-left y
                bw = float(parts[3])   # box width
                bh = float(parts[4])   # box height
            except ValueError:
                continue

            # Skip degenerate boxes
            if bw <= 0 or bh <= 0:
                continue

            # Clamp box to image bounds (xmax/ymax style then recompute w/h)
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(x + bw, img_w)
            y2 = min(y + bh, img_h)

            # Skip if clamping destroyed the box
            if x2 <= x1 or y2 <= y1:
                continue

            bw_c = x2 - x1
            bh_c = y2 - y1

            # Convert to normalized center format
            cx = (x1 + bw_c / 2) / img_w
            cy = (y1 + bh_c / 2) / img_h
            nw = bw_c / img_w
            nh = bh_c / img_h

            # Final bounds check
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                continue

            boxes.append((0, cx, cy, nw, nh))  # class_id=0 (person)

    return boxes


def main():
    print("Converting ExDARK People bbGt → YOLO format...")
    print(f"  Groundtruth dir : {GT_DIR}")
    print(f"  Images dir      : {IMG_DIR}")
    print(f"  Output dir      : {OUT_DIR}\n")

    ann_files = sorted(GT_DIR.glob('*.txt'))
    print(f"  Found {len(ann_files)} annotation files\n")

    converted, skipped, empty, no_image = 0, 0, 0, 0

    for ann_path in tqdm(ann_files, desc='ExDARK People'):
        # ann_path.stem strips the final .txt
        # e.g. "2015_06246.jpg.txt" → stem = "2015_06246.jpg"
        # e.g. "2015_06246.txt"     → stem = "2015_06246"
        stem = ann_path.stem

        # Resolve image stem — strip any image extension suffix
        img_stem = stem
        for img_ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
            if stem.endswith(img_ext):
                img_stem = stem[: -len(img_ext)]
                break

        # Get image dimensions — search all valid extensions
        size = get_image_size(img_stem)
        if size is None:
            no_image += 1
            continue

        img_w, img_h = size
        boxes = parse_bbgt(ann_path, img_w, img_h)

        # Output label file uses clean stem (no extension)
        out_txt = OUT_DIR / (img_stem + '.txt')

        if not boxes:
            empty += 1
            out_txt.write_text('')
            continue

        with open(out_txt, 'w') as f:
            for cls_id, cx, cy, bw, bh in boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        converted += 1

    print(f"\n✅ ExDARK conversion complete")
    print(f"   Converted  : {converted}")
    print(f"   Empty      : {empty}")
    print(f"   No image   : {no_image}")
    print(f"   Skipped    : {skipped}")
    print(f"   Total out  : {converted + empty}")

    # Verify a sample
    sample = next(OUT_DIR.glob('*.txt'), None)
    if sample:
        print(f"\nSample label ({sample.name}):")
        print(sample.read_text().strip())


if __name__ == '__main__':
    main()