"""
build_dataset_yaml.py
=====================
Generates dataset.yaml for YOLOv11s fine-tuning.
Combines LLVIP + ExDARK People into one unified dataset.

Strategy:
  - LLVIP train (12,025) → primary training set
  - ExDARK People (609)  → appended to training set
  - LLVIP test  (3,463)  → validation set

Output: data/processed/annotations_yolo/dataset.yaml
"""

import os
import shutil
from pathlib import Path
import yaml
import random

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
LLVIP_IMGS_TRAIN = ROOT / 'data' / 'raw'       / 'llvip' / 'visible'  / 'train'
LLVIP_IMGS_TEST  = ROOT / 'data' / 'raw'       / 'llvip' / 'visible'  / 'test'
EXDARK_IMGS      = ROOT / 'data' / 'raw'       / 'exdark' / 'images'  / 'People'
LLVIP_LBLS_TRAIN = ROOT / 'data' / 'processed' / 'annotations_yolo'   / 'llvip' / 'train'
LLVIP_LBLS_TEST  = ROOT / 'data' / 'processed' / 'annotations_yolo'   / 'llvip' / 'test'
EXDARK_LBLS      = ROOT / 'data' / 'processed' / 'annotations_yolo'   / 'exdark'

# Unified dataset directories
DATASET_ROOT  = ROOT / 'data' / 'processed' / 'yolo_dataset'
TRAIN_IMGS    = DATASET_ROOT / 'images' / 'train'
VAL_IMGS      = DATASET_ROOT / 'images' / 'val'
TRAIN_LBLS    = DATASET_ROOT / 'labels' / 'train'
VAL_LBLS      = DATASET_ROOT / 'labels' / 'val'
YAML_OUT      = DATASET_ROOT / 'dataset.yaml'

for d in [TRAIN_IMGS, VAL_IMGS, TRAIN_LBLS, VAL_LBLS]:
    d.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path):
    """Try symlink first, fall back to copy (Windows compatibility)."""
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def link_split(img_dir: Path, lbl_dir: Path,
               out_img_dir: Path, out_lbl_dir: Path,
               prefix: str = ''):
    """Link all images + labels from a source split into unified dataset dirs."""
    VALID_EXT = {'.jpg', '.jpeg', '.png'}
    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in VALID_EXT]
    linked = 0

    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue

        # Add prefix to avoid name collisions between datasets
        new_stem    = prefix + img_path.stem if prefix else img_path.stem
        dst_img     = out_img_dir / (new_stem + img_path.suffix)
        dst_lbl     = out_lbl_dir / (new_stem + '.txt')

        symlink_or_copy(img_path, dst_img)
        symlink_or_copy(lbl_path, dst_lbl)
        linked += 1

    return linked


def main():
    print("Building unified YOLO dataset...")
    print(f"  Output → {DATASET_ROOT}\n")

    # Training set: LLVIP train + ExDARK People
    n_llvip_train = link_split(
        LLVIP_IMGS_TRAIN, LLVIP_LBLS_TRAIN,
        TRAIN_IMGS, TRAIN_LBLS,
        prefix='llvip_'
    )
    n_exdark = link_split(
        EXDARK_IMGS, EXDARK_LBLS,
        TRAIN_IMGS, TRAIN_LBLS,
        prefix='exdark_'
    )

    # Validation set: LLVIP test only
    n_val = link_split(
        LLVIP_IMGS_TEST, LLVIP_LBLS_TEST,
        VAL_IMGS, VAL_LBLS,
        prefix='llvip_'
    )

    total_train = n_llvip_train + n_exdark
    print(f"  Train : {total_train} images")
    print(f"    └── LLVIP   : {n_llvip_train}")
    print(f"    └── ExDARK  : {n_exdark}")
    print(f"  Val   : {n_val} images")

    # Write dataset.yaml
    config = {
        'path'  : str(DATASET_ROOT).replace('\\', '/'),
        'train' : 'images/train',
        'val'   : 'images/val',
        'nc'    : 1,
        'names' : {0: 'person'},
    }

    with open(YAML_OUT, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ dataset.yaml written → {YAML_OUT}")
    print(f"\nContents:")
    print(YAML_OUT.read_text())


if __name__ == '__main__':
    main()