"""
splits.py
=========
Official UCF-Crime Action_Recognition_splits loader + train/test leakage check.
Moved (logic unchanged) out of extract_clips.py.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .common import (SPLITS_DIR, UCF_DIR, VIDEOS_DIR, UCF_FOLDER_TO_LABEL,
                     LABELS, VALID_FOLDS, VIDEO_EXTENSIONS)


def load_official_fold(fold: int, verbose: bool = True) -> dict:
    """
    Parse Action_Recognition_splits/train_NNN.txt and test_NNN.txt.
    Returns {'train': {'Fighting': [(folder, stem), ...], ...}, 'test': {...}}
    Folder assignment done by scanning VIDEOS_DIR on disk — not the .txt label
    column which varies between dataset versions.
    """
    assert fold in VALID_FOLDS, f"fold must be one of {VALID_FOLDS}"

    train_file = SPLITS_DIR / f'train_{fold:03d}.txt'
    test_file  = SPLITS_DIR / f'test_{fold:03d}.txt'

    if not SPLITS_DIR.exists():
        raise FileNotFoundError(
            f"Action_Recognition_splits not found: {SPLITS_DIR}\n"
            f"Place at {UCF_DIR}/Action_Recognition_splits/"
        )
    for f in [train_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(f"Split file missing: {f}")

    if verbose:
        print(f"\n[Fold {fold}] Building stem→folder lookup from {VIDEOS_DIR}")
    stem_to_folder: dict = {}
    for ucf_folder in UCF_FOLDER_TO_LABEL:
        folder_dir = VIDEOS_DIR / ucf_folder
        if not folder_dir.exists():
            print(f"  [WARN] Not found on disk: {folder_dir}")
            continue
        for p in folder_dir.iterdir():
            if p.suffix in VIDEO_EXTENSIONS:
                stem_to_folder[p.stem] = ucf_folder

    result = {'train': defaultdict(list), 'test': defaultdict(list)}
    skipped_other = skipped_missing = 0

    for split_key, txt_file in [('train', train_file), ('test', test_file)]:
        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname      = line.split()[0]
                stem       = Path(fname).stem
                ucf_folder = stem_to_folder.get(stem)
                if ucf_folder is None:
                    skipped_missing += 1
                    continue
                label = UCF_FOLDER_TO_LABEL.get(ucf_folder)
                if label is None:
                    skipped_other += 1   # Assault etc. → silently ignored
                    continue
                result[split_key][label].append((ucf_folder, stem))

    result = {'train': dict(result['train']), 'test': dict(result['test'])}

    # Leakage check
    train_stems = {s for v in result['train'].values() for _, s in v}
    test_stems  = {s for v in result['test'].values()  for _, s in v}
    overlap = train_stems & test_stems
    if overlap:
        raise RuntimeError(
            f"Fold {fold}: {len(overlap)} stems appear in BOTH splits.\n"
            f"First 5: {sorted(overlap)[:5]}"
        )

    if verbose:
        print(f"[Fold {fold}] Split loaded ✅  "
              f"(skipped {skipped_other} non-target, {skipped_missing} missing)")
        print(f"\n  {'Label':10s}  {'Train':>5}  {'Test':>5}")
        print(f"  {'-'*28}")
        for label in LABELS:
            tr = len(result['train'].get(label, []))
            te = len(result['test'].get(label, []))
            print(f"  {label:10s}  {tr:5d}  {te:5d}")
        print(f"  ✅ No train/test leakage\n")
    return result


def split_of(stem: str, fold: int = 1):
    """'train' / 'test' / None for a video stem in a given official fold."""
    fs = load_official_fold(fold, verbose=False)
    for sp in ('train', 'test'):
        if any(stem == s for v in fs[sp].values() for _, s in v):
            return sp
    return None
