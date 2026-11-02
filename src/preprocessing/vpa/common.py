"""
common.py
=========
Shared helpers for the VPA (video processing & analytics) pipeline:
paths, config loading, the ONE brightness measure, letterboxing, video listing.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[3]
UCF_DIR     = ROOT / 'data' / 'raw' / 'ucf_crime'
VIDEOS_DIR  = UCF_DIR / 'videos'
SPLITS_DIR  = UCF_DIR / 'Action_Recognition_splits'
TEMPORAL_GT = UCF_DIR / 'annotations' / 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
ZERODCE_W   = ROOT / 'models' / 'zerodce' / 'stage2_best.pth'
YOLO_FT_W   = ROOT / 'models' / 'yolov11' / 'finetune_llvip_exdark' / 'weights' / 'best.pt'
YOLO_COCO_W = ROOT / 'models' / 'yolov11' / 'yolo11s.pt'
DOCS_DIR    = ROOT / 'docs' / 'vpa_samples'
CLIPS_DIR        = ROOT / 'data' / 'processed' / 'clips_vpa'
CLIPS_SAMPLE_DIR = ROOT / 'data' / 'processed' / 'clips_vpa_sample'
CACHE_DIR   = ROOT / 'outputs' / 'vpa_cache'
CONFIG_PATH = Path(__file__).with_name('config.yaml')

UCF_FOLDER_TO_LABEL = {
    'Fighting'           : 'Fighting',
    'Robbery'            : 'Robbery',
    'Vandalism'          : 'Vandalism',
    'Normal_Videos_event': 'Normal',
}
LABELS           = ['Fighting', 'Robbery', 'Vandalism', 'Normal']
ABNORMAL_LABELS  = ('Fighting', 'Robbery', 'Vandalism')
VALID_FOLDS      = [1, 2, 3, 4]
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.MP4', '.AVI', '.MKV'}

_CFG = None


def load_config(path: Path | None = None, reload: bool = False) -> dict:
    """Load config.yaml once (cached)."""
    global _CFG
    if _CFG is None or reload or path is not None:
        with open(path or CONFIG_PATH, 'r', encoding='utf-8') as f:
            _CFG = yaml.safe_load(f)
    return _CFG


# ── Brightness: the single shared measure ──────────────────────────────────────
def mean_luma(frame_bgr: np.ndarray) -> float:
    """
    Mean luminance Y (0-255) of a BGR uint8 frame (ITU-R BT.601, via cv2 gray).
    This is THE brightness measure for the whole project; extract_clips.py,
    infer_zerodce.py and backend/utils.py should all call it (or the identical
    formula) with the same threshold from config.yaml -> enhance.dark_threshold.
    """
    return float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())


def is_dark(frame_bgr: np.ndarray, threshold: float | None = None) -> bool:
    if threshold is None:
        threshold = load_config()['enhance']['dark_threshold']
    return mean_luma(frame_bgr) < threshold


# ── Geometry ───────────────────────────────────────────────────────────────────
def letterbox(img: np.ndarray, size=224, pad_value: int = 0,
              interpolation: int = cv2.INTER_AREA):
    """
    Aspect-preserving resize + pad to (size, size). Returns (out, scale, (pad_x, pad_y)).
    INTER_AREA is the anti-aliased choice for down-scaling; falls back to
    INTER_LINEAR when enlarging.
    """
    if isinstance(size, int):
        size = (size, size)
    tw, th = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = interpolation if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.full((th, tw) + img.shape[2:], pad_value, dtype=img.dtype)
    px, py = (tw - nw) // 2, (th - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, (px, py)


# ── Video listing ──────────────────────────────────────────────────────────────
def list_videos() -> list:
    """All target-class UCF-Crime videos on disk: [{stem, folder, label, path}]."""
    out = []
    for folder, label in UCF_FOLDER_TO_LABEL.items():
        d = VIDEOS_DIR / folder
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix in VIDEO_EXTENSIONS:
                out.append(dict(stem=p.stem, folder=folder, label=label, path=str(p)))
    return out


def find_video(stem: str):
    for v in list_videos():
        if v['stem'] == stem:
            return v
    return None


def load_temporal_gt() -> dict:
    """
    Parse UCF-Crime temporal annotations.
    Returns {stem: [(start_frame, end_frame), ...]} (frame indices at the ORIGINAL 30 FPS;
    -1 entries dropped). Empty dict if the file is missing.
    """
    if not TEMPORAL_GT.exists():
        return {}
    gt = {}
    for line in TEMPORAL_GT.read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        stem = Path(parts[0]).stem
        nums = [int(x) for x in parts[2:6]]
        segs = [(nums[i], nums[i + 1]) for i in (0, 2) if nums[i] >= 0 and nums[i + 1] >= 0]
        gt[stem] = segs
    return gt


def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
