"""
legacy.py
=========
Re-creates the EXISTING pre-processing operations (as they were before the VPA pipeline) so each can be
shown with a sample input and output:

  X1  frame_sampling_step        old `step = round(src_fps / 12)` sampling
  X2  resize_stretch_vs_letterbox old `cv2.resize(frame, (640, 640))` vs aspect-preserving letterbox
  X3  zerodce_dark_frame         dark frame before / after Zero-DCE++ (existing ZeroDCEInference class)
  X6  yolo_annotation_boxes      LLVIP / ExDARK YOLO labels drawn on the image (verify_annotations drawing code)

(X4 dedup and X5 validity-check samples live in temporal.py, next to the code that produces the clips.)
These helpers are demonstration-only; the real pipeline uses the new modules.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, ZERODCE_W, letterbox, load_config, mean_luma
from . import viz


def legacy_first_frames(path: str, n: int = 6):
    """First n frames the OLD sampler would keep (cv2, step = round(fps/12))."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / 12))
    out, i = [], 0
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append(f)
        i += 1
    cap.release()
    return out, fps, step


def sample_frame_sampling(video: str, path: str, n_src_frames: int, src_fps: float):
    """X1 — the old fixed-step sampling: original timeline vs kept frames."""
    if not viz.should_save(video):
        return
    frames, fps, step = legacy_first_frames(path, 6)
    idx = np.arange(0, min(n_src_frames, int(fps * 2)), step)

    def plot():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 2.4))
        t_all = np.arange(int(fps * 1.5)) / fps
        ax.eventplot([t_all, idx[idx / fps <= 1.5] / fps], lineoffsets=[1, 0], linelengths=0.7,
                     colors=[viz.INK2, viz.C['orange']])
        ax.set_yticks([1, 0]); ax.set_yticklabels(['source frames', f'kept (step={step})'])
        ax.set_xlabel('time (s)')
        ax.set_title(f'OLD sampling: every {step}-th frame of a {fps:.0f} FPS video -> effective {fps / step:.1f} FPS (target 12)')
        fig.tight_layout()
        return fig

    viz.save_io_sample(
        'X1', 'frame_sampling_step', video, inputs=frames[:1], outputs=frames, cols=3,
        output_labels=[f'#{i * step}  t={i * step / fps:.3f}s' for i in range(len(frames))],
        params=dict(rule='step = round(src_fps / 12)', src_fps=fps, step=step),
        metrics=dict(src_fps=fps, step=step, effective_fps=round(fps / step, 3), target_fps=12,
                     speed_error_pct=round(100 * (fps / step - 12) / 12, 1)),
        note=(f'EXISTING op (extract_clips._load_frames): keep every round(src_fps/12)-th decoded frame; {fps:.0f} FPS -> step {step} -> {fps / step:.1f} FPS.\n'
              'Why it is a gap: motion speed then differs from video to video and a stride-4 clip no longer spans ~5.3 s.\n'
              'Notice: kept frames are evenly spaced but at the wrong rate; replaced by S2 (timestamp-based).'),
        plot=plot, labels=('source frame 0', 'frames kept by the old rule'))


def sample_resize(video: str, frame: np.ndarray):
    """X2 — old stretch to 640x640 vs aspect-preserving letterbox."""
    if not viz.should_save(video):
        return
    stretched = cv2.resize(frame, (640, 640))
    lb, scale, pad = letterbox(frame, 640, pad_value=0, interpolation=cv2.INTER_LINEAR)
    h, w = frame.shape[:2]
    viz.save_io_sample(
        'X2', 'resize_stretch_vs_letterbox', video, inputs=stretched, outputs=lb,
        params=dict(old='cv2.resize(frame,(640,640))', new='letterbox 640x640 (pad black)', src_size=f'{w}x{h}'),
        metrics=dict(src_aspect=round(w / h, 3), old_horizontal_stretch=round((640 / w) / (640 / h), 3),
                     new_scale=round(scale, 3), new_pad_xy=pad),
        note=(f'EXISTING op: every frame was stretched to 640x640; a {w}x{h} frame (4:3) gets 33% horizontal stretch.\n'
              'Why it is a gap: distorted body shapes; letterboxing keeps the aspect ratio.\n'
              'Notice: people look fatter on the left; the letterboxed version keeps true proportions (black bars).'),
        labels=('OLD: stretched 640x640', 'NEW: letterboxed 640x640'))


def sample_zerodce_dark_frame(video: str, frames: list, device: str = 'cuda', cache: dict | None = None):
    """X3 — the darkest frame of the video through the EXISTING ZeroDCEInference (forced on for the demo)."""
    if not viz.should_save(video) or not frames:
        return
    sys.path.insert(0, str(ROOT / 'src'))
    from inference.infer_zerodce import ZeroDCEInference
    lum = [mean_luma(f) for f in frames]
    ok = [k for k, v in enumerate(lum) if v >= 10] or list(range(len(lum)))      # skip fully black (fade) frames
    i = min(ok, key=lambda k: lum[k])
    legacy_thr = 50
    cache = cache if cache is not None else {}
    if 'legacy_zerodce' not in cache:
        cache['legacy_zerodce'] = ZeroDCEInference(str(ZERODCE_W), device=device, brightness_threshold=255,
                                                   output_size=640, enhance_size=256)
    enh = cache['legacy_zerodce']
    before, _ = cv2.resize(frames[i], (640, 640)), None
    after, was = enh.enhance(frames[i])
    viz.save_io_sample(
        'X3', 'zerodce_dark_frame', video, inputs=before, outputs=after,
        params=dict(weights='stage2_best.pth', legacy_brightness_threshold=legacy_thr, enhance_size=256,
                    forced_for_demo=True),
        metrics=dict(frame_index=i, luma_before=round(lum[i], 2), luma_after=round(mean_luma(after), 2),
                     would_trigger_at_legacy_threshold=bool(lum[i] < legacy_thr)),
        note=(f'EXISTING op: Zero-DCE++ (stage-2 weights) brightens the darkest frame of the video (luma {lum[i]:.0f}); trigger was grayscale-mean < 50.\n'
              'Why: low-light CCTV hides people from YOLO/VideoMAE.\n'
              'Notice: shadows lift while highlights are preserved; if the video is bright the demo forces the enhancer on.'),
        labels=('before (darkest frame, 640 stretch)', 'after Zero-DCE++ (existing path)'))


def sample_annotation_boxes(n: int = 2, seed: int = 42):
    """X6 — YOLO annotation conversion check using verify_annotations.draw_boxes_on_image (skip if data missing)."""
    if not viz.should_save('ALL'):
        return
    sys.path.insert(0, str(ROOT / 'src'))
    from preprocessing import verify_annotations as va
    rng = random.Random(seed)
    sets = [('llvip', va.LLVIP_IMG_TRAIN, va.LLVIP_LABELS_TRAIN), ('exdark', va.EXDARK_IMG, va.EXDARK_LABELS)]
    ins, outs, used = [], [], []
    for name, img_dir, lbl_dir in sets:
        if not img_dir.exists() or not lbl_dir.exists():
            print(f'  [X6] {name}: data missing -> skipped')
            continue
        imgs = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        imgs = [p for p in imgs if (lbl_dir / (p.stem + '.txt')).exists() and (lbl_dir / (p.stem + '.txt')).stat().st_size > 0]
        for p in rng.sample(imgs, min(n, len(imgs))):
            tmp = viz.sample_dir('X6', 'yolo_annotation_boxes', 'ALL') / f'_{name}_{p.stem}.jpg'
            va.draw_boxes_on_image(p, lbl_dir / (p.stem + '.txt'), tmp)
            outs.append(cv2.imread(str(tmp))); ins.append(cv2.imread(str(p))); used.append(f'{name}/{p.stem}')
            tmp.unlink()
    if not ins:
        return
    viz.save_io_sample(
        'X6', 'yolo_annotation_boxes', 'ALL', inputs=ins, outputs=outs, tile_h=260, cols=2,
        params=dict(code='verify_annotations.draw_boxes_on_image', images=used),
        metrics=dict(images=used),
        note=('EXISTING op: LLVIP XML / ExDARK bbGt annotations converted to YOLO txt, drawn back on the image to verify the conversion.\n'
              'Why: a wrong conversion silently trains the person detector on shifted boxes.\n'
              'Notice: green boxes sit tightly on people in the original images (top row = raw image, bottom row = boxes drawn).'),
        labels=('original images', 'converted YOLO boxes drawn'))
