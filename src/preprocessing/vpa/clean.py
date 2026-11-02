"""
clean.py  (S3 — structural cleaning)
====================================
  • detect_shot_cuts       HSV-histogram distance + edge-change ratio (classical, per plan)
  • detect_frozen_black    duplicated / blank frames
  • detect_letterbox       black-bar detection from row/col variance profiles (+ apply_crop)
  • detect_overlay         timestamp / logo / caption detection from temporal stroke frequency
  • remove_overlay         cv2.inpaint (Telea) on the detected overlay strokes
  • clean_video            runs all four in order and returns cleaned frames + signals

All functions take frames as a list / (T,H,W,3) uint8 BGR array.
Modern alternatives (TransNetV2, EAST/PaddleOCR) are optional; see DECISIONS.md for status.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .common import load_config
from . import viz


# ═══════════════════════════════════════════════════════════════════════════════
# Letterbox / black bars
# ═══════════════════════════════════════════════════════════════════════════════

def detect_letterbox(frames, cfg: dict | None = None) -> dict:
    """
    Row / column variance profiles over sampled frames (cf. ffmpeg cropdetect).
    A bar row/col is flat black: mean luma < bar_mean_max and std < bar_std_max
    across all sampled frames and the whole row/col.
    Returns box=(x0,y0,x1,y1) of the content (x1,y1 exclusive) and the profiles.
    """
    c = (cfg or load_config())['clean']['letterbox']
    n = len(frames)
    idx = np.linspace(0, n - 1, min(c['sample_frames'], n)).round().astype(int)
    G = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in idx]).astype(np.float32)
    H, W = G.shape[1:]
    row_mean, row_std = G.mean(axis=(0, 2)), G.std(axis=(0, 2))
    col_mean, col_std = G.mean(axis=(0, 1)), G.std(axis=(0, 1))

    def bar(mean, std):
        return (mean < c['bar_mean_max']) & (std < c['bar_std_max'])

    def run_from_start(mask):
        k = 0
        while k < len(mask) and mask[k]:
            k += 1
        return k

    rb, cb = bar(row_mean, row_std), bar(col_mean, col_std)
    top, bottom = run_from_start(rb), run_from_start(rb[::-1])
    left, right = run_from_start(cb), run_from_start(cb[::-1])
    lim_h, lim_w = int(H * c['max_crop_frac']), int(W * c['max_crop_frac'])
    top, bottom = min(top, lim_h), min(bottom, lim_h)
    left, right = min(left, lim_w), min(right, lim_w)
    # ignore hair-thin bars (compression ringing) — only crop when substantial
    top    = top    if top    >= c['min_bar_px'] else 0
    bottom = bottom if bottom >= c['min_bar_px'] else 0
    left   = left   if left   >= c['min_bar_px'] else 0
    right  = right  if right  >= c['min_bar_px'] else 0
    # compression smears the bar edge by a pixel or two: trim a small safety margin on cropped sides only
    sp = c.get('safety_px', 2)
    top, bottom, left, right = [(v + sp) if v else 0 for v in (top, bottom, left, right)]
    # keep even sizes so downstream scalers behave
    x0, y0, x1, y1 = left, top, W - right, H - bottom
    if (x1 - x0) % 2: x1 -= 1
    if (y1 - y0) % 2: y1 -= 1
    return dict(box=(int(x0), int(y0), int(x1), int(y1)), bars=dict(top=top, bottom=bottom, left=left, right=right),
                has_bars=bool(top or bottom or left or right), frame_size=(W, H),
                row_mean=row_mean, row_std=row_std, col_mean=col_mean, col_std=col_std)


def apply_crop(frames, box):
    x0, y0, x1, y1 = box
    return [f[y0:y1, x0:x1] for f in frames]


# ═══════════════════════════════════════════════════════════════════════════════
# Timestamp / overlay text
# ═══════════════════════════════════════════════════════════════════════════════

def _strokes(gray: np.ndarray, thr: int, k: int, hi: int = 185, lo: int = 70) -> np.ndarray:
    """Thin, very high-contrast strokes that are also near-saturated (burnt-in text is white or black)."""
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)
    return ((th > thr) & (gray >= hi)) | ((bh > thr) & (gray <= lo))


def detect_overlay_gray(grays, cfg: dict | None = None) -> dict:
    """
    Classical overlay detector. Text / logos / camera captions are thin high-contrast strokes that
    appear at the SAME place in most frames (even when digits tick, the stroke *positions* keep
    recurring), while moving content does not. So:
        stroke map per frame (top-hat | black-hat)  →  per-pixel frequency over time
        → keep frequent pixels → close horizontally (merge characters) → connected components
        → keep text-like blobs (size, aspect, density) lying in the outer border band.
    """
    c = (cfg or load_config())['clean']['overlay']
    grays = [np.asarray(g) for g in grays]
    H, W = grays[0].shape
    freq = np.zeros((H, W), np.float32)
    for g in grays:
        freq += _strokes(g, c['stroke_thr'], c['stroke_kernel'], c['bright_min'], c['dark_max'])
    freq /= max(1, len(grays))

    cand = (freq >= c['freq_min']).astype(np.uint8)
    merged = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(merged, connectivity=8)
    bf = c['border_frac']
    region = np.zeros((H, W), np.uint8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < c['min_w'] or not (c['min_h'] <= h <= c['max_h']):
            continue
        if w / h < c['min_aspect']:
            continue
        dens = float(cand[y:y + h, x:x + w].mean())
        if not (c['min_density'] <= dens <= c['max_density']):
            continue
        cx, cy = x + w / 2, y + h / 2
        if not (cy < H * bf or cy > H * (1 - bf) or cx < W * bf or cx > W * (1 - bf)):
            continue
        region[lab == i] = 1
        comps.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), density=round(dens, 3)))
    area_frac = float(region.mean())
    if area_frac > c['max_area_frac']:       # implausibly large → almost surely scene texture
        region[:] = 0
        comps = []
        area_frac = 0.0
    region = cv2.dilate(region, np.ones((3, 3), np.uint8))
    return dict(region=region, components=comps, area_frac=area_frac, n_components=len(comps),
                freq=freq, score=float(area_frac * 1000 + len(comps)))


def detect_overlay(frames, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    n = len(frames)
    idx = np.linspace(0, n - 1, min(cfg['profile']['overlay_frames'], n)).round().astype(int)
    grays = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in idx]
    return detect_overlay_gray(grays, cfg)


def remove_overlay(frames, region: np.ndarray, cfg: dict | None = None, return_masks: bool = False):
    """Inpaint (Telea) the text strokes that lie inside `region`, frame by frame."""
    c = (cfg or load_config())['clean']['overlay']
    if region is None or not region.any():
        return (list(frames), None) if return_masks else list(frames)
    out, masks = [], []
    k = np.ones((3, 3), np.uint8)
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        m = (_strokes(g, c['stroke_thr'], c['stroke_kernel'], c['bright_min'], c['dark_max']).astype(np.uint8) & region)
        m = cv2.dilate(m, k, iterations=2)
        out.append(cv2.inpaint(f, m, c['inpaint_radius'], cv2.INPAINT_TELEA) if m.any() else f)
        masks.append(m)
    return (out, masks) if return_masks else out


# ═══════════════════════════════════════════════════════════════════════════════
# Shot-boundary detection
# ═══════════════════════════════════════════════════════════════════════════════

def _edge_change_ratio(e0: np.ndarray, e1: np.ndarray, kern) -> float:
    n0, n1 = int(e0.sum()), int(e1.sum())
    if n0 < 30 or n1 < 30:           # almost no edges (dark / blank) → let the histogram decide
        return 1.0
    d0, d1 = cv2.dilate(e0, kern), cv2.dilate(e1, kern)
    rho_in  = 1.0 - float((e1 & d0).sum()) / n1
    rho_out = 1.0 - float((e0 & d1).sum()) / n0
    return max(rho_in, rho_out)


def frame_signals(frames, cfg: dict | None = None) -> dict:
    """Per-transition HSV-histogram distance and edge-change ratio (index t = frame t-1 → t)."""
    c = (cfg or load_config())['clean']['shot_cut']
    bins = list(c['hist_bins'])
    kern = np.ones((5, 5), np.uint8)
    T = len(frames)
    hd, ecr = np.zeros(T), np.zeros(T)
    prev_h = prev_e = None
    for t, f in enumerate(frames):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
        cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0), 80, 160) > 0
        e = e.astype(np.uint8)
        if prev_h is not None:
            hd[t] = cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA)
            ecr[t] = _edge_change_ratio(prev_e, e, kern)
        prev_h, prev_e = h, e
    return dict(hist_dist=hd, ecr=ecr)


def transnetv2_probs(frames, models: dict | None = None):
    """TransNetV2 (pretrained, PyTorch port, weights ship in the pip wheel) single-frame transition probabilities.
    Runs on CPU: its Conv3D kernels are not available on this GPU/torch-nightly combination (TODO: retry on GPU)."""
    import torch
    from transnetv2_pytorch import TransNetV2
    models = models if models is not None else {}
    if 'transnetv2' not in models:
        models['transnetv2'] = TransNetV2(device='cpu')
    x = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (48, 27), interpolation=cv2.INTER_AREA) for f in frames])
    single, _ = models['transnetv2'].predict_frames(torch.from_numpy(x), quiet=True)
    return np.asarray(single).reshape(-1)


def detect_shot_cuts(frames, cfg: dict | None = None, signals: dict | None = None, models: dict | None = None,
                     method: str | None = None) -> dict:
    """
    Classical cut at transition t (frame t starts a new shot) when
        hist_dist[t] > max(hist_abs_min, localMedian + k*localMAD)   AND   ecr[t] > ecr_min
    followed by non-maximum suppression within `min_gap_frames`.
    method 'transnetv2' / 'both' adds TransNetV2 (probability > 0.5; 'both' = union, cuts within 3 frames merged).
    """
    c = (cfg or load_config())['clean']['shot_cut']
    method = method or c.get('method', 'classical')
    sig = signals or frame_signals(frames, cfg)
    hd, ecr = sig['hist_dist'], sig['ecr']
    T = len(hd)
    cuts_cls = []
    thr = np.zeros(T)
    if T >= 3:
        win = min(49, T if T % 2 else T - 1)
        med = median_filter(hd, size=win, mode='nearest')
        mad = median_filter(np.abs(hd - med), size=win, mode='nearest')
        thr = np.maximum(c['hist_abs_min'], med + c['hist_mad_k'] * 1.4826 * mad)
        cand = np.where((hd > thr) & (ecr > c['ecr_min']))[0]
        cand = cand[cand > 0]
        for t in sorted(cand, key=lambda i: -hd[i]):
            if all(abs(t - u) >= c['min_gap_frames'] for u in cuts_cls):
                cuts_cls.append(int(t))
        cuts_cls.sort()
    probs, cuts_tn, tn_error = None, [], None
    if method in ('transnetv2', 'both') and T >= 3:
        try:
            probs = transnetv2_probs(frames, models)
            cuts_tn = [int(i) for i in np.where(probs > 0.5)[0] if i > 0]
            keep = []                                               # a cut spans 1-2 frames of high probability: keep the peak
            for i in sorted(cuts_tn, key=lambda i: -probs[i]):
                if all(abs(i - j) > 3 for j in keep):
                    keep.append(i)
            cuts_tn = sorted(keep)
        except Exception as e:                                       # logged as TODO; classical result still returned
            tn_error = f'{type(e).__name__}: {e}'
            method = 'classical'
    if method == 'classical':
        cuts = cuts_cls
    elif method == 'transnetv2':
        cuts = cuts_tn
    else:
        cuts = []
        for i in sorted(cuts_cls + cuts_tn):
            if not cuts or i - cuts[-1] > 3:
                cuts.append(i)
    return dict(cuts=cuts, thr=thr, hist_dist=hd, ecr=ecr, cuts_classical=cuts_cls, cuts_transnetv2=cuts_tn,
                transnet_probs=probs, method=method, transnet_error=tn_error)


def shot_ids_from_cuts(n_frames: int, cuts) -> np.ndarray:
    ids = np.zeros(n_frames, np.int32)
    for c in cuts:
        ids[c:] += 1
    return ids


def pyscenedetect_cuts(video_path: str, fps_target: float = 12.0) -> list:
    """Cross-check with PySceneDetect's ContentDetector. Returns cut times in seconds."""
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(str(video_path), ContentDetector(threshold=27.0, min_scene_len=15))
        return [float(s[0].get_seconds()) for s in scenes[1:]]
    except Exception as e:                                  # pragma: no cover
        return [f'error: {e}']


# ═══════════════════════════════════════════════════════════════════════════════
# Frozen / black frames
# ═══════════════════════════════════════════════════════════════════════════════

def detect_frozen_black(frames, cfg: dict | None = None) -> dict:
    c = (cfg or load_config())['clean']
    T = len(frames)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diff = np.zeros(T)
    for t in range(1, T):
        diff[t] = float(np.abs(gray[t].astype(np.int16) - gray[t - 1].astype(np.int16)).mean())
    mean = np.array([g.mean() for g in gray])
    std = np.array([g.std() for g in gray])
    black = (mean < c['black']['mean_max']) & (std < c['black']['std_max'])
    ident = diff < c['frozen']['diff_thr']
    ident[0] = False
    frozen = np.zeros(T, bool)                 # only runs of >= min_run identical frames
    t = 0
    runs = []
    while t < T:
        if ident[t]:
            s = t
            while t < T and ident[t]:
                t += 1
            if t - s >= c['frozen']['min_run']:
                frozen[s - 1:t] = True
                runs.append((s - 1, t - 1))
        else:
            t += 1
    return dict(frozen=frozen, black=black, diff=diff, mean=mean, std=std, frozen_runs=runs,
                n_frozen=int(frozen.sum()), n_black=int(black.sum()))


def bad_frame_mask(cl: dict, n_frames: int, cut_margin: int = 2, black_margin: int = 3) -> np.ndarray:
    """Frames that must not drive activity analysis: frozen, black (+ the fade frames after), around cuts."""
    bad = np.zeros(n_frames, bool)
    fz = cl.get('frozen')
    if fz is not None:
        bad |= fz['frozen']
        blk = fz['black'].copy()
        for k in range(1, black_margin + 1):
            blk[k:] |= fz['black'][:-k]
        bad |= blk
    for c in cl.get('cuts', []):
        bad[max(0, c - cut_margin):c + cut_margin + 1] = True
    return bad


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def clean_video(frames, cfg: dict | None = None, video: str | None = None, native_fps: float = 12.0,
                do_letterbox=True, do_overlay=True, do_cuts=True, do_frozen=True,
                video_path: str | None = None, models: dict | None = None) -> dict:
    """Run S3 and (if sample saving is on) save every operation's input/output sample."""
    cfg = cfg or load_config()
    frames = list(frames)
    res = dict(frames=frames, crop_box=(0, 0, frames[0].shape[1], frames[0].shape[0]),
               overlay=None, cuts=[], shot_ids=np.zeros(len(frames), np.int32), frozen=None)
    raw = frames

    if do_letterbox and cfg['clean']['letterbox']['enabled']:
        lb = detect_letterbox(frames, cfg)
        frames = apply_crop(frames, lb['box']) if lb['has_bars'] else frames
        res.update(letterbox=lb, crop_box=lb['box'])
        if video:
            sample_letterbox(video, raw, frames, lb, cfg)

    if do_overlay and cfg['clean']['overlay']['enabled']:
        ov = detect_overlay(frames, cfg)
        before = frames
        frames, masks = remove_overlay(frames, ov['region'], cfg, return_masks=True)
        res['overlay'] = ov
        if video:
            sample_overlay(video, before, frames, ov, masks, cfg)

    if do_cuts:
        sd = detect_shot_cuts(frames, cfg, models=models)
        res.update(cuts=sd['cuts'], shot_ids=shot_ids_from_cuts(len(frames), sd['cuts']), shot_signals=sd)
        if video:
            sample_shot_cuts(video, frames, sd, cfg, video_path=video_path, fps=native_fps)

    if do_frozen:
        fz = detect_frozen_black(frames, cfg)
        res['frozen'] = fz
        if video:
            sample_frozen_black(video, frames, fz, cfg)

    res['frames'] = frames
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# Sample saving
# ═══════════════════════════════════════════════════════════════════════════════

def _mid(frames):
    return frames[len(frames) // 2]


def sample_letterbox(video, before, after, lb, cfg):
    if not viz.should_save(video):
        return
    f0 = _mid(before)
    x0, y0, x1, y1 = lb['box']
    marked = f0.copy()
    cv2.rectangle(marked, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)

    def plot():
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
        axs[0].plot(lb['row_mean'], label='row mean'); axs[0].plot(lb['row_std'], label='row std')
        axs[0].axvspan(0, lb['bars']['top'], color=viz.C['red'], alpha=.25)
        axs[0].axvspan(len(lb['row_mean']) - lb['bars']['bottom'], len(lb['row_mean']), color=viz.C['red'], alpha=.25)
        axs[0].set_title('row profile (y)'); axs[0].legend()
        axs[1].plot(lb['col_mean'], label='col mean'); axs[1].plot(lb['col_std'], label='col std')
        axs[1].axvspan(0, lb['bars']['left'], color=viz.C['red'], alpha=.25)
        axs[1].axvspan(len(lb['col_mean']) - lb['bars']['right'], len(lb['col_mean']), color=viz.C['red'], alpha=.25)
        axs[1].set_title('column profile (x)')
        fig.tight_layout()
        return fig

    viz.save_io_sample(
        'S3', 'letterbox_crop', video, inputs=marked, outputs=_mid(after),
        params=dict(bar_mean_max=cfg['clean']['letterbox']['bar_mean_max'],
                    bar_std_max=cfg['clean']['letterbox']['bar_std_max']),
        metrics=dict(box=lb['box'], bars=lb['bars'], has_bars=lb['has_bars'],
                     size_before=list(f0.shape[1::-1]), size_after=list(_mid(after).shape[1::-1])),
        note=('Detects black letter/pillar-box bars from flat-black row/column profiles and crops them.\n'
              'Why: bars burn part of the 224x224 model input on nothing.\n'
              'Notice: green box = kept content; red bands in plot.png = detected bars (none -> unchanged).'),
        plot=plot, labels=('input (green = content box)', 'output (cropped)'))


def sample_overlay(video, before, after, ov, masks, cfg):
    if not viz.should_save(video):
        return
    i = len(before) // 2
    mask_vis = cv2.cvtColor((ov['region'] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    boxed = before[i].copy()
    for cmp in ov['components']:
        cv2.rectangle(boxed, (cmp['x'], cmp['y']), (cmp['x'] + cmp['w'], cmp['y'] + cmp['h']), (0, 0, 255), 1)
    inp_mask = (masks[i] * 255).astype(np.uint8) if masks else np.zeros(before[i].shape[:2], np.uint8)

    def plot():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        im = ax.imshow(ov['freq'], cmap='magma'); ax.axis('off')
        ax.set_title('stroke frequency over time (bright = recurring text-like)')
        fig.colorbar(im, ax=ax, fraction=0.04)
        return fig

    clip_n = min(18, len(before))
    viz.save_io_sample(
        'S3', 'overlay_text_mask', video, inputs=boxed,
        outputs=[mask_vis, after[i]], output_labels=['detected region', 'inpainted (Telea)'],
        params=dict(method='classical (top-hat/black-hat stroke frequency + CC filter) + Telea inpaint',
                    freq_min=cfg['clean']['overlay']['freq_min'],
                    border_frac=cfg['clean']['overlay']['border_frac']),
        metrics=dict(n_components=ov['n_components'], area_frac=round(ov['area_frac'], 5),
                     components=ov['components'],
                     mean_abs_change_in_masked_px=float(
                         np.abs(before[i].astype(np.int16) - after[i].astype(np.int16)).mean())),
        note=('Finds timestamps / logos / captions as recurring high-contrast strokes in the border band, then inpaints them.\n'
              'Why: a camera-specific timestamp is a shortcut the model can learn instead of the action.\n'
              'Notice: red boxes = detected overlay, middle = region mask, right = frame after inpainting; none found -> unchanged.'),
        plot=plot, input_clip=before[:clip_n], output_clip=after[:clip_n], labels=('input', 'region + inpainted output'))


def sample_shot_cuts(video, frames, sd, cfg, video_path=None, fps=12.0):
    if not viz.should_save(video):
        return
    cuts = sd['cuts']

    def plot():
        import matplotlib.pyplot as plt
        t = np.arange(len(sd['hist_dist'])) / fps
        has_tn = sd.get('transnet_probs') is not None
        fig, axs = plt.subplots(3 if has_tn else 2, 1, figsize=(9, 5.6 if has_tn else 4.2), sharex=True)
        axs[0].plot(t, sd['hist_dist'], label='HSV hist distance (Bhattacharyya)')
        axs[0].plot(t, sd['thr'], '--', color=viz.INK2, label='adaptive threshold')
        axs[1].plot(t, sd['ecr'], color=viz.C['orange'], label='edge-change ratio')
        axs[1].axhline(cfg['clean']['shot_cut']['ecr_min'], ls='--', color=viz.INK2)
        if has_tn:
            axs[2].plot(t, sd['transnet_probs'], color=viz.C['violet'], label='TransNetV2 transition probability')
            axs[2].axhline(0.5, ls='--', color=viz.INK2)
        for ax in axs:
            for c in cuts:
                ax.axvline(c / fps, color=viz.C['red'], alpha=.7)
            ax.legend(loc='upper right')
        axs[-1].set_xlabel('time (s, 12 FPS timeline)')
        fig.suptitle(f'{len(cuts)} shot cut(s) (red) = {sd["method"]}: classical {len(sd["cuts_classical"])}, TransNetV2 {len(sd["cuts_transnetv2"])}', fontsize=9)
        fig.tight_layout()
        return fig

    if cuts:
        c = cuts[int(np.argmax(sd['hist_dist'][cuts]))]
        inp, out = frames[c - 1], frames[c]
        lab = (f'frame {c-1} (last of shot)', f'frame {c} (first of next shot)')
        clip = frames[max(0, c - 8):c + 8]
    else:
        c = int(np.argmax(sd['hist_dist']))
        inp, out = frames[max(0, c - 1)], frames[c]
        lab = (f'frame {c-1} (largest-change pair, no cut)', f'frame {c}')
        clip = frames[max(0, c - 8):c + 8]
    ps = pyscenedetect_cuts(video_path) if video_path else []
    viz.save_io_sample(
        'S3', 'shot_boundary', video, inputs=inp, outputs=out,
        params=dict(detector='HSV hist + edge-change ratio', **{k: v for k, v in cfg['clean']['shot_cut'].items() if k not in ('hist_bins', 'method')}),
        metrics=dict(n_cuts=len(cuts), cut_frames=cuts, cut_times_s=[round(c_ / fps, 2) for c_ in cuts], method=sd['method'],
                     cuts_classical=sd['cuts_classical'], cuts_transnetv2=sd['cuts_transnetv2'], transnetv2_error=sd['transnet_error'],
                     pyscenedetect_cut_times_s=ps, max_hist_dist=float(np.max(sd['hist_dist']))),
        note=('Finds hard cuts by combining an HSV histogram distance with an edge-change ratio, then splits the video into shots.\n'
              'Why: a clip spanning a cut mixes two unrelated scenes (edited/compilation footage).\n'
              'Notice: the red lines in plot.png; left/right images are the frames either side of the strongest cut.'),
        plot=plot, input_clip=clip, labels=lab)


def sample_frozen_black(video, frames, fz, cfg):
    if not viz.should_save(video):
        return
    T = len(frames)

    def plot():
        import matplotlib.pyplot as plt
        t = np.arange(T) / 12.0
        fig, axs = plt.subplots(2, 1, figsize=(9, 3.8), sharex=True)
        axs[0].plot(t, fz['diff'], label='mean |frame diff|')
        axs[0].axhline(cfg['clean']['frozen']['diff_thr'], ls='--', color=viz.INK2)
        axs[0].set_ylim(0, max(2, float(np.percentile(fz['diff'], 99))))
        axs[1].plot(t, fz['mean'], color=viz.C['aqua'], label='mean luma')
        for ax, m, col in ((axs[0], fz['frozen'], viz.C['red']), (axs[1], fz['black'], viz.C['violet'])):
            ymax = ax.get_ylim()[1]
            ax.fill_between(t, 0, ymax, where=m, color=col, alpha=.35, step='mid')
            ax.legend(loc='upper right')
        axs[1].set_xlabel('time (s)')
        fig.suptitle(f'frozen frames: {fz["n_frozen"]} (red), black frames: {fz["n_black"]} (violet)', fontsize=9)
        fig.tight_layout()
        return fig

    bad = np.where(fz['frozen'] | fz['black'])[0]
    i = int(bad[len(bad) // 2]) if len(bad) else int(np.argmin(fz['diff'][1:]) + 1)
    lab_o = 'flagged frame' if len(bad) else 'lowest-diff frame (nothing flagged)'
    viz.save_io_sample(
        'S3', 'frozen_black_frames', video, inputs=viz.filmstrip(frames, 6, 100),
        outputs=frames[i], params=dict(diff_thr=cfg['clean']['frozen']['diff_thr'],
                                       min_run=cfg['clean']['frozen']['min_run'],
                                       black_mean_max=cfg['clean']['black']['mean_max']),
        metrics=dict(n_frames=T, n_frozen=fz['n_frozen'], n_black=fz['n_black'],
                     frozen_runs=fz['frozen_runs'][:10], frozen_frac=round(fz['n_frozen'] / T, 4)),
        note=('Flags duplicated (frame-diff ~ 0) and blank frames on the 12 FPS timeline.\n'
              'Why: they look like "no motion" and would contaminate clips; the quality gate (S8) rejects clips that contain many.\n'
              'Notice: shaded spans on the timeline are flagged frames (none = clean video).'),
        plot=plot, labels=('overview (6 frames)', lab_o))
