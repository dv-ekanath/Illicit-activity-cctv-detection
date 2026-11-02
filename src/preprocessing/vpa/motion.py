"""
motion.py  (S6 — motion analysis)
==================================
Per-frame motion features on the 12-FPS (cleaned, enhanced) frames:

  background subtraction : MOG2 -> HSV shadow removal -> morphological open/close -> connected components
                           => foreground mask, blob count, foreground ratio
  dense optical flow     : Farneback (classical) [+ RAFT-small on sample frames for comparison]
                           => mean magnitude, direction entropy, divergence, curl
  Motion History Image   : compact picture of the last second of motion
  motion-energy signal   : one value per frame (mean flow magnitude, px per 12-FPS step), consumed by S8
"""

from __future__ import annotations

import cv2
import numpy as np

from .common import load_config
from . import viz


# ═══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════════

def remove_shadows(frame_bgr: np.ndarray, bg_bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    HSV shadow rule: a cast shadow darkens V (ratio in [v_low, v_high]) but keeps hue and saturation.
    Foreground pixels that satisfy the rule are removed from the mask.
    """
    c = cfg['motion']['shadow']
    hf = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ratio = hf[..., 2] / (hb[..., 2] + 1e-3)
    dh = np.abs(hf[..., 0] - hb[..., 0])
    dh = np.minimum(dh, 180 - dh)
    ds = np.abs(hf[..., 1] - hb[..., 1])
    shadow = (ratio >= c['v_low']) & (ratio <= c['v_high']) & (ds <= c['s_tol']) & (dh <= c['h_tol'])
    out = mask.copy()
    out[shadow] = 0
    return out, shadow


def clean_mask(mask: np.ndarray, cfg: dict):
    """Morphological opening (remove speckle) then closing (fill holes); drop tiny blobs."""
    m = cfg['motion']
    op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m['morph']['open'],) * 2)
    cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m['morph']['close'],) * 2)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, op)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cl)
    return opened, closed


def label_components(mask: np.ndarray, min_area: int):
    n, lab, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    filt = keep[lab]
    lab2 = np.where(filt, lab, 0)
    blobs = [dict(x=int(stats[i, 0]), y=int(stats[i, 1]), w=int(stats[i, 2]), h=int(stats[i, 3]),
                  area=int(stats[i, 4])) for i in range(1, n) if keep[i]]
    return filt.astype(np.uint8) * 255, lab2, blobs


def farneback(prev_gray: np.ndarray, gray: np.ndarray, cfg: dict) -> np.ndarray:
    f = cfg['motion']['flow']
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, f['pyr_scale'], f['levels'], f['winsize'],
                                        f['iterations'], f['poly_n'], f['poly_sigma'], 0)


def flow_features(flow: np.ndarray, cfg: dict) -> dict:
    """Mean magnitude, direction entropy (0 = coherent .. 1 = chaotic), mean |divergence|, mean |curl|."""
    m = cfg['motion']
    u, v = flow[..., 0], flow[..., 1]
    mag = np.hypot(u, v)
    moving = mag > m['flow_min_mag']
    frac = float(moving.mean())
    ent = div = curl = 0.0
    if moving.sum() > 0.005 * mag.size:
        ang = np.arctan2(v, u)[moving]
        w = mag[moving]
        h, _ = np.histogram(ang, bins=m['entropy_bins'], range=(-np.pi, np.pi), weights=w)
        p = h / (h.sum() + 1e-9)
        p = p[p > 0]
        ent = float(-(p * np.log(p)).sum() / np.log(m['entropy_bins']))
        us, vs = cv2.GaussianBlur(u, (0, 0), 2.0), cv2.GaussianBlur(v, (0, 0), 2.0)
        dudx, dudy = np.gradient(us, axis=1), np.gradient(us, axis=0)
        dvdx, dvdy = np.gradient(vs, axis=1), np.gradient(vs, axis=0)
        div = float(np.abs(dudx + dvdy)[moving].mean())
        curl = float(np.abs(dvdx - dudy)[moving].mean())
    return dict(mag=float(mag.mean()), moving_frac=frac, entropy=ent, div=div, curl=curl)


def flow_to_hsv(flow: np.ndarray, max_mag: float = 6.0) -> np.ndarray:
    """Optical flow -> BGR colour image (hue = direction, brightness = magnitude)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def colorize_labels(lab: np.ndarray) -> np.ndarray:
    n = int(lab.max()) + 1
    rng = np.random.RandomState(3)
    pal = np.vstack([[0, 0, 0], rng.randint(60, 255, (max(n, 1), 3))]).astype(np.uint8)
    return pal[np.minimum(lab, len(pal) - 1)]


def mhi_to_bgr(mhi: np.ndarray, tau: float) -> np.ndarray:
    g = np.clip(mhi / tau * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)


# ═══════════════════════════════════════════════════════════════════════════════
# RAFT (modern flow, sample frames only)
# ═══════════════════════════════════════════════════════════════════════════════

class RaftFlow:
    """torchvision RAFT-small (pretrained, no training). Loaded lazily; failure is non-fatal."""

    def __init__(self, device: str = 'cuda'):
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.torch = torch
        self.dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        w = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=w).to(self.dev).eval()
        self.tfm = w.transforms()

    def __call__(self, f0_bgr: np.ndarray, f1_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = f0_bgr.shape[:2]
        H8, W8 = max(8, int(round(H / 8)) * 8), max(8, int(round(W / 8)) * 8)

        def prep(f):
            f = cv2.resize(f, (W8, H8), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(f[:, :, ::-1].copy()).permute(2, 0, 1)[None].to(self.dev)
        a, b = self.tfm(prep(f0_bgr), prep(f1_bgr))
        with torch.no_grad():
            flow = self.model(a, b)[-1][0].permute(1, 2, 0).cpu().numpy()
        flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= W / W8
        flow[..., 1] *= H / H8
        return flow


# ═══════════════════════════════════════════════════════════════════════════════
# Whole-video analysis
# ═══════════════════════════════════════════════════════════════════════════════

def cheap_energy(frames) -> np.ndarray:
    """Mean abs gray difference per frame (used to pick the showcase frame)."""
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    e = np.zeros(len(g))
    for t in range(1, len(g)):
        e[t] = cv2.absdiff(g[t], g[t - 1]).mean()
    return e


def analyze_motion(frames, cfg: dict | None = None, showcase: int | None = None, window: int = 8,
                   cam_shifts=None) -> dict:
    """
    Returns per-frame arrays (len T): fg_ratio, blob_count, flow_mag, flow_entropy, flow_div, flow_curl,
    motion_energy (= flow_mag), plus a `show` dict with intermediate images around `showcase`.
    `cam_shifts` (optional, T x 2) are S5 per-frame camera translations subtracted from the flow.
    """
    cfg = cfg or load_config()
    m = cfg['motion']
    T = len(frames)
    mog = cv2.createBackgroundSubtractorMOG2(history=m['mog2']['history'], varThreshold=m['mog2']['var_threshold'],
                                             detectShadows=False)
    for f in frames[:min(m['mog2']['warmup_frames'], T)]:        # prime the background model
        mog.apply(f)
    out = {k: np.zeros(T) for k in ('fg_ratio', 'blob_count', 'flow_mag', 'flow_entropy', 'flow_div',
                                     'flow_curl', 'flow_moving_frac', 'shadow_frac')}
    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)     # last DISTINCT frame (flow reference)
    ref_t, dup_run = 0, 0
    last_flow, last_ff = None, None
    dup_thr = cfg['clean']['frozen']['diff_thr']
    out['duplicate'] = np.zeros(T, bool)
    mhi = np.zeros(gray_prev.shape, np.float32)
    tau = float(m['mhi_duration'])
    area = float(gray_prev.size)
    lo = hi = None
    if showcase is not None:
        lo, hi = max(1, showcase - window), min(T - 1, showcase + window)
    show = dict(idx=showcase, frames=[], raw_mask=[], noshadow=[], morph=[], comps=[], flow_hsv=[], mhi=[],
                final_mask=[], flows=[])
    for t in range(T):
        f = frames[t]
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        raw = mog.apply(f)
        raw = (raw > 0).astype(np.uint8) * 255
        bg = mog.getBackgroundImage()
        nosh, shadow = remove_shadows(f, bg, raw, cfg) if bg is not None else (raw, np.zeros_like(raw, bool))
        opened, closed = clean_mask(nosh, cfg)
        final, lab, blobs = label_components(closed, m['min_blob_area'])
        out['fg_ratio'][t] = (final > 0).sum() / area
        out['blob_count'][t] = len(blobs)
        out['shadow_frac'][t] = float((shadow & (raw > 0)).sum() / max((raw > 0).sum(), 1))
        mhi = np.where(final > 0, tau, np.maximum(mhi - 1, 0))
        if t > 0:
            # Many CCTV sources are 10-15 FPS unique frames stored at 30 FPS, so resampling to 12 FPS can pick the
            # same picture twice. Flow is therefore measured against the last DISTINCT frame and divided by the
            # number of 12-FPS steps elapsed (motion per step); isolated duplicates hold the previous value, long
            # runs (truly static / frozen) give 0.
            if cv2.absdiff(gray, gray_prev).mean() < dup_thr:
                dup_run += 1
                out['duplicate'][t] = True
                if last_ff is not None and dup_run <= 2:
                    flow, ff = last_flow, last_ff
                else:
                    flow, ff = None, dict(mag=0.0, moving_frac=0.0, entropy=0.0, div=0.0, curl=0.0)
            else:
                flow = farneback(gray_prev, gray, cfg) / float(t - ref_t)
                if cam_shifts is not None:
                    flow = flow - np.asarray(cam_shifts[t], np.float32)[None, None, :]
                ff = flow_features(flow, cfg)
                last_flow, last_ff = flow, ff
                gray_prev, ref_t, dup_run = gray, t, 0
            out['flow_mag'][t], out['flow_entropy'][t] = ff['mag'], ff['entropy']
            out['flow_div'][t], out['flow_curl'][t], out['flow_moving_frac'][t] = ff['div'], ff['curl'], ff['moving_frac']
            if lo is not None and lo <= t <= hi:
                fl = flow if flow is not None else np.zeros(gray.shape + (2,), np.float32)
                show['flows'].append(fl)
                show['flow_hsv'].append(flow_to_hsv(fl))
        if lo is not None and lo <= t <= hi:
            show['frames'].append(f)
            show['raw_mask'].append(raw)
            show['noshadow'].append(nosh)
            show['morph'].append(closed)
            show['final_mask'].append(final)
            show['comps'].append(colorize_labels(lab))
            show['mhi'].append(mhi_to_bgr(mhi, tau))
    out['motion_energy'] = out['flow_mag'].copy()
    out['global_change'] = out['fg_ratio'] > m.get('global_change_fg', 0.55)   # fades / flashes / scene swaps
    out['show'] = show
    return out


def pick_showcase(res: dict, bad: np.ndarray | None = None, smooth: int = 7, margin: int = 12,
                  prefer: tuple | None = None) -> int:
    """Frame of peak (smoothed) motion energy, ignoring cleaning-flagged and global-change frames."""
    e = res['motion_energy'].copy()
    ok = ~res['global_change']
    if bad is not None:
        ok &= ~bad
    e[~ok] = 0
    k = np.ones(smooth) / smooth
    es = np.convolve(e, k, mode='same')
    es[:margin] = 0
    if len(es) > margin:
        es[-4:] = 0
    if prefer is not None:                                   # e.g. the ground-truth event window (frame indices)
        a, b = max(0, int(prefer[0])), min(len(es), int(prefer[1]))
        if b > a and es[a:b].max() > 0:
            keep = np.zeros_like(es); keep[a:b] = es[a:b]; es = keep
    return int(np.argmax(es)) if es.max() > 0 else len(e) // 2


def raft_vs_farneback(frames, idxs, cfg: dict | None = None, device: str = 'cuda', raft: RaftFlow | None = None,
                      example_t: int | None = None):
    """RAFT flow on a handful of frame pairs and its agreement with Farneback (log TODO on failure)."""
    cfg = cfg or load_config()
    try:
        raft = raft or RaftFlow(device)
    except Exception as e:                                                  # pragma: no cover
        return dict(error=f'RAFT unavailable: {type(e).__name__}: {e}')
    mags_r, mags_f, ex = [], [], None
    for t in idxs:
        fr = raft(frames[t - 1], frames[t])
        fb = farneback(cv2.cvtColor(frames[t - 1], cv2.COLOR_BGR2GRAY), cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY), cfg)
        mags_r.append(float(np.hypot(fr[..., 0], fr[..., 1]).mean()))
        mags_f.append(float(np.hypot(fb[..., 0], fb[..., 1]).mean()))
        if ex is None or t == example_t:
            ex = (t, fr, fb)
    corr = float(np.corrcoef(mags_r, mags_f)[0, 1]) if len(mags_r) > 2 and np.std(mags_r) > 0 and np.std(mags_f) > 0 else None
    return dict(pairs=list(map(int, idxs)), raft_mag=mags_r, farneback_mag=mags_f, corr=corr,
                example=ex, raft_model='raft_small (torchvision, pretrained)')


# ═══════════════════════════════════════════════════════════════════════════════
# Sample saving
# ═══════════════════════════════════════════════════════════════════════════════

def _signal_plot(res, fps=12.0, gt=None, title='', keys=('fg_ratio', 'flow_mag', 'flow_entropy', 'flow_div', 'flow_curl', 'blob_count')):
    import matplotlib.pyplot as plt
    T = len(res['flow_mag'])
    t = np.arange(T) / fps
    n = len(keys)
    fig, axs = plt.subplots(n, 1, figsize=(9, 1.5 * n + 0.6), sharex=True)
    names = dict(fg_ratio='foreground ratio', flow_mag='mean flow magnitude (px/step)',
                 flow_entropy='flow direction entropy', flow_div='mean |divergence|', flow_curl='mean |curl|',
                 blob_count='blob count', motion_energy='motion energy')
    for ax, k, col in zip(np.atleast_1d(axs), keys, viz.SERIES * 2):
        ax.plot(t, res[k], color=col, lw=1.2)
        ax.set_ylabel(names.get(k, k), fontsize=7)
        if gt:
            for s, e in gt:
                ax.axvspan(s, e, color=viz.C['red'], alpha=.12)
    np.atleast_1d(axs)[-1].set_xlabel('time (s)')
    fig.suptitle(title + ('  (red band = ground-truth event)' if gt else ''), fontsize=9)
    fig.tight_layout()
    return fig


def save_motion_samples(video: str, frames, res: dict, cfg: dict, gt_seconds=None, raft_info: dict | None = None):
    if not viz.should_save(video):
        return
    sh = res['show']
    if not sh['frames']:
        return
    k = len(sh['frames']) // 2
    f = sh['frames'][k]
    sig_plot = lambda: _signal_plot(res, gt=gt_seconds, title=f'{video}: S6 per-frame motion signals')
    m3 = lambda im: cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    # background subtraction pipeline
    viz.save_io_sample(
        'S6', 'background_subtraction', video, inputs=f,
        outputs=[m3(sh['raw_mask'][k]), m3(sh['noshadow'][k]), m3(sh['morph'][k]), sh['comps'][k]],
        output_labels=['MOG2 raw', 'shadow removed (HSV)', 'open + close', 'connected components'],
        params=dict(mog2=cfg['motion']['mog2'], shadow=cfg['motion']['shadow'], morph=cfg['motion']['morph'],
                    min_blob_area=cfg['motion']['min_blob_area']),
        metrics=dict(showcase_frame=sh['idx'], fg_ratio=float(res['fg_ratio'][sh['idx']]),
                     blob_count=int(res['blob_count'][sh['idx']]), shadow_share_of_raw_fg=float(res['shadow_frac'][sh['idx']]),
                     mean_fg_ratio=float(res['fg_ratio'].mean()), mean_blob_count=float(res['blob_count'].mean())),
        note=('MOG2 foreground -> HSV shadow removal (shadow = darker V, same H and S) -> morphological opening/closing -> connected components.\n'
              'Why: gives a clean foreground mask, blob count and foreground ratio (how much of the scene is moving).\n'
              'Notice: raw mask (left) has shadow/speckle; later panels are cleaner; colours in the last panel are separate blobs.'),
        plot=lambda: _signal_plot(res, gt=gt_seconds, title=f'{video}: foreground ratio / blobs',
                                  keys=('fg_ratio', 'blob_count')),
        input_clip=sh['frames'], output_clip=[m3(x) for x in sh['final_mask']],
        labels=('input frame (showcase = peak motion)', 'mask stages'), cols=4, tile_h=170)

    # optical flow
    flow_out = [sh['flow_hsv'][min(k, len(sh['flow_hsv']) - 1)]]
    lab = ['Farneback (hue=direction)']
    metrics = dict(showcase_frame=sh['idx'], mean_mag=float(res['flow_mag'][sh['idx']]),
                   direction_entropy=float(res['flow_entropy'][sh['idx']]), mean_abs_div=float(res['flow_div'][sh['idx']]),
                   mean_abs_curl=float(res['flow_curl'][sh['idx']]))
    if raft_info and raft_info.get('example'):
        t_ex, fr, fb = raft_info['example']
        flow_out.append(flow_to_hsv(fr)); lab.append(f'RAFT-small (pair {t_ex})')
        metrics.update(raft_vs_farneback_corr=raft_info['corr'], raft_pairs=raft_info['pairs'],
                       raft_model=raft_info['raft_model'])
    elif raft_info:
        metrics['raft'] = raft_info.get('error')
    viz.save_io_sample(
        'S6', 'optical_flow', video, inputs=[sh['frames'][max(k - 1, 0)], f], input_labels=['t-1', 't'],
        outputs=flow_out, output_labels=lab,
        params=dict(flow_used='Farneback' + (' + RAFT-small' if raft_info and raft_info.get('example') else ''),
                    **cfg['motion']['flow'], entropy_bins=cfg['motion']['entropy_bins']),
        metrics=metrics,
        note=('Dense optical flow between consecutive 12-FPS frames, summarised as magnitude, direction entropy (chaotic vs coherent) and divergence/curl.\n'
              'Why: fights are fast and chaotic (high entropy), walking is coherent; these features drive the activity score (S8).\n'
              'Notice: HSV colours encode direction, brightness encodes speed; plot.png shows the per-frame features over the whole video.'),
        plot=sig_plot, output_clip=sh['flow_hsv'], input_clip=sh['frames'], labels=('frame pair', 'flow'))

    # MHI
    viz.save_io_sample(
        'S6', 'motion_history_image', video, inputs=f, outputs=[sh['mhi'][k]], output_labels=['MHI (bright = most recent motion)'],
        params=dict(duration_frames=cfg['motion']['mhi_duration'], source='cleaned foreground mask'),
        metrics=dict(showcase_frame=sh['idx'], mhi_active_share=float((cv2.cvtColor(sh['mhi'][k], cv2.COLOR_BGR2GRAY) > 20).mean())),
        note=('Motion History Image: each foreground pixel is stamped with the newest time it moved and fades over 1 s.\n'
              'Why: summarises recent motion direction/extent in one picture (mainly a figure for the report).\n'
              'Notice: bright trails show where and how recently people moved.'),
        output_clip=sh['mhi'], input_clip=sh['frames'], labels=('frame', 'MHI'))

    # motion-energy signal
    lowi = int(np.argmin(res['motion_energy'][1:]) + 1)
    hii = int(np.argmax(res['motion_energy']))
    viz.save_io_sample(
        'S6', 'motion_energy_signal', video, inputs=[frames[lowi], frames[hii]], input_labels=[f'lowest energy (t={lowi / 12:.1f}s)', f'highest energy (t={hii / 12:.1f}s)'],
        outputs=None, plot=lambda: _signal_plot(res, gt=gt_seconds, title=f'{video}: motion-energy signal', keys=('motion_energy', 'flow_entropy', 'fg_ratio')),
        compare=_signal_plot(res, gt=gt_seconds, title=f'{video}: motion-energy signal (flow magnitude), direction entropy, foreground ratio',
                             keys=('motion_energy', 'flow_entropy', 'fg_ratio')),
        params=dict(definition='mean Farneback flow magnitude per 12-FPS step (px)'),
        metrics=dict(mean=float(res['motion_energy'].mean()), p95=float(np.percentile(res['motion_energy'], 95)),
                     max=float(res['motion_energy'].max()), t_max_s=hii / 12.0),
        note=('Collapses the video to one motion-energy value per frame (mean flow magnitude) plus companions.\n'
              'Why: this signal (with the interaction signals) feeds the activity score and action proposals in S8.\n'
              'Notice: peaks mark bursts of movement; red band (if present) is the UCF-Crime ground-truth event.'))
