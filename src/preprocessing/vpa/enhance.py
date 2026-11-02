"""
enhance.py  (S4 — photometric enhancement; S5 — camera-motion compensation lives at the bottom)
================================================================================================
  1. ONE brightness measure everywhere  : mean luma Y (common.mean_luma) vs ONE threshold (config enhance.dark_threshold,
     set from the S1 statistics). The per-frame decision is median-smoothed over time and uses hysteresis, so a video
     does not flip between 'dark' and 'bright' frame by frame.
  2. Zero-DCE++ (existing stage-2 weights, no retraining) on dark runs, at native resolution, with the per-frame
     curve parameters smoothed over time (EMA) so brightness does not jump.
  3. Temporal denoising of the enhanced run (FastDVDnet pretrained if it can be fetched, else cv2 fastNlMeans multi-frame).
  4. Luma deflickering (zero-phase EMA of the frame brightness) on the enhanced run.
Baseline for comparison only: CLAHE.
"""

from __future__ import annotations

import importlib.util
import sys
import urllib.request

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .common import ROOT, ZERODCE_W, load_config, mean_luma
from . import viz


# ═══════════════════════════════════════════════════════════════════════════════
# Brightness decision
# ═══════════════════════════════════════════════════════════════════════════════

def luma_series(frames) -> np.ndarray:
    return np.array([mean_luma(f) for f in frames])


def dark_decision(luma: np.ndarray, cfg: dict | None = None, threshold: float | None = None):
    """Median-smoothed luma + hysteresis -> boolean 'dark' per frame (plus the smoothed series)."""
    c = (cfg or load_config())['enhance']
    thr = c['dark_threshold'] if threshold is None else threshold
    win = int(c['smooth_window']) | 1
    sm = median_filter(luma, size=min(win, max(1, len(luma) | 1)), mode='nearest')
    dark = np.zeros(len(luma), bool)
    state = False
    for t, v in enumerate(sm):
        if not state and v < thr:
            state = True
        elif state and v > thr + c['hysteresis']:
            state = False
        dark[t] = state
    dark &= luma >= cfg['clean']['black']['mean_max']          # black (fade) frames carry no scene: never enhance them
    return dark, sm


def runs_of(mask: np.ndarray, shot_ids=None):
    """Contiguous True runs as (start, end_exclusive); runs are also split at shot boundaries."""
    out, t, n = [], 0, len(mask)
    while t < n:
        if mask[t]:
            s = t
            while t < n and mask[t] and (shot_ids is None or shot_ids[t] == shot_ids[s]):
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def is_monochrome(frames, thr: float = 10.0) -> bool:
    idx = np.linspace(0, len(frames) - 1, min(10, len(frames))).round().astype(int)
    sat = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)[..., 1].mean() for i in idx]
    return bool(np.median(sat) < thr)


def flicker_index(frames) -> float:
    """Mean absolute change of the frame brightness between consecutive frames (0 = steady)."""
    y = luma_series(frames)
    return float(np.abs(np.diff(y)).mean()) if len(y) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Zero-DCE++ with temporally smoothed curve parameters
# ═══════════════════════════════════════════════════════════════════════════════

class ZeroDCEEnhancer:
    def __init__(self, device: str = 'cuda'):
        import torch
        sys.path.insert(0, str(ROOT / 'src'))
        from training.zerodce_model import ZeroDCE            # read-only import of the existing network
        self.torch = torch
        self.dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ZeroDCE(num_iterations=8).to(self.dev)
        ckpt = torch.load(str(ZERODCE_W), map_location=self.dev)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def enhance_run(self, frames_bgr, alpha_ema: float = 0.6, keep_mono: bool = False):
        """Sequentially enhance a run of frames; curve parameters are EMA-smoothed (alpha_ema=0 -> plain per-frame)."""
        torch = self.torch
        out, prev = [], None
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.dev)
            with torch.no_grad():
                if self.dev.type == 'cuda':
                    with torch.autocast('cuda'):
                        _, alphas = self.model(x)
                else:
                    _, alphas = self.model(x)
                a = torch.cat([t.float() for t in alphas], dim=1)           # (1, 24, H, W)
                if prev is not None and alpha_ema > 0:
                    a = alpha_ema * prev + (1 - alpha_ema) * a
                prev = a
                y = x
                for i in range(a.shape[1] // 3):
                    ai = a[:, 3 * i:3 * i + 3]
                    y = y + ai * (y - y * y)
                y = y.clamp(0, 1)
            res = (y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            if keep_mono:                                                    # IR / grayscale source: no colour cast
                g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            out.append(bgr)
        return out


def clahe_frames(frames_bgr, clip: float = 3.0, tile: int = 8):
    """Baseline only: CLAHE on the L channel."""
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    out = []
    for f in frames_bgr:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cl.apply(lab[..., 0])
        out.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal denoising
# ═══════════════════════════════════════════════════════════════════════════════

FASTDVD_DIR = ROOT / 'models' / 'fastdvdnet'
_FASTDVD_URL = 'https://raw.githubusercontent.com/m-tassano/fastdvdnet/master/'


def ensure_fastdvdnet() -> bool:
    """Fetch the MIT-licensed FastDVDnet definition + pretrained weights into models/ (git-ignored) on first use."""
    FASTDVD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for fn in ('models.py', 'model.pth'):
            p = FASTDVD_DIR / fn
            if not p.exists() or p.stat().st_size < 1000:
                urllib.request.urlretrieve(_FASTDVD_URL + fn, p)
        return True
    except Exception as e:
        print(f'  [FastDVDnet] download failed: {e}')
        return False


class FastDVDnetDenoiser:
    def __init__(self, device: str = 'cuda'):
        import torch
        if not ensure_fastdvdnet():
            raise RuntimeError('FastDVDnet files unavailable')
        spec = importlib.util.spec_from_file_location('fastdvdnet_models', FASTDVD_DIR / 'models.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.torch = torch
        self.dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.net = mod.FastDVDnet(num_input_frames=5)
        sd = torch.load(FASTDVD_DIR / 'model.pth', map_location='cpu')
        sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        self.net.load_state_dict(sd)
        self.net.to(self.dev).eval()

    def __call__(self, frames_bgr, sigma: float):
        torch = self.torch
        n = len(frames_bgr)
        H, W = frames_bgr[0].shape[:2]
        ph, pw = (-H) % 4, (-W) % 4
        ts = []
        for f in frames_bgr:
            x = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
            ts.append(torch.nn.functional.pad(x[None], (0, pw, 0, ph), mode='replicate')[0])
        out = []
        nm = torch.full((1, 1, H + ph, W + pw), float(sigma) / 255.0, device=self.dev)
        for t in range(n):
            idx = [min(max(t + d, 0), n - 1) for d in (-2, -1, 0, 1, 2)]
            x = torch.cat([ts[i] for i in idx], dim=0)[None].to(self.dev)
            with torch.no_grad():
                y = self.net(x, nm)[0, :, :H, :W].clamp(0, 1)
            out.append(cv2.cvtColor((y.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        return out


def estimate_sigma(frames_bgr) -> float:
    from .profile import noise_sigma
    idx = np.linspace(0, len(frames_bgr) - 1, min(5, len(frames_bgr))).round().astype(int)
    return float(np.median([noise_sigma(cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)) for i in idx]))


def denoise_fastnl(frames_bgr, cfg: dict):
    """Classical multi-frame non-local means (uses the neighbouring frames)."""
    d = cfg['enhance']['denoise']
    n = len(frames_bgr)
    out = []
    for t in range(n):
        w = d['window']
        half = w // 2
        lo, hi = max(0, t - half), min(n, t + half + 1)
        # keep the window odd and centred on t
        r = min(t - lo, hi - 1 - t)
        lo, hi = t - r, t + r + 1
        imgs = frames_bgr[lo:hi]
        if len(imgs) == 1:
            out.append(cv2.fastNlMeansDenoisingColored(imgs[0], None, d['h'], d['h_color'], 7, 21))
        else:
            out.append(cv2.fastNlMeansDenoisingColoredMulti(imgs, t - lo, len(imgs), None, d['h'], d['h_color'], 7, 21))
    return out


def temporal_denoise(frames_bgr, cfg: dict, models: dict | None = None, device: str = 'cuda', method: str | None = None):
    method = method or cfg['enhance']['denoise']['method']
    models = models if models is not None else {}
    if method == 'none' or len(frames_bgr) < 1:
        return list(frames_bgr), 'none'
    if method == 'fastdvdnet':
        try:
            if 'fastdvdnet' not in models:
                models['fastdvdnet'] = FastDVDnetDenoiser(device)
            sigma = float(np.clip(estimate_sigma(frames_bgr), 5.0, 55.0))
            return models['fastdvdnet'](frames_bgr, sigma), 'fastdvdnet'
        except Exception as e:                                           # logged in DECISIONS.md; classical fallback
            print(f'  [S4] FastDVDnet unavailable ({e}); falling back to fastNlMeans')
            models['fastdvdnet_failed'] = True
            return denoise_fastnl(frames_bgr, cfg), 'fastnlmeans'
    return denoise_fastnl(frames_bgr, cfg), 'fastnlmeans'


# ═══════════════════════════════════════════════════════════════════════════════
# Deflicker (brightness smoothing)
# ═══════════════════════════════════════════════════════════════════════════════

def deflicker_luma(frames_bgr, cfg: dict):
    """Zero-phase EMA (forward + backward average) of the log mean-luma; per-frame gamma moves each frame to it."""
    c = cfg['enhance']['deflicker']
    y = np.maximum(luma_series(frames_bgr), 1.0)
    ly = np.log(y)
    a = float(c['luma_ema'])
    f = ly.copy(); b = ly.copy()
    for t in range(1, len(ly)):
        f[t] = a * f[t - 1] + (1 - a) * ly[t]
    for t in range(len(ly) - 2, -1, -1):
        b[t] = a * b[t + 1] + (1 - a) * ly[t]
    target = np.exp((f + b) / 2)
    out = []
    base = np.arange(256) / 255.0
    lo_g, hi_g = 1.0 / c['max_gain'], c['max_gain']

    def apply(fr, g):
        return cv2.LUT(fr, (255.0 * base ** g).clip(0, 255).astype(np.uint8))

    for fr, yt, tg in zip(frames_bgr, y, target):
        if abs(mean_luma(fr) - tg) < 0.03:
            out.append(fr)
            continue
        lo, hi = lo_g, hi_g                                     # mean luma decreases monotonically with gamma: bisect
        if mean_luma(apply(fr, lo)) < tg:
            out.append(apply(fr, lo)); continue
        if mean_luma(apply(fr, hi)) > tg:
            out.append(apply(fr, hi)); continue
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if mean_luma(apply(fr, mid)) > tg:
                lo = mid
            else:
                hi = mid
        out.append(apply(fr, 0.5 * (lo + hi)))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Whole-video enhancement
# ═══════════════════════════════════════════════════════════════════════════════

def process_run(run_frames, cfg: dict, models: dict, device: str = 'cuda', alpha_ema: float | None = None,
                denoise: bool = True, deflicker: bool = True, mono: bool | None = None) -> dict:
    """Zero-DCE++ -> temporal denoise -> deflicker on one contiguous run. Returns frames + the stages' outputs."""
    e = cfg['enhance']
    if 'zerodce' not in models:
        models['zerodce'] = ZeroDCEEnhancer(device)
    mono = is_monochrome(run_frames) if mono is None else mono
    enh = models['zerodce'].enhance_run(run_frames, e['deflicker']['alpha_ema'] if alpha_ema is None else alpha_ema,
                                        keep_mono=bool(e.get('keep_mono', True) and mono))
    stages = dict(zerodce=enh)
    method = 'none'
    if denoise:
        enh, method = temporal_denoise(enh, cfg, models, device)
        stages['denoised'] = enh
    if deflicker:
        enh = deflicker_luma(enh, cfg)
        stages['deflickered'] = enh
    return dict(frames=enh, stages=stages, denoise_method=method, mono=mono)


def enhance_video(frames, shot_ids=None, cfg: dict | None = None, models: dict | None = None, device: str = 'cuda',
                  force: bool = False) -> dict:
    cfg = cfg or load_config()
    models = models if models is not None else {}
    luma0 = luma_series(frames)
    dark, sm = dark_decision(luma0, cfg)
    if force:
        dark[:] = True
    out = list(frames)
    enhanced = np.zeros(len(frames), bool)
    runs = runs_of(dark, shot_ids)
    info = []
    for s, e in runs:
        r = process_run(frames[s:e], cfg, models, device)
        out[s:e] = r['frames']
        enhanced[s:e] = True
        info.append(dict(start=s, end=e, denoise=r['denoise_method'], mono=r['mono']))
    return dict(frames=out, dark=dark, enhanced=enhanced, luma_before=luma0, luma_smooth=sm,
                luma_after=luma_series(out), runs=info)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics: NIQE / BRISQUE (pyiqa), YOLO person recall proxy
# ═══════════════════════════════════════════════════════════════════════════════

def iqa_scores(frames_bgr, models: dict, device: str = 'cuda', n: int = 8) -> dict:
    """NIQE and BRISQUE (lower = better) on up to n evenly spaced frames. Returns None values if pyiqa fails."""
    try:
        import torch
        import pyiqa
        dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        for nm in ('niqe', 'brisque'):
            if nm not in models:
                models[nm] = pyiqa.create_metric(nm, device=dev)
        idx = np.linspace(0, len(frames_bgr) - 1, min(n, len(frames_bgr))).round().astype(int)
        res = {}
        for nm in ('niqe', 'brisque'):
            vals = []
            for i in idx:
                x = torch.from_numpy(cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).to(dev)
                try:
                    vals.append(float(models[nm](x)))
                except Exception:                                  # NIQE's SVD fails on degenerate (flat/black) frames
                    vals.append(float('nan'))
            res[nm] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
        return res
    except Exception as e:
        if 'iqa_error' not in models:
            print(f'  [S4] pyiqa NIQE/BRISQUE unavailable: {type(e).__name__}: {e}')
            models['iqa_error'] = str(e)
        return dict(niqe=None, brisque=None)


def variant_metrics(frames_bgr, models: dict, device: str, detector=None) -> dict:
    m = dict(mean_luma=float(luma_series(frames_bgr).mean()), flicker_index=flicker_index(frames_bgr),
             noise_sigma=estimate_sigma(frames_bgr))
    m.update(iqa_scores(frames_bgr, models, device))
    if detector is not None:
        idx = np.linspace(0, len(frames_bgr) - 1, min(12, len(frames_bgr))).round().astype(int)
        m['yolo_persons_per_frame'] = float(detector.count_persons([frames_bgr[i] for i in idx]).mean())
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Sample saving
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_low_light(frames_bgr, gain: float = 0.5, gamma: float = 1.6, noise_sigma: float = 2.5, seed: int = 0):
    """DEMO ONLY: darken + add sensor-like noise so the enhancement chain can be shown on a video that is not dark."""
    rng = np.random.RandomState(seed)
    out = []
    for f in frames_bgr:
        x = gain * (f.astype(np.float32) / 255.0) ** gamma * 255.0
        x = x + rng.normal(0, noise_sigma, x.shape)
        out.append(np.clip(x, 0, 255).astype(np.uint8))
    return out


def _demo_window(luma: np.ndarray, enhanced: np.ndarray, n: int = 36):
    """Window to show: centre of the longest enhanced run, else the darkest window (demo is then 'forced')."""
    T = len(luma)
    n = min(n, T)
    runs = runs_of(enhanced)
    if runs and enhanced.mean() >= 0.10:            # tiny dark spans are fades/transitions, not a dark video: simulate instead
        s, e = max(runs, key=lambda r: r[1] - r[0])
        mid = (s + e) // 2
        a = int(np.clip(mid - n // 2, s if e - s >= n else 0, max(0, T - n)))
        return a, a + n, False
    sm = np.convolve(luma, np.ones(n) / n, mode='valid')
    a = int(np.argmin(sm))
    return a, a + n, True


def save_enhance_samples(video: str, frames, res: dict, cfg: dict, models: dict, device: str = 'cuda',
                         shot_ids=None, detector=None):
    if not viz.should_save(video):
        return
    import matplotlib.pyplot as plt
    thr = cfg['enhance']['dark_threshold']
    T = len(frames)
    luma = res['luma_before']

    # 1 ── the single brightness measure
    valid = np.where(luma >= cfg['clean']['black']['mean_max'])[0]
    i_dark = int(valid[np.argmin(luma[valid])]) if len(valid) else int(np.argmin(luma))
    i_bright = int(np.argmax(luma))
    hsv_v = lambda f: float(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 2].mean())

    def luma_plot():
        fig, ax = plt.subplots(figsize=(9, 3))
        t = np.arange(T) / 12
        ax.plot(t, luma, color=viz.INK2, lw=0.8, label='mean luma Y (per frame)')
        ax.plot(t, res['luma_smooth'], color=viz.C['blue'], label='median-smoothed')
        ax.axhline(thr, ls='--', color=viz.C['red'], label=f'dark threshold = {thr}')
        ax.fill_between(t, 0, 255, where=res['dark'], color=viz.C['yellow'], alpha=.25, label='dark -> enhance')
        ax.set_ylim(0, max(120, luma.max() * 1.05)); ax.set_xlabel('time (s)'); ax.legend(ncol=4, fontsize=7)
        fig.tight_layout()
        return fig
    viz.save_io_sample(
        'S4', 'brightness_measure', video, inputs=[frames[i_dark], frames[i_bright]],
        input_labels=[f'darkest: Y={luma[i_dark]:.0f}', f'brightest: Y={luma[i_bright]:.0f}'], outputs=None,
        params=dict(measure='mean luma Y (BT.601)', dark_threshold=thr, smooth_window=cfg['enhance']['smooth_window'],
                    hysteresis=cfg['enhance']['hysteresis']),
        metrics=dict(frac_dark_frames=float(res['dark'].mean()), n_dark_runs=len(res['runs']),
                     old_extract_clips_would_enhance=bool(luma[i_dark] < 50), old_infer_zerodce_would_enhance=bool(luma[i_dark] < 80),
                     darkest_frame=dict(gray_mean=float(luma[i_dark]), hsv_v_mean=hsv_v(frames[i_dark])),
                     brightest_frame=dict(gray_mean=float(luma[i_bright]), hsv_v_mean=hsv_v(frames[i_bright]))),
        note=('One brightness measure (mean luma Y) and one threshold (from S1) decide, with temporal smoothing and hysteresis, which frames are enhanced.\n'
              'Why: the old code used grayscale mean < 50 (extract_clips), < 80 (infer_zerodce) and HSV-V (backend) - three different answers for the same frame.\n'
              'Notice: yellow bands in plot.png = frames sent to Zero-DCE++ (none for bright videos).'),
        plot=luma_plot, labels=('input frames', ''))

    # 2 ── comparison on a demo window (forced when the video has no dark frames)
    a, b, forced = _demo_window(luma, res['enhanced'])
    orig_win = frames[a:b]
    win = simulate_low_light(orig_win) if forced else orig_win          # bright video -> SIMULATED low light (labelled)
    mono = is_monochrome(win)
    variants = {'raw': win, 'CLAHE': clahe_frames(win)}
    plain = process_run(win, cfg, models, device, alpha_ema=0.0, denoise=False, deflicker=False, mono=mono)
    full = process_run(win, cfg, models, device)
    variants['Zero-DCE++'] = plain['frames']
    variants['Zero-DCE++ + denoise + deflicker'] = full['frames']
    mets = {k: variant_metrics(v, models, device, detector) for k, v in variants.items()}
    if forced:
        mets['original bright frames (reference)'] = variant_metrics(orig_win, models, device, detector)
    k_mid = len(win) // 2

    def fmt(m):
        s = f"Y={m['mean_luma']:.0f}  flicker={m['flicker_index']:.2f}"
        if m.get('niqe') is not None:
            s += f"\nNIQE={m['niqe']:.2f}  BRISQUE={m['brisque']:.1f}"
        if 'yolo_persons_per_frame' in m:
            s += f"  persons/frame={m['yolo_persons_per_frame']:.2f}"
        return s
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.0))
    for ax, (name, fr) in zip(axs.ravel(), variants.items()):
        ax.imshow(cv2.cvtColor(fr[k_mid], cv2.COLOR_BGR2RGB)); ax.axis('off')
        ax.set_title(f'{name}\n{fmt(mets[name])}', fontsize=8)
    fig.suptitle(f'{video} — frames {a}-{b} ({"SIMULATED low light: video is not dark" if forced else "real dark run"}; '
                 f'monochrome kept={mono}; denoiser={full["denoise_method"]})', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    viz.save_io_sample(
        'S4', 'enhancement_comparison', video, inputs=win[k_mid], outputs=full['frames'][k_mid],
        params=dict(compare=list(variants), dark_threshold=thr, alpha_ema=cfg['enhance']['deflicker']['alpha_ema'],
                    denoiser=full['denoise_method'], simulated_low_light=forced),
        metrics=mets, compare=fig,
        input_clip=win, output_clip=full['frames'],
        note=('4-way comparison: raw vs CLAHE (baseline only) vs Zero-DCE++ vs Zero-DCE++ with temporal denoising and deflickering.\n'
              'Why: brightening dark footage also brightens its noise and per-frame curves flicker; the extra steps fix both.\n'
              'Notice: NIQE/BRISQUE (lower = better), flicker index (brightness change between frames) and YOLO persons/frame under each panel.'),
        labels=('raw', 'final (Zero-DCE++ + denoise + deflicker)'))

    # 3 ── temporal denoise (zoomed crop)
    h, w = win[k_mid].shape[:2]
    cy, cx = h // 2, w // 2
    crop = lambda f: cv2.resize(f[cy - h // 5:cy + h // 5, cx - w // 5:cx + w // 5], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    viz.save_io_sample(
        'S4', 'temporal_denoise', video, inputs=crop(plain['frames'][k_mid]), outputs=crop(full['stages']['denoised'][k_mid]),
        params=dict(method=full['denoise_method'], window=cfg['enhance']['denoise']['window'], zoom='3x centre crop'),
        metrics=dict(noise_sigma_before=estimate_sigma(plain['frames']), noise_sigma_after=estimate_sigma(full['stages']['denoised']),
                     niqe_before=mets['Zero-DCE++'].get('niqe'), niqe_after=None),
        note=('Multi-frame (temporal) denoising of the enhanced frames (FastDVDnet pretrained, else cv2 fastNlMeansMulti).\n'
              'Why: Zero-DCE++ amplifies sensor noise in dark footage; noise differs between frames while the scene does not, so neighbouring frames help.\n'
              'Notice: grain in the zoomed crop (left) is reduced on the right; compare noise_sigma in metrics.json.'),
        input_clip=[crop(f) for f in plain['frames']], output_clip=[crop(f) for f in full['stages']['denoised']],
        labels=('Zero-DCE++ only (3x crop)', 'after temporal denoise'))

    # 4 ── deflicker
    def flicker_plot():
        fig, ax = plt.subplots(figsize=(9, 3))
        t = np.arange(len(win)) / 12
        for (nm, fr), col in zip(variants.items(), (viz.INK2, viz.C['yellow'], viz.C['orange'], viz.C['blue'])):
            ax.plot(t, luma_series(fr), color=col, label=f'{nm} (flicker {mets[nm]["flicker_index"]:.2f})')
        ax.set_xlabel('time (s)'); ax.set_ylabel('frame mean luma'); ax.legend(fontsize=7)
        fig.tight_layout()
        return fig
    viz.save_io_sample(
        'S4', 'deflicker', video, inputs=viz.filmstrip(plain['frames'], 6, 90), outputs=viz.filmstrip(full['frames'], 6, 90),
        params=dict(luma_ema=cfg['enhance']['deflicker']['luma_ema'], curve_param_ema=cfg['enhance']['deflicker']['alpha_ema'],
                    max_gain=cfg['enhance']['deflicker']['max_gain']),
        metrics=dict(flicker_before=mets['Zero-DCE++']['flicker_index'], flicker_after=mets['Zero-DCE++ + denoise + deflicker']['flicker_index'],
                     flicker_raw=mets['raw']['flicker_index']),
        note=('Temporal deflicker: Zero-DCE++ curve parameters are EMA-smoothed across frames and the frame brightness is smoothed (zero-phase EMA).\n'
              'Why: per-frame enhancement makes brightness jump; the motion stages and VideoMAE would read that flicker as motion.\n'
              'Notice: the blue brightness curve in plot.png is smoother than the orange (plain Zero-DCE++); flicker index in the legend.'),
        plot=flicker_plot, labels=('Zero-DCE++ per frame', 'smoothed final'))

    # 5 ── metrics table
    cols = ['variant', 'mean luma', 'flicker idx', 'noise sigma', 'NIQE', 'BRISQUE', 'persons/frame']
    rows = [[k, f"{m['mean_luma']:.1f}", f"{m['flicker_index']:.3f}", f"{m['noise_sigma']:.2f}",
             '-' if m.get('niqe') is None else f"{m['niqe']:.2f}", '-' if m.get('brisque') is None else f"{m['brisque']:.1f}",
             f"{m.get('yolo_persons_per_frame', float('nan')):.2f}"] for k, m in mets.items()]
    tab = viz.save_table_image(rows, cols, f'{video}: enhancement metrics (lower is better for flicker/noise/NIQE/BRISQUE)' + (' - SIMULATED low light' if forced else ''),
                               col_w=[0.34] + [0.11] * 6)
    viz.save_io_sample('S4', 'enhancement_metrics', video, inputs=None, outputs=tab, compare=tab,
                       params=dict(frames=f'{a}-{b}', simulated_low_light=forced), metrics=mets,
                       note=('Numbers behind the 4-way comparison: brightness, flicker index, noise sigma, NIQE, BRISQUE and YOLO persons per frame.\n'
                             'Why: shows the effect of each step with measurements, not just pictures.\n'
                             'Notice: flicker and noise drop with the temporal steps; persons/frame is a recall proxy (no ground truth on UCF-Crime).'))
    return mets
