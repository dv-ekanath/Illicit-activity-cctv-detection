"""
profile.py  (S1 — video profiling and quality assessment)
==========================================================
One decode pass per video (PyAV) collects
  • stream metadata   : codec, resolution, nominal/average FPS, VFR flag, bitrate, duration, frame count
  • I/P/B frame shares
  • per-frame brightness (luma), frame-difference, HSV-histogram distance (cut candidates)
  • sampled quality stats: sharpness (Laplacian variance), noise sigma (Immerkaer), blockiness
  • camera-motion estimate (phase correlation between frames 0.1 s apart)
  • letterbox bars + timestamp/overlay score (via the S3 detectors on ~40 sampled frames)
and the dataset-level analytics that SET the thresholds used later (dark threshold, blur floor)
and pick the fixed sample set (docs/vpa_samples/sample_set.json).

Modern VQA (DOVER / FAST-VQA) is optional — see DECISIONS.md.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import cv2
import numpy as np

from .common import (load_config, list_videos, load_temporal_gt, CACHE_DIR, DOCS_DIR, LABELS)
from . import clean, viz

_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame quality measures
# ═══════════════════════════════════════════════════════════════════════════════

def central(gray: np.ndarray, margin: float = 0.15) -> np.ndarray:
    h, w = gray.shape[:2]
    return gray[int(h * margin):int(h * (1 - margin)), int(w * margin):int(w * (1 - margin))]


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian on the central region (excludes letterbox bars)."""
    return float(cv2.Laplacian(central(gray), cv2.CV_64F).var())


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer fast noise-level estimate."""
    g = central(gray).astype(np.float32)
    h, w = g.shape
    s = np.abs(cv2.filter2D(g, -1, _NOISE_K)).sum()
    return float(np.sqrt(np.pi / 2) * s / (6.0 * (w - 2) * (h - 2)))


def blockiness(gray: np.ndarray, b: int = 8) -> float:
    """Ratio of mean |horizontal gradient| on 8-px block boundaries to elsewhere (1.0 = no blocking)."""
    g = central(gray).astype(np.float32)
    d = np.abs(g[:, 1:] - g[:, :-1])
    cols = np.arange(d.shape[1])
    edge = (cols + 1) % b == 0
    return float(d[:, edge].mean() / (d[:, ~edge].mean() + 1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
# One-video profile (runs in worker processes)
# ═══════════════════════════════════════════════════════════════════════════════

def profile_video(path: str, cfg: dict | None = None) -> dict:
    cv2.setNumThreads(1)
    cfg = cfg or load_config()
    pc = cfg['profile']
    out = dict(path=str(path), stem=Path(path).stem, ok=False)
    try:
        with av.open(str(path)) as cont:
            st = cont.streams.video[0]
            st.thread_type = 'AUTO'
            tb = float(st.time_base)
            avg = float(st.average_rate) if st.average_rate else 0.0
            guessed = float(st.guessed_rate) if st.guessed_rate else avg
            dur = float(st.duration * tb) if st.duration else (cont.duration / 1e6 if cont.duration else 0.0)
            nom_frames = st.frames or int(round(dur * (avg or 25)))
            out.update(codec=st.codec_context.name, width=st.codec_context.width, height=st.codec_context.height,
                       pix_fmt=st.codec_context.pix_fmt, fps_avg=round(avg, 3), fps_guessed=round(guessed, 3),
                       bit_rate=int(st.bit_rate or cont.bit_rate or 0), duration_s=round(dur, 3),
                       file_mb=round(Path(path).stat().st_size / 1e6, 2))

            sample_step = max(1, nom_frames // pc['overlay_frames'])
            hann = None
            ptypes = {}
            luma, sdiff, hdist, pts = [], [], [], []
            sharp, noise, block, p5, p95 = [], [], [], [], []
            shifts, resp = [], []
            samples = []
            prev_gray = prev_hist = None
            cam_prev = None
            n = 0
            for fr in cont.decode(st):
                pt = {1: 'I', 2: 'P', 3: 'B'}.get(int(fr.pict_type), 'other')   # PyAV exposes AVPictureType as int
                ptypes[pt] = ptypes.get(pt, 0) + 1
                bgr = fr.to_ndarray(format='bgr24')
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                luma.append(float(gray.mean()))
                if fr.pts is not None:
                    pts.append(fr.pts * tb)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
                cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
                if prev_gray is not None:
                    sdiff.append(float(cv2.absdiff(gray, prev_gray).mean()))
                    hdist.append(float(cv2.compareHist(prev_hist, h, cv2.HISTCMP_BHATTACHARYYA)))
                prev_gray, prev_hist = gray, h

                if n % pc['quality_every'] == 0:
                    sharp.append(sharpness(gray)); noise.append(noise_sigma(gray)); block.append(blockiness(gray))
                    c = central(gray)
                    p5.append(float(np.percentile(c, 5))); p95.append(float(np.percentile(c, 95)))
                # camera motion: phase correlation between frame n and n+3 every 30 frames
                if n % 30 == 0:
                    cam_prev = gray.astype(np.float32)
                elif n % 30 == 3 and cam_prev is not None:
                    if hann is None:
                        hann = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
                    (dx, dy), r = cv2.phaseCorrelate(cam_prev, gray.astype(np.float32), hann)
                    shifts.append(float(np.hypot(dx, dy))); resp.append(float(r))
                    cam_prev = None
                if n % sample_step == 0 and len(samples) < pc['overlay_frames']:
                    samples.append(bgr)
                n += 1
        out['n_frames'] = n
        luma = np.asarray(luma)
        sdiff, hdist = np.asarray(sdiff), np.asarray(hdist)
        total = max(1, sum(ptypes.values()))
        out['ipb'] = {k: ptypes.get(k, 0) for k in ('I', 'P', 'B')}
        out['ipb_frac'] = {k: round(v / total, 4) for k, v in out['ipb'].items()}
        dts = np.diff(pts) if len(pts) > 2 else np.array([1 / (avg or 25)])
        med = float(np.median(dts)) if len(dts) else 0.04
        out['vfr'] = bool(np.mean(np.abs(dts - med) > 0.25 * med) > 0.02) if len(dts) else False
        out['fps_measured'] = round(1 / med, 3) if med > 0 else None
        out['luma_mean'] = round(float(luma.mean()), 2)
        out['luma_median'] = round(float(np.median(luma)), 2)
        out['luma_p10'] = round(float(np.percentile(luma, 10)), 2)
        out['luma_p90'] = round(float(np.percentile(luma, 90)), 2)
        out['luma_hist'] = np.histogram(luma, bins=64, range=(0, 256))[0].tolist()
        out['sharp_median'] = round(float(np.median(sharp)), 2) if sharp else None
        out['sharp_p10'] = round(float(np.percentile(sharp, 10)), 2) if sharp else None
        out['noise_median'] = round(float(np.median(noise)), 3) if noise else None
        out['block_median'] = round(float(np.median(block)), 3) if block else None
        out['contrast_median'] = round(float(np.median(np.asarray(p95) - np.asarray(p5))), 2) if p5 else None
        out['frozen_frac'] = round(float((sdiff < cfg['clean']['frozen']['diff_thr']).mean()), 4) if len(sdiff) else 0
        out['black_frac'] = round(float((luma < cfg['clean']['black']['mean_max']).mean()), 4)
        out['motion_diff_median'] = round(float(np.median(sdiff)), 3) if len(sdiff) else 0.0
        # cut candidates at native frame rate (S3 refines these on the 12-FPS timeline)
        if len(hdist) > 10:
            med_h = float(np.median(hdist)); mad_h = float(np.median(np.abs(hdist - med_h)))
            thr = max(0.45, med_h + 8 * 1.4826 * mad_h)
            cand = [int(i) for i in np.where(hdist > thr)[0]]
            kept = []
            for i in sorted(cand, key=lambda j: -hdist[j]):
                if all(abs(i - j) > 8 for j in kept):
                    kept.append(i)
            out['cut_candidates'] = len(kept)
            out['cut_max_hdist'] = round(float(hdist[kept].max()), 3) if kept else 0.0
            out['cut_frames'] = sorted(kept)[:20]
        else:
            out.update(cut_candidates=0, cut_max_hdist=0.0, cut_frames=[])
        # camera motion
        if shifts:
            sh = np.asarray(shifts)
            out['cam_shift_median'] = round(float(np.median(sh)), 3)
            out['cam_moving_frac'] = round(float((sh >= pc['moving_camera_shift_px']).mean()), 3)
        else:
            out['cam_shift_median'], out['cam_moving_frac'] = 0.0, 0.0
        # letterbox + overlay (S3 detectors)
        lb = clean.detect_letterbox(samples, cfg)
        out['bars'] = lb['bars']; out['has_bars'] = lb['has_bars']
        crop = clean.apply_crop(samples, lb['box']) if lb['has_bars'] else samples
        ov = clean.detect_overlay(crop, cfg)
        out['overlay_components'] = ov['n_components']
        out['overlay_area_frac'] = round(ov['area_frac'], 5)
        out['overlay_score'] = round(ov['score'], 3)
        out['ok'] = True
    except Exception as e:                                           # corrupt / unreadable → flagged, not fatal
        out['error'] = f'{type(e).__name__}: {e}'
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset driver + cache
# ═══════════════════════════════════════════════════════════════════════════════

def profile_dataset(force: bool = False, workers: int | None = None, limit: int | None = None) -> list:
    """Profile every UCF-Crime target video (cached at outputs/vpa_cache/profiles.json)."""
    cfg = load_config()
    cache = CACHE_DIR / 'profiles.json'
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vids = list_videos()[:limit] if limit else list_videos()
    workers = workers or cfg['profile']['workers']
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(profile_video, v['path'], None): v for v in vids}
        for i, f in enumerate(as_completed(futs), 1):
            v = futs[f]
            r = f.result()
            r.update(label=v['label'], folder=v['folder'])
            res.append(r)
            if i % 25 == 0:
                print(f'  [S1] profiled {i}/{len(vids)}  ({time.time() - t0:.0f}s)', flush=True)
    res.sort(key=lambda r: r['stem'])
    cache.write_text(json.dumps(res))
    print(f'[S1] profiled {len(res)} videos in {time.time() - t0:.0f}s ({sum(not r["ok"] for r in res)} failed)')
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset analytics: thresholds + sample-set selection
# ═══════════════════════════════════════════════════════════════════════════════

def dataset_summary(profiles: list) -> dict:
    ok = [p for p in profiles if p.get('ok')]
    cfg = load_config()
    res = dict(n_videos=len(profiles), n_ok=len(ok), n_failed=len(profiles) - len(ok),
               failed=[p['stem'] for p in profiles if not p.get('ok')])
    res['fps'] = sorted({round(p['fps_avg'], 2) for p in ok})
    res['resolutions'] = sorted({f"{p['width']}x{p['height']}" for p in ok})
    res['codecs'] = sorted({p['codec'] for p in ok})
    res['vfr_videos'] = sum(p['vfr'] for p in ok)
    res['with_letterbox_bars'] = sum(p['has_bars'] for p in ok)
    res['with_overlay'] = sum(p['overlay_components'] > 0 for p in ok)
    res['with_cut_candidates'] = sum(p['cut_candidates'] > 0 for p in ok)
    res['moving_camera_videos'] = sum(p['cam_moving_frac'] >= cfg['profile']['moving_camera_frac'] for p in ok)
    res['moving_camera_fraction'] = round(res['moving_camera_videos'] / max(1, len(ok)), 4)
    per = {}
    for lb in LABELS:
        q = [p for p in ok if p['label'] == lb]
        if not q:
            continue
        g = lambda k: np.array([p[k] for p in q if p.get(k) is not None], float)
        per[lb] = dict(n=len(q), duration_median_s=float(np.median(g('duration_s'))),
                       luma_median=float(np.median(g('luma_median'))), sharp_median=float(np.median(g('sharp_median'))),
                       noise_median=float(np.median(g('noise_median'))), bars=sum(p['has_bars'] for p in q),
                       overlay=sum(p['overlay_components'] > 0 for p in q))
    res['per_class'] = per
    return res


def derive_thresholds(profiles: list) -> dict:
    """
    Data-driven thresholds from the pooled frame-level statistics.
      dark_threshold : the luma below which the darkest `dark_percentile` % of ALL frames fall (rounded to 5,
                       clamped to [35, 70]). Otsu is reported for reference but the pooled luma distribution is
                       broad/unimodal, so a split-based rule is unreliable (it says 76 and would enhance ~1/3 of
                       the dataset); a tail rule targets genuinely under-exposed / night-time frames.
      blur_lap_var   : 5th percentile of per-video median Laplacian variance.
    """
    ok = [p for p in profiles if p.get('ok')]
    cfg = load_config()
    h = np.sum([p['luma_hist'] for p in ok], axis=0).astype(np.float64)       # 64 bins of width 4
    centers = np.arange(64) * 4 + 2
    cdf = np.cumsum(h) / h.sum()
    pct = cfg['profile'].get('dark_percentile', 5)
    tail = float(np.interp(pct / 100.0, cdf, centers + 2))                    # upper edge of the bin
    sub = h[centers < 110]
    best, best_t = -1, 50
    for k in range(2, len(sub) - 2):
        w0, w1 = sub[:k].sum(), sub[k:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (sub[:k] * centers[:k]).sum() / w0
        m1 = (sub[k:] * centers[k:len(sub)]).sum() / w1
        var_b = w0 * w1 * (m0 - m1) ** 2
        if var_b > best:
            best, best_t = var_b, centers[k] - 2
    frames_below = {t: float(h[centers < t].sum() / h.sum()) for t in (30, 40, 50, 60, 70, 80)}
    sharp = np.array([p['sharp_median'] for p in ok if p.get('sharp_median') is not None])
    return dict(dark_threshold=float(np.clip(round(tail / 5) * 5, 35, 70)), dark_percentile=pct,
                dark_threshold_tail_raw=tail, dark_threshold_otsu=float(best_t),
                frac_frames_below=frames_below,
                blur_lap_var_p5=float(np.percentile(sharp, 5)), blur_lap_var_p10=float(np.percentile(sharp, 10)),
                sharp_median=float(np.median(sharp)),
                noise_median=float(np.median([p['noise_median'] for p in ok if p.get('noise_median') is not None])))


def select_sample_set(profiles: list, seed: int = 42, max_seconds: float = 300.0,
                      verify_cuts: bool = True) -> list:
    """
    Selection rule (see DECISIONS.md):
      eligible  = decodable, 20 s <= duration <= max_seconds (whole video fits the sample-run cap)
      dark      = eligible video with the lowest median luma
      overlay   = highest S3 overlay score among non-dark eligible videos
      cut       = strongest cut-candidate video whose 12-FPS S3 detection confirms 1..8 cuts
      moving    = highest camera-motion fraction, only if >= profile.moving_camera_frac
      per class = among still-unused eligible videos of the class, those WITH temporal ground truth (abnormal
                  classes) whose GT event lies inside the analysed window, nearest to the class median of
                  (median luma, sharpness) in z-units; seeded random tie-break among the 3 nearest.
    """
    from . import decode
    cfg = load_config()
    rng = np.random.RandomState(seed)
    gt = load_temporal_gt()
    ok = [p for p in profiles if p.get('ok')]
    elig = [p for p in ok if 20 <= p['duration_s'] <= max_seconds]
    chosen, used = [], set()

    def add(p, role, why):
        chosen.append(dict(stem=p['stem'], label=p['label'], role=role, why=why, path=p['path'],
                           duration_s=p['duration_s'], luma_median=p['luma_median'],
                           has_gt=bool(gt.get(p['stem'])), overlay_components=p['overlay_components'],
                           cut_candidates=p['cut_candidates'], cam_moving_frac=p['cam_moving_frac']))
        used.add(p['stem'])

    free = lambda: [p for p in elig if p['stem'] not in used]
    d = min(free(), key=lambda p: p['luma_median'])
    add(d, 'very_dark', f"lowest median luma among eligible videos ({d['luma_median']})")
    ov = max([p for p in free() if p['luma_median'] >= 50], key=lambda p: (p['overlay_score'], p['stem']))
    add(ov, 'timestamp_overlay', f"highest overlay score ({ov['overlay_score']}, {ov['overlay_components']} text blobs)")
    cands = sorted([p for p in free() if p['cut_candidates'] > 0], key=lambda p: -p['cut_max_hdist'] * min(p['cut_candidates'], 3))
    picked = False
    for p in cands[:12]:
        if verify_cuts:
            try:
                dec = decode.decode_12fps(p['path'], max_seconds=max_seconds)
                n = len(clean.detect_shot_cuts(dec['frames'], cfg, method='classical')['cuts'])
            except Exception:
                continue
        else:
            n = 1
        if 1 <= n <= 8:
            add(p, 'shot_cut', f"strongest cut candidate (hist dist {p['cut_max_hdist']}); S3 confirms {n} cut(s) at 12 FPS")
            picked = True
            break
    if not picked:
        chosen.append(dict(role='shot_cut', stem=None, why='no video with a confirmed cut'))
    mv = max(free(), key=lambda p: p['cam_moving_frac'])
    if mv['cam_moving_frac'] >= cfg['profile']['moving_camera_frac']:
        add(mv, 'moving_camera', f"highest camera-motion fraction ({mv['cam_moving_frac']})")
    else:
        chosen.append(dict(role='moving_camera', stem=None,
                           why=f"no video reaches cam_moving_frac >= {cfg['profile']['moving_camera_frac']} "
                               f"(max {mv['cam_moving_frac']})"))
    for lb in LABELS:
        pool = [p for p in free() if p['label'] == lb and 30 <= p['duration_s'] <= 240]
        if lb != 'Normal':
            pool = [p for p in pool if gt.get(p['stem']) and
                    max(e for _, e in gt[p['stem']]) / (p['fps_avg'] or 30) <= max_seconds - 5]
        if not pool:
            pool = [p for p in free() if p['label'] == lb]
        allc = [p for p in ok if p['label'] == lb]
        mu_l, sd_l = np.mean([p['luma_median'] for p in allc]), np.std([p['luma_median'] for p in allc]) + 1e-6
        mu_s, sd_s = np.mean([p['sharp_median'] for p in allc]), np.std([p['sharp_median'] for p in allc]) + 1e-6
        dist = lambda p: np.hypot((p['luma_median'] - mu_l) / sd_l, (p['sharp_median'] - mu_s) / sd_s)
        top = sorted(pool, key=dist)[:3]
        p = top[rng.randint(len(top))]
        add(p, f'class_{lb}', f"typical {lb} video (nearest to class median brightness/sharpness; "
                              f"{'has temporal GT' if gt.get(p['stem']) else 'no GT'})")
    return chosen


# ═══════════════════════════════════════════════════════════════════════════════
# Sample saving
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_COLORS = dict(zip(LABELS, viz.SERIES))


def plot_dataset_histograms(profiles: list, thresholds: dict | None = None):
    import matplotlib.pyplot as plt
    ok = [p for p in profiles if p.get('ok')]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    def hist(ax, key, title, bins=20, log=False, xlabel=None, vline=None):
        vals = {lb: [p[key] for p in ok if p['label'] == lb and p.get(key) is not None] for lb in LABELS}
        allv = np.concatenate([v for v in vals.values() if len(v)])
        edges = np.linspace(allv.min(), np.percentile(allv, 99) if log else allv.max(), bins + 1)
        for lb in LABELS:
            ax.hist(np.clip(vals[lb], edges[0], edges[-1]), bins=edges, histtype='step', lw=1.8,
                    color=CLASS_COLORS[lb], label=f'{lb} ({len(vals[lb])})')
        if vline is not None:
            ax.axvline(vline, ls='--', color=viz.INK2)
        ax.set_title(title); ax.set_xlabel(xlabel or key)
    hist(axs[0, 0], 'duration_s', 'duration (s, clipped at p99)', log=True)
    hist(axs[0, 1], 'luma_median', 'median brightness (luma Y)', vline=(thresholds or {}).get('dark_threshold'))
    hist(axs[0, 2], 'sharp_median', 'sharpness (Laplacian var, clipped at p99)', log=True)
    hist(axs[1, 0], 'noise_median', 'noise sigma (Immerkaer)')
    hist(axs[1, 1], 'bit_rate', 'bit rate (b/s, clipped at p99)', log=True)
    axs[0, 1].legend(loc='upper right')
    res = {}
    for p in ok:
        res[f"{p['width']}x{p['height']} @{p['fps_avg']:.0f}fps"] = res.get(f"{p['width']}x{p['height']} @{p['fps_avg']:.0f}fps", 0) + 1
    ks = list(res)
    axs[1, 2].bar(ks, [res[k] for k in ks], color=viz.C['blue'])
    axs[1, 2].set_title('resolution / FPS (videos)')
    for ax in axs.ravel():
        ax.grid(True, axis='y')
    fig.suptitle('S1 dataset profile — 300 UCF-Crime videos by class', fontsize=11)
    fig.tight_layout()
    return fig


def _middle_frame(path: str):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, f = cap.read()
    cap.release()
    return f if ok else np.zeros((240, 320, 3), np.uint8)


def save_dataset_samples(profiles: list, thresholds: dict, summary: dict):
    if not viz.should_save('ALL'):
        return
    ok = [p for p in profiles if p.get('ok')]
    rows = [[lb, v['n'], f"{v['duration_median_s']:.0f}", f"{v['luma_median']:.0f}", f"{v['sharp_median']:.0f}",
             f"{v['noise_median']:.2f}", v['bars'], v['overlay']] for lb, v in summary['per_class'].items()]
    tab = viz.save_table_image(rows, ['class', 'videos', 'median dur (s)', 'median luma', 'median sharp', 'median noise',
                                      'with bars', 'with overlay'], 'Per-class S1 summary')
    viz.save_io_sample(
        'S1', 'dataset_profile', 'ALL', inputs=None, outputs=tab,
        params=dict(videos=summary['n_videos'], decoder='PyAV'),
        metrics=dict(summary=summary, thresholds=thresholds), compare=tab,
        note=('Dataset-level profile of all UCF-Crime videos used: histograms of duration, brightness, sharpness, noise, bit rate, resolution and FPS per class.\n'
              'Why: these numbers set the thresholds used later (dark threshold, blur floor) so they are not guesses.\n'
              f"Notice: dark threshold chosen = {thresholds['dark_threshold']} (darkest {thresholds['dark_percentile']}% of frames; Otsu would say {thresholds['dark_threshold_otsu']:.0f}); all videos are {summary['fps']} FPS, {summary['resolutions']}."),
        plot=lambda: plot_dataset_histograms(profiles, thresholds))


def save_video_samples(p: dict, profiles: list, thresholds: dict):
    """S1 samples for one (sample-set) video: metadata table, I/P/B chart, quality stats vs class."""
    v = p['stem']
    if not viz.should_save(v):
        return
    import matplotlib.pyplot as plt
    frame = _middle_frame(p['path'])
    meta = [['codec', p['codec']], ['resolution', f"{p['width']}x{p['height']}"], ['pix_fmt', p['pix_fmt']],
            ['fps avg / measured', f"{p['fps_avg']} / {p['fps_measured']}"], ['variable FPS', p['vfr']],
            ['bit rate (kb/s)', f"{p['bit_rate'] / 1000:.0f}"], ['duration (s)', p['duration_s']],
            ['frames', p['n_frames']], ['file size (MB)', p['file_mb']]]
    tab = viz.save_table_image(meta, ['field', 'value'], f'{v} — stream metadata (PyAV)', col_w=[0.45, 0.55])
    viz.save_io_sample(
        'S1', 'metadata', v, inputs=frame, outputs=tab, params=dict(decoder='PyAV'),
        metrics={k: p[k] for k in ('codec', 'width', 'height', 'fps_avg', 'fps_measured', 'vfr', 'bit_rate',
                                   'duration_s', 'n_frames')},
        note='Reads container/stream metadata (codec, resolution, nominal vs measured FPS, VFR flag, bitrate, duration).\n'
             'Why: decides how S2 must resample and flags corrupt or unusable videos.\n'
             'Notice: nominal and measured FPS agree and the video is constant-frame-rate.',
        labels=('middle frame', 'metadata'))

    def ipb_fig():
        fig, ax = plt.subplots(figsize=(6, 3.2))
        names = [v] + ['class mean']
        cls = [q for q in profiles if q.get('ok') and q['label'] == p['label']]
        cm = {k: np.mean([q['ipb_frac'][k] for q in cls]) for k in 'IPB'}
        bottoms = np.zeros(2)
        for k, col in zip('IPB', (viz.C['blue'], viz.C['orange'], viz.C['aqua'])):
            vals = np.array([p['ipb_frac'][k], cm[k]])
            ax.barh(names, vals, left=bottoms, color=col, label=f'{k}-frames', edgecolor=viz.SURFACE, linewidth=2)
            bottoms += vals
        ax.set_xlim(0, 1); ax.set_xlabel('share of frames'); ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0))
        fig.tight_layout()
        return fig
    viz.save_io_sample('S1', 'ipb_frame_types', v, inputs=None, outputs=None, compare=ipb_fig(),
                       params={}, metrics=dict(ipb=p['ipb'], ipb_frac=p['ipb_frac']),
                       note='Counts I/P/B picture types over the whole video.\n'
                            'Why: tells how much of the stream is intra-coded vs predicted (compression level / possible blockiness).\n'
                            'Notice: bars are the shares for this video vs. its class mean.')

    def qfig():
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
        ok = [q for q in profiles if q.get('ok')]
        for ax, key, ttl in zip(axs, ('luma_median', 'sharp_median', 'noise_median', 'block_median'),
                                ('brightness (median luma)', 'sharpness (Laplacian var)', 'noise sigma', 'blockiness')):
            for lb in LABELS:
                vals = [q[key] for q in ok if q['label'] == lb]
                ax.hist(vals, bins=18, histtype='step', color=CLASS_COLORS[lb], label=lb, lw=1.5)
            ax.axvline(p[key], color=viz.C['red'], lw=2)
            ax.set_title(ttl)
        axs[0].axvline(thresholds['dark_threshold'], ls='--', color=viz.INK2)
        axs[0].legend(fontsize=7)
        fig.suptitle(f'{v} (red line) against all videos per class', fontsize=9)
        fig.tight_layout()
        return fig
    q = dict(luma_median=p['luma_median'], luma_p10=p['luma_p10'], sharp_median=p['sharp_median'],
             noise_median=p['noise_median'], block_median=p['block_median'], contrast=p['contrast_median'],
             frozen_frac=p['frozen_frac'], black_frac=p['black_frac'], cam_moving_frac=p['cam_moving_frac'],
             overlay_components=p['overlay_components'], bars=p['bars'])
    viz.save_io_sample('S1', 'quality_stats', v, inputs=frame, outputs=None, compare=None, plot=qfig, params={},
                       metrics=q,
                       note='Per-video brightness, sharpness (Laplacian variance), noise (Immerkaer sigma), blockiness, frozen/black share, camera motion.\n'
                            'Why: sets data-driven thresholds (dark threshold, blur floor) and flags poor videos.\n'
                            'Notice: plot.png places this video (red) within its class distribution.')


def write_profile_csv(profiles: list, path: Path):
    import csv
    keys = ['stem', 'label', 'width', 'height', 'fps_avg', 'fps_measured', 'vfr', 'codec', 'bit_rate', 'duration_s',
            'n_frames', 'luma_median', 'sharp_median', 'noise_median', 'block_median', 'frozen_frac', 'black_frac',
            'cut_candidates', 'cam_moving_frac', 'has_bars', 'overlay_components', 'overlay_score', 'ok']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for p in profiles:
            w.writerow([p.get(k) for k in keys])
