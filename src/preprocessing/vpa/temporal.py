"""
temporal.py  (S8 — temporal analytics and clip selection)
==========================================================
  activity_score      weighted motion energy / foreground ratio / interaction score, Savitzky-Golay smoothed
  propose_segments    hysteresis thresholding + peak detection -> high-activity segments (temporal action proposals)
  temporal_iou        proposals vs UCF-Crime temporal ground truth
  sample_clips        abnormal: inside proposals; Normal: uniform + hard negatives; never across a shot cut
  quality_gate        frozen / black / blurry / low-motion-for-abnormal / cut / illumination-change rejection
  deduplicate_clips   ORIGINAL motion-fingerprint deduplication (threshold 0.99), kept as is (+ similarity scores)
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks

from .common import ABNORMAL_LABELS, load_config, letterbox
from . import actors, viz

DEDUP_THRESHOLD = 0.99
FPS = 12.0


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication (original behaviour; exposes the similarity matrix too)
# ═══════════════════════════════════════════════════════════════════════════════

def _fingerprint(clip: np.ndarray) -> np.ndarray:
    diffs = np.abs(np.diff(clip.astype(np.float32), axis=0))
    f = diffs.mean(axis=0).flatten()
    return f / (np.linalg.norm(f) + 1e-8)


def dedup_indices(clips: list, threshold: float = DEDUP_THRESHOLD):
    """Kept indices (greedy, in order) + full cosine-similarity matrix of the motion fingerprints."""
    if len(clips) == 0:
        return [], np.zeros((0, 0))
    nfps = np.stack([_fingerprint(c) for c in clips])
    sim = nfps @ nfps.T
    kept = [0]
    for i in range(1, len(clips)):
        if not any(float(sim[i, j]) > threshold for j in kept):
            kept.append(i)
    return kept, sim


def deduplicate_clips(clips: list, threshold: float = DEDUP_THRESHOLD) -> list:
    """
    Motion-fingerprint dedup. Uses mean absolute frame-diff — not mean frame,
    which is nearly identical for all clips from fixed-camera CCTV footage.
    """
    if len(clips) <= 1:
        return clips
    kept, _ = dedup_indices(clips, threshold)
    return [clips[k] for k in kept]


# ═══════════════════════════════════════════════════════════════════════════════
# Activity score and action proposals
# ═══════════════════════════════════════════════════════════════════════════════

def activity_score(motion: dict | None, inter: dict | None, bad: np.ndarray | None, T: int, cfg: dict | None = None) -> dict:
    """act = w_m*motion + w_f*foreground + w_i*interaction (each in [0,1]); invalid frames -> 0; SG-smoothed."""
    cfg = cfg or load_config()
    c = cfg['temporal']
    w = c['weights']
    z = np.zeros(T)
    me = np.clip(motion['motion_energy'] / c['motion_ref'], 0, 1) if motion is not None else z
    fg = np.clip(motion['fg_ratio'] / c['fg_ref'], 0, 1) if motion is not None else z
    it = np.clip(inter['interaction'], 0, 1) if inter is not None else z
    raw = w['motion'] * me + w['fg'] * fg + w['interaction'] * it
    valid = np.ones(T, bool)
    if bad is not None:
        valid &= ~bad
    if motion is not None:
        valid &= ~motion['global_change']
    raw = np.where(valid, raw, 0.0)
    win = int(c['sg_window']) | 1
    win = min(win, T if T % 2 else T - 1)
    smooth = np.clip(savgol_filter(raw, win, min(c['sg_poly'], win - 1)), 0, 1) if win >= 5 else raw.copy()
    return dict(raw=raw, smooth=smooth, motion=me, fg=fg, interaction=it, valid=valid)


def propose_segments(act: np.ndarray, cfg: dict | None = None, fps: float = FPS) -> dict:
    """
    Hysteresis thresholding (a region above `low` is kept only if it contains a sample above `high` / a prominent
    peak), merge close segments, drop short ones, pad. Returns segments as (start, end) frame indices (end exclusive).
    """
    cfg = cfg or load_config()
    c = cfg['temporal']
    hi, lo = c['hysteresis']['high'], c['hysteresis']['low']
    T = len(act)
    above = act > lo
    segs, t = [], 0
    while t < T:
        if above[t]:
            s = t
            while t < T and above[t]:
                t += 1
            segs.append((s, t))
        else:
            t += 1
    peaks, _ = find_peaks(act, height=hi, prominence=c['peak_prominence'])
    keep = [(s, e) for s, e in segs if ((act[s:e] > hi).any() or ((peaks >= s) & (peaks < e)).any())]
    merged = []
    for s, e in keep:
        if merged and s - merged[-1][1] <= c['merge_gap_s'] * fps:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    merged = [(s, e) for s, e in merged if e - s >= c['min_segment_s'] * fps]
    pad = int(c['pad_s'] * fps)
    padded = [(max(0, s - pad), min(T, e + pad)) for s, e in merged]
    out = []
    for s, e in padded:                                   # padding can make neighbours touch
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return dict(segments=out, peaks=peaks.tolist(), method='hysteresis+peaks')


def fallback_segments(act: np.ndarray, cfg: dict | None = None, fps: float = FPS, k: int = 3) -> list:
    """Abnormal video with no proposal: take the top-k highest-activity windows (rank-based)."""
    span = int((cfg or load_config())['decode']['clip_len'] * 4 * 1.0)
    T = len(act)
    if T < span:
        return [(0, T)]
    ma = np.convolve(act, np.ones(span) / span, mode='valid')
    segs = []
    for i in np.argsort(-ma):
        if all(abs(i - s) >= span for s, _ in segs):
            segs.append((int(i), int(i) + span))
        if len(segs) == k:
            break
    return sorted(segs)


def temporal_iou(segs: list, gt_frames: list, T: int) -> dict:
    """Frame-level temporal IoU between the union of the proposals and the union of the GT intervals."""
    p = np.zeros(T, bool)
    g = np.zeros(T, bool)
    for s, e in segs:
        p[s:e] = True
    for s, e in gt_frames:
        g[max(0, s):min(T, e)] = True
    inter, union = int((p & g).sum()), int((p | g).sum())
    return dict(tiou=inter / union if union else 0.0, precision=inter / max(int(p.sum()), 1),
                recall=inter / max(int(g.sum()), 1), pred_frames=int(p.sum()), gt_frames=int(g.sum()))


# ═══════════════════════════════════════════════════════════════════════════════
# Clip sampling
# ═══════════════════════════════════════════════════════════════════════════════

def clip_span(cfg: dict | None = None) -> int:
    d = (cfg or load_config())['decode']
    return (d['clip_len'] - 1) * d['clip_stride'] + 1


def clip_indices(start: int, cfg: dict | None = None) -> np.ndarray:
    d = (cfg or load_config())['decode']
    return start + d['clip_stride'] * np.arange(d['clip_len'])


def valid_starts(T: int, shot_ids: np.ndarray | None, cfg: dict | None = None, step: int = 2, shot_aware: bool = True) -> np.ndarray:
    span = clip_span(cfg)
    if T < span:
        return np.zeros(0, int)
    st = np.arange(0, T - span + 1, step)
    if shot_aware and shot_ids is not None:
        st = st[shot_ids[st] == shot_ids[st + span - 1]]
    return st


def farthest_point(starts: np.ndarray, scores: np.ndarray, n: int, min_sep: int) -> list:
    """Pick n starts spread over the candidates (max-min distance), first = highest score, min separation enforced."""
    if len(starts) == 0 or n <= 0:
        return []
    chosen = [int(np.argmax(scores))]
    while len(chosen) < n:
        d = np.min(np.abs(starts[:, None] - starts[chosen][None, :]), axis=1).astype(float)
        d[d < min_sep] = -1
        if d.max() < 0:
            break
        chosen.append(int(np.argmax(d + 1e-3 * scores)))
    return [int(starts[i]) for i in chosen]


def sample_clips(label: str, split: str, T: int, act: dict | None, segs: list, shot_ids, cfg: dict | None = None,
                 rng: np.random.RandomState | None = None, oversample: float = 1.5, shot_aware: bool = True) -> list:
    """Candidate clip starts. Abnormal: inside the segments. Normal: uniform + hard negatives."""
    cfg = cfg or load_config()
    c = cfg['temporal']
    rng = rng or np.random.RandomState(cfg['general']['seed'])
    span = clip_span(cfg)
    n = int(np.ceil(c['clips_per_video'][split] * oversample))
    st = valid_starts(T, shot_ids, cfg, shot_aware=shot_aware)
    if len(st) == 0:
        return []
    sm = act['smooth'] if act is not None else np.zeros(T)
    csum = np.concatenate([[0], np.cumsum(sm)])
    score = (csum[st + span] - csum[st]) / span            # mean activity per window
    out = []
    jit = c['temporal_jitter']

    stset = set(st.tolist())

    def jitter(s):
        j = int(s + rng.randint(-jit, jit + 1))
        return j if j in stset else int(s)

    if label in ABNORMAL_LABELS:
        centre = st + span // 2
        inseg = np.zeros(len(st), bool)
        for a, b in segs:
            inseg |= (centre >= a) & (centre < b)
        kind = 'proposal'
        if not inseg.any():                                 # nothing proposed: best windows overall
            inseg = score >= np.percentile(score, 80)
            kind = 'fallback_rank'
        cs, sc = st[inseg], score[inseg]
        picks = farthest_point(cs, sc, n, c['min_start_sep'])
        out = [dict(start=jitter(s), kind=kind) for s in picks]
    else:
        n_hard = int(round(c['hard_negative_frac'] * n))
        n_uni = n - n_hard
        idx = np.linspace(0, len(st) - 1, min(n_uni, len(st))).round().astype(int)
        uni = [int(st[i]) for i in idx]
        out = [dict(start=jitter(s), kind='uniform') for s in uni]
        order = np.argsort(-score)
        hard = []
        for i in order:
            s = int(st[i])
            if all(abs(s - u['start']) >= c['min_start_sep'] for u in out) and all(abs(s - h) >= c['min_start_sep'] for h in hard):
                hard.append(s)
            if len(hard) >= n_hard:
                break
        hard_c = [dict(start=s, kind='hard_negative') for s in hard]
        k = int(round(c['hard_negative_frac'] * c['clips_per_video'][split]))
        # priority order (truncation keeps the head): the hard-negative quota first, then uniform picks, then spare hard ones
        out = hard_c[:k] + [o for o in out] + hard_c[k:]
    lookup = dict(zip(st.tolist(), score.tolist()))
    for o in out:
        o['score'] = float(lookup.get(o['start'], 0.0))
    return out                                              # priority order, NOT time order


def uniform_candidates(T: int, n: int, cfg: dict | None = None, rng=None) -> list:
    """The OLD sampler: n evenly spaced segments with jitter; ignores shots (kept for the old-vs-new comparison)."""
    cfg = cfg or load_config()
    rng = rng or np.random.RandomState(cfg['general']['seed'])
    span = clip_span(cfg)
    if T < span:
        return []
    seg = T / n
    jit = cfg['temporal']['temporal_jitter']
    out = []
    for i in range(n):
        s = int(i * seg) + int(rng.randint(-jit, jit + 1))
        out.append(dict(start=int(np.clip(s, 0, T - span)), kind='uniform_legacy', score=0.0))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Rendering + quality gate
# ═══════════════════════════════════════════════════════════════════════════════

def render_clip(frames, boxes, idxs, mode: str = 'actor', size: int = 224) -> np.ndarray:
    """(16, size, size, 3) uint8 RGB clip from BGR frames: actor crop | letterbox | legacy stretch."""
    out = []
    for i in idxs:
        f = frames[int(i)]
        if mode == 'actor' and boxes is not None:
            im = actors.render_crop(f, boxes[int(i)], size)
        elif mode == 'letterbox' or (mode == 'actor' and boxes is None):
            im, _, _ = letterbox(f, size, 0, cv2.INTER_AREA)
        else:
            im = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return np.stack(out)


def old_valid(clip_u8: np.ndarray, cfg: dict | None = None) -> bool:
    """The legacy extract_clips._valid(): shape, range and clip.std() > 0.01 (on float [0,1])."""
    cfg = cfg or load_config()
    c = clip_u8.astype(np.float32) / 255.0
    return bool(c.shape[1:] == (224, 224, 3) and c.max() <= 1 and c.min() >= 0 and c.std() > cfg['temporal']['gate']['old_std_min'])


def lap_var_series(frames) -> np.ndarray:
    from .profile import sharpness
    return np.array([sharpness(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames])


def quality_gate(start: int, label: str, ctx: dict, cfg: dict | None = None) -> list:
    """
    Reasons to reject the clip starting at `start` (empty list = pass). ctx supplies per-frame arrays:
    frozen, black, luma, lap (lazily cached per frame), motion_energy, global_change, shot_ids.
    """
    cfg = cfg or load_config()
    g = cfg['temporal']['gate']
    span = clip_span(cfg)
    sl = slice(start, start + span)
    idx = clip_indices(start, cfg)
    reasons = []
    if ctx['frozen'][sl].mean() > g['frozen_frac_max']:
        reasons.append('frozen')
    if ctx['luma'][idx].mean() < g['black_luma_max'] or ctx['black'][sl].mean() > 0.2:
        reasons.append('black')
    if np.mean([ctx['lap'](i) for i in idx]) < g['blur_lap_var_min']:
        reasons.append('blurry')
    if label in ABNORMAL_LABELS and ctx['motion_energy'][sl].mean() < g['abnormal_min_motion']:
        reasons.append('low_motion')
    sid = ctx['shot_ids']
    if sid[start] != sid[start + span - 1]:
        reasons.append('shot_cut')
    if ctx['global_change'][sl].mean() > 0.2:
        reasons.append('illumination_change')
    return reasons


def make_gate_ctx(frames, clean_res: dict | None, motion: dict | None, luma: np.ndarray) -> dict:
    T = len(frames)
    fz = (clean_res or {}).get('frozen')
    cache = {}

    def lap(i):
        if i not in cache:
            cache[i] = lap_var_series([frames[i]])[0]
        return cache[i]
    return dict(frozen=fz['frozen'] if fz else np.zeros(T, bool), black=fz['black'] if fz else np.zeros(T, bool),
                luma=luma, lap=lap, motion_energy=motion['motion_energy'] if motion is not None else np.ones(T),
                global_change=motion['global_change'] if motion is not None else np.zeros(T, bool),
                shot_ids=(clean_res or {}).get('shot_ids', np.zeros(T, np.int32)))


# ═══════════════════════════════════════════════════════════════════════════════
# S8 driver
# ═══════════════════════════════════════════════════════════════════════════════

def run_s8(frames, label: str, split: str, motion: dict | None, inter: dict | None, clean_res: dict | None,
           crops: dict | None, luma: np.ndarray, gt_seconds: list, cfg: dict | None = None, mode: str | None = None,
           rng=None, use_proposals: bool = True) -> dict:
    cfg = cfg or load_config()
    T = len(frames)
    mode = mode or cfg['package']['framing']
    rng = rng or np.random.RandomState(cfg['general']['seed'])
    bad = None
    if clean_res is not None:
        from .clean import bad_frame_mask
        bad = bad_frame_mask(clean_res, T)
    act = activity_score(motion, inter, bad, T, cfg)
    prop = propose_segments(act['smooth'], cfg)
    segs = prop['segments']
    method = prop['method']
    if label in ABNORMAL_LABELS and not segs and use_proposals and motion is not None:
        segs = fallback_segments(act['smooth'], cfg)
        method = 'fallback_rank'
    if not use_proposals:
        segs, method = [], 'uniform'
    gt_frames = [(int(a * FPS), int(b * FPS)) for a, b in gt_seconds]
    gt_eval = temporal_iou(segs, gt_frames, T) if gt_frames else None
    sid = (clean_res or {}).get('shot_ids')
    gctx = make_gate_ctx(frames, clean_res, motion, luma)
    boxes = crops['boxes'] if crops is not None else None
    span = clip_span(cfg)
    n_target = cfg['temporal']['clips_per_video'][split]

    cands = sample_clips(label, split, T, act, segs, sid, cfg, rng) if use_proposals else \
        [dict(c, kind='uniform') for c in uniform_candidates(T, int(np.ceil(1.5 * n_target)), cfg, rng)]
    for cnd in cands:
        cnd['idx'] = clip_indices(cnd['start'], cfg)
        cnd['reasons'] = quality_gate(cnd['start'], label, gctx, cfg)
        cnd['passed'] = not cnd['reasons']
    passed = [c for c in cands if c['passed']]
    rendered = [render_clip(frames, boxes, c['idx'], mode, cfg['package']['clip_size']) for c in passed]
    kept_idx, sim = dedup_indices(rendered, cfg['temporal']['dedup_threshold'])
    order = sorted(kept_idx[:n_target], key=lambda i: passed[i]['start'])      # keep the highest-priority n, store in time order
    final = [passed[i] for i in order]
    clips = [rendered[i] for i in order]
    for c in cands:
        c['selected'] = False
    for c in final:
        c['selected'] = True
    for i, c in enumerate(passed):
        c['dedup_dropped'] = i not in kept_idx
    return dict(act=act, segments=segs, proposal_method=method, gt_eval=gt_eval, gt_frames=gt_frames, candidates=cands,
                passed=passed, rendered=rendered, kept_idx=kept_idx, sim=sim, selected=final, clips=clips, bad=bad,
                gate_ctx=gctx, n_target=n_target, T=T, mode=mode, span=span,
                reject_counts={r: sum(r in c['reasons'] for c in cands) for r in
                               ('frozen', 'black', 'blurry', 'low_motion', 'shot_cut', 'illumination_change')},
                n_candidates=len(cands), n_passed=len(passed), n_after_dedup=len(kept_idx))


# ═══════════════════════════════════════════════════════════════════════════════
# Sample saving
# ═══════════════════════════════════════════════════════════════════════════════

def _clip_strip(clip_rgb: np.ndarray, n: int = 8, h: int = 96, label: str | None = None) -> np.ndarray:
    fr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in clip_rgb]
    s = viz.filmstrip(fr, n, h, labels=[str(i) for i in range(len(fr))])
    return viz.put_label(s, label, 0.4) if label else s


def save_temporal_samples(video: str, frames, s8: dict, cfg: dict, label: str, gt_seconds: list, legacy_cmp: dict | None = None):
    if not viz.should_save(video):
        return
    import matplotlib.pyplot as plt
    T = s8['T']
    t = np.arange(T) / FPS
    act = s8['act']
    c = cfg['temporal']

    def timeline(show_clips=True, title=''):
        fig, axs = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
        ax = axs[0]
        ax.plot(t, act['raw'], color=viz.INK2, lw=0.7, label='raw score')
        ax.plot(t, act['smooth'], color=viz.C['blue'], lw=1.8, label='Savitzky-Golay smoothed')
        ax.axhline(c['hysteresis']['high'], ls='--', color=viz.C['red'], lw=1, label='high thr')
        ax.axhline(c['hysteresis']['low'], ls=':', color=viz.C['orange'], lw=1, label='low thr')
        for k, (a, b) in enumerate(s8['segments']):
            ax.axvspan(a / FPS, b / FPS, color=viz.C['aqua'], alpha=.28, label='proposed segment' if k == 0 else None)
        for k, (a, b) in enumerate(gt_seconds):
            ax.axvspan(a, b, ymin=0, ymax=0.06, color=viz.C['red'], alpha=.9, label='ground-truth event' if k == 0 else None)
        ax.set_ylabel('activity score'); ax.legend(ncol=3, fontsize=7, loc='upper right')
        ge = s8['gt_eval']
        ax.set_title(title + (f"  tIoU={ge['tiou']:.2f} (precision {ge['precision']:.2f}, recall {ge['recall']:.2f})" if ge else '  (no ground truth)'), fontsize=9)
        ax2 = axs[1]
        ax2.set_ylim(0, 3); ax2.set_yticks([0.5, 1.5, 2.5]); ax2.set_yticklabels(['not kept', 'selected', 'old uniform'], fontsize=7)
        for cnd in s8['candidates']:
            ax2.plot([cnd['start'] / FPS, (cnd['start'] + s8['span']) / FPS], [1.5 if cnd.get('selected') else 0.5] * 2,
                     color=viz.C['aqua'] if cnd.get('selected') else viz.C['red'], lw=3, solid_capstyle='butt')
        if legacy_cmp:
            for cnd in legacy_cmp['cands']:
                ax2.plot([cnd['start'] / FPS, (cnd['start'] + s8['span']) / FPS], [2.5, 2.5], color=viz.C['orange'], lw=3, alpha=.6, solid_capstyle='butt')
        ax2.set_xlabel('time (s)')
        fig.tight_layout()
        return fig

    sel = s8['selected']
    mid = lambda i: frames[int(np.clip(i, 0, T - 1))]
    hi_t = int(np.argmax(act['smooth']))
    lo_t = int(np.argmin(np.where(act['valid'], act['smooth'], 9)))
    viz.save_io_sample(
        'S8', 'activity_score', video, inputs=[mid(lo_t), mid(hi_t)],
        input_labels=[f'lowest activity t={lo_t / FPS:.1f}s', f'peak activity t={hi_t / FPS:.1f}s'], outputs=None,
        compare=timeline(title=f'{video} [{label}] activity score + proposals'),
        plot=lambda: timeline(title=f'{video} [{label}]'),
        params=dict(weights=c['weights'], motion_ref=c['motion_ref'], fg_ref=c['fg_ref'], sg_window=c['sg_window'], sg_poly=c['sg_poly'],
                    hysteresis=c['hysteresis'], peak_prominence=c['peak_prominence']),
        metrics=dict(n_segments=len(s8['segments']), segments_s=[(round(a / FPS, 1), round(b / FPS, 1)) for a, b in s8['segments']],
                     proposal_method=s8['proposal_method'], gt_s=gt_seconds, gt_eval=s8['gt_eval'], frames=T,
                     valid_frac=float(act['valid'].mean())),
        note=('Per-frame activity score = weighted motion energy + foreground ratio + interaction signal, smoothed (Savitzky-Golay); hysteresis thresholds and peaks give action proposals.\n'
              'Why: replaces blind uniform sampling - clips of abnormal classes are taken where something actually happens (temporal action proposals).\n'
              'Notice: green bands = proposed segments, thick red bar at the bottom = UCF-Crime ground truth (test videos only); tIoU in the title.'))

    # proposals / sampling comparison
    seg_frames = []
    for a, b in s8['segments'][:4]:
        seg_frames += [mid(a), mid((a + b) // 2), mid(b - 1)]
    viz.save_io_sample(
        'S8', 'motion_guided_sampling', video, inputs=[mid(c_['start'] + s8['span'] // 2) for c_ in (legacy_cmp['cands'][:8] if legacy_cmp else [])] or [mid(T // 2)],
        input_labels=[f"old uniform t={c_['start'] / FPS:.0f}s" for c_ in (legacy_cmp['cands'][:8] if legacy_cmp else [])] or None,
        outputs=[_clip_strip(c_['clip'], 4, 90, f"start {c_['start'] / FPS:.1f}s ({c_['kind']})") for c_ in
                 [dict(start=x['start'], kind=x['kind'], clip=s8['clips'][i]) for i, x in enumerate(sel[:6])]] if sel else None,
        compare=timeline(title=f'{video} [{label}] old uniform vs motion-guided clip starts'),
        params=dict(label=label, clips_per_video=c['clips_per_video'], min_start_sep=c['min_start_sep'],
                    hard_negative_frac=c['hard_negative_frac'] if label not in ABNORMAL_LABELS else None),
        metrics=dict(n_selected=len(sel), kinds={k: sum(x['kind'] == k for x in sel) for k in {x['kind'] for x in sel}},
                     in_gt_frac=float(np.mean([any(a <= x['start'] + s8['span'] // 2 < b for a, b in s8['gt_frames']) for x in sel])) if s8['gt_frames'] and sel else None,
                     legacy_in_gt_frac=float(np.mean([any(a <= x['start'] + s8['span'] // 2 < b for a, b in s8['gt_frames']) for x in legacy_cmp['cands']])) if s8['gt_frames'] and legacy_cmp and legacy_cmp['cands'] else None),
        note=('Abnormal classes: clips are sampled from inside the proposed segments (farthest-point spread, never across a shot cut); Normal: uniform + high-motion hard negatives.\n'
              'Why: UCF-Crime Fighting videos are mostly normal footage, so uniform sampling gives clips labelled Fighting that show no fighting (label noise).\n'
              'Notice: in the timeline the bottom row compares old uniform starts (orange) with the new selected starts (green); left images = old picks, right = new clips.'),
        labels=('old uniform picks (mid-frames)', 'new selected clips (filmstrips)'), tile_h=130)

    # selected clips filmstrip
    strips = [_clip_strip(s8['clips'][i], 8, 84, f"clip {i}  t={x['start'] / FPS:.1f}-{(x['start'] + s8['span']) / FPS:.1f}s  {x['kind']}  act={x['score']:.2f}")
              for i, x in enumerate(sel[:8])]
    viz.save_io_sample(
        'S8', 'selected_clips', video, inputs=None, outputs=np.vstack(strips) if strips else None,
        params=dict(clip_len=cfg['decode']['clip_len'], stride=cfg['decode']['clip_stride'], framing=s8['mode'], size=cfg['package']['clip_size']),
        metrics=dict(n_selected=len(sel), starts_s=[round(x['start'] / FPS, 1) for x in sel], target=s8['n_target']),
        note=('The final clips (16 frames, stride 4 at 12 FPS, ~5.3 s) as filmstrips of the exact 224x224 model input.\n'
              'Why: this is what VideoMAE will be trained on - actor-centred, cleaned, enhanced.\n'
              'Notice: each row = one clip (8 of its 16 frames); labels give start/end time, selection kind and mean activity.'),
        output_clip=[cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in s8['clips'][0]] if s8['clips'] else None)

    # quality gate rejections
    rej = [x for x in s8['candidates'] if x['reasons']]
    if legacy_cmp:
        rej += [x for x in legacy_cmp['cands'] if x['reasons'] and not any(abs(x['start'] - y['start']) < 2 for y in rej)]
    tiles = []
    for x in rej[:8]:
        cl = x.get('clip_for_show')
        im = frames[int(x['idx'][len(x['idx']) // 2])] if cl is None else cv2.cvtColor(cl[len(cl) // 2], cv2.COLOR_RGB2BGR)
        tiles.append(viz.put_label(im, f"t={x['start'] / FPS:.1f}s: {','.join(x['reasons'])}", 0.38))
    cnt = s8['reject_counts']
    cnt_old = {k: sum(k in x['reasons'] for x in legacy_cmp['cands']) if legacy_cmp else 0 for k in cnt}
    rows = [[k, cnt[k], cnt_old[k]] for k in cnt]
    n_old = len(legacy_cmp['cands']) if legacy_cmp else 0
    tab = viz.save_table_image(rows, ['rule', f'new candidates ({s8["n_candidates"]})', f'old uniform ({n_old})'],
                               f'{video}: clips rejected per rule', col_w=[0.4, 0.3, 0.3])
    viz.save_io_sample(
        'S8', 'quality_gate', video, inputs=tiles or None, outputs=tab, cols=4, tile_h=150,
        params=dict(**c['gate']), metrics=dict(reject_counts=cnt, reject_counts_old_uniform=cnt_old, n_candidates=s8['n_candidates'], n_passed=s8['n_passed'],
                                             rejected=[dict(start_s=round(x['start'] / FPS, 1), reasons=x['reasons']) for x in rej[:30]]),
        note=('Each candidate clip is rejected if it is frozen, black, blurry (Laplacian variance below the S1-derived floor), has almost no motion although the class is abnormal, spans a shot cut, or sits in a fade/flash.\n'
              'Why: replaces the old `clip.std() > 0.01` check, which cannot see these defects.\n'
              'Notice: tiles = middle frame of rejected candidate clips with the reason(s) (candidates include old-style uniform ones so defects show up); table = rejections per rule.'),
        labels=('rejected candidates', 'rejections per rule'))

    # X4: dedup and X5: old vs new validity
    kept = set(s8['kept_idx'])
    sim = s8['sim']
    if len(s8['rendered']) > 1:
        tiles = []
        for j, (cnd, clip) in enumerate(zip(s8['passed'], s8['rendered'])):
            prev = [sim[j, k] for k in s8['kept_idx'] if k < j]
            ms = max(prev) if prev else 0.0
            tiles.append(viz.put_label(cv2.cvtColor(clip[8], cv2.COLOR_RGB2BGR), f"{'KEEP' if j in kept else 'DROP'} sim={ms:.3f}", 0.4))

        def sim_fig():
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            im = ax.imshow(sim, vmin=min(0.8, float(sim.min())), vmax=1.0, cmap='viridis')
            ax.set_title(f'motion-fingerprint cosine similarity (drop if > {cfg["temporal"]["dedup_threshold"]})', fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.04)
            fig.tight_layout()
            return fig
        viz.save_io_sample(
            'X4', 'dedup', video, inputs=tiles, outputs=[tiles[i] for i in sorted(kept)], cols=6, tile_h=110, plot=sim_fig,
            params=dict(existing_function='deduplicate_clips', threshold=cfg['temporal']['dedup_threshold'], fingerprint='mean |frame diff| per pixel'),
            metrics=dict(before=len(s8['rendered']), after=len(kept), max_off_diagonal_similarity=float(np.max(sim - np.eye(len(sim)) * 9)),
                         mean_similarity=float(sim[np.triu_indices(len(sim), 1)].mean())),
            note=('EXISTING op (unchanged): motion-fingerprint deduplication drops a clip whose mean-abs-frame-difference image is more than 0.99 cosine-similar to an already kept clip.\n'
                  'Why: heavily overlapping clips from static CCTV add no information and bias training.\n'
                  'Notice: tiles show each candidate with KEEP/DROP and the highest similarity to an earlier kept clip; heat-map in plot.png.'),
            labels=('candidate clips (before)', 'kept clips (after)'))
    if legacy_cmp:
        tiles = []
        rows = []
        diff = 0
        for k, x in enumerate(legacy_cmp['cands']):
            ov, nv = x['old_valid'], not x['reasons']
            diff += ov and not nv
            rows.append([k, f"{x['start'] / FPS:.1f}s", 'pass' if ov else 'fail', 'pass' if nv else 'FAIL: ' + ','.join(x['reasons'])])
        for x in [x for x in legacy_cmp['cands'] if x['old_valid'] and x['reasons']][:6]:
            tiles.append(viz.put_label(cv2.cvtColor(x['clip'][8], cv2.COLOR_RGB2BGR), f"old OK, new REJECT: {','.join(x['reasons'])}", 0.36))
        tab = viz.save_table_image(rows[:20], ['#', 'start', 'OLD std>0.01', 'NEW quality gate'], f'{video}: same {len(rows)} uniform clips, old vs new check',
                                   col_w=[0.08, 0.15, 0.22, 0.55])
        viz.save_io_sample(
            'X5', 'validity_old_vs_new', video, inputs=tiles or None, outputs=tab, cols=3, tile_h=150, compare=None,
            params=dict(old='std>0.01', new='quality gate', clips='old uniform'),
            metrics=dict(n_clips=len(legacy_cmp['cands']), old_pass=sum(x['old_valid'] for x in legacy_cmp['cands']),
                         new_pass=sum(not x['reasons'] for x in legacy_cmp['cands']), accepted_by_old_rejected_by_new=int(diff)),
            note=('EXISTING check (std > 0.01) vs the new quality gate on the SAME old-style uniform clips.\n'
                  'Why: std > 0.01 passes almost everything - frozen footage, blur, fades and clips that straddle a cut all have enough pixel variance.\n'
                  'Notice: tiles = clips the old check accepted and the new gate rejects (with the reason); table lists every clip.'),
            labels=('accepted by OLD, rejected by NEW', 'per-clip verdicts'))


def legacy_candidates(frames, boxes, label, split, clean_res, motion, luma, cfg, rng, mode='actor') -> dict:
    """Old-style uniform candidates (n = clips_per_video), judged by BOTH the old std check and the new gate."""
    T = len(frames)
    n = cfg['temporal']['clips_per_video'][split]
    cands = uniform_candidates(T, n, cfg, rng)
    gctx = make_gate_ctx(frames, clean_res, motion, luma)
    for c in cands:
        c['idx'] = clip_indices(c['start'], cfg)
        c['clip'] = render_clip(frames, boxes, c['idx'], mode, cfg['package']['clip_size'])
        c['old_valid'] = old_valid(c['clip'], cfg)
        c['reasons'] = quality_gate(c['start'], label, gctx, cfg)
        c['clip_for_show'] = c['clip']
    return dict(cands=cands)
