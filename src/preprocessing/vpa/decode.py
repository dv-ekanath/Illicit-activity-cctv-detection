"""
decode.py  (S2 — decoding and resampling to exactly 12 FPS)
============================================================
Decodes with PyAV and picks, for every output instant t_k = k / 12 s, the source frame whose
presentation timestamp is nearest (what ffmpeg's `fps` filter does). This is correct for 25 FPS,
30 FPS and variable-frame-rate sources, unlike the old `step = round(src_fps / 12)` which turned
30 FPS into 15 FPS and 25 FPS into 12.5 FPS.
"""

from __future__ import annotations

import av
import cv2
import numpy as np

from .common import load_config
from . import viz


def legacy_indices(src_fps: float, n_src: int, target_fps: float = 12.0):
    """The OLD extract_clips._load_frames rule: every `step`-th frame, step = round(src_fps/12)."""
    step = max(1, round(src_fps / target_fps))
    idx = np.arange(0, n_src, step)
    return idx, idx / src_fps, src_fps / step          # indices, times, effective fps


def decode_12fps(path: str, target_fps: float | None = None, max_seconds: float | None = None,
                 start_seconds: float = 0.0, to_bgr: bool = True) -> dict:
    """
    Timestamp-based resampling.
    Returns dict(frames=[BGR uint8], t_target, t_src, src_idx, src_fps, vfr, n_src_frames, ...).
    """
    cfg = load_config()
    fps = float(target_fps or cfg['decode']['target_fps'])
    frames, t_src, src_idx, t_tgt = [], [], [], []
    with av.open(str(path)) as cont:
        st = cont.streams.video[0]
        st.thread_type = 'AUTO'
        tb = float(st.time_base)
        avg = float(st.average_rate) if st.average_rate else 25.0
        t0 = None
        k = int(round(start_seconds * fps))
        prev = None                       # (t, idx, av.VideoFrame)
        n_seen = 0
        dts = []
        last_t = None

        def emit(fr, t, i, tk):
            frames.append(fr.to_ndarray(format='bgr24') if to_bgr else fr.to_ndarray(format='rgb24'))
            t_src.append(t); src_idx.append(i); t_tgt.append(tk)

        end_t = None if max_seconds is None else start_seconds + max_seconds
        for fr in cont.decode(st):
            t = fr.pts * tb if fr.pts is not None else n_seen / avg
            if t0 is None:
                t0 = t
            t -= t0
            if last_t is not None:
                dts.append(t - last_t)
            last_t = t
            cur = (t, n_seen, fr)
            n_seen += 1
            while k / fps <= t + 1e-9 and (end_t is None or k / fps < end_t):
                tk = k / fps
                if prev is not None and (tk - prev[0]) < (cur[0] - tk):
                    emit(prev[2], prev[0], prev[1], tk)
                else:
                    emit(cur[2], cur[0], cur[1], tk)
                k += 1
            prev = cur
            if end_t is not None and k / fps >= end_t:
                break
        dts = np.asarray(dts) if dts else np.array([1 / avg])
        med = float(np.median(dts))
        vfr = bool(np.mean(np.abs(dts - med) > 0.25 * med) > 0.02)
    t_src, t_tgt = np.asarray(t_src), np.asarray(t_tgt)
    return dict(frames=frames, t_target=t_tgt, t_src=t_src, src_idx=np.asarray(src_idx),
                src_fps=1.0 / med if med > 0 else avg, nominal_fps=avg, vfr=vfr,
                n_src_frames=n_seen, fps=fps,
                max_timing_error_s=float(np.max(np.abs(t_src - t_tgt))) if len(t_src) else 0.0,
                width=int(frames[0].shape[1]) if frames else 0, height=int(frames[0].shape[0]) if frames else 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Sample saving
# ═══════════════════════════════════════════════════════════════════════════════

def sample_resample(video: str, dec: dict, legacy_first_frames=None, window_s: float = 1.2):
    """S2 sample: kept-frame timestamps, old vs new sampling, effective FPS."""
    if not viz.should_save(video):
        return
    src_fps = dec['src_fps']
    n_src = dec['n_src_frames']
    l_idx, l_t, l_fps = legacy_indices(dec['nominal_fps'], n_src)      # the old code read cv2's nominal FPS (30.0 exactly)
    n_new = len(dec['frames'])
    t_new = dec['t_target']
    eff_new = (n_new - 1) / (t_new[-1] - t_new[0]) if n_new > 1 else 0
    gaps_new = np.diff(dec['t_src'])

    def plot():
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(9, 3.8), sharex=True)
        src_t = np.arange(int(window_s * src_fps) + 1) / src_fps
        ideal = np.arange(int(window_s * 12) + 1) / 12
        lm = l_t[l_t <= window_s]
        nm = dec['t_src'][dec['t_src'] <= window_s]
        axs[0].eventplot([src_t, lm], lineoffsets=[1, 0], linelengths=0.7,
                         colors=[viz.INK2, viz.C['orange']])
        axs[0].set_yticks([1, 0]); axs[0].set_yticklabels(['source frames', f'OLD step={max(1, round(dec["nominal_fps"] / 12))}'])
        axs[0].set_title(f'first {window_s}s — OLD sampling keeps {l_fps:.1f} FPS')
        axs[1].eventplot([ideal, nm], lineoffsets=[1, 0], linelengths=0.7,
                         colors=[viz.INK2, viz.C['blue']])
        axs[1].set_yticks([1, 0]); axs[1].set_yticklabels(['ideal t=k/12', 'NEW (nearest pts)'])
        axs[1].set_title(f'NEW timestamp-based sampling keeps {eff_new:.2f} FPS')
        axs[1].set_xlabel('time (s)')
        fig.tight_layout()
        return fig

    n_show = 6
    old_frames = legacy_first_frames
    new_frames = dec['frames'][:n_show]
    in_lab = [f'old #{int(i)} t={t:.3f}s' for i, t in zip(l_idx[:n_show], l_t[:n_show])]
    out_lab = [f'new #{int(i)} t={t:.3f}s' for i, t in zip(dec['src_idx'][:n_show], dec['t_src'][:n_show])]
    viz.save_io_sample(
        'S2', 'resample_12fps', video,
        inputs=old_frames if old_frames else new_frames[:1], input_labels=in_lab if old_frames else None,
        outputs=new_frames, output_labels=out_lab, cols=3,
        params=dict(target_fps=12, source_fps=round(src_fps, 3), decoder='PyAV', rule='nearest pts to t=k/12'),
        metrics=dict(source_fps=round(src_fps, 3), vfr=dec['vfr'], old_effective_fps=round(l_fps, 3),
                     new_effective_fps=round(eff_new, 3), n_frames_old=int(len(l_idx)), n_frames_new=n_new,
                     max_timing_error_ms=round(dec['max_timing_error_s'] * 1000, 2),
                     new_source_gap_frames_min_max=[int(np.round(gaps_new.min() * src_fps)),
                                                    int(np.round(gaps_new.max() * src_fps))] if len(gaps_new) else None),
        note=(f'Resamples to exactly 12 FPS by picking the source frame nearest to each t = k/12 s (OLD: every round({src_fps:.0f}/12)-th frame = {l_fps:.1f} FPS).\n'
              'Why: VideoMAE stride 4 only spans the intended ~5.3 s if every video really is 12 FPS; otherwise motion speed differs per video.\n'
              'Notice: plot.png — old kept frames are evenly 2 source frames apart (15 FPS), new alternate 2/3 apart (12 FPS).'),
        plot=plot, labels=('OLD (step) first frames', 'NEW (timestamp) first frames'))
