"""
runner.py
=========
Per-video orchestration of the VPA stages S2 -> S3 -> S6 -> S4 -> S7 -> S8 -> S9 (S1 is dataset-level and is
driven from extract_clips.py).  Used by extract_clips.py and by the walkthrough notebook.

Every stage can be switched off (ctx.stages); later stages degrade gracefully when an earlier one is off.
"""

from __future__ import annotations

import time
import traceback
import zlib
from dataclasses import dataclass, field

import numpy as np

from .common import load_config, mean_luma
from . import decode, clean, enhance, actors, legacy, motion, package, temporal, viz

ALL_STAGES = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']


@dataclass
class Ctx:
    cfg: dict = field(default_factory=load_config)
    stages: set = field(default_factory=lambda: set(ALL_STAGES))
    max_seconds: float | None = None
    device: str = 'cuda'
    fold: int = 1
    out_root: object = None
    gt: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)         # lazily loaded heavy models (Zero-DCE, YOLO, RAFT ...)
    legacy_samples: bool = True
    fold_splits: dict | None = None            # {fold: 'train'|'test'} when one video serves several folds
    rng_seed: int = 42

    def on(self, s: str) -> bool:
        return s in self.stages


def process_video(video: dict, split: str, ctx: Ctx, verbose: bool = True) -> dict:
    """
    video: {stem, label, path, folder}.  Returns a result dict with every intermediate artefact
    (frames, signals, clips ...) so the notebook can display them.
    """
    t0 = time.time()
    stem, path, label = video['stem'], video['path'], video['label']
    cfg = ctx.cfg
    res = dict(stem=stem, label=label, split=split, stages_done=[], errors={})

    def log(msg):
        if verbose:
            print(f'  [{stem}] {msg}  ({time.time() - t0:.0f}s)', flush=True)

    # ── S2 ────────────────────────────────────────────────────────────────────
    dec = decode.decode_12fps(path, max_seconds=ctx.max_seconds)
    frames = dec['frames']
    res.update(dec=dec, raw_frames=frames)
    res['stages_done'].append('s2')
    log(f"S2 decoded {len(frames)} frames @12 FPS from {dec['src_fps']:.1f} FPS source")
    if viz.should_save(stem):
        old_frames, _, _ = legacy.legacy_first_frames(path, 6)
        decode.sample_resample(stem, dec, legacy_first_frames=old_frames)
        if ctx.legacy_samples:
            legacy.sample_frame_sampling(stem, path, dec['n_src_frames'], dec['src_fps'])
            legacy.sample_resize(stem, frames[len(frames) // 2])
            legacy.sample_zerodce_dark_frame(stem, frames, ctx.device, ctx.models)

    # ── S3 ────────────────────────────────────────────────────────────────────
    if ctx.on('s3'):
        cl = clean.clean_video(frames, cfg, video=stem, video_path=path, models=ctx.models)
        frames = cl['frames']
        res['clean'] = cl
        res['stages_done'].append('s3')
        log(f"S3 crop={cl['crop_box']} overlay_blobs={cl['overlay']['n_components'] if cl['overlay'] else 0} "
            f"cuts={len(cl['cuts'])} frozen={cl['frozen']['n_frozen']} black={cl['frozen']['n_black']}")
    res['frames_clean'] = frames

    # ── S4 ────────────────────────────────────────────────────────────────────
    if ctx.on('s4'):
        sid = res['clean']['shot_ids'] if 'clean' in res else None
        en = enhance.enhance_video(frames, sid, cfg, ctx.models, ctx.device)
        if viz.should_save(stem):
            enhance.save_enhance_samples(stem, frames, en, cfg, ctx.models, ctx.device, sid, actors.get_detector(ctx.models, ctx.device))
        frames = en['frames']
        res['enhance'] = en
        res['stages_done'].append('s4')
        log(f"S4 dark frames {en['dark'].mean():.0%}, enhanced runs {len(en['runs'])} "
            f"(denoise: {[r['denoise'] for r in en['runs']][:1]})")
    res['frames_enh'] = frames

    # ── S6 ────────────────────────────────────────────────────────────────────
    gt_sec = [(a / 30.0, b / 30.0) for a, b in ctx.gt.get(stem, [])]
    res['gt_seconds'] = gt_sec
    if ctx.on('s6'):
        T = len(frames)
        mo = motion.analyze_motion(frames, cfg)                      # features for the whole video
        bad = clean.bad_frame_mask(res['clean'], T) if 'clean' in res else None
        showcase = motion.pick_showcase(mo, bad, prefer=(gt_sec[0][0] * 12, gt_sec[0][1] * 12) if gt_sec else None)
        if viz.should_save(stem):                                    # re-run a short window to capture the stage images
            a_, b_ = max(0, showcase - 80), min(T, showcase + 9)
            sub = motion.analyze_motion(frames[a_:b_], cfg, showcase=showcase - a_)
            mo['show'] = sub['show']
            mo['show']['idx'] = showcase
        raft_info = None
        if viz.should_save(stem) and cfg['motion']['raft']['enabled']:
            idxs = np.unique(np.r_[np.linspace(2, T - 1, cfg['motion']['raft']['max_pairs']).round().astype(int), showcase])
            if 'raft' not in ctx.models:
                try:
                    ctx.models['raft'] = motion.RaftFlow(ctx.device)
                except Exception as ex:                      # logged as TODO in DECISIONS.md
                    ctx.models['raft'] = None
                    print(f'  [RAFT] unavailable: {ex}')
            if ctx.models['raft'] is not None:
                raft_info = motion.raft_vs_farneback(frames, idxs, cfg, ctx.device, ctx.models['raft'], example_t=showcase)
        motion.save_motion_samples(stem, frames, mo, cfg, gt_seconds=gt_sec, raft_info=raft_info)
        res['motion'] = mo
        res['stages_done'].append('s6')
        log(f"S6 motion: mean flow {mo['flow_mag'].mean():.3f}px, mean fg {mo['fg_ratio'].mean():.3f}, "
            f"mean entropy {mo['flow_entropy'].mean():.2f}")
    # ── S7 ────────────────────────────────────────────────────────────────────
    if ctx.on('s7'):
        ac = cfg['actors']
        det = actors.get_detector(ctx.models, ctx.device)
        dets = det.detect(frames, conf=ac['tracker']['track_low_thresh'])      # low conf: ByteTrack's second association
        tracks = actors.track_video(dets, cfg)
        sig = actors.interaction_signals(tracks, cfg)
        crops = actors.actor_crop_boxes(tracks, frames[0].shape[:2], cfg)
        if viz.should_save(stem):
            vis = [d[d[:, 4] >= ac['conf']] for d in dets]
            actors.save_actor_samples(stem, frames, vis, tracks, sig, crops, cfg, gt_seconds=gt_sec, detector_kind=det.kind,
                                      prefer=(gt_sec[0][0] * 12, gt_sec[0][1] * 12) if gt_sec else None)
        res.update(dets=dets, tracks=tracks, interaction=sig, crops=crops, detector_kind=det.kind)
        res['stages_done'].append('s7')
        log(f"S7 {det.kind}: {sig['n_people'].mean():.2f} people/frame, {len({int(r[4]) for t in tracks for r in t})} tracks, "
            f"crop jitter {crops['jitter_raw']:.1f}->{crops['jitter_smooth']:.1f}px")
    # ── S8 / S9 (once per fold the video appears in: only the per-split clip count differs, so S2-S7 are shared) ──
    if ctx.on('s8'):
        luma = res['enhance']['luma_after'] if 'enhance' in res else enhance.luma_series(frames)
        targets = ctx.fold_splits or {ctx.fold: split}
        first = True
        for fold, sp in targets.items():
            rng = np.random.RandomState((cfg['general']['seed'] + zlib.crc32(stem.encode())) % (2 ** 31))
            s8 = temporal.run_s8(frames, label, sp, res.get('motion'), res.get('interaction'), res.get('clean'),
                                 res.get('crops'), luma, gt_sec, cfg, rng=rng)
            res['s8'], res['split'] = s8, sp
            if first and viz.should_save(stem):
                leg = temporal.legacy_candidates(frames, res['crops']['boxes'] if 'crops' in res else None, label, sp,
                                                 res.get('clean'), res.get('motion'), luma, cfg,
                                                 np.random.RandomState(cfg['general']['seed']))
                temporal.save_temporal_samples(stem, frames, s8, cfg, label, gt_sec, leg)
            ge = s8['gt_eval']
            tiou_txt = ', tIoU %.2f' % ge['tiou'] if ge else ''
            log('S8 fold %s/%s %s: %d segments%s; clips cand %d -> gate %d -> dedup %d -> kept %d; rejects %s' % (
                fold, sp, s8['proposal_method'], len(s8['segments']), tiou_txt, s8['n_candidates'], s8['n_passed'],
                s8['n_after_dedup'], len(s8['clips']), s8['reject_counts']))
            if ctx.on('s9') and ctx.out_root is not None:
                pk = package.save_video_clips(res, ctx.out_root, fold, cfg)
                res.setdefault('package_all', []).append(pk)
                if first:
                    package.save_package_sample(stem, res, pk['files'], cfg)
                log(f"S9 fold {fold}: wrote {len(pk['files'])} clips ({cfg['package']['clip_dtype']}) -> "
                    f"{pk['files'][0].parent if pk['files'] else '-'}")
            first = False
        res['stages_done'] += ['s8'] + (['s9'] if ctx.on('s9') and ctx.out_root is not None else [])
        if 'package_all' in res:
            res['package'] = dict(rows=[r for pk in res['package_all'] for r in pk['rows']],
                                  vrow=res['package_all'][0]['vrow'], files=res['package_all'][0]['files'],
                                  vrows=[pk['vrow'] for pk in res['package_all']])
    res['seconds'] = time.time() - t0
    return res


def prepare_detector_choice(videos: list, ctx: Ctx, max_seconds: float = 90.0) -> str:
    """Decode (+letterbox-crop, +enhance if dark) the given videos and let actors.select_detector pick the detector."""
    frames_by_video = {}
    for v in videos:
        dec = decode.decode_12fps(v['path'], max_seconds=max_seconds)
        fr = dec['frames']
        lb = clean.detect_letterbox(fr, ctx.cfg)
        if lb['has_bars']:
            fr = clean.apply_crop(fr, lb['box'])
        fr = enhance.enhance_video(fr, None, ctx.cfg, ctx.models, ctx.device)['frames']
        frames_by_video[v['stem']] = fr
    return actors.select_detector(frames_by_video, ctx.models, ctx.device)
