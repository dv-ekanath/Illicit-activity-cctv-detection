"""
extract_clips.py  (v13 — VPA pipeline driver)
===============================================================
4 classes: Fighting, Robbery, Vandalism, Normal  (Assault removed)

Runs the video-processing-and-analytics stages in order (modules in src/preprocessing/vpa/):

  S1  profile.py   video profiling & quality assessment (dataset-level, sets thresholds, picks the sample set)
  S2  decode.py    timestamp-based resampling to exactly 12 FPS
  S3  clean.py     shot cuts, frozen/black frames, letterbox bars, timestamp-overlay masking
  S4  enhance.py   one brightness measure -> Zero-DCE++ -> temporal denoise -> deflicker  (S5 camera comp.)
  S6  motion.py    background subtraction, optical flow, MHI, motion-energy signal
  S7  actors.py    detection + ByteTrack, interaction signals, actor-centred crop
  S8  temporal.py  activity score, action proposals, clip sampling, quality gate, dedup
  S9  package.py   uint8 clips, JSON sidecars, dataset analytics report

Official Action_Recognition_splits loading + the train/test leakage check (vpa/splits.py) and the
motion-fingerprint deduplication (vpa/temporal.py) are the original implementations, kept and re-exported here.

EXAMPLES
  # sample run (8 fixed videos, saves every operation's input/output samples + INDEX.md)
  python src/preprocessing/extract_clips.py --sample-set --save-samples
  # switch stages off
  python src/preprocessing/extract_clips.py --sample-set --skip s7,s5
  # full extraction (all folds, all videos)  -- NOT run as part of the VPA build
  python src/preprocessing/extract_clips.py --all-folds

OUTPUT STRUCTURE (data/processed/clips_vpa/, sample runs: clips_vpa_sample/)
  fold_N/train|test/<Label>/scene/<stem>_cNNNN.npy   (16,224,224,3) uint8 RGB   + <stem>_cNNNN.json sidecar
"""

import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _s in (sys.stdout, sys.stderr):                       # Windows consoles default to cp1252; some prints use arrows
    try:
        _s.reconfigure(encoding='utf-8')
    except Exception:
        pass

from preprocessing.vpa.common import (                                    # noqa: E402
    ROOT, UCF_DIR, VIDEOS_DIR, SPLITS_DIR, ZERODCE_W, CLIPS_DIR, CLIPS_SAMPLE_DIR, DOCS_DIR, CACHE_DIR,
    UCF_FOLDER_TO_LABEL, LABELS, VALID_FOLDS, VIDEO_EXTENSIONS, load_config, find_video, load_temporal_gt)
from preprocessing.vpa.splits import load_official_fold, split_of         # noqa: E402,F401
from preprocessing.vpa.temporal import deduplicate_clips                  # noqa: E402,F401
from preprocessing.vpa import profile, viz, runner, legacy                # noqa: E402

# ── Clip parameters (kept for backwards compatibility; real values live in vpa/config.yaml) ──────
_C = load_config()
TARGET_FPS        = _C['decode']['target_fps']
CLIP_LEN          = _C['decode']['clip_len']
STRIDE            = _C['decode']['clip_stride']
CLIP_SIZE         = (_C['package']['clip_size'],) * 2
CLIPS_PER_VIDEO   = dict(_C['temporal']['clips_per_video'])
DEDUP_THRESHOLD   = _C['temporal']['dedup_threshold']
MIN_FRAMES_NEEDED = CLIP_LEN * STRIDE

SAMPLE_SET_JSON = DOCS_DIR / 'sample_set.json'


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_clips(fold: int, split: str = None, clips_dir: Path = CLIPS_DIR):
    print(f"\n{'='*70}")
    print(f"  Clip Verification — Fold {fold}  (CLIP_LEN={CLIP_LEN})  [{clips_dir}]")
    print(f"{'='*70}")
    for sp in ([split] if split else ['train', 'test']):
        print(f"\n  Split: {sp}")
        print(f"  {'Label':10s}  {'Clips':>6}  {'Videos':>7}  {'Clips/vid':>9}  Status")
        print(f"  {'-'*50}")
        for label in LABELS:
            d = clips_dir / f'fold_{fold}' / sp / label / 'scene'
            if not d.exists():
                print(f"  {label:10s}  NOT FOUND")
                continue
            clip_files = list(d.glob('*.npy'))
            stems = {re.sub(r'_c\d{4}$', '', p.stem) for p in clip_files}
            errors = 0
            for p in clip_files[:20]:
                try:
                    c = np.load(p)
                    if c.shape != (CLIP_LEN, *CLIP_SIZE, 3):
                        errors += 1
                    elif c.dtype == np.float32 and (c.max() > 1.0 or c.min() < 0.0):
                        errors += 1
                    elif c.std() < 0.01:
                        errors += 1
                except Exception:
                    errors += 1
            status = "✅" if errors == 0 else f"❌ {errors} errors"
            print(f"  {label:10s}  {len(clip_files):6d}  {len(stems):7d}  "
                  f"{len(clip_files) / max(1, len(stems)):9.1f}  {status}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Stage S1 (dataset level) + sample set
# ═══════════════════════════════════════════════════════════════════════════════

def run_s1(args, ctx):
    """Profile all videos, derive thresholds, (re)select the fixed sample set, save S1 samples."""
    print("\n[S1] Video profiling & quality assessment")
    profiles = profile.profile_dataset(force=args.force_profile)
    summary = profile.dataset_summary(profiles)
    thr = profile.derive_thresholds(profiles)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / 'thresholds.json').write_text(json.dumps(thr, indent=2))
    print(f"[S1] dark threshold (data-driven) = {thr['dark_threshold']}  | blur Laplacian-var p5 = {thr['blur_lap_var_p5']:.0f}"
          f"  | moving-camera videos: {summary['moving_camera_videos']}/{summary['n_ok']}")

    if SAMPLE_SET_JSON.exists() and not args.reselect:
        sample = json.loads(SAMPLE_SET_JSON.read_text())
    else:
        sample = profile.select_sample_set(profiles, seed=ctx.cfg['general']['seed'],
                                           max_seconds=ctx.cfg['general']['sample_max_seconds'])
        SAMPLE_SET_JSON.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE_SET_JSON.write_text(json.dumps(sample, indent=2))
        print(f"[S1] sample set written -> {SAMPLE_SET_JSON}")
    for s in sample:
        print(f"      {s.get('role'):18s} {s.get('stem')}  — {s.get('why')}")

    if args.save_samples:
        stems = [s['stem'] for s in sample if s.get('stem')]
        viz.configure(True, stems=stems)
        profile.write_profile_csv(profiles, DOCS_DIR / 'S1_profiles.csv')
        profile.save_dataset_samples(profiles, thr, summary)
        by = {p['stem']: p for p in profiles}
        for s in sample:
            if s.get('stem'):
                profile.save_video_samples(by[s['stem']], profiles, thr)
    return profiles, sample, thr, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_stages(args) -> set:
    st = set(runner.ALL_STAGES)
    if args.stages:
        st = {s.strip().lower() for s in args.stages.split(',') if s.strip()}
    if args.skip:
        st -= {s.strip().lower() for s in args.skip.split(',') if s.strip()}
    return st


def _run_one(v, split, ctx, all_rows, all_vrows):
    """Process one video; a failing video is logged and never stops the run. Keeps only the small report rows."""
    try:
        r = runner.process_video(v, split, ctx)
        if 'package' in r:
            all_rows += r['package']['rows']
            all_vrows += r['package'].get('vrows', [r['package']['vrow']])
        del r
    except Exception as e:
        import traceback
        print(f"  [{v['stem']}] ERROR: {e}")
        traceback.print_exc()


def process_folds(folds, args, ctx, all_rows, all_vrows):
    """
    Official-split driven extraction (leakage check inside load_official_fold). Every video appears in all four folds
    (train in three, test in one), and only the per-split clip count differs, so each video is analysed ONCE (S2-S7)
    and S8/S9 are run for every fold it belongs to.
    """
    membership = {}                                    # stem -> {fold: split}
    order = {lb: {} for lb in LABELS}
    for fold in folds:
        fs = load_official_fold(fold)
        for split in ('train', 'test'):
            if args.split and split != args.split:
                continue
            for label in LABELS:
                if args.cls and label != args.cls:
                    continue
                for _, stem in fs[split].get(label, []):
                    membership.setdefault(stem, {})[fold] = split
                    order[label][stem] = True
    for label in LABELS:
        stems = sorted(order[label])
        if args.max_videos:
            stems = stems[:args.max_videos]
        print(f"\n[{label}] {len(stems)} videos")
        for i, stem in enumerate(stems):
            v = find_video(stem)
            if v is None:
                print(f"  [{i + 1}/{len(stems)}] NOT FOUND: {stem}")
                continue
            ctx.fold_splits = membership[stem]
            print(f"  [{i + 1}/{len(stems)}] {stem}  folds: {membership[stem]}")
            _run_one(v, next(iter(membership[stem].values())), ctx, all_rows, all_vrows)
    ctx.fold_splits = None
    for fold in folds:
        verify_clips(fold, args.split, ctx.out_root)


def main(args):
    cfg = load_config()
    stages = parse_stages(args)
    folds_to_run = list(VALID_FOLDS) if args.all_folds else [args.fold]

    if args.show_split:
        for fold in folds_to_run:
            load_official_fold(fold)
        return
    if args.verify_only:
        for fold in folds_to_run:
            verify_clips(fold, args.split, CLIPS_SAMPLE_DIR if args.sample_set else CLIPS_DIR)
        return

    max_sec = args.max_seconds if args.max_seconds is not None else (
        cfg['general']['sample_max_seconds'] if args.sample_set else cfg['general']['full_max_seconds'])
    out_root = Path(args.out_dir) if args.out_dir else (CLIPS_SAMPLE_DIR if args.sample_set else CLIPS_DIR)
    if args.clip_dtype:
        cfg['package']['clip_dtype'] = args.clip_dtype
    ctx = runner.Ctx(cfg=cfg, stages=stages, max_seconds=max_sec, device=args.device, fold=args.fold,
                     out_root=out_root, gt=load_temporal_gt())
    if args.save_samples:
        viz.configure(True)

    print(f"\n{'='*70}\n  extract_clips v13 — VPA pipeline | stages: {sorted(stages)}"
          f"\n  mode: {'SAMPLE SET' if args.sample_set else 'full'} | max_seconds={max_sec} | out: {out_root}\n{'='*70}")

    sample = None
    if 's1' in stages and (args.sample_set or args.profile):
        profiles, sample, thr, summary = run_s1(args, ctx)
    elif args.sample_set:
        sample = json.loads(SAMPLE_SET_JSON.read_text())
    if args.profile and not args.sample_set:
        return

    all_rows, all_vrows = [], []
    if args.sample_set:
        sample_stems = [x['stem'] for x in sample if x.get('stem')]
        stems = list(sample_stems)
        if args.videos:
            wanted = [w.strip() for w in args.videos.split(',')]
            stems = [s_ for s_ in stems if any(w in s_ for w in wanted)]
        if args.max_videos:
            stems = stems[:args.max_videos]
        if args.save_samples:
            viz.configure(True, stems=stems)
        fs = load_official_fold(args.fold, verbose=False)
        if args.save_samples:
            legacy.sample_annotation_boxes()           # X6: existing annotation-conversion check (dataset level)
        if 's7' in stages and (args.reselect_detector or not (CACHE_DIR / 'detector_choice.json').exists()):
            kind = runner.prepare_detector_choice([find_video(s_) for s_ in sample_stems], ctx)
            print(f"[S7] detector choice: {kind}")
        for stem in stems:
            v = find_video(stem)
            sp = 'test' if any(stem == s for vv in fs['test'].values() for _, s in vv) else 'train'
            _run_one(v, sp, ctx, all_rows, all_vrows)
    else:
        if 's7' in stages and not (CACHE_DIR / 'detector_choice.json').exists():
            fs = load_official_fold(args.fold, verbose=False)
            probe = [find_video(stem) for lb in LABELS for _, stem in fs['train'].get(lb, [])[:2]]
            print(f"[S7] detector choice: {runner.prepare_detector_choice([p for p in probe if p], ctx)}")
        process_folds(folds_to_run, args, ctx, all_rows, all_vrows)

    if 's9' in stages and all_rows:
        from preprocessing.vpa import package
        rep = package.build_report(all_rows, all_vrows, out_root / '_vpa_report', save_samples=args.save_samples)
        print(f"[S9] dataset report -> {out_root / '_vpa_report'}  ({rep['n_clips']} clips, {rep['n_videos']} videos)")
    if args.save_samples:
        print(f"\nIndex -> {viz.build_index()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract_clips v13 — VPA pipeline (S1–S9)')
    fold_grp = parser.add_mutually_exclusive_group()
    fold_grp.add_argument('--fold', type=int, choices=VALID_FOLDS, default=1)
    fold_grp.add_argument('--all-folds', action='store_true')
    parser.add_argument('--split', choices=['train', 'test'], default=None)
    parser.add_argument('--class', dest='cls', default=None, choices=LABELS)
    parser.add_argument('--max-videos', type=int, default=None)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--show-split', action='store_true')
    # VPA options
    parser.add_argument('--sample-set', action='store_true', help='run only the fixed sample set (sample_set.json)')
    parser.add_argument('--save-samples', action='store_true', help='save per-operation input/output samples')
    parser.add_argument('--stages', default=None, help='comma list of stages to run, e.g. s1,s2,s3 (default all)')
    parser.add_argument('--skip', default=None, help='comma list of stages to switch off, e.g. s5,s7')
    parser.add_argument('--max-seconds', type=float, default=None, help='analysis cap per video (seconds)')
    parser.add_argument('--videos', default=None, help='sample-set subset: comma list of stem substrings')
    parser.add_argument('--out-dir', default=None, help='clip output root (default clips_vpa/, sample runs clips_vpa_sample/)')
    parser.add_argument('--clip-dtype', choices=['uint8', 'float32'], default=None, help='override package.clip_dtype')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--profile', action='store_true', help='run S1 only (profile + thresholds + sample set)')
    parser.add_argument('--force-profile', action='store_true')
    parser.add_argument('--reselect', action='store_true', help='re-pick the sample set')
    parser.add_argument('--reselect-detector', action='store_true', help='re-run the fine-tuned vs COCO detector check')
    main(parser.parse_args())
