"""
package.py  (S9 — packaging and dataset analytics report)
=========================================================
  save_video_clips   writes (16,224,224,3) clips as uint8 .npy (4x smaller than the legacy float32) + a JSON
                     sidecar per clip (shot id, motion stats, people count, crop box, brightness, enhanced?, ...)
  build_report       per-class distributions of motion energy, flow entropy, people count, brightness, activity
                     and how many clips each quality-gate rule rejected  (report.json / report.md / plots)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .common import LABELS, load_config
from . import temporal, viz

PIPELINE_VERSION = 'vpa-1.0'


def _clip_stats(res: dict, c: dict, cfg: dict) -> dict:
    """Per-clip statistics over the 61-frame span of the clip."""
    span = temporal.clip_span(cfg)
    a, b = c['start'], c['start'] + span
    idx = c['idx']
    mo, it, en = res.get('motion'), res.get('interaction'), res.get('enhance')
    st = {}
    if mo is not None:
        st.update(motion_energy_mean=float(mo['motion_energy'][a:b].mean()), motion_energy_max=float(mo['motion_energy'][a:b].max()),
                  flow_entropy_mean=float(mo['flow_entropy'][a:b].mean()), flow_div_mean=float(mo['flow_div'][a:b].mean()),
                  flow_curl_mean=float(mo['flow_curl'][a:b].mean()), fg_ratio_mean=float(mo['fg_ratio'][a:b].mean()),
                  blob_count_mean=float(mo['blob_count'][a:b].mean()))
    if it is not None:
        st.update(people_count_mean=float(it['n_people'][a:b].mean()), people_count_max=int(it['n_people'][a:b].max()),
                  min_person_distance_mean=float(np.minimum(it['min_dist'][a:b], 6).mean()), max_iou_mean=float(it['max_iou'][a:b].mean()),
                  speed_max_mean=float(it['speed_max'][a:b].mean()), interaction_mean=float(it['interaction'][a:b].mean()))
    if en is not None:
        st.update(brightness_before=float(en['luma_before'][idx].mean()), brightness_after=float(en['luma_after'][idx].mean()),
                  enhanced_fraction=float(en['enhanced'][idx].mean()))
    st['activity_mean'] = float(res['s8']['act']['smooth'][a:b].mean())
    return st


def build_sidecar(res: dict, c: dict, i: int, fold: int, cfg: dict) -> dict:
    s8, dec = res['s8'], res['dec']
    cl = res.get('clean') or {}
    idx = [int(v) for v in c['idx']]
    boxes = res['crops']['boxes'] if 'crops' in res else None
    return dict(
        pipeline=PIPELINE_VERSION, video=res['stem'], label=res['label'], split=res['split'], fold=fold, clip_index=i,
        start_frame_12fps=int(c['start']), end_frame_12fps=int(c['start'] + s8['span'] - 1),
        start_time_s=round(c['start'] / 12.0, 3), end_time_s=round((c['start'] + s8['span']) / 12.0, 3),
        frame_indices_12fps=idx, source_frame_indices=[int(dec['src_idx'][k]) for k in idx],
        source_times_s=[round(float(dec['t_src'][k]), 4) for k in idx],
        shot_id=int(cl['shot_ids'][c['start']]) if 'shot_ids' in cl else 0, n_shots_in_video=int(cl['shot_ids'].max() + 1) if 'shot_ids' in cl else 1,
        selection_kind=c['kind'], window_activity=round(float(c['score']), 4),
        stats={k: round(v, 4) if isinstance(v, float) else v for k, v in _clip_stats(res, c, cfg).items()},
        crop_boxes_xyxy=[[int(v) for v in boxes[k]] for k in idx] if boxes is not None else None,
        framing=s8['mode'], video_content_box_xyxy=[int(v) for v in cl.get('crop_box', [])],
        overlay_removed=bool(cl.get('overlay') and cl['overlay']['n_components'] > 0),
        quality_gate=dict(passed=True, reasons=[]), proposal_method=s8['proposal_method'],
        dtype=cfg['package']['clip_dtype'], shape=[16, cfg['package']['clip_size'], cfg['package']['clip_size'], 3], color='RGB',
    )


def save_video_clips(res: dict, out_root: Path, fold: int, cfg: dict | None = None) -> dict:
    """Write every selected clip + sidecar. Returns report rows."""
    cfg = cfg or load_config()
    s8 = res['s8']
    out_dir = Path(out_root) / f'fold_{fold}' / res['split'] / res['label'] / 'scene'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, written = [], []
    for i, (c, clip) in enumerate(zip(s8['selected'], s8['clips'])):
        arr = clip.astype(np.float32) / 255.0 if cfg['package']['clip_dtype'] == 'float32' else clip
        stem = f"{res['stem']}_c{i:04d}"
        np.save(out_dir / f'{stem}.npy', arr)
        side = build_sidecar(res, c, i, fold, cfg)
        if cfg['package']['sidecar']:
            (out_dir / f'{stem}.json').write_text(json.dumps(side, indent=1))
        written.append(out_dir / f'{stem}.npy')
        rows.append(dict(video=res['stem'], label=res['label'], split=res['split'], kind=c['kind'], **side['stats']))
    vrow = dict(video=res['stem'], label=res['label'], split=res['split'], n_candidates=s8['n_candidates'], n_gate_passed=s8['n_passed'],
                n_after_dedup=s8['n_after_dedup'], n_saved=len(written), reject_counts=s8['reject_counts'],
                proposal_method=s8['proposal_method'], n_segments=len(s8['segments']),
                tiou=s8['gt_eval']['tiou'] if s8['gt_eval'] else None,
                seconds_analysed=len(res['dec']['frames']) / 12.0)
    return dict(files=written, rows=rows, vrow=vrow)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset analytics
# ═══════════════════════════════════════════════════════════════════════════════

REPORT_KEYS = [('motion_energy_mean', 'motion energy (px/step)'), ('flow_entropy_mean', 'flow direction entropy'),
               ('people_count_mean', 'people count'), ('brightness_after', 'brightness (luma Y)'),
               ('activity_mean', 'activity score'), ('fg_ratio_mean', 'foreground ratio')]


def build_report(rows: list, vrows: list, out_dir: Path | None = None, save_samples: bool = True) -> dict:
    """Per-class distributions + gate-rejection counts. Writes report.json / report.md next to the clips."""
    import matplotlib.pyplot as plt
    out = dict(n_clips=len(rows), n_videos=len({x['video'] for x in vrows}), n_video_fold_entries=len(vrows),
               per_class={}, gate_rejections={}, dedup={}, proposals={})
    for lb in LABELS:
        r = [x for x in rows if x['label'] == lb]
        v = [x for x in vrows if x['label'] == lb]
        if not r and not v:
            continue
        pc = dict(clips=len(r), videos=len({x['video'] for x in v}))
        for k, _ in REPORT_KEYS:
            vals = np.array([x[k] for x in r if k in x], float)
            if len(vals):
                pc[k] = dict(mean=float(vals.mean()), p10=float(np.percentile(vals, 10)), p50=float(np.percentile(vals, 50)),
                             p90=float(np.percentile(vals, 90)))
        pc['selection_kinds'] = {k: sum(x['kind'] == k for x in r) for k in {x['kind'] for x in r}}
        out['per_class'][lb] = pc
        rej = {}
        for x in v:
            for k, n in x['reject_counts'].items():
                rej[k] = rej.get(k, 0) + n
        out['gate_rejections'][lb] = rej
        out['dedup'][lb] = dict(before=sum(x['n_gate_passed'] for x in v), after=sum(x['n_after_dedup'] for x in v))
        ti = [x['tiou'] for x in v if x.get('tiou') is not None]
        out['proposals'][lb] = dict(videos_with_gt=len(ti), mean_tiou=float(np.mean(ti)) if ti else None)

    colors = dict(zip(LABELS, viz.SERIES))

    def fig():
        f, axs = plt.subplots(2, 4, figsize=(15, 6.4))
        for ax, (k, nm) in zip(axs.ravel(), REPORT_KEYS):
            data = [[x[k] for x in rows if x['label'] == lb and k in x] for lb in LABELS]
            labs = [lb for lb, d in zip(LABELS, data) if d]
            data = [d for d in data if d]
            if data:
                bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, showfliers=False)
                for patch, lb in zip(bp['boxes'], labs):
                    patch.set_facecolor(colors[lb]); patch.set_alpha(.55)
            ax.set_title(nm, fontsize=9)
        ax = axs.ravel()[6]
        rules = ['frozen', 'black', 'blurry', 'low_motion', 'shot_cut', 'illumination_change']
        bottom = np.zeros(len(rules))
        for lb in LABELS:
            vals = np.array([out['gate_rejections'].get(lb, {}).get(r, 0) for r in rules], float)
            ax.bar(range(len(rules)), vals, bottom=bottom, color=colors[lb], label=lb, edgecolor=viz.SURFACE, linewidth=1.5)
            bottom += vals
        ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules, rotation=35, ha='right', fontsize=7)
        ax.set_title('candidates rejected by the quality gate', fontsize=9); ax.legend(fontsize=7)
        ax = axs.ravel()[7]
        w = 0.38
        xs = np.arange(len(LABELS))
        ax.bar(xs - w / 2, [out['dedup'].get(lb, {}).get('before', 0) for lb in LABELS], w, color=viz.C['blue'], label='before dedup')
        ax.bar(xs + w / 2, [out['dedup'].get(lb, {}).get('after', 0) for lb in LABELS], w, color=viz.C['aqua'], label='after dedup')
        ax.set_xticks(xs); ax.set_xticklabels(LABELS, fontsize=8); ax.set_title('clips per class (gate-passed)', fontsize=9); ax.legend(fontsize=7)
        f.suptitle(f'VPA dataset analytics report — {len(rows)} clips from {out["n_videos"]} videos', fontsize=11)
        f.tight_layout()
        return f

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'report.json').write_text(json.dumps(out, indent=1))
        lines = ['# VPA dataset report', '', f'{len(rows)} clips from {out["n_videos"]} videos ({len(vrows)} video-fold entries).', '',
                 '| class | videos | clips | motion energy (p50) | flow entropy (p50) | people (p50) | brightness (p50) | activity (p50) |', '|---|---|---|---|---|---|---|---|']
        for lb, pc in out['per_class'].items():
            g = lambda k: f"{pc[k]['p50']:.2f}" if k in pc else '-'
            lines.append(f"| {lb} | {pc['videos']} | {pc['clips']} | {g('motion_energy_mean')} | {g('flow_entropy_mean')} | {g('people_count_mean')} | {g('brightness_after')} | {g('activity_mean')} |")
        lines += ['', '## Quality-gate rejections (candidate clips)', '', '| class | ' + ' | '.join(['frozen', 'black', 'blurry', 'low_motion', 'shot_cut', 'illumination_change']) + ' |', '|---|---|---|---|---|---|---|']
        for lb, rj in out['gate_rejections'].items():
            lines.append(f'| {lb} | ' + ' | '.join(str(rj.get(r, 0)) for r in ['frozen', 'black', 'blurry', 'low_motion', 'shot_cut', 'illumination_change']) + ' |')
        (out_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    if save_samples and viz.should_save('ALL'):
        viz.save_io_sample(
            'S9', 'dataset_report', 'ALL', inputs=None, outputs=None, compare=fig(), params=dict(rows=len(rows)),
            metrics=out,
            note=('Per-class distributions of motion energy, flow entropy, people count, brightness, activity and foreground ratio over the packaged clips, plus how many candidates each quality-gate rule rejected and the dedup effect.\n'
                  'Why: shows that the processed dataset differs by class in the expected ways and makes ablations / error analysis possible later.\n'
                  'Notice: abnormal classes should show higher motion energy / activity than Normal; the sample set has only 8 videos so the boxes are small.'))
    return out


def save_package_sample(video: str, res: dict, out_files: list, cfg: dict):
    """S9 sample: one saved clip as a filmstrip + its JSON sidecar rendered as an image + storage-size comparison."""
    if not viz.should_save(video) or not out_files:
        return
    import matplotlib.pyplot as plt
    f = out_files[len(out_files) // 2]
    clip = np.load(f)
    sc_path = Path(f).with_suffix('.json')
    side = json.loads(sc_path.read_text()) if sc_path.exists() else {}
    show = {k: v for k, v in side.items() if k not in ('frame_indices_12fps', 'source_frame_indices', 'source_times_s', 'crop_boxes_xyxy')}
    show['crop_boxes_xyxy'] = side.get('crop_boxes_xyxy', [None])[0:1] and f"[16 boxes] first={side['crop_boxes_xyxy'][0]}"
    txt = json.dumps(show, indent=1)

    fig, ax = plt.subplots(figsize=(7.2, 0.105 * (txt.count('\n') + 3) + 0.3))
    ax.axis('off')
    ax.text(0, 1, txt, va='top', ha='left', family='monospace', fontsize=6.3)
    fig.tight_layout()
    u8 = clip if clip.dtype == np.uint8 else (clip * 255).round().astype(np.uint8)
    strip = viz.filmstrip([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in u8], 8, 112, labels=[f'f{i}' for i in range(16)])
    size_u8, size_f32 = u8.nbytes, u8.size * 4
    viz.save_io_sample(
        'S9', 'clip_package', video, inputs=strip, outputs=None, compare=None, plot=lambda: fig,
        params=dict(dtype=str(clip.dtype), shape=list(clip.shape), file=Path(f).name, sidecar=sc_path.name),
        metrics=dict(bytes_uint8=size_u8, bytes_float32_legacy=size_f32, saving_factor=size_f32 / size_u8, file_bytes=Path(f).stat().st_size,
                     sidecar=side),
        note=('The final clip is stored as uint8 (16,224,224,3) RGB .npy (4x smaller than float32, lossless since pixels began as 8-bit) with a JSON sidecar of shot id, motion/people stats, crop boxes, brightness and enhancement flags.\n'
              'Why: ablations and error analysis need to know exactly how each clip was made; the legacy loader expects float32 [0,1] (needs /255 once uint8 clips are used for training).\n'
              'Notice: left = the saved clip (8 of 16 frames), plot.png = the sidecar; metrics.json has the size comparison.'),
        labels=('saved clip', ''))
