"""
actors.py  (S7 — actor analysis)
================================
  Detector        existing YOLOv11s (fine-tuned best.pt, or the COCO yolo11s.pt fallback) — NO retraining
  track_video     ByteTrack (ultralytics.trackers.byte_tracker.BYTETracker) over the per-frame detections
  interaction_signals   people count, minimum normalised distance, max box overlap, per-track speed, interaction score
  actor_crops     union-of-interacting-people box + margin, zero-phase smoothed, square-expanded, letterboxed -> 224
  select_detector       compares the two detectors on sample frames and records the choice
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np

from .common import YOLO_FT_W, YOLO_COCO_W, CACHE_DIR, load_config, letterbox
from . import viz


# ═══════════════════════════════════════════════════════════════════════════════
# Detection
# ═══════════════════════════════════════════════════════════════════════════════

class Detector:
    """Thin batch wrapper around an ultralytics YOLO model returning PERSON boxes only."""

    def __init__(self, weights=None, device: str = 'cuda', conf: float | None = None, imgsz: int | None = None,
                 kind: str = 'finetuned'):
        from ultralytics import YOLO
        cfg = load_config()['actors']
        self.kind = kind
        self.weights = str(weights or (YOLO_FT_W if kind == 'finetuned' else YOLO_COCO_W))
        self.model = YOLO(self.weights)
        self.device = device
        self.conf = conf if conf is not None else cfg['conf']
        self.imgsz = imgsz or cfg['imgsz']
        names = {int(k): v for k, v in self.model.names.items()}
        wanted = set(cfg['person_class_names'])
        self.person_ids = [k for k, v in names.items() if v in wanted] or [0]

    def detect(self, frames_bgr, batch: int = 16, conf: float | None = None):
        """-> list (per frame) of float arrays (N, 6): x1, y1, x2, y2, conf, cls."""
        out = []
        for i in range(0, len(frames_bgr), batch):
            res = self.model.predict(list(frames_bgr[i:i + batch]), conf=conf if conf is not None else self.conf,
                                     imgsz=self.imgsz, classes=self.person_ids, device=self.device, verbose=False)
            for r in res:
                if r.boxes is None or len(r.boxes) == 0:
                    out.append(np.zeros((0, 6), np.float32))
                else:
                    out.append(np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                                          r.boxes.cls.cpu().numpy()[:, None]]).astype(np.float32))
        return out

    def count_persons(self, frames_bgr, batch: int = 16) -> np.ndarray:
        return np.array([len(d) for d in self.detect(frames_bgr, batch)])


def get_detector(models: dict, device: str = 'cuda', kind: str | None = None) -> Detector:
    """Cached detector; `kind` defaults to the persisted choice (select_detector) or the config."""
    cfg = load_config()['actors']
    kind = kind or models.get('detector_kind') or _persisted_choice() or \
        ('finetuned' if cfg['detector'] == 'auto' else cfg['detector'])
    key = f'detector_{kind}'
    if key not in models:
        models[key] = Detector(device=device, kind=kind)
    models['detector'] = models[key]
    return models[key]


def _persisted_choice():
    p = CACHE_DIR / 'detector_choice.json'
    return json.loads(p.read_text())['kind'] if p.exists() else None


def _confirm_frac(a_list, b_list, thr: float = 0.5) -> float:
    """Share of boxes in `a_list` (per-frame arrays) that overlap (IoU >= thr) a box of the other detector."""
    tot = hit = 0
    for a, b in zip(a_list, b_list):
        for x in a:
            tot += 1
            hit += any(_iou(x[:4], y[:4]) >= thr for y in b)
    return hit / tot if tot else 0.0


def _implausible_frac(dets_list, frame_area: float, max_area: float = 0.35) -> float:
    tot = bad = 0
    for d in dets_list:
        for x in d:
            tot += 1
            bad += (x[2] - x[0]) * (x[3] - x[1]) > max_area * frame_area
    return bad / tot if tot else 0.0


def select_detector(video_frames: dict, models: dict, device: str = 'cuda', save: bool = True) -> str:
    """
    video_frames: {stem: [frames]} (enhanced where dark). Runs the fine-tuned and the COCO detector on 8 frames per
    video. There is no person ground truth for UCF-Crime, so box quality is judged by
      * cross-confirmation: share of a detector's boxes that the OTHER detector also finds (IoU >= 0.5), and
      * plausibility: share of boxes covering > 35 % of the frame (a standing person never does).
    COCO is chosen when the fine-tuned model's boxes are confirmed clearly less often or are often implausible;
    otherwise the fine-tuned model (the project's own detector) is kept.
    """
    cfg = load_config()['actors']
    dets = {k: Detector(device=device, kind=k) for k in ('finetuned', 'coco')}
    rows, imgs = [], {'finetuned': [], 'coco': []}
    allres = {'finetuned': [], 'coco': []}
    areas = []
    nfr = 0
    for stem, frames in video_frames.items():
        idx = np.linspace(0, len(frames) - 1, 8).round().astype(int)
        sel = [frames[i] for i in idx]
        areas.append(float(sel[0].shape[0] * sel[0].shape[1]))
        res = {k: d.detect(sel, conf=cfg['conf']) for k, d in dets.items()}
        for k in res:
            allres[k] += res[k]
        if save and viz.should_save(stem):
            for k in res:
                cnt = [len(r) for r in res[k]]
                j = int(np.argmax(cnt)) if max(cnt) else 0
                imgs[k].append(draw_boxes(sel[j], res[k][j], (0, 255, 0) if k == 'finetuned' else (0, 128, 255),
                                          label=f'{k}: {cnt[j]}'))
        nfr += len(sel)
        rows.append([stem[:22], f"{np.mean([len(r) for r in res['finetuned']]):.2f}", f"{np.mean([len(r) for r in res['coco']]):.2f}",
                     f"{_confirm_frac(res['finetuned'], res['coco']):.2f}", f"{_confirm_frac(res['coco'], res['finetuned']):.2f}"])
    conf_ft = _confirm_frac(allres['finetuned'], allres['coco'])
    conf_coco = _confirm_frac(allres['coco'], allres['finetuned'])
    imp_ft = _implausible_frac(allres['finetuned'], float(np.mean(areas)))
    imp_coco = _implausible_frac(allres['coco'], float(np.mean(areas)))
    n_ft = sum(len(r) for r in allres['finetuned'])
    n_coco = sum(len(r) for r in allres['coco'])
    kind = 'coco' if (conf_ft + 0.10 < conf_coco or imp_ft > 0.15) else 'finetuned'
    models['detector_kind'] = kind
    summary = dict(kind=kind, finetuned_persons=n_ft, coco_persons=n_coco, finetuned_boxes_confirmed_by_coco=conf_ft,
                   coco_boxes_confirmed_by_finetuned=conf_coco, finetuned_implausible_frac=imp_ft,
                   coco_implausible_frac=imp_coco, frames=nfr)
    if save:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / 'detector_choice.json').write_text(json.dumps(summary))
        rows.append(['ALL', f"{n_ft / nfr:.2f}", f"{n_coco / nfr:.2f}", f"{conf_ft:.2f}", f"{conf_coco:.2f}"])
        if viz.should_save('ALL') and imgs['finetuned']:
            tab = viz.save_table_image(rows, ['video', 'FT persons/fr', 'COCO persons/fr', 'FT boxes confirmed', 'COCO boxes confirmed'],
                                       f'Detector check (conf {cfg["conf"]}): chosen = {kind}; implausibly large boxes FT {imp_ft:.0%} / COCO {imp_coco:.0%}',
                                       col_w=[0.28, 0.16, 0.18, 0.19, 0.19])
            viz.save_io_sample(
                'S7', 'detector_check', 'ALL', inputs=imgs['finetuned'], outputs=imgs['coco'], input_labels=None,
                tile_h=190, cols=4, compare=None,
                params=dict(conf=cfg['conf'], imgsz=cfg['imgsz'],
                            rule='COCO if fine-tuned boxes are confirmed >10 pts less often by COCO than vice versa, or >15 % implausible'),
                metrics=summary | dict(per_video=rows),
                note=(f'Visual + numeric check of the existing fine-tuned YOLOv11s against COCO yolo11s.pt on enhanced UCF-Crime frames (both without any training); chosen detector: {kind}.\n'
                      'Why: the fine-tuned model was trained on night-time LLVIP/ExDARK images, UCF-Crime CCTV is another domain; there is no person ground truth, so cross-confirmation and box plausibility are used.\n'
                      'Notice: each tile is the sampled frame with most people for that detector (green = fine-tuned, orange = COCO).'),
                plot=lambda: _table_fig(tab), labels=('fine-tuned best.pt', 'COCO yolo11s.pt'))
    return kind


def _table_fig(img):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.35 * 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis('off')
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Tracking (ByteTrack)
# ═══════════════════════════════════════════════════════════════════════════════

class _Dets:
    """Minimal results object accepted by BYTETracker.update (xywh / xyxy / conf / cls + boolean indexing)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy, self.conf, self.cls = xyxy.astype(np.float32), conf.astype(np.float32), cls.astype(np.float32)

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return _Dets(self.xyxy[i], self.conf[i], self.cls[i])


def track_video(det_list, cfg: dict | None = None, fps: float = 12.0):
    """ByteTrack over per-frame detections (N,6). Returns per-frame arrays (M, 7): x1,y1,x2,y2,id,score,cls."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    cfg = cfg or load_config()
    tr = BYTETracker(SimpleNamespace(**cfg['actors']['tracker']), frame_rate=int(fps))
    BYTETracker.reset_id() if hasattr(BYTETracker, 'reset_id') else None
    out = []
    for d in det_list:
        dets = _Dets(d[:, :4], d[:, 4], d[:, 5]) if len(d) else _Dets(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
        r = tr.update(dets)
        out.append(np.asarray(r)[:, :7].astype(np.float32) if len(r) else np.zeros((0, 7), np.float32))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction signals
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def interaction_signals(tracks, cfg: dict | None = None) -> dict:
    """
    Per frame: people count, min centre distance between two people in units of mean person height
    (scale-invariant, 9 = nobody to compare), max IoU, mean / max speed (person-heights per 12-FPS step),
    and `interaction` in [0,1] = proximity * (0.4 + 0.6 * speed) for >= 2 people, 0.5 * speed for one person.
    """
    cfg = cfg or load_config()
    dh = cfg['actors']['interact_dist_h']
    T = len(tracks)
    sig = {k: np.zeros(T) for k in ('n_people', 'min_dist', 'max_iou', 'speed_mean', 'speed_max', 'interaction')}
    sig['min_dist'][:] = 9.0
    prev = {}
    for t, tr in enumerate(tracks):
        n = len(tr)
        sig['n_people'][t] = n
        cur = {}
        speeds = []
        for r in tr:
            cx, cy, h = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2, max(r[3] - r[1], 1.0)
            cur[int(r[4])] = (cx, cy, h)
            if int(r[4]) in prev:
                px, py, ph = prev[int(r[4])]
                speeds.append(np.hypot(cx - px, cy - py) / ((h + ph) / 2))
        prev = cur
        if speeds:
            sig['speed_mean'][t], sig['speed_max'][t] = float(np.mean(speeds)), float(np.max(speeds))
        if n >= 2:
            md, mi = 9.0, 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = tr[i], tr[j]
                    hm = ((a[3] - a[1]) + (b[3] - b[1])) / 2
                    d = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) / max(hm, 1.0)
                    md = min(md, d)
                    mi = max(mi, _iou(a, b))
            sig['min_dist'][t], sig['max_iou'][t] = md, mi
            prox = float(np.clip(1.0 - md / dh, 0, 1))
            sp = float(np.clip(sig['speed_max'][t] / 0.25, 0, 1))
            sig['interaction'][t] = prox * (0.4 + 0.6 * sp)
        elif n == 1:
            sig['interaction'][t] = 0.5 * float(np.clip(sig['speed_max'][t] / 0.25, 0, 1))
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# Actor-centred crop
# ═══════════════════════════════════════════════════════════════════════════════

def _cluster_box(tr: np.ndarray, dh: float):
    """Union box of the people that interact (centre distance < dh heights); all people if nobody does."""
    if len(tr) == 0:
        return None
    if len(tr) == 1:
        return tr[0, :4].copy()
    n = len(tr)
    cen = np.stack([(tr[:, 0] + tr[:, 2]) / 2, (tr[:, 1] + tr[:, 3]) / 2], 1)
    hh = tr[:, 3] - tr[:, 1]
    adj = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(cen[i] - cen[j])) < dh * (hh[i] + hh[j]) / 2:
                adj[i, j] = adj[j, i] = True
    seen, comps = np.zeros(n, bool), []
    for i in range(n):
        if not seen[i]:
            stack, comp = [i], []
            seen[i] = True
            while stack:
                k = stack.pop()
                comp.append(k)
                for m in np.where(adj[k] & ~seen)[0]:
                    seen[m] = True
                    stack.append(m)
            comps.append(comp)
    best = max(comps, key=lambda c: (len(c), sum(hh[k] for k in c)))
    sel = tr[best] if len(best) >= 2 else tr
    return np.array([sel[:, 0].min(), sel[:, 1].min(), sel[:, 2].max(), sel[:, 3].max()])


def _zero_phase_ema(x: np.ndarray, a: float) -> np.ndarray:
    f, b = x.copy(), x.copy()
    for t in range(1, len(x)):
        f[t] = a * f[t - 1] + (1 - a) * x[t]
    for t in range(len(x) - 2, -1, -1):
        b[t] = a * b[t + 1] + (1 - a) * x[t]
    return (f + b) / 2


def actor_crop_boxes(tracks, frame_hw, cfg: dict | None = None) -> dict:
    """
    Per-frame square-ish crop window (x0,y0,x1,y1) around the interacting people (+ margin), smoothed over
    time (zero-phase EMA) so it does not jitter. Frames with nobody use the whole frame (fallback), and the
    smoothing blends into / out of it.
    """
    cfg = cfg or load_config()
    c = cfg['actors']['crop']
    H, W = frame_hw
    T = len(tracks)
    cx, cy, s = np.zeros(T), np.zeros(T), np.zeros(T)
    found = np.zeros(T, bool)
    for t, tr in enumerate(tracks):
        b = _cluster_box(tr, cfg['actors']['interact_dist_h'])
        if b is None:
            cx[t], cy[t], s[t] = W / 2, H / 2, max(H, W)
            continue
        found[t] = True
        w, h = b[2] - b[0], b[3] - b[1]
        side = max(w, h) * (1 + 2 * c['margin'])
        s[t] = float(np.clip(side, c['min_side'], max(H, W)))
        cx[t], cy[t] = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    raw = np.stack([cx, cy, s], 1)
    sm = np.stack([_zero_phase_ema(raw[:, k], c['ema']) for k in range(3)], 1)
    boxes = np.zeros((T, 4), int)
    for t in range(T):
        half = sm[t, 2] / 2
        x0, y0, x1, y1 = sm[t, 0] - half, sm[t, 1] - half, sm[t, 0] + half, sm[t, 1] + half
        # slide the window inside the frame where possible; clip the remainder (letterbox pads it)
        if x0 < 0: x1 -= x0; x0 = 0
        if y0 < 0: y1 -= y0; y0 = 0
        if x1 > W: x0 -= x1 - W; x1 = W
        if y1 > H: y0 -= y1 - H; y1 = H
        boxes[t] = [int(max(0, round(x0))), int(max(0, round(y0))), int(min(W, round(x1))), int(min(H, round(y1)))]
    jit = lambda a: float(np.abs(np.diff(a[:, :2], axis=0)).mean()) if len(a) > 1 else 0.0
    return dict(boxes=boxes, raw=raw, smooth=sm, found=found, jitter_raw=jit(raw), jitter_smooth=jit(sm),
                person_found_frac=float(found.mean()))


def render_crop(frame_bgr: np.ndarray, box, size: int = 224) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame_bgr
    out, _, _ = letterbox(crop, size, 0, cv2.INTER_AREA)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing + sample saving
# ═══════════════════════════════════════════════════════════════════════════════

def _id_color(i: int):
    rng = np.random.RandomState(int(i) * 7 + 3)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def draw_boxes(frame, dets, color=(0, 255, 0), label: str | None = None):
    im = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d[:4]]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
        cv2.putText(im, f'{d[4]:.2f}', (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return viz.put_label(im, label) if label else im


def draw_tracks(frame, tracks_hist, t, tail: int = 24):
    """Boxes + IDs at frame t and the centroid trajectories of the last `tail` frames."""
    im = frame.copy()
    paths = {}
    for u in range(max(0, t - tail), t + 1):
        for r in tracks_hist[u]:
            paths.setdefault(int(r[4]), []).append((int((r[0] + r[2]) / 2), int(r[3])))
    for i, pts in paths.items():
        if len(pts) > 1:
            cv2.polylines(im, [np.array(pts, np.int32)], False, _id_color(i), 1, cv2.LINE_AA)
    for r in tracks_hist[t]:
        col = _id_color(int(r[4]))
        cv2.rectangle(im, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), col, 1)
        cv2.putText(im, f'ID {int(r[4])}', (int(r[0]), max(int(r[1]) - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return im


def pick_actor_showcase(sig: dict, prefer=None, smooth: int = 9) -> int:
    e = np.convolve(sig['interaction'] + 0.05 * sig['n_people'], np.ones(smooth) / smooth, mode='same')
    e[:12] = 0
    if prefer is not None:
        a, b = max(0, int(prefer[0])), min(len(e), int(prefer[1]))
        if b > a and e[a:b].max() > 0:
            k = np.zeros_like(e); k[a:b] = e[a:b]; e = k
    return int(np.argmax(e)) if e.max() > 0 else len(e) // 2


def save_actor_samples(video: str, frames, dets, tracks, sig: dict, crops: dict, cfg: dict, gt_seconds=None,
                       detector_kind: str = 'finetuned', prefer=None, other_dets=None):
    if not viz.should_save(video):
        return
    import matplotlib.pyplot as plt
    T = len(frames)
    t0 = pick_actor_showcase(sig, prefer)
    a, b = max(0, t0 - 9), min(T, t0 + 9)
    f = frames[t0]
    all_ids = {int(r[4]) for tr in tracks for r in tr}
    lengths = {}
    for tr in tracks:
        for r in tr:
            lengths[int(r[4])] = lengths.get(int(r[4]), 0) + 1

    # detection
    viz.save_io_sample(
        'S7', 'person_detection', video, inputs=f, outputs=draw_boxes(f, dets[t0], label=f'{len(dets[t0])} person(s)'),
        params=dict(detector=detector_kind, conf=cfg['actors']['conf'], imgsz=cfg['actors']['imgsz'], retrained=False),
        metrics=dict(frame=t0, persons=int(len(dets[t0])), mean_persons_per_frame=float(np.mean([len(d) for d in dets])),
                     frames_with_person=float(np.mean([len(d) > 0 for d in dets]))),
        note=(f'Existing YOLOv11s ({detector_kind}) finds people on the enhanced frame (no retraining).\n'
              'Why: boxes feed the tracker, the interaction signals and the actor-centred crop.\n'
              'Notice: boxes are on the enhanced/cleaned 12-FPS frame of the highest-activity moment.'),
        input_clip=frames[a:b], output_clip=[draw_boxes(frames[u], dets[u]) for u in range(a, b)],
        labels=('input frame', 'YOLO person boxes'))

    # tracking
    viz.save_io_sample(
        'S7', 'bytetrack_ids', video, inputs=draw_boxes(f, dets[t0], label='detections only'),
        outputs=draw_tracks(f, tracks, t0), params=dict(tracker='ByteTrack (ultralytics BYTETracker)', **cfg['actors']['tracker']),
        metrics=dict(n_tracks=len(all_ids), mean_track_len_frames=float(np.mean(list(lengths.values()))) if lengths else 0,
                     max_track_len_frames=int(max(lengths.values())) if lengths else 0, frame=t0,
                     tracks_in_frame=int(len(tracks[t0]))),
        note=('ByteTrack links the YOLO detections across frames, giving each person a persistent ID and a trajectory.\n'
              'Why: speed and distance between people (interaction signals) need identities over time; this also replaces the never-committed tracker.py.\n'
              'Notice: IDs stay constant while people move; coloured tails are the last 2 s of foot positions.'),
        input_clip=[draw_boxes(frames[u], dets[u]) for u in range(a, b)], output_clip=[draw_tracks(frames[u], tracks, u) for u in range(a, b)],
        labels=('detections (no identity)', 'tracks: IDs + trajectories'))

    # interaction signals plot
    def sig_fig():
        fig, axs = plt.subplots(5, 1, figsize=(9, 7), sharex=True)
        t = np.arange(T) / 12
        spec = [('n_people', 'people count'), ('min_dist', 'min distance (person heights)'), ('max_iou', 'max box overlap (IoU)'),
                ('speed_max', 'max speed (heights/step)'), ('interaction', 'interaction score')]
        for ax, (k, nm), col in zip(axs, spec, viz.SERIES * 2):
            y = np.minimum(sig[k], 6) if k == 'min_dist' else sig[k]
            ax.plot(t, y, color=col, lw=1.1); ax.set_ylabel(nm, fontsize=7)
            if gt_seconds:
                for s_, e_ in gt_seconds:
                    ax.axvspan(s_, e_, color=viz.C['red'], alpha=.12)
        axs[-1].set_xlabel('time (s)')
        fig.suptitle(f'{video}: interaction signals' + (' (red band = ground-truth event)' if gt_seconds else ''), fontsize=9)
        fig.tight_layout()
        return fig
    lo, hi = int(np.argmin(sig['interaction'][12:]) + 12), t0
    viz.save_io_sample(
        'S7', 'interaction_signals', video, inputs=[draw_tracks(frames[lo], tracks, lo), draw_tracks(frames[hi], tracks, hi)],
        input_labels=[f'low interaction (t={lo / 12:.1f}s)', f'high interaction (t={hi / 12:.1f}s)'], outputs=None,
        plot=sig_fig, compare=sig_fig(),
        params=dict(interact_dist_h=cfg['actors']['interact_dist_h'], speed_unit='person heights per 12-FPS step'),
        metrics=dict({k: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)), max=float(np.max(v))) for k, v in sig.items()}),
        note=('Per frame: people count, minimum distance between people (in person heights), box overlap, speed per track, and a combined interaction score.\n'
              'Why: fights and robberies involve people getting close and moving fast; this signal joins motion energy in the activity score (S8).\n'
              'Notice: the interaction score rises where people are close AND moving (red band = ground-truth event if annotated).'))

    # actor crop
    boxes = crops['boxes']
    marked = f.copy()
    x0, y0, x1, y1 = boxes[t0]
    cv2.rectangle(marked, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1)
    size = cfg['package']['clip_size']
    letter, _, _ = letterbox(f, size, 0, cv2.INTER_AREA)
    crop224 = render_crop(f, boxes[t0], size)
    viz.save_io_sample(
        'S7', 'actor_centred_crop', video, inputs=[marked, letter], input_labels=['frame (yellow = crop window)', 'plain letterbox 224'],
        outputs=crop224,
        params=dict(margin=cfg['actors']['crop']['margin'], min_side=cfg['actors']['crop']['min_side'], ema=cfg['actors']['crop']['ema'],
                    resize='letterbox + INTER_AREA', fallback='full frame when nobody is detected'),
        metrics=dict(frame=t0, crop_box=[int(v) for v in boxes[t0]], person_found_frac=crops['person_found_frac'],
                     center_jitter_px_raw=crops['jitter_raw'], center_jitter_px_smoothed=crops['jitter_smooth'],
                     crop_side_px=int(max(x1 - x0, y1 - y0))),
        note=('Crops the union of the interacting people plus a margin, smooths the window over time, letterboxes it and resizes to 224x224 with INTER_AREA.\n'
              'Why: in 320x240 CCTV people are tiny; shrinking the whole frame to 224 throws away the detail the action model needs.\n'
              'Notice: right = the actual model input; jitter_px (metrics) drops after smoothing; with nobody detected the full frame is used.'),
        input_clip=[cv2.rectangle(frames[u].copy(), tuple(boxes[u][:2]), (int(boxes[u][2]) - 1, int(boxes[u][3]) - 1), (0, 255, 255), 1) for u in range(a, b)],
        output_clip=[render_crop(frames[u], boxes[u], size) for u in range(a, b)],
        labels=('full frame / plain letterbox', 'actor-centred 224x224 crop'))
