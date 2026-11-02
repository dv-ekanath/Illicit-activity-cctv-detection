"""
viz.py
======
Sample-saving helpers for the VPA pipeline.

Every operation (existing or new) calls `save_io_sample(...)`, which writes

  docs/vpa_samples/<stage>_<op>/<video_stem>/
      input.png / output.png     representative frame(s)
      compare.png                input vs output side by side, labelled, params in title
      input.gif / output.gif     short clips (<= 3 s, small) for temporal operations
      plot.png                   signal plots where relevant
      params.json / metrics.json
      README.md                  what / why / what to notice

and registers the sample so `build_index()` can regenerate docs/vpa_samples/INDEX.md.
Everything is a no-op unless `configure(enabled=True)` was called (--save-samples).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
if 'ipykernel' not in sys.modules:          # keep the inline backend inside notebooks, use Agg for scripts
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from .common import DOCS_DIR, safe_name

# ── Palette (validated categorical order from the dataviz reference) ───────────
SURFACE = '#fcfcfb'
INK     = '#0b0b0b'
INK2    = '#52514e'
GRID    = '#e3e2dd'
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100',
         magenta='#e87ba4', green='#008300', violet='#4a3aa7', red='#e34948')
SERIES = [C['blue'], C['orange'], C['aqua'], C['yellow'], C['magenta'], C['violet']]

_STATE = dict(enabled=False, root=DOCS_DIR, stems=None, gif_max_frames=18,
              gif_max_width=256, gif_fps=6)


def apply_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.frameon': False, 'legend.fontsize': 8, 'lines.linewidth': 1.6,
        'axes.prop_cycle': matplotlib.cycler(color=SERIES), 'font.size': 9,
    })


apply_style()


# ── Configuration ──────────────────────────────────────────────────────────────
def configure(enabled: bool = True, root: Path | None = None, stems=None,
              gif_max_frames: int | None = None):
    """Turn sample saving on/off. `stems` restricts saving to those video stems (None = all)."""
    _STATE['enabled'] = enabled
    if root is not None:
        _STATE['root'] = Path(root)
    _STATE['stems'] = set(stems) if stems else None
    if gif_max_frames:
        _STATE['gif_max_frames'] = gif_max_frames


def should_save(video: str | None = None) -> bool:
    if not _STATE['enabled']:
        return False
    return video in (None, 'ALL') or _STATE['stems'] is None or video in _STATE['stems']


# ── Image helpers ──────────────────────────────────────────────────────────────
def to_bgr(img) -> np.ndarray:
    """Any HxW / HxWx1 / HxWx3 array (uint8 or float) -> HxWx3 BGR uint8."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        if 0.0 <= lo and hi <= 1.0:
            a = a * 255.0
        else:
            a = (a - lo) / (hi - lo + 1e-8) * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 1:
        a = cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(a)


def put_label(img: np.ndarray, text: str, scale: float = 0.45) -> np.ndarray:
    out = to_bgr(img).copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(out, (0, 0), (tw + 8, th + 8), (0, 0, 0), -1)
    cv2.putText(out, text, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mosaic(frames, cols: int | None = None, tile_h: int = 180, labels=None) -> np.ndarray:
    """Tile frames (list of images) into a grid, each resized to a common height."""
    if isinstance(frames, np.ndarray) and frames.ndim in (2, 3) and \
            (frames.ndim == 2 or frames.shape[-1] in (1, 3)):
        return to_bgr(frames)
    tiles = []
    for i, f in enumerate(frames):
        f = to_bgr(f)
        h, w = f.shape[:2]
        f = cv2.resize(f, (max(1, int(w * tile_h / h)), tile_h), interpolation=cv2.INTER_AREA)
        if labels is not None:
            f = put_label(f, str(labels[i]))
        tiles.append(f)
    if not tiles:
        return np.zeros((tile_h, tile_h, 3), np.uint8)
    cols = cols or len(tiles)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        if len(row) < cols:
            row = row + [np.zeros_like(row[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    return np.vstack(rows)


def filmstrip(frames, n: int = 8, tile_h: int = 112, labels=None) -> np.ndarray:
    """n evenly spaced frames from a sequence, in one row."""
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    lab = [labels[i] if labels is not None else str(i) for i in idx]
    return mosaic([frames[i] for i in idx], cols=len(idx), tile_h=tile_h, labels=lab)


def fig_to_bgr(fig, dpi: int = 90) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(np.ascontiguousarray(buf), cv2.COLOR_RGB2BGR)


def _save_png(path: Path, img_bgr: np.ndarray):
    """256-colour palette PNG: ~3x smaller than truecolour, visually fine for documentation samples."""
    pil = Image.fromarray(cv2.cvtColor(to_bgr(img_bgr), cv2.COLOR_BGR2RGB))
    pil = pil.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
    pil.save(path, optimize=True)


def save_gif(path: Path, frames, fps: int | None = None, max_frames: int | None = None,
             max_width: int | None = None):
    """Small GIF from a list of BGR frames (<= ~3 s, downscaled, 64-colour palette)."""
    max_frames = max_frames or _STATE['gif_max_frames']
    max_width  = max_width or _STATE['gif_max_width']
    fps        = fps or _STATE['gif_fps']
    frames = list(frames)
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    pil = []
    for f in frames:
        f = to_bgr(f)
        h, w = f.shape[:2]
        if w > max_width:
            f = cv2.resize(f, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(
            colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pil.append(im)
    if pil:
        pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000 / fps),
                    loop=0, optimize=True, disposal=2)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f'<ndarray {o.shape} {o.dtype}>'
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _dump_json(path: Path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ── Registry / index ───────────────────────────────────────────────────────────
def _registry_path() -> Path:
    return _STATE['root'] / '_registry.json'


def _register(stage, op, video, rel_dir, what):
    p = _registry_path()
    reg = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    key = f'{stage}_{op}'
    ent = reg.setdefault(key, dict(stage=stage, op=op, what=what, videos={}))
    ent['what'] = what or ent['what']
    ent['videos'][video] = rel_dir
    _dump_json(p, reg)


def build_index(root: Path | None = None) -> Path:
    """Regenerate INDEX.md (stage | operation | what it shows | links) from the registry."""
    root = Path(root or _STATE['root'])
    p = root / '_registry.json'
    reg = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

    def sort_key(k):
        st = reg[k]['stage']
        return (0 if st.startswith('S') else 1, st, k)

    lines = ['# VPA sample index', '',
             'Every row is one pre-training operation; each link opens that operation\'s '
             '`compare.png` (input vs output) for one sample video. The folder also holds '
             '`input/output.png|gif`, `plot.png`, `params.json`, `metrics.json`, `README.md`.', '',
             '| Stage | Operation | What it shows | Samples (per video) |',
             '|---|---|---|---|']
    for k in sorted(reg, key=sort_key):
        e = reg[k]
        links = ', '.join(f'[{v}]({rel}/compare.png)' for v, rel in sorted(e['videos'].items()))
        lines.append(f"| {e['stage']} | {e['op']} | {e['what']} | {links} |")
    out = root / 'INDEX.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


# ── The main helper ────────────────────────────────────────────────────────────
def sample_dir(stage: str, op: str, video: str) -> Path:
    d = _STATE['root'] / f'{safe_name(stage)}_{safe_name(op)}' / safe_name(video)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_io_sample(stage: str, op: str, video: str, inputs=None, outputs=None,
                   params: dict | None = None, metrics: dict | None = None,
                   note: str = '', *, plot=None, compare=None,
                   input_clip=None, output_clip=None,
                   labels=('input', 'output'), tile_h: int = 200, cols=None,
                   input_labels=None, output_labels=None):
    """
    Save one operation's sample (see module docstring). Returns the sample dir or None.

    inputs / outputs : image, or list of images (-> mosaic), BGR uint8 / gray / float
    plot             : matplotlib Figure (-> plot.png) or callable returning a Figure
    compare          : optional pre-built Figure or image replacing the auto side-by-side
    input_clip / output_clip : lists of frames -> input.gif / output.gif
    note             : 2-3 lines "what it does / why / what to notice" (README.md)
    """
    if not should_save(video):
        return None
    params = params or {}
    metrics = metrics or {}
    d = sample_dir(stage, op, video)

    in_img = out_img = None
    if inputs is not None:
        in_img = mosaic(inputs, cols=cols, tile_h=tile_h, labels=input_labels) \
            if isinstance(inputs, (list, tuple)) else to_bgr(inputs)
        _save_png(d / 'input.png', in_img)
    if outputs is not None:
        out_img = mosaic(outputs, cols=cols, tile_h=tile_h, labels=output_labels) \
            if isinstance(outputs, (list, tuple)) else to_bgr(outputs)
        _save_png(d / 'output.png', out_img)

    # compare.png
    if compare is not None:
        cmp_bgr = fig_to_bgr(compare) if hasattr(compare, 'savefig') else to_bgr(compare)
        _save_png(d / 'compare.png', cmp_bgr)
    elif in_img is not None and out_img is not None:
        ptxt = ', '.join(f'{k}={_short(v)}' for k, v in list(params.items())[:6])
        w = max(8, 4.2 * (in_img.shape[1] / in_img.shape[0] + out_img.shape[1] / out_img.shape[0]) / 1.33)
        fig, axs = plt.subplots(1, 2, figsize=(min(w, 14), 4.4),
                                gridspec_kw=dict(width_ratios=[in_img.shape[1] / in_img.shape[0],
                                                               out_img.shape[1] / out_img.shape[0]]))
        for ax, im, lab in zip(axs, (in_img, out_img), labels):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(lab, fontsize=10)
            ax.axis('off')
        fig.suptitle(f'{stage} · {op}   [{video}]\n{ptxt}', fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _save_png(d / 'compare.png', fig_to_bgr(fig))
    elif in_img is not None or out_img is not None:
        _save_png(d / 'compare.png', in_img if in_img is not None else out_img)

    if input_clip is not None and len(input_clip):
        save_gif(d / 'input.gif', input_clip)
    if output_clip is not None and len(output_clip):
        save_gif(d / 'output.gif', output_clip)
    if plot is not None:
        fig = plot() if callable(plot) else plot
        _save_png(d / 'plot.png', fig_to_bgr(fig))

    _dump_json(d / 'params.json', params)
    _dump_json(d / 'metrics.json', metrics)
    (d / 'README.md').write_text(
        f'# {stage} · {op}  ({video})\n\n{note.strip()}\n\n'
        f'Files: ' + ', '.join(sorted(p.name for p in d.iterdir() if p.name != 'README.md')) + '\n',
        encoding='utf-8')

    rel = d.relative_to(_STATE['root']).as_posix()
    what = note.strip().split('\n')[0] if note else op
    _register(stage, op, video, rel, what)
    return d


def _short(v):
    if isinstance(v, float):
        return f'{v:.3g}'
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 4:
        return f'[{len(v)} items]'
    return str(v)


def save_table_image(rows: list, header: list, title: str = '', col_w=None) -> np.ndarray:
    """Render a small table (list of row lists) to a BGR image."""
    n = len(rows) + 1
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(header)), 0.34 * n + 0.8))
    ax.axis('off')
    tb = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=header,
                  loc='center', cellLoc='left', colWidths=col_w)
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.3)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor('#eceae4')
            cell.set_text_props(weight='bold')
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig_to_bgr(fig)
