"""
evaluate_tta.py 
=====================================================================

During training evaluation, each .npy clip is run through the model once.
TTA runs each clip through the model 4 times with different orientations:
  View 1: original
  View 2: horizontal flip          (mirrors left/right)
  View 3: temporal reverse         (action plays backwards)
  View 4: h-flip + temporal reverse

The 4 softmax outputs are averaged before argmax.

OUTPUT
──────
  Per-fold results (acc, macro F1, per-class F1, confusion matrix)
  4-fold mean ± std
  Saved to: models/videomae/tta_results.json
"""

import sys
import json
import argparse
import platform
import statistics
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.amp import autocast
from transformers import VideoMAEForVideoClassification

ROOT        = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / 'models' / 'videomae'
CLIPS_DIR   = ROOT / 'data'   / 'processed' / 'clips'

sys.path.append(str(Path(__file__).resolve().parents[1]))

CLASSES       = ['Fighting', 'Robbery', 'Vandalism', 'Normal']
NUM_CLASSES   = len(CLASSES)
VALID_FOLDS   = [1, 2, 3, 4]
CLIP_LEN      = 16
CLIP_SIZE     = 224
VIDEOMAE_MEAN = [0.485, 0.456, 0.406]
VIDEOMAE_STD  = [0.229, 0.224, 0.225]
DEFAULT_MODEL = 'MCG-NJU/videomae-base-finetuned-kinetics'
IS_WINDOWS    = platform.system() == 'Windows'


# ═══════════════════════════════════════════════════════════════════════════════
# Model  (identical to training definition)
# ═══════════════════════════════════════════════════════════════════════════════

class VideoMAEFineTuned(nn.Module):
    def __init__(self, base: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.videomae   = base.videomae
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = base.classifier

    def forward(self, pixel_values):
        out    = self.videomae(pixel_values)
        pooled = out.last_hidden_state.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return type('Out', (), {'logits': logits})()


def load_fold_model(fold: int, device: torch.device) -> nn.Module:
    """
    Load videomae_best.pth for the given fold.
    Dropout is set to 0.0 — no stochasticity at inference time.
    """
    ckpt_path = WEIGHTS_DIR / f'fold_{fold}' / 'videomae_best.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train fold {fold} first with train_videomae.py --fold {fold}"
        )

    # Build base
    local_pretrained = ROOT / 'models' / 'videomae' / 'pretrained'
    if local_pretrained.exists():
        base = VideoMAEForVideoClassification.from_pretrained(
            str(local_pretrained),
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
    else:
        base = VideoMAEForVideoClassification.from_pretrained(
            DEFAULT_MODEL,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )

    model = VideoMAEFineTuned(base, dropout=0.0).to(device)

    # Load saved weights — handle AveragedModel (SWA) prefix
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['state_dict']
    clean = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)

    if missing:
        print(f"    [Fold {fold}] Missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"    [Fold {fold}] Unexpected keys (first 3): {unexpected[:3]}")

    model.eval()
    m   = ckpt.get('metrics', {})
    f1  = m.get('macro_f1', 0)
    acc = m.get('val_acc', 0)
    ep  = ckpt.get('epoch', '?')
    print(f"  [Fold {fold}] Loaded ✅  epoch={ep}  "
          f"saved_F1={f1:.2f}%  saved_Acc={acc:.2f}%")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# TTA views  (applied to one clip, not crossing fold boundaries)
# ═══════════════════════════════════════════════════════════════════════════════

def make_tta_views(clip_npy: np.ndarray,
                   use_temporal_reverse: bool = True) -> list:
    """
    Generate TTA views from a single stored (16, 224, 224, 3) clip.

    Views:
      1. Original
      2. Horizontal flip  — model was trained with this augmentation (step 4),
                            so it handles this view confidently
      3. Temporal reverse — model sees action played backwards; partial
                            temporal symmetry in VideoMAE helps here
      4. H-flip + temporal reverse  (only if use_temporal_reverse=True)

    With use_temporal_reverse=False: 2 views (original + h-flip)
    With use_temporal_reverse=True:  4 views (all above)

    Returns list of (16, 3, 224, 224) normalised float32 tensors.
    """
    norm = T.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD)

    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        # arr: (16, 224, 224, 3) float32 [0,1]
        t = torch.from_numpy(arr).float().permute(0, 3, 1, 2)  # (16,3,H,W)
        return torch.stack([norm(t[i]) for i in range(CLIP_LEN)])

    original  = clip_npy
    hflip     = clip_npy[:, :, ::-1, :].copy()   # flip W axis
    trev      = clip_npy[::-1].copy()             # reverse T axis
    hflip_rev = hflip[::-1].copy()

    views = [to_tensor(original), to_tensor(hflip)]
    if use_temporal_reverse:
        views.append(to_tensor(trev))
        views.append(to_tensor(hflip_rev))

    return views   # list of (16, 3, 224, 224) tensors


# ═══════════════════════════════════════════════════════════════════════════════
# Per-clip inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_clip(model: nn.Module, clip_path: Path, device: torch.device,
                 use_tta: bool, use_temporal_reverse: bool) -> np.ndarray:
    """
    Predict class probabilities for one .npy clip file.
    Returns averaged softmax vector shape (NUM_CLASSES,).
    """
    raw = np.load(clip_path)   # (16, 224, 224, 3) float32

    if not use_tta:
        # Single view — identical to training evaluation
        norm   = T.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD)
        tensor = torch.from_numpy(raw).float().permute(0, 3, 1, 2)
        tensor = torch.stack([norm(tensor[i]) for i in range(CLIP_LEN)])
        views  = [tensor]
    else:
        views = make_tta_views(raw, use_temporal_reverse)

    probs_list = []
    for view in views:
        x = view.unsqueeze(0).to(device)   # (1, 16, 3, 224, 224)
        with autocast('cuda' if device.type == 'cuda' else 'cpu'):
            out   = model(pixel_values=x)
            probs = F.softmax(out.logits, dim=-1)
        probs_list.append(probs.squeeze(0).cpu().float().numpy())

    return np.mean(probs_list, axis=0)   # (NUM_CLASSES,)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds: list, labels: list) -> dict:
    total   = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    acc     = 100 * correct / max(1, total)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    cc = defaultdict(int)
    ct = defaultdict(int)

    for p, l in zip(preds, labels):
        ct[l] += 1
        if p == l:
            tp[l] += 1
            cc[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1

    f1_scores = {}
    for i in range(NUM_CLASSES):
        pr = tp[i] / max(1, tp[i] + fp[i])
        rc = tp[i] / max(1, tp[i] + fn[i])
        f1_scores[CLASSES[i]] = round(200 * pr * rc / max(1e-8, pr + rc), 2)

    macro_f1 = round(sum(f1_scores.values()) / NUM_CLASSES, 4)
    per_acc  = {CLASSES[i]: round(100 * cc[i] / max(1, ct[i]), 2)
                for i in range(NUM_CLASSES)}

    conf = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for p, l in zip(preds, labels):
        conf[l][p] += 1

    return {
        'acc'      : round(acc, 4),
        'macro_f1' : macro_f1,
        'f1_scores': f1_scores,
        'per_acc'  : per_acc,
        'conf'     : conf,
        'n'        : total,
    }


def print_fold_result(fold: int, m: dict, n_views: int):
    abbr = [c[:4] for c in CLASSES]
    print(f"\n  {'─'*60}")
    print(f"  Fold {fold} | Acc={m['acc']:.2f}%  MacroF1={m['macro_f1']:.2f}%"
          f"  ({n_views} TTA views/clip)")
    print(f"  {'─'*60}")
    print(f"  {'Class':10s}  {'Acc':>7}  {'F1':>7}")
    for cls in CLASSES:
        print(f"  {cls:10s}  {m['per_acc'][cls]:>6.1f}%  "
              f"{m['f1_scores'][cls]:>6.1f}%")
    print(f"\n  Confusion matrix (row=true, col=pred):")
    print(f"             " + "  ".join(f"{x:>5}" for x in abbr))
    for i, row in enumerate(m['conf']):
        print(f"  {abbr[i]:>5} |   " + "  ".join(f"{v:5d}" for v in row))


def print_summary(fold_metrics: dict, n_views: int,
                  v14_baseline: dict):
    folds     = sorted(fold_metrics.keys())
    f1_vals   = [fold_metrics[f]['macro_f1'] for f in folds]
    acc_vals  = [fold_metrics[f]['acc']       for f in folds]
    mean_f1   = sum(f1_vals)  / len(f1_vals)
    mean_acc  = sum(acc_vals) / len(acc_vals)
    std_f1    = statistics.stdev(f1_vals)  if len(f1_vals) > 1 else 0.0
    std_acc   = statistics.stdev(acc_vals) if len(acc_vals) > 1 else 0.0

    # Per-class means
    class_means = {}
    for cls in CLASSES:
        vals = [fold_metrics[f]['f1_scores'][cls] for f in folds]
        class_means[cls] = sum(vals) / len(vals)

    print(f"\n{'='*65}")
    print(f"  4-FOLD TTA RESULTS  ({n_views} views/clip, fold-matched model)")
    print(f"{'='*65}")
    print(f"  {'Fold':>5}  {'Acc':>7}  {'MacroF1':>9}  "
          + "  ".join(f"{c[:5]:>7}" for c in CLASSES))
    print(f"  {'-'*60}")
    for f in folds:
        m = fold_metrics[f]
        print(f"  {f:>5}  {m['acc']:>6.2f}%  {m['macro_f1']:>8.2f}%  "
              + "  ".join(f"{m['f1_scores'][c]:>6.2f}%" for c in CLASSES))
    print(f"  {'-'*60}")
    print(f"  {'Mean':>5}  {mean_acc:>6.2f}%  {mean_f1:>8.2f}%  "
          + "  ".join(f"{class_means[c]:>6.2f}%" for c in CLASSES))
    print(f"  {'Std':>5}  {std_acc:>6.2f}   {std_f1:>8.2f}   ")

    print(f"\n  ✅ Reportable result (TTA): "
          f"Macro F1 = {mean_f1:.2f}% ± {std_f1:.2f}")

    # Before/after comparison
    print(f"\n  BEFORE vs AFTER (per-class F1):")
    print(f"  {'Class':10s}  {'v14 (no TTA)':>14}  {'v14+TTA':>10}  {'Δ':>8}")
    print(f"  {'-'*48}")
    for cls in CLASSES:
        old = v14_baseline.get(cls, 0.0)
        new = class_means[cls]
        d   = new - old
        arrow = "▲" if d > 0.05 else ("▼" if d < -0.05 else "─")
        print(f"  {cls:10s}  {old:>13.2f}%  {new:>9.2f}%  "
              f"{arrow} {d:>+.2f}%")

    old_mean = sum(v14_baseline.values()) / len(v14_baseline)
    print(f"  {'─'*48}")
    print(f"  {'MacroF1':10s}  {old_mean:>13.2f}%  {mean_f1:>9.2f}%  "
          f"{'▲' if mean_f1 > old_mean else '▼'} {mean_f1-old_mean:>+.2f}%")
    print(f"  Std         {5.64:>13.2f}   {std_f1:>9.2f}  "
          f"  {'▲' if std_f1 < 5.64 else '▼'} {std_f1-5.64:>+.2f}")
    print(f"\n  NOTE: Each fold evaluated by its OWN model only.")
    print(f"        No cross-fold model usage — zero leakage.")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    if IS_WINDOWS:
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥  Device : {device}")
    if device.type == 'cuda':
        print(f"   GPU  : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM : "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"   OS   : {platform.system()}")

    folds     = args.folds
    use_tta   = not args.no_tta
    use_trev  = not args.no_temporal_reverse and use_tta
    n_views   = (4 if use_trev else 2) if use_tta else 1

    print(f"\n  Evaluation strategy:")
    print(f"    TTA enabled          : {use_tta}")
    if use_tta:
        print(f"    H-flip               : yes (always)")
        print(f"    Temporal reverse     : {use_trev}")
        print(f"    Total views/clip     : {n_views}")
    print(f"    Folds                : {folds}")
    print(f"    Model per fold       : fold_N/videomae_best.pth (own fold only)")
    print(f"    Leakage              : NONE — each fold uses only its own model")

    # v14 baseline for comparison table
    v14_baseline = {
        'Fighting' : 63.77,
        'Robbery'  : 70.06,
        'Vandalism': 68.73,
        'Normal'   : 78.29,
    }

    fold_metrics = {}

    for fold in folds:
        print(f"\n{'═'*60}")
        print(f"  FOLD {fold}  —  loading fold_{fold}/videomae_best.pth")
        print(f"{'═'*60}")

        try:
            model = load_fold_model(fold, device)
        except FileNotFoundError as e:
            print(f"  ⚠  {e}")
            continue

        test_dir = CLIPS_DIR / f'fold_{fold}' / 'test'
        if not test_dir.exists():
            print(f"  ⚠  Test dir not found: {test_dir}")
            continue

        preds  = []
        labels = []

        for cls_idx, cls_name in enumerate(CLASSES):
            scene_dir = test_dir / cls_name / 'scene'
            if not scene_dir.exists():
                print(f"    ⚠  {scene_dir} not found — skipping {cls_name}")
                continue

            clip_files = sorted(scene_dir.glob('*.npy'))
            if not clip_files:
                print(f"    ⚠  No .npy files in {scene_dir}")
                continue

            print(f"    {cls_name:10s}: {len(clip_files):4d} clips",
                  end='', flush=True)

            for clip_path in clip_files:
                prob = predict_clip(model, clip_path, device,
                                    use_tta, use_trev)
                preds.append(int(np.argmax(prob)))
                labels.append(cls_idx)

            print(f"  ✓")

        # Free the model from GPU before loading the next fold's model
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if not preds:
            print(f"  ⚠  No predictions for fold {fold}")
            continue

        m = compute_metrics(preds, labels)
        fold_metrics[fold] = m
        print_fold_result(fold, m, n_views)

    if not fold_metrics:
        print("\n❌ No folds evaluated. Check checkpoint paths.")
        return

    # Summary across all evaluated folds
    print_summary(fold_metrics, n_views, v14_baseline)

    # Save
    out_path = WEIGHTS_DIR / 'tta_results.json'
    save_obj = {
        'strategy': {
            'use_tta'             : use_tta,
            'use_temporal_reverse': use_trev,
            'n_views_per_clip'    : n_views,
            'folds'               : folds,
            'leakage'             : 'none — each fold evaluated by own model only',
        },
        'per_fold': {str(f): fold_metrics[f] for f in sorted(fold_metrics)},
        'summary' : {
            'mean_macro_f1': round(
                sum(fold_metrics[f]['macro_f1'] for f in fold_metrics)
                / len(fold_metrics), 4),
            'std_macro_f1' : round(
                statistics.stdev(
                    [fold_metrics[f]['macro_f1'] for f in fold_metrics]
                ) if len(fold_metrics) > 1 else 0.0, 4),
            'mean_acc': round(
                sum(fold_metrics[f]['acc'] for f in fold_metrics)
                / len(fold_metrics), 4),
            'per_class_f1_mean': {
                cls: round(
                    sum(fold_metrics[f]['f1_scores'][cls]
                        for f in fold_metrics) / len(fold_metrics), 4)
                for cls in CLASSES
            },
        },
    }
    with open(out_path, 'w') as fp:
        json.dump(save_obj, fp, indent=2)
    print(f"  📄 Results saved → {out_path}")

    mean_f1 = save_obj['summary']['mean_macro_f1']
    std_f1  = save_obj['summary']['std_macro_f1']
    print(f"\n🎉 Leak-free TTA result: Macro F1 = {mean_f1:.2f}% ± {std_f1:.2f}%")
    print(f"   v14 baseline (no TTA): Macro F1 = 70.21% ± 5.64%")
    print(f"   This is your honest, reportable number.\n")


if __name__ == '__main__':
    if IS_WINDOWS:
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(
        description='evaluate_tta.py — leak-free TTA for UCF-Crime 4-fold CV'
    )
    parser.add_argument(
        '--folds', type=int, nargs='+', default=[1, 2, 3, 4],
        help='Folds to evaluate (default: 1 2 3 4)'
    )
    parser.add_argument(
        '--no-tta', action='store_true', dest='no_tta',
        help='Disable TTA — reproduce training evaluation exactly (sanity check)'
    )
    parser.add_argument(
        '--no-temporal-reverse', action='store_true',
        dest='no_temporal_reverse',
        help='Use only 2 views (original + h-flip) instead of 4'
    )

    args = parser.parse_args()
    main(args)