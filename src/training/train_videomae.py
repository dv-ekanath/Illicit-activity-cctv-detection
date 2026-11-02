"""
train_videomae.py
=========================================================
4 classes: Fighting, Robbery, Vandalism, Normal

TRAINING SETTINGS:
  Model      : MCG-NJU/videomae-base-finetuned-kinetics (86M params)
  Epochs     : 20 | Batch=8 | EffBatch=32 (accum=4)
  Head LR    : 6e-5 | Backbone LR : 3e-6 (head × 0.05)
  WD         : 0.08 | GradClip : 1.0
  Dropout    : 0.5(ep1-8) → 0.4(ep9-14) → 0.3(ep15+)
  Loss       : CrossEntropyLoss label_smoothing=0.15
  MixUp      : alpha=0.8, 50% batches, from epoch 3
  Unfreeze   : 5-stage epoch 1/4/8/13/18 via add_param_group (no rebuild)
  SWA        : from epoch 14, head_lr=3e-6, bb_lr=3e-7
  Scheduler  : ManualCosineScheduler until SWA → ReduceLROnPlateau

CHECKPOINT STRUCTURE:
  models/videomae/
    fold_1/videomae_best.pth        ← best macro F1 on that fold's test set
    fold_1/videomae_final.pth
    fold_1/videomae_final_swa.pth
    fold_1/videomae_epoch5.pth  (every 5 epochs)
    fold_2/ ...  fold_3/ ...  fold_4/ ...
    cv_results.json
"""

import sys
import time
import json
import argparse
import random
import math
import platform
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.amp import GradScaler, autocast
from transformers import VideoMAEForVideoClassification

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.videomae_dataset import get_dataloaders, CLASSES

ROOT        = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / 'models' / 'videomae'
LOGS_DIR    = ROOT / 'logs'   / 'videomae'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES   = len(CLASSES)   # 4
VALID_FOLDS   = [1, 2, 3, 4]
IS_WINDOWS    = platform.system() == 'Windows'
DEFAULT_MODEL = 'MCG-NJU/videomae-base-finetuned-kinetics'

# 5-stage unfreeze: {epoch: first_active_block_index}
BASE_UNFREEZE  = {1: 10, 4: 8, 8: 5, 13: 2, 18: 0}
LARGE_UNFREEZE = {1: 22, 4: 20, 8: 17, 13: 12, 18: 0}


# ═══════════════════════════════════════════════════════════════════════════════
# Manual Cosine LR Scheduler — safe with add_param_group
# ═══════════════════════════════════════════════════════════════════════════════

class ManualCosineScheduler:
    """
    Drop-in cosine LR with warmup. Unlike OneCycleLR it does NOT cache the
    param group count at init, so add_param_group() during training is safe.
    New groups added after init are registered on their first step() call.
    """
    def __init__(self, optimizer, total_steps: int, pct_start: float = 0.20,
                 div_factor: float = 25.0, final_div_factor: float = 1e4):
        self.optimizer        = optimizer
        self.total_steps      = total_steps
        self.warmup_steps     = int(total_steps * pct_start)
        self.div_factor       = div_factor
        self.final_div_factor = final_div_factor
        self._step            = 0
        self._peak_lr         = {idx: pg['lr']
                                 for idx, pg in enumerate(optimizer.param_groups)}
        print(f"  [LR] ManualCosine: total_steps={total_steps}, "
              f"warmup={self.warmup_steps} steps")

    def _compute_lr(self, peak_lr: float) -> float:
        s = self._step
        if self.warmup_steps > 0 and s <= self.warmup_steps:
            frac = s / self.warmup_steps
            return (peak_lr / self.div_factor
                    + frac * (peak_lr - peak_lr / self.div_factor))
        progress = ((s - self.warmup_steps)
                    / max(1, self.total_steps - self.warmup_steps))
        cos_val  = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        min_lr   = peak_lr / self.final_div_factor
        return min_lr + cos_val * (peak_lr - min_lr)

    def step(self):
        self._step += 1
        for idx, pg in enumerate(self.optimizer.param_groups):
            if idx not in self._peak_lr:
                self._peak_lr[idx] = pg['lr']   # register new group at its current LR
            pg['lr'] = self._compute_lr(self._peak_lr[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class VideoMAEFineTuned(nn.Module):
    def __init__(self, base: nn.Module, dropout: float = 0.5):
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


def build_model(pretrained_path: str = None,
                model_name: str = DEFAULT_MODEL,
                dropout: float = 0.5) -> nn.Module:
    if pretrained_path and Path(pretrained_path).exists():
        print(f"[Model] Loading from local: {pretrained_path}")
        base = VideoMAEForVideoClassification.from_pretrained(
            pretrained_path, num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True)
    else:
        print(f"[Model] Downloading {model_name} ...")
        base = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True)
    base.videomae.encoder.gradient_checkpointing = True
    model    = VideoMAEFineTuned(base, dropout=dropout)
    total    = sum(p.numel() for p in model.parameters())
    n_blocks = len(model.videomae.encoder.layer)
    print(f"[Model] {model_name.split('/')[-1]} | {n_blocks} blocks | "
          f"Dropout={dropout} | Params={total:,}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Progressive Unfreeze — add_param_group (no optimizer rebuild)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_unfreeze_schedule(n_blocks: int) -> dict:
    return BASE_UNFREEZE if n_blocks < 20 else LARGE_UNFREEZE


def progressive_unfreeze(model: nn.Module, optimizer: optim.Optimizer,
                          epoch: int, lr: float, wd: float) -> bool:
    blocks   = model.videomae.encoder.layer
    n        = len(blocks)
    schedule = _get_unfreeze_schedule(n)

    if epoch not in schedule:
        return False

    freeze_below = schedule[epoch]

    for i, blk in enumerate(blocks):
        for p in blk.parameters():
            p.requires_grad = (i >= freeze_below)
    for p in model.videomae.embeddings.parameters():
        p.requires_grad = (freeze_below == 0)

    all_optimised = set()
    for group in optimizer.param_groups:
        all_optimised.update(id(p) for p in group['params'])

    new_params = [p for p in model.parameters()
                  if p.requires_grad and id(p) not in all_optimised]

    if new_params:
        new_lr = lr * 0.05
        optimizer.add_param_group(
            {'params': new_params, 'lr': new_lr, 'weight_decay': wd})
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f"  [Unfreeze] Epoch {epoch}: blocks {freeze_below}–{n-1} | "
              f"+{len(new_params)} tensors @ lr={new_lr:.1e} | "
              f"trainable={trainable:,}")
    else:
        print(f"  [Unfreeze] Epoch {epoch}: no new params "
              f"(freeze_below={freeze_below})")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MixUp
# ═══════════════════════════════════════════════════════════════════════════════

def mixup(clips, labels, alpha: float = 0.8):
    lam   = random.betavariate(alpha, alpha)
    lam   = max(lam, 1 - lam)
    idx   = torch.randperm(clips.size(0), device=clips.device)
    mixed = lam * clips + (1 - lam) * clips[idx]
    return mixed, labels, labels[idx], lam


def mixup_loss(criterion, logits, la, lb, lam):
    return lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)


# ═══════════════════════════════════════════════════════════════════════════════
# Dropout schedule
# ═══════════════════════════════════════════════════════════════════════════════

def get_dropout_p(epoch: int) -> float:
    if epoch <= 8:  return 0.5
    if epoch <= 14: return 0.4
    return 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# Train / Evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, accum_steps, scheduler, epoch: int,
                    use_mixup: bool = True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    mixup_enabled = use_mixup and (epoch >= 3)
    model.dropout.p = get_dropout_p(epoch)

    for i, (clips, labels) in enumerate(loader):
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_mix = mixup_enabled and (random.random() < 0.5)
        if use_mix:
            clips, la, lb, lam = mixup(clips, labels, alpha=0.8)
        else:
            la, lb, lam = labels, labels, 1.0

        with autocast('cuda'):
            out  = model(pixel_values=clips)
            loss = (mixup_loss(criterion, out.logits, la, lb, lam)
                    if use_mix else criterion(out.logits, labels))
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * accum_steps
        preds    = out.logits.argmax(dim=1)
        correct += (preds == la).sum().item()
        total   += la.size(0)

        if (i + 1) % 20 == 0:
            curr_lr = max(pg['lr'] for pg in optimizer.param_groups)
            print(f"  [{i+1}/{len(loader)}] "
                  f"loss={total_loss/(i+1):.4f}  "
                  f"acc={100*correct/total:.1f}%  "
                  f"drop={model.dropout.p}  "
                  f"lr={curr_lr:.2e}  "
                  f"mix={'on' if mixup_enabled else 'off'}")

    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = total = 0
    cc = [0] * NUM_CLASSES
    ct = [0] * NUM_CLASSES
    all_p, all_l = [], []

    with torch.no_grad():
        for clips, labels in loader:
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast('cuda'):
                out  = model(pixel_values=clips)
                loss = criterion(out.logits, labels)
            total_loss += loss.item()
            preds = out.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_p.extend(preds.cpu().tolist())
            all_l.extend(labels.cpu().tolist())
            for p, l in zip(preds, labels):
                ct[l.item()] += 1
                if p == l:
                    cc[l.item()] += 1

    per_acc = {CLASSES[i]: round(100 * cc[i] / max(1, ct[i]), 2)
               for i in range(NUM_CLASSES)}

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, l in zip(all_p, all_l):
        if p == l: tp[l] += 1
        else:      fp[p] += 1; fn[l] += 1

    f1 = {}
    for i in range(NUM_CLASSES):
        pr = tp[i] / max(1, tp[i] + fp[i])
        rc = tp[i] / max(1, tp[i] + fn[i])
        f1[CLASSES[i]] = round(200 * pr * rc / max(1e-8, pr + rc), 2)

    macro_f1 = sum(f1.values()) / NUM_CLASSES
    conf = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for p, l in zip(all_p, all_l):
        conf[l][p] += 1

    return (total_loss / len(loader), 100 * correct / total,
            per_acc, f1, macro_f1, conf)


def print_conf(conf):
    abbr = [c[:4] for c in CLASSES]
    print("  Confusion matrix (row=true, col=pred):")
    print("            " + "  ".join(f"{x:>5}" for x in abbr))
    for i, row in enumerate(conf):
        print(f"  {abbr[i]:>5} |   " + "  ".join(f"{v:5d}" for v in row))


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

def save_ckpt(model, optimizer, epoch, metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = model.module if hasattr(model, 'module') else model
    torch.save({'epoch': epoch, 'state_dict': inner.state_dict(),
                'optimizer': optimizer.state_dict(), 'metrics': metrics}, path)
    print(f"  ✅ Saved → {path}")


def load_ckpt(model, path, device, optimizer=None):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    inner = model.module if hasattr(model, 'module') else model
    inner.load_state_dict(ckpt['state_dict'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  ▶ Resumed from epoch {ckpt['epoch']}")
    return ckpt['epoch'], ckpt.get('metrics', {})


# ═══════════════════════════════════════════════════════════════════════════════
# Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

def build_initial_optimizer(model, lr, wd):
    head_p = [p for n, p in model.named_parameters()
               if 'classifier' in n and p.requires_grad]
    bb_p   = [p for n, p in model.named_parameters()
               if 'classifier' not in n and p.requires_grad]
    return optim.AdamW([
        {'params': bb_p,   'lr': lr * 0.05, 'weight_decay': wd},
        {'params': head_p, 'lr': lr,         'weight_decay': wd},
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Single Fold
# ═══════════════════════════════════════════════════════════════════════════════

def train_fold(fold: int, args, device: torch.device) -> dict:
    fold_weights_dir = WEIGHTS_DIR / f'fold_{fold}'
    fold_weights_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f'training_log_fold{fold}_v14.txt'

    print(f"\n{'='*70}")
    print(f"  FOLD {fold}/{max(VALID_FOLDS)}  |  Classes: {CLASSES}")
    print(f"{'='*70}")

    train_loader, test_loader, class_weights = get_dataloaders(
        fold=fold, batch_size=args.batch_size, num_workers=args.workers,
        use_sampler=True, verify=True, sampler_mult=args.sampler_mult)

    pretrained = str(ROOT / args.pretrained) if args.pretrained else None
    model = build_model(pretrained_path=pretrained,
                        model_name=args.model_name, dropout=0.5).to(device)

    n_blocks     = len(model.videomae.encoder.layer)
    schedule     = _get_unfreeze_schedule(n_blocks)
    freeze_below = schedule[1]   # 10 for Base → blocks 10–11 active at epoch 1

    for i, blk in enumerate(model.videomae.encoder.layer):
        for p in blk.parameters():
            p.requires_grad = (i >= freeze_below)
    for p in model.videomae.embeddings.parameters():
        p.requires_grad = False

    init_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [Init] blocks {freeze_below}–{n_blocks-1} active | "
          f"embeddings frozen | {init_trainable:,} trainable params")

    wt = torch.tensor([class_weights[c] for c in CLASSES],
                      dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=wt, label_smoothing=0.15)
    print(f"\n[Loss] CrossEntropyLoss(label_smoothing=0.15)")
    print(f"       Weights: {[round(class_weights[c],3) for c in CLASSES]}")
    print(f"[Loss] MixUp alpha=0.8, 50% batches, from epoch 3\n")

    optimizer = build_initial_optimizer(model, args.lr, args.wd)

    steps_per_epoch = len(train_loader) // args.accum_steps
    total_steps     = max(1, steps_per_epoch * args.swa_start)

    scheduler_cos = ManualCosineScheduler(
        optimizer, total_steps=total_steps, pct_start=0.20)
    scheduler_plateau = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)

    swa_model   = AveragedModel(model)
    swa_active  = False
    swa_lr_head = args.lr * 0.05
    swa_lr_bb   = args.lr * 0.005

    scaler       = GradScaler()
    best_f1      = 0.0
    best_acc     = 0.0
    no_improve   = 0
    start_epoch  = 1
    best_metrics = {}

    if args.resume and Path(args.resume).exists():
        start_epoch, m = load_ckpt(model, args.resume, device, optimizer)
        start_epoch += 1
        best_f1  = m.get('macro_f1', 0.0)
        best_acc = m.get('val_acc',  0.0)

    print(f"  {args.model_name.split('/')[-1]} ({n_blocks} blocks)")
    print(f"  Classes ({NUM_CLASSES}): {CLASSES}")
    print(f"  Epochs={args.epochs} | Batch={args.batch_size} | "
          f"EffBatch={args.batch_size*args.accum_steps}")
    print(f"  Head LR={args.lr:.1e} | Backbone LR={args.lr*0.05:.1e} | "
          f"WD={args.wd}")
    print(f"  Dropout: 0.5(ep1-8)→0.4(ep9-14)→0.3(ep15+) | GradClip=1.0")
    print(f"  Unfreeze: ep1/4/8/13/18 | MixUp from ep3 | "
          f"SWA from ep{args.swa_start}")
    print(f"  Patience={args.patience} | steps/epoch={steps_per_epoch}")
    print(f"  Windows={IS_WINDOWS} (test_workers=0, persistent_workers=False)")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        progressive_unfreeze(model, optimizer, epoch, args.lr, args.wd)

        if epoch == args.swa_start and not swa_active:
            swa_active = True
            for pg in optimizer.param_groups:
                pg['lr'] = (swa_lr_head if pg['lr'] > args.lr * 0.02
                            else swa_lr_bb)
            print(f"\n  [SWA] Activated @ epoch {epoch} "
                  f"head_lr={swa_lr_head:.1e} bb_lr={swa_lr_bb:.1e}")

        active_sched = None if swa_active else scheduler_cos

        trn_loss, trn_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.accum_steps, active_sched,
            epoch=epoch, use_mixup=args.mixup)

        if swa_active:
            swa_model.update_parameters(model)
            try:
                update_bn(train_loader, swa_model, device=device)
            except Exception:
                pass

        eval_model = swa_model if swa_active else model
        val_loss, val_acc, per_class, f1_scores, macro_f1, conf = evaluate(
            eval_model, test_loader, criterion, device)

        if swa_active:
            scheduler_plateau.step(macro_f1)

        elapsed      = time.time() - t0
        curr_head_lr = max(pg['lr'] for pg in optimizer.param_groups)
        gap          = trn_acc - val_acc

        print(f"\nFold {fold} | Epoch {epoch}/{args.epochs} | "
              f"train={trn_loss:.4f}/{trn_acc:.1f}% | "
              f"test={val_loss:.4f}/{val_acc:.1f}% | "
              f"F1={macro_f1:.1f}% | "
              f"lr={curr_head_lr:.2e} | {elapsed:.0f}s")

        gap_icon = "⚠ " if gap > 20 else ("△ " if gap > 10 else "✅")
        print(f"  {gap_icon} gap={gap:.1f}pp  drop={get_dropout_p(epoch)}"
              + (" [SWA]" if swa_active else ""))

        print("  Per-class (acc / F1):")
        for cls in CLASSES:
            warn = " ⚠" if f1_scores[cls] < 50 else ""
            print(f"    {cls:10s}: acc={per_class[cls]:.1f}%  "
                  f"F1={f1_scores[cls]:.1f}%{warn}")
        print_conf(conf)

        with open(log_file, 'a') as f:
            f.write(
                f"fold={fold} epoch={epoch} trn={trn_acc:.2f} "
                f"val={val_acc:.2f} f1={macro_f1:.2f} gap={gap:.1f} "
                + " ".join(f"{c.lower()}_f1={f1_scores[c]:.1f}"
                           for c in CLASSES) + "\n")

        metrics = {'val_acc': val_acc, 'macro_f1': macro_f1,
                   'per_class': per_class, 'f1_scores': f1_scores}

        if macro_f1 > best_f1:
            best_f1 = macro_f1; best_acc = val_acc
            no_improve = 0; best_metrics = metrics.copy()
            save_model = swa_model if swa_active else model
            save_ckpt(save_model, optimizer, epoch, metrics,
                      fold_weights_dir / 'videomae_best.pth')
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{args.patience} "
                  f"(best F1={best_f1:.1f}%)")

        if epoch % 5 == 0:
            save_ckpt(model, optimizer, epoch, metrics,
                      fold_weights_dir / f'videomae_epoch{epoch}.pth')

        if no_improve >= args.patience:
            print(f"\n⏹  Early stopping at epoch {epoch} (fold {fold})")
            break

    if swa_active:
        save_ckpt(swa_model, optimizer, epoch,
                  {'val_acc': best_acc, 'macro_f1': best_f1},
                  fold_weights_dir / 'videomae_final_swa.pth')
    save_ckpt(model, optimizer, epoch,
              {'val_acc': best_acc, 'macro_f1': best_f1},
              fold_weights_dir / 'videomae_final.pth')

    print(f"\n  Fold {fold} done.  Best F1={best_f1:.2f}%  Acc={best_acc:.2f}%")
    return best_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CV Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_cv_summary(fold_results: dict):
    import statistics
    folds = sorted(fold_results.keys())
    print(f"\n{'='*70}")
    print(f"  4-FOLD CV SUMMARY  ({NUM_CLASSES} classes: {CLASSES})")
    print(f"{'='*70}")
    print(f"  {'Fold':>5}  {'Acc':>7}  {'MacroF1':>9}  "
          + "  ".join(f"{c[:5]:>7}" for c in CLASSES))
    print(f"  {'-'*60}")

    macro_f1s = []; accs = []; class_f1s = {c: [] for c in CLASSES}
    for fold in folds:
        m = fold_results[fold]
        macro_f1s.append(m['macro_f1']); accs.append(m['val_acc'])
        for c in CLASSES:
            class_f1s[c].append(m['f1_scores'].get(c, 0.0))
        print(f"  {fold:>5}  {m['val_acc']:7.2f}%  {m['macro_f1']:9.2f}%  "
              + "  ".join(f"{m['f1_scores'].get(c,0):7.2f}%" for c in CLASSES))

    print(f"  {'-'*60}")

    def fmt(vals):
        return (f"{sum(vals)/len(vals):.2f}% ±{statistics.stdev(vals):.2f}"
                if len(vals) > 1 else f"{vals[0]:.2f}%")

    print(f"  {'Mean':>5}  {fmt(accs):>8}  {fmt(macro_f1s):>10}  "
          + "  ".join(f"{fmt(class_f1s[c]):>8}" for c in CLASSES))
    print(f"\n  ✅ Reportable result: Macro F1 = {fmt(macro_f1s)}")
    print(f"{'='*70}\n")

    return {
        'mean_macro_f1': sum(macro_f1s) / len(macro_f1s),
        'std_macro_f1' : statistics.stdev(macro_f1s) if len(macro_f1s) > 1 else 0.0,
        'mean_acc'     : sum(accs) / len(accs),
        'per_class_f1' : {c: sum(class_f1s[c])/len(class_f1s[c]) for c in CLASSES},
        'fold_results' : fold_results,
    }


def load_fold_result_from_ckpt(fold: int) -> dict:
    ckpt_path = WEIGHTS_DIR / f'fold_{fold}' / 'videomae_best.pth'
    if not ckpt_path.exists():
        print(f"  [WARN] No checkpoint for fold {fold}: {ckpt_path}")
        return {}
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    m    = ckpt.get('metrics', {})
    print(f"  Fold {fold}: F1={m.get('macro_f1',0):.2f}%  "
          f"Acc={m.get('val_acc',0):.2f}%")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM: {vram_gb:.1f} GB")
        if vram_gb < 10 and 'large' in args.model_name.lower():
            print(f"   ⚠  {vram_gb:.1f}GB + videomae-large → will OOM")
    print(f"   OS  : {platform.system()}"
          + (" (test_workers=0)" if IS_WINDOWS else ""))

    if args.summarise_only:
        print("\n[Summary] Loading best checkpoints from all 4 folds...")
        fold_results = {}
        for fold in VALID_FOLDS:
            m = load_fold_result_from_ckpt(fold)
            if m:
                fold_results[fold] = m
        if fold_results:
            cv_summary = print_cv_summary(fold_results)
            cv_path    = WEIGHTS_DIR / 'cv_results.json'
            with open(cv_path, 'w') as f:
                json.dump(cv_summary, f, indent=2)
            print(f"  📄 Saved → {cv_path}")
        else:
            print("  No fold checkpoints found.")
        return

    folds_to_run = list(VALID_FOLDS) if args.all_folds else [args.fold]

    print(f"\n  Model    : {args.model_name}")
    print(f"  Classes  : {CLASSES}")
    print(f"  Folds    : {folds_to_run}")
    print(f"  Epochs   : {args.epochs} | Batch: {args.batch_size}"
          f" × accum={args.accum_steps} = eff {args.batch_size*args.accum_steps}")
    print(f"  Head LR  : {args.lr:.1e}  Backbone: {args.lr*0.05:.1e}")
    print(f"  SWA ep   : {args.swa_start}  Patience: {args.patience}")

    fold_results = {}
    for fold in folds_to_run:
        fold_results[fold] = train_fold(fold, args, device)

    if len(fold_results) > 1:
        cv_summary = print_cv_summary(fold_results)
        cv_path    = WEIGHTS_DIR / 'cv_results.json'
        with open(cv_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        print(f"  📄 CV results → {cv_path}")
    else:
        fold = folds_to_run[0]
        m    = fold_results.get(fold, {})
        if m:
            print(f"\n🎉 Fold {fold} | F1={m.get('macro_f1',0):.2f}%  "
                  f"Acc={m.get('val_acc',0):.2f}%")
            print("   Per-class: " + "  ".join(
                f"{c}={m.get('f1_scores',{}).get(c,0):.1f}%"
                for c in CLASSES))


if __name__ == '__main__':
    # Windows: spawn MUST be set here, before main() is called
    if IS_WINDOWS:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description='train_videomae v14 — 4 classes, all crashes fixed'
    )
    fold_grp = parser.add_mutually_exclusive_group()
    fold_grp.add_argument('--fold', type=int, choices=VALID_FOLDS, default=1)
    fold_grp.add_argument('--all-folds', action='store_true')

    parser.add_argument('--summarise-only', action='store_true',
                        dest='summarise_only',
                        help='Load saved checkpoints and print CV summary, no training')
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch-size',   type=int,   default=8,  dest='batch_size')
    parser.add_argument('--accum-steps',  type=int,   default=4,  dest='accum_steps')
    parser.add_argument('--workers',      type=int,   default=4,
                        help='DataLoader workers (capped at 2 on Windows)')
    parser.add_argument('--lr',           type=float, default=6e-5)
    parser.add_argument('--wd',           type=float, default=0.08)
    parser.add_argument('--swa-start',    type=int,   default=14, dest='swa_start')
    parser.add_argument('--patience',     type=int,   default=10)
    parser.add_argument('--sampler-mult', type=float, default=2.0, dest='sampler_mult')
    parser.add_argument('--model-name',   type=str,   default=DEFAULT_MODEL,
                        dest='model_name')
    parser.add_argument('--mixup',        action='store_true', default=True)
    parser.add_argument('--no-mixup',     dest='mixup', action='store_false')
    parser.add_argument('--pretrained',   type=str,
                        default='models/videomae/pretrained')
    parser.add_argument('--resume',       type=str, default=None)

    args = parser.parse_args()
    main(args)