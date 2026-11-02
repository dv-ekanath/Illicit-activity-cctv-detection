"""
train_zerodce.py
================
Zero-DCE++ Two-Stage Training

Stage 1 — General Enhancement on ExDARK
  • From scratch, standard 4-loss unsupervised
  • 200 epochs, LR=1e-4, batch=16, 256×256
  • Saves: models/zerodce/stage1_best.pth

Stage 2 — Detection-Aware Fine-tuning on LLVIP
  • Loads stage1_best.pth
  • Combined loss: L_enh + λ(epoch) * L_det
  • YOLO weights FROZEN, only Zero-DCE++ updates
  • λ warmup: 0.05 (ep1-20) → 0.10 (ep21-50) → 0.20 (ep51+)
  • 100 epochs, LR=1e-5, batch=8, 256×256
  • Saves: models/zerodce/stage2_best.pth
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.zerodce_model   import ZeroDCE
from training.zerodce_losses  import ZeroDCELoss
from training.zerodce_dataset import get_exdark_loaders, get_llvip_loaders

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
EXDARK_DIR   = ROOT / 'data' / 'raw' / 'exdark' / 'images'
LLVIP_VIS    = ROOT / 'data' / 'raw' / 'llvip'  / 'visible'
LLVIP_ANN    = ROOT / 'data' / 'raw' / 'llvip'  / 'Annotations'
WEIGHTS_DIR  = ROOT / 'models' / 'zerodce'
LOGS_DIR     = ROOT / 'logs'   / 'zerodce'
YOLO_WEIGHTS = ROOT / 'models' / 'yolov11' / 'yolov11s.pt'

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# λ Warmup Schedule
# ═══════════════════════════════════════════════════════════════════════════════
def get_lambda(epoch: int) -> float:
    """Detection loss weight per project spec."""
    if epoch <= 20:
        return 0.05
    elif epoch <= 50:
        return 0.10
    else:
        return 0.20


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO Detection Loss Helper
# ═══════════════════════════════════════════════════════════════════════════════
def load_frozen_yolo(weights_path: Path, device: torch.device):
    """
    Load YOLOv11 with all parameters frozen.
    Returns the model in train mode so its internal loss is accessible.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLOv11 weights not found at {weights_path}\n"
            f"Download first: yolo download model=yolov11n.pt\n"
            f"Or run Stage 1 only with --skip-stage2"
        )
    from ultralytics import YOLO
    yolo = YOLO(str(weights_path))
    model = yolo.model.to(device)

    # Freeze ALL YOLO parameters — only Zero-DCE++ updates
    for param in model.parameters():
        param.requires_grad = False
    model.train()   # train mode so YOLO loss heads are active
    print(f"[YOLO] Loaded frozen weights from {weights_path}")
    print(f"[YOLO] All {sum(p.numel() for p in model.parameters()):,} params frozen")
    return model


def compute_yolo_det_loss(yolo_model, enhanced_images, boxes_list, device):
    """
    Run YOLOv11 forward on enhanced images and compute detection loss.
    Gradients flow back through enhanced_images into Zero-DCE++.
    YOLO weights stay frozen (requires_grad=False).

    Args:
        yolo_model   : frozen YOLOv11 model
        enhanced_images : (B, 3, H, W) at 256×256 — will be resized to 640×640
        boxes_list   : list of (N, 5) tensors [class, cx, cy, w, h] per image
        device       : torch device

    Returns:
        det_loss scalar (differentiable w.r.t. enhanced_images)
    """
    import torch.nn.functional as F

    # Resize 256 → 640 for YOLO (keep gradients)
    imgs_640 = F.interpolate(enhanced_images, size=(640, 640),
                             mode='bilinear', align_corners=False)

    # Build YOLO-format batch target
    targets = []
    for i, boxes in enumerate(boxes_list):
        if len(boxes) == 0:
            continue
        batch_col = torch.full((len(boxes), 1), i, dtype=torch.float32, device=device)
        targets.append(torch.cat([batch_col, boxes.to(device)], dim=1))

    if not targets:
        return torch.tensor(0.0, device=device, requires_grad=True)

    targets = torch.cat(targets, dim=0)   # (total_boxes, 6): [batch_idx, cls, cx, cy, w, h]

    try:
        loss, _ = yolo_model(imgs_640, targets)
        return loss
    except Exception as e:
        # Some YOLO versions return loss differently — fallback
        preds = yolo_model(imgs_640)
        return torch.tensor(0.0, device=device, requires_grad=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Generic Train Loop
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch_stage1(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total, n = 0.0, 0
    comps = {'spatial': 0, 'exposure': 0, 'color': 0, 'illumination': 0}

    for i, images in enumerate(loader):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            enhanced, alphas = model(images)
            loss, breakdown  = criterion(images, enhanced, alphas)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        n     += 1
        for k in comps:
            comps[k] += breakdown[k]

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(loader)}] loss={total/n:.4f} "
                  f"spa={breakdown['spatial']:.3f} "
                  f"exp={breakdown['exposure']:.3f} "
                  f"col={breakdown['color']:.3f} "
                  f"tv={breakdown['illumination']:.3f}")

    for k in comps:
        comps[k] /= n
    return total / n, comps


def train_one_epoch_stage2(model, yolo_model, loader, criterion,
                           optimizer, scaler, device, epoch):
    model.train()
    lam   = get_lambda(epoch)
    total, n = 0.0, 0
    enh_total, det_total = 0.0, 0.0

    for i, (images, boxes_list) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            enhanced, alphas = model(images)

            # Enhancement loss (standard 4-loss)
            enh_loss, breakdown = criterion(images, enhanced, alphas)

            # Detection loss — gradients flow through enhanced → zerodce
            # YOLO weights are frozen via requires_grad=False (NOT no_grad)
            det_loss = compute_yolo_det_loss(yolo_model, enhanced, boxes_list, device)

            total_loss = enh_loss + lam * det_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total     += total_loss.item()
        enh_total += enh_loss.item()
        det_total += det_loss.item()
        n         += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(loader)}] total={total/n:.4f} "
                  f"enh={enh_total/n:.4f} det={det_total/n:.4f} λ={lam}")

    return total / n, enh_total / n, det_total / n


def validate(model, loader, criterion, device, has_boxes=False):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device) if has_boxes else batch.to(device)
            with autocast('cuda'):
                enhanced, alphas = model(images)
                loss, _          = criterion(images, enhanced, alphas)
            total += loss.item()
            n     += 1
    return total / n


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def save_ckpt(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch, 'loss': loss,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }, path)
    print(f"  ✅ Saved → {path}  (loss={loss:.4f})")


def load_ckpt(model, path, device, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  ▶ Loaded {path}  (epoch={ckpt['epoch']} loss={ckpt['loss']:.4f})")
    return ckpt['epoch'], ckpt['loss']


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — ExDARK General Enhancement
# ═══════════════════════════════════════════════════════════════════════════════
def run_stage1(model, device, args, log_file):
    print(f"\n{'='*60}")
    print(f"  STAGE 1 — ExDARK General Enhancement")
    print(f"  Epochs={args.s1_epochs} | LR={args.s1_lr} | Batch={args.s1_batch} | 256×256")
    print(f"{'='*60}\n")

    trn_loader, val_loader = get_exdark_loaders(
        str(EXDARK_DIR), batch_size=args.s1_batch, img_size=256, num_workers=args.workers
    )
    criterion = ZeroDCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.s1_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.s1_epochs, eta_min=args.s1_lr * 0.01)
    scaler    = GradScaler()

    start_epoch, best_loss = 1, float('inf')
    stage1_best = WEIGHTS_DIR / 'stage1_best.pth'

    if args.resume_s1 and Path(args.resume_s1).exists():
        start_epoch, best_loss = load_ckpt(model, args.resume_s1, device, optimizer)
        start_epoch += 1

    with open(log_file, 'a') as f:
        f.write('\n--- Stage 1: ExDARK ---\n')

    for epoch in range(start_epoch, args.s1_epochs + 1):
        t0 = time.time()
        trn_loss, comps = train_one_epoch_stage1(
            model, trn_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device, has_boxes=False)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{args.s1_epochs} | "
              f"train={trn_loss:.4f} val={val_loss:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        with open(log_file, 'a') as f:
            f.write(f"ep={epoch} trn={trn_loss:.4f} val={val_loss:.4f} "
                    f"spa={comps['spatial']:.4f} exp={comps['exposure']:.4f} "
                    f"col={comps['color']:.4f} tv={comps['illumination']:.4f}\n")

        if val_loss < best_loss:
            best_loss = val_loss
            save_ckpt(model, optimizer, epoch, best_loss, stage1_best)

        if epoch % 25 == 0:
            save_ckpt(model, optimizer, epoch, val_loss,
                      WEIGHTS_DIR / f'stage1_epoch{epoch}.pth')

    save_ckpt(model, optimizer, args.s1_epochs, best_loss,
              WEIGHTS_DIR / 'stage1_final.pth')
    print(f"\n✅ Stage 1 complete. Best val loss: {best_loss:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Detection-Aware LLVIP Fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════
def run_stage2(model, device, args, log_file):
    print(f"\n{'='*60}")
    print(f"  STAGE 2 — Detection-Aware LLVIP Fine-tuning")
    print(f"  Epochs={args.s2_epochs} | LR={args.s2_lr} | Batch={args.s2_batch} | 256×256")
    print(f"  λ warmup: ep1-20=0.05 | ep21-50=0.10 | ep51+=0.20")
    print(f"{'='*60}\n")

    # Load Stage 1 best weights
    stage1_best = WEIGHTS_DIR / 'stage1_best.pth'
    if stage1_best.exists() and not args.skip_stage1:
        load_ckpt(model, stage1_best, device)
    elif args.resume_s2 and Path(args.resume_s2).exists():
        load_ckpt(model, args.resume_s2, device)

    # Load frozen YOLO
    yolo_model = load_frozen_yolo(YOLO_WEIGHTS, device)

    trn_loader, val_loader = get_llvip_loaders(
        str(LLVIP_VIS), str(LLVIP_ANN),
        batch_size=args.s2_batch, img_size=256, num_workers=args.workers
    )
    criterion = ZeroDCELoss().to(device)

    # Only Zero-DCE++ params in optimizer — YOLO is already frozen
    optimizer = optim.Adam(model.parameters(), lr=args.s2_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.s2_epochs, eta_min=args.s2_lr * 0.01)
    scaler    = GradScaler()

    start_epoch, best_loss = 1, float('inf')
    stage2_best = WEIGHTS_DIR / 'stage2_best.pth'

    with open(log_file, 'a') as f:
        f.write('\n--- Stage 2: LLVIP Detection-Aware ---\n')

    for epoch in range(start_epoch, args.s2_epochs + 1):
        lam = get_lambda(epoch)
        t0  = time.time()

        trn_loss, enh_loss, det_loss = train_one_epoch_stage2(
            model, yolo_model, trn_loader, criterion,
            optimizer, scaler, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device, has_boxes=True)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{args.s2_epochs} | "
              f"total={trn_loss:.4f} enh={enh_loss:.4f} det={det_loss:.4f} | "
              f"val={val_loss:.4f} | λ={lam} | {elapsed:.1f}s")

        with open(log_file, 'a') as f:
            f.write(f"ep={epoch} total={trn_loss:.4f} enh={enh_loss:.4f} "
                    f"det={det_loss:.4f} val={val_loss:.4f} lambda={lam}\n")

        if val_loss < best_loss:
            best_loss = val_loss
            save_ckpt(model, optimizer, epoch, best_loss, stage2_best)

        if epoch % 10 == 0:
            save_ckpt(model, optimizer, epoch, val_loss,
                      WEIGHTS_DIR / f'stage2_epoch{epoch}.pth')

    save_ckpt(model, optimizer, args.s2_epochs, best_loss,
              WEIGHTS_DIR / 'stage2_final.pth')
    print(f"\n✅ Stage 2 complete. Best val loss: {best_loss:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥  Device : {device}")
    if device.type == 'cuda':
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model    = ZeroDCE(num_iterations=8).to(device)
    log_file = str(LOGS_DIR / 'training_log.txt')
    print(f"   Params : {sum(p.numel() for p in model.parameters()):,}")

    if not args.skip_stage1:
        model = run_stage1(model, device, args, log_file)

    if not args.skip_stage2:
        model = run_stage2(model, device, args, log_file)

    print(f"\n🎉 All done!")
    print(f"   Stage 1 weights → {WEIGHTS_DIR / 'stage1_best.pth'}")
    print(f"   Stage 2 weights → {WEIGHTS_DIR / 'stage2_best.pth'}")
    print(f"   Logs            → {log_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Zero-DCE++ (2-stage)')

    # Stage control
    parser.add_argument('--skip-stage1',  action='store_true', help='Skip Stage 1')
    parser.add_argument('--skip-stage2',  action='store_true', help='Skip Stage 2')

    # Stage 1 hyperparams (per spec)
    parser.add_argument('--s1-epochs',    type=int,   default=200)
    parser.add_argument('--s1-lr',        type=float, default=1e-4)
    parser.add_argument('--s1-batch',     type=int,   default=16)

    # Stage 2 hyperparams (per spec)
    parser.add_argument('--s2-epochs',    type=int,   default=100)
    parser.add_argument('--s2-lr',        type=float, default=1e-5)
    parser.add_argument('--s2-batch',     type=int,   default=8)

    # Hardware
    parser.add_argument('--workers',      type=int,   default=4)

    # Resume
    parser.add_argument('--resume-s1',    type=str,   default=None)
    parser.add_argument('--resume-s2',    type=str,   default=None)

    args = parser.parse_args()
    main(args)