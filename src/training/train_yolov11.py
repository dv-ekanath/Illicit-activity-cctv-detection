"""
train_yolov11.py
================
YOLOv11s Fine-tuning for Person Detection
  Base    : COCO pretrained yolo11s.pt
  Dataset : LLVIP (12,025 train) + ExDARK People (609) → 12,634 total train
  Val     : LLVIP test (3,463)
  Classes : person only (nc=1)
  Epochs  : 50
  Input   : 640×640

RTX 5060 (8GB VRAM) tuned:
  batch=16, workers=4, amp=True
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
WEIGHTS      = ROOT / 'models'  / 'yolov11'   / 'yolo11s.pt'
DATASET_YAML = ROOT / 'data'    / 'processed' / 'yolo_dataset' / 'dataset.yaml'
OUTPUT_DIR   = ROOT / 'models'  / 'yolov11'
LOGS_DIR     = ROOT / 'logs'    / 'yolov11'

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 55)
    print("  YOLOv11s Fine-tuning — Person Detection")
    print("=" * 55)
    print(f"  Weights  : {WEIGHTS}")
    print(f"  Dataset  : {DATASET_YAML}")
    print(f"  Output   : {OUTPUT_DIR}")
    print()

    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"YOLOv11s weights not found at {WEIGHTS}\n"
            f"Run: python -c \"from ultralytics import YOLO; YOLO('yolo11s.pt')\"\n"
            f"Then move yolo11s.pt to models/yolov11/yolo11s.pt"
        )

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {DATASET_YAML}\n"
            f"Run: python src/preprocessing/build_dataset_yaml.py"
        )

    model = YOLO(str(WEIGHTS))

    results = model.train(
        data        = str(DATASET_YAML),
        epochs      = 50,
        imgsz       = 640,
        batch       = 16,           # RTX 5060 8GB — reduce to 8 if OOM
        workers     = 4,
        device      = 0,            # GPU 0
        amp         = True,         # mixed precision — saves VRAM
        optimizer   = 'AdamW',
        lr0         = 1e-3,         # initial LR
        lrf         = 0.01,         # final LR = lr0 * lrf
        momentum    = 0.937,
        weight_decay= 0.0005,
        warmup_epochs    = 3,
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,

        # Augmentation — tuned for night/surveillance domain
        hsv_h       = 0.015,        # hue shift (minimal — night footage)
        hsv_s       = 0.3,          # saturation
        hsv_v       = 0.4,          # brightness variance (important for night)
        flipud      = 0.0,          # no vertical flip (CCTV is fixed)
        fliplr      = 0.5,
        mosaic      = 0.8,          # mosaic augmentation
        mixup       = 0.1,

        # Saving
        project     = str(OUTPUT_DIR),
        name        = 'finetune_llvip_exdark',
        save        = True,
        save_period = 10,           # save checkpoint every 10 epochs

        # Validation
        val         = True,
        plots       = True,         # save training curves

        # Class names override
        # nc=1 is set in dataset.yaml
    )

    print("\n✅ Training complete!")
    print(f"   Best weights → {OUTPUT_DIR}/finetune_llvip_exdark/weights/best.pt")
    print(f"   Last weights → {OUTPUT_DIR}/finetune_llvip_exdark/weights/last.pt")

    # Print final mAP
    print(f"\n   mAP50     : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95  : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == '__main__':
    main()