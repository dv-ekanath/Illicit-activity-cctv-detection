# 🔍 Real-Time Illicit Activity Detection System

A research-grade surveillance AI prototype that detects illicit activity (fighting, robbery, anomalies) in real time using CCTV footage, with bounding box overlays, confidence scoring, and an alerting pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0_nightly-EE4C2C?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Model Stack](#model-stack)
- [Datasets](#datasets)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Evaluation Metrics](#evaluation-metrics)

---

## Overview

This system processes live CCTV streams and classifies activity into four categories:

| Class | Description |
|-------|-------------|
| `Fighting` | Physical altercations between persons |
| `Robbery` | Theft or armed robbery events |
| `Vandalism` | Property destruction or damage |
| `Normal` | No anomalous activity |

Key features:
- **Low-light enhancement** via Zero-DCE++ — two-stage trained (visual quality + detection-aware), triggered when frame brightness < 50
- **Person detection** via fine-tuned YOLOv11s (mAP50: 0.907)
- **Person tracking** via ByteTrack (persistent IDs across frames)
- **Action recognition** via VideoMAE fine-tuned on UCF-Crime with 4-fold CV (Macro F1: 70.21% ± 5.64%)
- **Alert pipeline** with 30-second auto-clipped evidence videos uploaded to Supabase Storage
- **SMS/MMS alerts** via Twilio with video evidence attached
- **Live dashboard** via React + Supabase Realtime

---

## Pipeline

```
CCTV Stream (RTSP / mp4 / webcam)
    ↓
Frame Extraction @ 12 FPS (OpenCV)
    ↓
Zero-DCE++ Low-Light Enhancement  ← triggered if brightness < 50
    ↓                                 uses stage2_best.pth (detection-aware)
YOLOv11s Object Detection         ← person bounding boxes
    ↓
Annotated Frame → WebSocket → React Frontend (live view)
    ↓
Person detected ≥ 3 consecutive frames → fill 16-frame buffer
    ↓
Redis Stream (XADD)               ← 16-frame clip + 30s context + camera metadata
    ↓
VideoMAE Worker (XREAD)           ← polls Redis continuously
    ↓
VideoMAE Inference                ← Fighting / Robbery / Vandalism / Normal
    ↓
Confidence ≥ 0.40 + class ≠ Normal
    ↓
    ├── Supabase Storage           ← upload 30s evidence clip (.mp4)
    ├── Supabase PostgreSQL        ← insert incident row (camera, class, confidence, timestamp, clip_url)
    ├── Supabase Realtime          ← broadcast to frontend dashboard
    └── Twilio MMS                 ← SMS alert + video evidence to phone
```

---

### 🔦 Zero-DCE++ Two-Stage Training

Zero-DCE++ is trained in two sequential stages before being used in the pipeline. This is an **offline training process** — the resulting weights (`stage2_best.pth`) are frozen and loaded at inference time.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 — Visual Quality Training  (ExDARK, 200 epochs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ExDARK Dark Images (7,363 images, 12 classes)
    ↓
Zero-DCE++ (randomly initialized)
    ↓
L_total = 1.0·L_spa + 10.0·L_exp + 0.5·L_col + 200.0·L_tv
    ↓
Learns to produce visually bright, color-balanced,
noise-free enhanced images
    ↓
Saved → models/zerodce/stage1_best.pth  (val loss: 0.3348)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2 — Detection-Aware Fine-Tuning  (LLVIP, 100 epochs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLVIP Night Images (15,488 visible images)
    ↓
Zero-DCE++ (loaded from stage1_best.pth)
    ↓
Enhanced Image
    ↓              ↓
Enhancement    Frozen YOLOv11s (fine-tuned teacher)
Losses         ← domain-adapted to LLVIP/ExDARK
    ↓              ↓
L_enh          L_det (detection confidence signal)
    ↓              ↓
L_total = L_enh + λ · L_det

λ warmup schedule:
  Epochs  1–20  →  λ = 0.05  (stabilize first)
  Epochs 21–50  →  λ = 0.10  (gradually increase)
  Epochs 51–100 →  λ = 0.20  (full detection signal)

Zero-DCE++ learns to enhance images specifically to
maximize YOLOv11s person detection confidence.
YOLOv11s weights remain FROZEN throughout.
    ↓
Saved → models/zerodce/stage2_best.pth  (val loss: 0.1233)
                ↑
        ← USED IN PIPELINE
```

**Why two stages?**
- Stage 1 ensures Zero-DCE++ first learns what a good enhanced image looks like, establishing a stable base.
- Stage 2 then refines that enhancement objective toward detection utility — without Stage 1, the detection loss alone would destabilize training early on.
- The λ warmup in Stage 2 prevents the detection signal from overwhelming enhancement quality before the model has stabilized.

---

## Project Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── ucf_crime/                    # UCF-Crime videos + Action_Recognition_splits
│   │   ├── rwf2000/                      # RWF-2000 fight/nonfight videos
│   │   ├── shanghaitech/                 # ShanghaiTech campus videos
│   │   ├── exdark/                       # ExDARK low-light images
│   │   └── llvip/                        # LLVIP visible images + XML annotations
│   └── processed/
│       ├── clips/                        # (16, 224, 224, 3) float32 .npy clips
│       │   └── fold_N/train|test/ClassName/scene/*.npy
│       └── annotations_yolo/             # Converted YOLO format labels
│           ├── llvip/
│           └── exdark/
├── models/
│   ├── zerodce/
│   │   ├── stage1_best.pth               # ExDARK visual quality stage
│   │   └── stage2_best.pth               # LLVIP detection-aware stage ← pipeline
│   ├── yolov11/
│   │   ├── finetune_llvip_exdark/weights/best.pt
│   │   └── best.onnx                     # ONNX export ← pipeline
│   └── videomae/
│       ├── fold_1/videomae_best.pth
│       ├── fold_2/videomae_best.pth
│       ├── fold_3/videomae_best.pth
│       ├── fold_4/videomae_best.pth
│       └── pretrained/                   # MCG-NJU/videomae-base-finetuned-kinetics
├── src/
│   ├── preprocessing/
│   │   ├── extract_clips.py              # VPA pipeline driver (S1-S9, per-stage on/off flags)
│   │   ├── vpa/                          # video processing & analytics modules + config.yaml (see docs/vpa_samples/)
│   │   ├── convert_llvip_to_yolo.py
│   │   ├── convert_exdark_to_yolo.py
│   │   ├── build_dataset_yaml.py
│   │   └── verify_annotations.py
│   ├── training/
│   │   ├── zerodce_model.py
│   │   ├── zerodce_losses.py
│   │   ├── zerodce_dataset.py
│   │   ├── train_zerodce.py
│   │   ├── train_yolov11.py
│   │   ├── videomae_dataset.py           # ClipDataset + augmentation pipeline
│   │   ├── train_videomae.py             # 4-fold CV training + SWA
│   │   └── evaluate_tta.py               # Leak-free TTA evaluation
│   ├── inference/
│   │   ├── infer_zerodce.py
│   │   └── infer_yolov11.py
│   └── pipeline/                         # (empty: the planned tracker.py was never committed;
│                                         #  ByteTrack now lives in src/preprocessing/vpa/actors.py)
├── backend/
│   ├── app.py                            # FastAPI + WebSocket stream server
│   ├── worker_videomae.py                # Redis XREAD → VideoMAE inference worker
│   └── incident_handler.py              # Supabase upload + Twilio alert
├── frontend/                             # React dashboard
│   └── src/
│       └── App.jsx                       # Live feed + Supabase Realtime incident log
├── logs/
│   ├── zerodce/training_log.txt
│   └── videomae/training_log_fold{N}_v14.txt
└── outputs/
    ├── alerts/                           # Incident metadata logs
    └── clips/                            # 30s evidence .mp4 files
```

---

## Model Stack

### 1. Zero-DCE++ — Low-Light Enhancement
- **Type:** Unsupervised fine-tuning
- **Architecture:** 8-iteration curve estimation (79,416 params)
- **Training data:** ExDARK (Stage 1) → LLVIP visible (Stage 2)
- **Loss:** L_spa + L_exp + L_col + L_tv + λ·L_det (detection-aware)
- **Framework:** PyTorch
- **Trigger:** Only when mean frame brightness < threshold

### 2. YOLOv11s — Object Detection
- **Type:** Supervised fine-tuning on COCO pretrained base
- **Classes:** person, bag, weapon
- **Training data:** LLVIP (12,025) + ExDARK People (609) = 12,634 images
- **Input size:** 640×640
- **Framework:** Ultralytics
- **Export:** ONNX for inference

### 3. ByteTrack — Person Tracking
- **Status:** implemented in `src/preprocessing/vpa/actors.py` (ultralytics `BYTETracker`) as part of the pre-training VPA pipeline; the live backend integration (`src/pipeline/tracker.py`) was never committed
- **Type:** Plug-and-play (no training required)
- **Input:** YOLOv11 bounding boxes
- **Output:** Persistent person IDs across frames

### 4. VideoMAE — Action Recognition
- **Type:** Supervised fine-tuning with 4-fold cross-validation
- **Base:** MCG-NJU/videomae-base-finetuned-kinetics (86M parameters)
- **Training data:** UCF-Crime — Fighting, Robbery, Vandalism, Normal (official 4-fold ActionRecognition splits)
- **Input:** (16, 224, 224, 3) float32 clips, stride-4 sampled at 12 FPS
- **Clip extraction:** ZeroDCE enhancement → annotation-guided / segmented / sliding-window strategy per class → motion energy filter → cosine similarity deduplication → disk augmentation (3× for Vandalism/Normal)
- **Augmentation (train):** speed perturbation, temporal jitter, random crop 192→224, horizontal flip, color jitter, Gaussian noise, random grayscale
- **Loss:** CrossEntropyLoss (label smoothing=0.15) + inverse-frequency class weights
- **Optimizer:** AdamW — head LR 6e-5, backbone LR 3e-6, WD 0.08
- **Scheduler:** ManualCosineScheduler → ReduceLROnPlateau after SWA
- **Progressive unfreeze:** 5 stages at epochs 1, 4, 8, 13, 18
- **Dropout schedule:** 0.5 (ep 1–8) → 0.4 (ep 9–14) → 0.3 (ep 15+)
- **MixUp:** alpha=0.8, 50% of batches, from epoch 3
- **SWA:** Stochastic Weight Averaging from epoch 14
- **Result: Macro F1 = 70.21% ± 5.64% across 4 folds**

---

## Datasets

| Dataset | Purpose | Size | Location |
|---------|---------|------|----------|
| UCF-Crime | Primary crime classification (4-fold CV) | ~24.5GB | `data/raw/ucf_crime/` |
| ExDARK | Zero-DCE++ Stage 1 + YOLOv11 detection | ~1.5GB | `data/raw/exdark/` |
| LLVIP (visible) | Zero-DCE++ Stage 2 + YOLOv11 fine-tuning | ~13GB | `data/raw/llvip/` |

**Total storage:** ~46GB

---

## Results

### Zero-DCE++ (Stage 2 — pipeline model)
| Metric | Value |
|--------|-------|
| Enhancement Loss | 2.39 → 0.09 (converged) |
| Val Loss | 0.1233 |
| Checkpoint | `models/zerodce/stage2_best.pth` |

### YOLOv11s
| Metric | Value |
|--------|-------|
| mAP50 | **0.907** ✅ |
| mAP50-95 | **0.534** ✅ |
| Precision | 0.869 |
| Recall | 0.850 |
| Inference Speed | 2.0 ms/frame |
| Checkpoint | `models/yolov11/best.onnx` |

### VideoMAE — 4-Fold Cross-Validation (v14)
| Class | Macro F1 |
|-------|----------|
| Fighting | 63.77% ± 10.48 |
| Robbery | 70.06% ± 7.10 |
| Vandalism | 68.73% ± 9.32 |
| Normal | 78.29% ± 3.58 |
| **Overall Macro F1** | **70.21% ± 5.64%** ✅ |

---

## Setup

### Prerequisites
- Windows OS, NVIDIA GPU (8GB+ VRAM recommended)
- Python 3.11
- CUDA 12.8+
- Conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/illicit-detection.git
cd illicit-detection

# Create and activate conda environment
conda create -n illicit_detect python=3.11
conda activate illicit_detect

# Install PyTorch (nightly with CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Follow the download instructions in [`docs/dataset_setup.md`](docs/dataset_setup.md) for UCF-Crime, RWF-2000, ShanghaiTech, ExDARK, and LLVIP.

---

## Usage

### Annotation Conversion
```bash
# LLVIP XML → YOLO
python src/preprocessing/convert_llvip_to_yolo.py

# ExDARK bbGt → YOLO (People class only)
python src/preprocessing/convert_exdark_to_yolo.py

# Build unified dataset YAML
python src/preprocessing/build_dataset_yaml.py
```

### Training

```bash
# Stage 1: Zero-DCE++ on ExDARK
python src/training/train_zerodce.py --stage 1

# Stage 2: Detection-aware fine-tuning on LLVIP
python src/training/train_zerodce.py --stage 2

# YOLOv11s fine-tuning
python src/training/train_yolov11.py

# Extract clips for all 4 folds with the VPA pipeline (S1-S9; see "Video processing & analytics" below)
python src/preprocessing/extract_clips.py --all-folds

# VideoMAE 4-fold CV training
python src/training/train_videomae.py --all-folds --epochs 20

# Single fold
python src/training/train_videomae.py --fold 2 --epochs 20

# Print CV summary from saved checkpoints (no retraining)
python src/training/train_videomae.py --summarise-only
```

### Evaluation

```bash
# Leak-free TTA evaluation (all 4 folds, recommended)
python src/training/evaluate_tta.py

# Single fold TTA
python src/training/evaluate_tta.py --fold 1

# Reproduce training eval exactly (sanity check, no TTA)
python src/training/evaluate_tta.py --no-tta
```

### Inference

```bash
# Zero-DCE++ inference
python src/inference/infer_zerodce.py --input path/to/video

# YOLOv11s inference
python src/inference/infer_yolov11.py --input path/to/video
```

---

## Video processing & analytics (pre-training pipeline)

Everything that happens to a video **before** deep-learning training is explicit, measurable and documented
(`docs/vpa_samples/`): profiling (S1), exact-12-FPS resampling (S2), structural cleaning (S3), enhancement (S4),
motion analysis (S6), actor analysis (S7), temporal analytics / clip selection (S8) and packaging (S9);
S5 (camera compensation) is skipped because only 2 of 300 videos have a moving camera.

```bash
# sample run: 8 fixed videos, saves an input/output sample of EVERY operation + docs/vpa_samples/INDEX.md
python src/preprocessing/extract_clips.py --sample-set --save-samples
# switch stages on/off
python src/preprocessing/extract_clips.py --sample-set --skip s7
# walkthrough notebook (one video, S1 -> S9)
jupyter notebook notebooks/vpa_walkthrough.ipynb
```
Settings live in `src/preprocessing/vpa/config.yaml`; extra dependencies in `requirements-vpa.txt`;
the summary, decisions and TODOs are in `docs/vpa_samples/{SUMMARY,DECISIONS,PROGRESS,SETUP_REPORT}.md`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn (Python) |
| Stream Handling | OpenCV |
| ML Inference | PyTorch / ONNX Runtime |
| Message Queue | Redis Streams (XADD / XREAD) |
| Database | Supabase PostgreSQL |
| Video Storage | Supabase Storage |
| Realtime Events | Supabase Realtime |
| Alerting | Twilio SMS / MMS |
| Frontend | React + WebSockets + Supabase Realtime |

---

## Roadmap

| Week | Task | Status |
|------|------|--------|
| 1 | Environment setup, dataset download, preprocessing | ✅ Done |
| 2 | Zero-DCE++ two-stage training + YOLOv11 fine-tuning | ✅ Done |
| 3 | ByteTrack integration + clip extraction pipeline | ✅ Done |
| 4 | VideoMAE 4-fold CV training + TTA evaluation | ✅ Done |
| 5 | Full pipeline integration (FastAPI + Redis + VideoMAE worker) | ✅ Done |
| 6 | Alert pipeline — Supabase Storage + PostgreSQL + Twilio MMS | ✅ Done |
| 7 | React dashboard + Supabase Realtime incident feed | ✅ Done |
| 8 | End-to-end evaluation (mAP, F1, latency) + report writing | ⏳ Pending |

---

## Evaluation Metrics

- **mAP** — YOLOv11 object detection
- **F1 Score + Confusion Matrix** — VideoMAE V2 classification
- **AUC-ROC** — CLIP anomaly head
- **Inference Latency** — ms/frame end-to-end
- **False Positive Rate** — alert pipeline

---

## Alert Logic

- Alert triggers if confidence score > **0.75** for **10+ consecutive frames**
- Auto-clips **30 seconds** of evidence → `outputs/clips/`
- Logs all incidents to PostgreSQL with camera ID, timestamp, confidence
- Cooldown period enforced between alerts to prevent duplicates

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5060 (8GB VRAM) |
| OS | Windows |
| Python | 3.11 |
| PyTorch | 2.12.0 nightly |
| CUDA | 12.8 (on 13.2 driver) |
| Conda env | `illicit_detect` |

---

## Acknowledgements

- [UCF-Crime Dataset](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) — University of Central Florida
- [RWF-2000 Dataset](https://huggingface.co/datasets/DanJoshua/RWF-2000)
- [ShanghaiTech Campus Dataset](https://svip-lab.github.io/dataset/campus_dataset.html)
- [ExDARK Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- [LLVIP Dataset](https://github.com/bupt-ai-cz/LLVIP)
- [VideoMAE V2](https://huggingface.co/MCG-NJU/videomae-base) — MCG-NJU
- [Zero-DCE++](https://li-chongyi.github.io/Proj_Zero-DCE++.html)
- [YOLOv11](https://github.com/ultralytics/ultralytics) — Ultralytics
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [CLIP](https://github.com/openai/CLIP) — OpenAI