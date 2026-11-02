# VPA Setup Report

Written at the start of the autonomous build (2026-10-07). Everything below was verified on this machine.

## Repository
- Git repo root is `illicit_detection/` (branch `main`, clean tree, 4 commits). The parent folder `Deep Learning Project (Dont Delete)` is *not* a repo.
- `plan.md` is not on disk (it was supplied in the task message). `backend/` and `backend/utils.py` **do not exist** in this checkout, neither do `frontend/` or `src/pipeline/tracker.py` (`src/pipeline/` holds only `.gitkeep`). Backend brightness inconsistency is therefore taken from the task description, not verified.
- `.gitignore` ignores `*.png`, `*.mp4`, `data/`, `models/`, `outputs/` ...; I added negations so `docs/vpa_samples/**` and `notebooks/` are tracked.
- Existing code read: `src/preprocessing/*`, `src/inference/infer_zerodce.py`, `infer_yolov11.py`, `src/training/zerodce_model.py` (read-only, imported for the Zero-DCE++ network), `videomae_dataset.py` (read-only, to learn the clip format).
- `src/training/videomae_dataset.py` does `torch.from_numpy(np.load(p)).float()` with **no /255**, i.e. it expects float32 in [0,1]. The plan's uint8 clip storage needs a one-line loader change before training (out of scope today, logged in DECISIONS).

## Data
| Item | Finding |
|---|---|
| `data/raw/ucf_crime/videos/` | Fighting 50, Robbery 150, Vandalism 50, Normal_Videos_event 50 (300 videos) |
| Video format | **All 300 are 30 FPS, 320x240** (cv2 metadata) - so `step=round(30/12)=2` -> real 15 FPS bug affects every video |
| Durations (s, min/median/max) | Fighting 27/118/1055, Robbery 9/66/509, Vandalism 14/65/420, Normal 7/54/449 (~32k s total) |
| `Action_Recognition_splits/` | train_001..004, test_001..004 (532 train + 168 test lines per fold, includes non-target classes) |
| Temporal annotation | `annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt` (290 lines) **present**, plus `Temporal_Anomaly_Annotation.txt` in the ucf_crime root. Frame indices at 30 FPS. |
| LLVIP / ExDARK | present (`data/raw/llvip`, `data/raw/exdark`, converted YOLO labels in `data/processed/annotations_yolo`) |
| RWF-2000 | present (`data/raw/rwf2000/{train,test}`), not used |
| `data/processed/clips/` | does not exist (no clips extracted on this machine) |

## Models
- Zero-DCE++: `models/zerodce/stage2_best.pth` OK
- YOLOv11s fine-tuned: `models/yolov11/finetune_llvip_exdark/weights/best.pt` OK; COCO `models/yolov11/yolo11s.pt` also present (fallback detector)
- VideoMAE weights exist but are not touched.

## Compute / environment
- Env: conda `illicit_detect` (`C:\Users\Admin\anaconda3\envs\illicit_detect\python.exe`, Python 3.11.14), torch 2.12.0.dev+cu128, **CUDA available (RTX 5060 8 GB)**, OpenCV 4.13, numpy 2.4.2, ultralytics 8.4.21.
- CPU i5-14600K, 32 GB RAM, 74 GB free on E:.
- No `ffprobe`/`ffmpeg` binaries -> S1 metadata is read with **PyAV** instead (equivalent data).
- Installed today: `av, scikit-image, scenedetect, imageio, nbconvert, nbformat, pyiqa` (torch/numpy/opencv untouched). See `requirements-vpa.txt`.
- Downloads reachable: torchvision RAFT weights, FastDVDnet repo files, `transnetv2-pytorch` wheel.

## Implementation plan
Order S1 -> S2 -> S3 -> S6 -> S4 -> S7 -> S8 -> S9 as specified. Shared helpers in `vpa/common.py` (config, paths, brightness, letterbox), `vpa/splits.py` (official-split loader + leakage check moved from `extract_clips.py`), `vpa/viz.py` (sample saving), `vpa/runner.py` (per-video orchestration, used by both `extract_clips.py` and the notebook). Settings in `vpa/config.yaml`. After every stage: run on the sample set, look at the images, `git commit`, append to `PROGRESS.md`. All decisions/fallbacks go to `DECISIONS.md`.

## Outcome (added at the end of the build)
- Installed in the `illicit_detect` env: `av, scikit-image, scenedetect, imageio, nbconvert, nbformat, pyiqa, transnetv2-pytorch (--no-deps)` (+ their small dependencies; torch / numpy / opencv were not changed). Models fetched on first use: RAFT-small (torchvision), FastDVDnet (`models/fastdvdnet/`, git-ignored), NIQE/BRISQUE weights (pyiqa cache).
- All planned items for the day are implemented or logged: see `SUMMARY.md` (what was built, TODOs, full-extraction command/time/disk) and `DECISIONS.md` (every choice with alternatives).
