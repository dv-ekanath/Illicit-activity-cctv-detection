# VPA pre-training pipeline — summary (2026-10-07/08)

Everything requested for today is built, run on the 8-video sample set, committed locally (no push), and documented.
No model was trained; `src/training/` and `backend/` (which does not exist in this checkout) were not touched.

## What was built

| Stage | Module (`src/preprocessing/vpa/`) | Operations (each has an input/output sample in `docs/vpa_samples/`, see `INDEX.md`) |
|---|---|---|
| S1 | `profile.py` | PyAV metadata, I/P/B shares, brightness / sharpness / noise / blockiness, camera-motion, bars + overlay scores for all 300 videos (8 min, 8 workers); data-driven dark threshold (**50**) and blur floor; fixed **sample set** (`sample_set.json`, rule in `DECISIONS.md`) |
| S2 | `decode.py` | timestamp-based resampling to exactly **12.00 FPS** (the old rule gave 15 FPS on these 30-FPS videos) |
| S3 | `clean.py` | letterbox-bar crop, timestamp/logo overlay detection + Telea inpainting, shot cuts (HSV+edge-change **and TransNetV2**), frozen/black frames |
| S4 | `enhance.py` | one brightness measure (mean luma Y) + one threshold, Zero-DCE++ (existing weights) with EMA-smoothed curves, **FastDVDnet** temporal denoise (fastNlMeans fallback), zero-phase luma deflicker; raw / CLAHE / Zero-DCE++ / full-chain comparison with NIQE, BRISQUE, flicker index, noise sigma, YOLO persons |
| S5 | — | **skipped by rule**: 2/300 videos (0.7 %) have a moving camera (< 5 %); logged |
| S6 | `motion.py` | MOG2 → HSV shadow removal → morphology → connected components; Farneback (+ RAFT-small on sample pairs); direction entropy, divergence, curl; MHI; motion-energy signal |
| S7 | `actors.py` | person detection (**COCO yolo11s chosen over the fine-tuned best.pt**, evidence in `S7_detector_check`), ByteTrack IDs/trajectories, interaction signals, smoothed actor-centred 224 crop |
| S8 | `temporal.py` | activity score (Savitzky–Golay), hysteresis/peak action proposals + tIoU vs UCF-Crime ground truth, motion-guided clip sampling (Normal: uniform + hard negatives), quality gate, original dedup kept |
| S9 | `package.py` | uint8 `(16,224,224,3)` clips + JSON sidecar per clip, per-class dataset report |
| — | `common.py`, `splits.py`, `viz.py`, `legacy.py`, `runner.py`, `config.yaml` | shared brightness function / letterbox, **official-split loader + leakage check (moved, unchanged)**, `save_io_sample` + INDEX generator, re-creation of the existing ops, per-video orchestration, all thresholds |

Also: `extract_clips.py` is now the driver (`--sample-set`, `--save-samples`, `--stages/--skip`, `--all-folds`, `--out-dir`, `--clip-dtype` …),
`notebooks/vpa_walkthrough.ipynb` (executed end to end, 0 errors, 20 inline figures), `requirements-vpa.txt`, README fixes (tracker line, usage path).
**Existing operations re-created with samples:** old step sampling (X1), stretch-vs-letterbox (X2), Zero-DCE++ dark frame (X3), dedup (X4), old `std>0.01` vs new gate (X5), YOLO annotation boxes (X6).
35 operations × 8 videos (+ dataset-level ones) = 797 PNG / 160 GIF samples, ~100 MB.

## Results worth showing (sample set)

- **12 FPS fix**: old sampling = 15.0 FPS, new = 12.00 FPS (source-frame gaps 2/3/2/3, max timing error 16.7 ms).
- **Hidden data property found**: many videos are 10–15 *unique* FPS stored at 30 FPS, so naive 12-FPS resampling repeats frames and made motion energy a 0/high sawtooth → flow is now measured against the last distinct frame.
- **S3**: 206/300 videos have letterbox bars, 218/300 an overlay blob; 3 cuts found in `Vandalism038`, 5 in `Normal_Videos_722`; 68 frozen frames in `Fighting018` (title card).
- **S4** (on the 8 demo windows; 7 use clearly labelled *simulated* low light, 1 is real dark footage): Zero-DCE++ amplifies noise (σ 2.7→10.2 in one example) and the temporal denoise lowers it again in **8/8** (10.2→6.2); NIQE improves with the full chain vs Zero-DCE++ alone in **7/8**; the flicker index improves in **5/8** (it is already tiny in static dark scenes and slightly worse in 3, e.g. 0.019→0.042 on the real dark video). CLAHE (baseline only) amplifies noise most.
- **S7 finding**: the fine-tuned YOLO (mAP50 0.907 on LLVIP) does **not transfer** to UCF-Crime frames — 34 % of its boxes are frame-sized, only 14 % are confirmed by COCO — so COCO `yolo11s.pt` is used (still no training).
- **S8**: tIoU of the proposals with the ground truth — Robbery137 **0.96**, Fighting018 0.35, Vandalism017 0.15 (recall 1.0 in all three; precision is limited because motion ≠ crime; thresholds were tuned on only these 3 videos). The new gate rejects frozen/low-motion clips that the old `std>0.01` check accepts (X5, Fighting018).
- **S9**: clips are 2.4 MB (uint8) instead of 9.6 MB (float32).

## TODO / limitations (all logged with reasons in `DECISIONS.md`)

1. **Training loader needs one line**: `videomae_dataset.py` expects float32 in [0,1] (no `/255`). Either divide by 255 there, or extract with `--clip-dtype float32`. (Training code untouched today.)
2. **S5 camera compensation not built** (2 videos); `Normal_Videos_913` shows the failure (whole frame "moves"). Cheap to add later.
3. **EAST / PaddleOCR** overlay detector, **DOVER / FAST-VQA**, **pose**, **NeuFlow/SEA-RAFT**: not done (classical / alternative used). TransNetV2 runs on **CPU** (Conv3D failed on the GPU with this torch nightly).
4. **`infer_zerodce.py` (threshold 80) and the backend (HSV-V) are not changed**; they should call `vpa.common.is_dark` with threshold 50. The backend folder is absent from this checkout, so training/inference parity could not be verified.
5. The overlay detector over-detects on textured static borders (harmless inpainting of a little real content); activity-score weights/thresholds were hand-set from three annotated videos.
6. Sample runs analyse ≤ 300 s per video (all 8 are shorter); the full run has no cap.
7. Samples are all fold-1 *train* videos; the dataset report of the sample set (106 clips, 8 videos) is only a demonstration of the report.

## Full-dataset extraction (NOT run today)

```bash
conda activate illicit_detect      # or: C:\Users\Admin\anaconda3\envs\illicit_detect\python.exe
cd "E:\Deep Learning Project (Dont Delete)\illicit_detection"
python src/preprocessing/extract_clips.py --all-folds            # clips -> data/processed/clips_vpa/fold_N/{train,test}/<Label>/scene/
# optional: --save-samples (adds samples, slower) | --skip s7,s4 | --clip-dtype float32 | --class Fighting | --fold 1
```
- **Scope**: the 4 official folds contain the same 200 videos (6.6 h of video, 23,800 s); each video is analysed **once** (S2–S7) and S8/S9 run for each fold it belongs to (train 16 clips, test 8), ~4× cheaper than the old per-fold loop.
- **Time**: measured 0.37 s of compute per second of video on the sample set (RTX 5060 + i5-14600K) + ~0.04 s for TransNetV2 on CPU → **≈ 2.5–3.5 hours** total. Peak RAM ≈ 3 GB for the longest (1055 s) video.
- **Disk**: uint8 clips ≈ 2.4 MB each; ≈ 2.5 k clips per fold → **≈ 6 GB per fold, ≈ 24 GB for 4 folds** (+ ~40 MB JSON sidecars); float32 would be ≈ 96 GB. Free space now: ≈ 70 GB.
- Models downloaded on first use: RAFT-small (4 MB), FastDVDnet (10 MB), NIQE/BRISQUE weights; TransNetV2 ships in the pip wheel. `outputs/vpa_cache/` holds the S1 profile cache, thresholds and detector choice.

## Commits (local, `illicit_detection/`)

S1 → S2+S3 → S6 → S4 → S7 → S8+S9 (+ final: TransNetV2, full-run driver, notebook, docs). See `git log`.
