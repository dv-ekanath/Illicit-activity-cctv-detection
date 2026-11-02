# S4 · enhancement_comparison  (Normal_Videos_913_x264)

4-way comparison: raw vs CLAHE (baseline only) vs Zero-DCE++ vs Zero-DCE++ with temporal denoising and deflickering.
Why: brightening dark footage also brightens its noise and per-frame curves flicker; the extra steps fix both.
Notice: NIQE/BRISQUE (lower = better), flicker index (brightness change between frames) and YOLO persons/frame under each panel.

Files: compare.png, input.gif, input.png, metrics.json, output.gif, output.png, params.json
