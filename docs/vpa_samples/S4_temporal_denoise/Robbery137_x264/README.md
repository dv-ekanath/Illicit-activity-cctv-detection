# S4 · temporal_denoise  (Robbery137_x264)

Multi-frame (temporal) denoising of the enhanced frames (FastDVDnet pretrained, else cv2 fastNlMeansMulti).
Why: Zero-DCE++ amplifies sensor noise in dark footage; noise differs between frames while the scene does not, so neighbouring frames help.
Notice: grain in the zoomed crop (left) is reduced on the right; compare noise_sigma in metrics.json.

Files: compare.png, input.gif, input.png, metrics.json, output.gif, output.png, params.json
