# X4 · dedup  (Robbery137_x264)

EXISTING op (unchanged): motion-fingerprint deduplication drops a clip whose mean-abs-frame-difference image is more than 0.99 cosine-similar to an already kept clip.
Why: heavily overlapping clips from static CCTV add no information and bias training.
Notice: tiles show each candidate with KEEP/DROP and the highest similarity to an earlier kept clip; heat-map in plot.png.

Files: compare.png, input.png, metrics.json, output.png, params.json, plot.png
