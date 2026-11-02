# S7 · actor_centred_crop  (Vandalism032_x264)

Crops the union of the interacting people plus a margin, smooths the window over time, letterboxes it and resizes to 224x224 with INTER_AREA.
Why: in 320x240 CCTV people are tiny; shrinking the whole frame to 224 throws away the detail the action model needs.
Notice: right = the actual model input; jitter_px (metrics) drops after smoothing; with nobody detected the full frame is used.

Files: compare.png, input.gif, input.png, metrics.json, output.gif, output.png, params.json
