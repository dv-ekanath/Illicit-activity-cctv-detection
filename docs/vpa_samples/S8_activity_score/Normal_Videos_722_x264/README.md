# S8 · activity_score  (Normal_Videos_722_x264)

Per-frame activity score = weighted motion energy + foreground ratio + interaction signal, smoothed (Savitzky-Golay); hysteresis thresholds and peaks give action proposals.
Why: replaces blind uniform sampling - clips of abnormal classes are taken where something actually happens (temporal action proposals).
Notice: green bands = proposed segments, thick red bar at the bottom = UCF-Crime ground truth (test videos only); tIoU in the title.

Files: compare.png, input.png, metrics.json, params.json, plot.png
