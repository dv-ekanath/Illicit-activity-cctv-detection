# S3 · frozen_black_frames  (Normal_Videos_722_x264)

Flags duplicated (frame-diff ~ 0) and blank frames on the 12 FPS timeline.
Why: they look like "no motion" and would contaminate clips; the quality gate (S8) rejects clips that contain many.
Notice: shaded spans on the timeline are flagged frames (none = clean video).

Files: compare.png, input.png, metrics.json, output.png, params.json, plot.png
