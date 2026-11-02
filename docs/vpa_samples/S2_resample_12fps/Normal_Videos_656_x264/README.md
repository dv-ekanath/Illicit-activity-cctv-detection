# S2 · resample_12fps  (Normal_Videos_656_x264)

Resamples to exactly 12 FPS by picking the source frame nearest to each t = k/12 s (OLD: every round(30/12)-th frame = 15.0 FPS).
Why: VideoMAE stride 4 only spans the intended ~5.3 s if every video really is 12 FPS; otherwise motion speed differs per video.
Notice: plot.png — old kept frames are evenly 2 source frames apart (15 FPS), new alternate 2/3 apart (12 FPS).

Files: compare.png, input.png, metrics.json, output.png, params.json, plot.png
