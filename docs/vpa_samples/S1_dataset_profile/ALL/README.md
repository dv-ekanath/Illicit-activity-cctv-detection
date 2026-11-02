# S1 · dataset_profile  (ALL)

Dataset-level profile of all UCF-Crime videos used: histograms of duration, brightness, sharpness, noise, bit rate, resolution and FPS per class.
Why: these numbers set the thresholds used later (dark threshold, blur floor) so they are not guesses.
Notice: dark threshold chosen = 50.0 (darkest 5% of frames; Otsu would say 76); all videos are [30.0] FPS, ['320x240'].

Files: compare.png, metrics.json, output.png, params.json, plot.png
