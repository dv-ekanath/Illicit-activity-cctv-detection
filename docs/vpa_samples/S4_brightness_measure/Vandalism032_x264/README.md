# S4 · brightness_measure  (Vandalism032_x264)

One brightness measure (mean luma Y) and one threshold (from S1) decide, with temporal smoothing and hysteresis, which frames are enhanced.
Why: the old code used grayscale mean < 50 (extract_clips), < 80 (infer_zerodce) and HSV-V (backend) - three different answers for the same frame.
Notice: yellow bands in plot.png = frames sent to Zero-DCE++ (none for bright videos).

Files: compare.png, input.png, metrics.json, params.json, plot.png
