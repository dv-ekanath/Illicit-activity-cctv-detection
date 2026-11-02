# S9 · clip_package  (Vandalism032_x264)

The final clip is stored as uint8 (16,224,224,3) RGB .npy (4x smaller than float32, lossless since pixels began as 8-bit) with a JSON sidecar of shot id, motion/people stats, crop boxes, brightness and enhancement flags.
Why: ablations and error analysis need to know exactly how each clip was made; the legacy loader expects float32 [0,1] (needs /255 once uint8 clips are used for training).
Notice: left = the saved clip (8 of 16 frames), plot.png = the sidecar; metrics.json has the size comparison.

Files: compare.png, input.png, metrics.json, params.json, plot.png
