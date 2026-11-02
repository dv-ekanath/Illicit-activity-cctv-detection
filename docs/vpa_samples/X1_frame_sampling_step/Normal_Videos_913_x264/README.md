# X1 · frame_sampling_step  (Normal_Videos_913_x264)

EXISTING op (extract_clips._load_frames): keep every round(src_fps/12)-th decoded frame; 30 FPS -> step 2 -> 15.0 FPS.
Why it is a gap: motion speed then differs from video to video and a stride-4 clip no longer spans ~5.3 s.
Notice: kept frames are evenly spaced but at the wrong rate; replaced by S2 (timestamp-based).

Files: compare.png, input.png, metrics.json, output.png, params.json, plot.png
