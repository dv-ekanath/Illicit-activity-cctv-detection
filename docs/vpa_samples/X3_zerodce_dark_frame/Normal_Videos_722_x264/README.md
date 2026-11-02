# X3 · zerodce_dark_frame  (Normal_Videos_722_x264)

EXISTING op: Zero-DCE++ (stage-2 weights) brightens the darkest frame of the video (luma 11); trigger was grayscale-mean < 50.
Why: low-light CCTV hides people from YOLO/VideoMAE.
Notice: shadows lift while highlights are preserved; if the video is bright the demo forces the enhancer on.

Files: compare.png, input.png, metrics.json, output.png, params.json
