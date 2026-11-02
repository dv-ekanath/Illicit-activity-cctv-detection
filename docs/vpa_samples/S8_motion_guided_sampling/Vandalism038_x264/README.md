# S8 · motion_guided_sampling  (Vandalism038_x264)

Abnormal classes: clips are sampled from inside the proposed segments (farthest-point spread, never across a shot cut); Normal: uniform + high-motion hard negatives.
Why: UCF-Crime Fighting videos are mostly normal footage, so uniform sampling gives clips labelled Fighting that show no fighting (label noise).
Notice: in the timeline the bottom row compares old uniform starts (orange) with the new selected starts (green); left images = old picks, right = new clips.

Files: compare.png, input.png, metrics.json, output.png, params.json
