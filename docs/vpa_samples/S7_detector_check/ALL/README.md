# S7 · detector_check  (ALL)

Visual + numeric check of the existing fine-tuned YOLOv11s against COCO yolo11s.pt on enhanced UCF-Crime frames (both without any training); chosen detector: coco.
Why: the fine-tuned model was trained on night-time LLVIP/ExDARK images, UCF-Crime CCTV is another domain; there is no person ground truth, so cross-confirmation and box plausibility are used.
Notice: each tile is the sampled frame with most people for that detector (green = fine-tuned, orange = COCO).

Files: compare.png, input.png, metrics.json, output.png, params.json, plot.png
