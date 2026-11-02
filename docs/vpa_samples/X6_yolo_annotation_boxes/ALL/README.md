# X6 · yolo_annotation_boxes  (ALL)

EXISTING op: LLVIP XML / ExDARK bbGt annotations converted to YOLO txt, drawn back on the image to verify the conversion.
Why: a wrong conversion silently trains the person detector on shifted boxes.
Notice: green boxes sit tightly on people in the original images (top row = raw image, bottom row = boxes drawn).

Files: compare.png, input.png, metrics.json, output.png, params.json
