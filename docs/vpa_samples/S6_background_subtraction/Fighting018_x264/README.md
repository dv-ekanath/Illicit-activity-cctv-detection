# S6 · background_subtraction  (Fighting018_x264)

MOG2 foreground -> HSV shadow removal (shadow = darker V, same H and S) -> morphological opening/closing -> connected components.
Why: gives a clean foreground mask, blob count and foreground ratio (how much of the scene is moving).
Notice: raw mask (left) has shadow/speckle; later panels are cleaner; colours in the last panel are separate blobs.

Files: compare.png, input.gif, input.png, metrics.json, output.gif, output.png, params.json, plot.png
