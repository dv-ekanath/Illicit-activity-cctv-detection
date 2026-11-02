# S8 · quality_gate  (Normal_Videos_722_x264)

Each candidate clip is rejected if it is frozen, black, blurry (Laplacian variance below the S1-derived floor), has almost no motion although the class is abnormal, spans a shot cut, or sits in a fade/flash.
Why: replaces the old `clip.std() > 0.01` check, which cannot see these defects.
Notice: tiles = middle frame of rejected candidate clips with the reason(s) (candidates include old-style uniform ones so defects show up); table = rejections per rule.

Files: compare.png, metrics.json, output.png, params.json
