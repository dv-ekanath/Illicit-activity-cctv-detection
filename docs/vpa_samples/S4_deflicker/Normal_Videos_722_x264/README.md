# S4 · deflicker  (Normal_Videos_722_x264)

Temporal deflicker: Zero-DCE++ curve parameters are EMA-smoothed across frames and the frame brightness is smoothed (zero-phase EMA).
Why: per-frame enhancement makes brightness jump; the motion stages and VideoMAE would read that flicker as motion.
Notice: the blue brightness curve in plot.png is smoother than the orange (plain Zero-DCE++); flicker index in the legend.

Files: compare.png, input.png, metrics.json, output.png, params.json, plot.png
