# S9 · dataset_report  (ALL)

Per-class distributions of motion energy, flow entropy, people count, brightness, activity and foreground ratio over the packaged clips, plus how many candidates each quality-gate rule rejected and the dedup effect.
Why: shows that the processed dataset differs by class in the expected ways and makes ablations / error analysis possible later.
Notice: abnormal classes should show higher motion energy / activity than Normal; the sample set has only 8 videos so the boxes are small.

Files: compare.png, metrics.json, params.json
