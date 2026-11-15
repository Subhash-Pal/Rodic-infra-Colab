# RODIC V3 - sensorless long-range edge pipeline

One 4K RGB camera, no LiDAR / radar, 5-15 m stand-off (bridges, dams, tunnels).
Reuses V2's recommender, PDF exporter and video sampler (`edge_drone_pipeline/_v2.py`).

```
frame -> frame_conditioner -> sahi_rtdetr_engine -> active_zoom_tracker -> sensorless_metrology
      -> analyzer (V2 contract) -> irc_recommender -> vlm_arbitration_bridge (optional)
```

| Module | What it does |
|---|---|
| `frame_conditioner.py` | Blur gate (Laplacian variance at width 960), exposure gate, LAB-CLAHE |
| `sahi_rtdetr_engine.py` | 640x640 tiles, 20 % overlap, batched detection, non-maximum merging back to 4K coordinates |
| `active_zoom_tracker.py` | IoU tracker, zoom only after 3 consecutive frames; Mode A gimbal pixel-offset to yaw / pitch (pinhole maths), Mode B padded ROI crop + unsharp |
| `sensorless_metrology.py` | GSD = D x p / f (or from a reference object), crack width from skeleton + distance transform |
| `vlm_arbitration_bridge.py` | Request builder, prompt, SQLite offline queue with flush; you inject the Gemini `sender` |
| `pipeline.py`, `batch_runner.py`, `run_batch.py` | Frame pipeline; resumable batch of images / video with one PDF |

`config.yaml` holds every parameter. All values are untested starting values.

```
python v3/run_batch.py --input /data/survey --out /data/out --detector rtdetr --weights best.pt --gsd-mm-per-px 0.8
python v3/run_batch.py --input /data/survey --out /data/out --detector heuristic   # demo only
```

## The limit to tell the jury

Width cannot be measured below about 2 pixels. With D = 10 m, p = 3.4 um, f = 8.8 mm the GSD is ~3.9 mm/px: a 0.2 mm hairline is far sub-pixel and a 2 mm crack is half a pixel. At 10 m only cracks wider than ~8 mm can be sized. To size 0.2 mm you need ~0.1 mm/px (closer, longer focal length, optical zoom). Below the limit the code reports an upper bound and `resolvable=False`, and the recommender marks severity provisional. Example camera numbers are placeholders; use the real datasheet.

## Verified vs not verified

- Tested on synthetic data: tiling covers every 4K pixel, merge logic, tracker gating (once, after 3 consecutive frames), gimbal maths (e.g. 45 deg for a 500 px offset at 90 deg HFOV), GSD formulas, offline queue, batch resume / error isolation, PDF generation.
- Measured width on synthetic diagonal cracks came out about 10-15 % BELOW the rendered width (area / length). Underestimating is the unsafe direction. No real-image validation yet; calibrate against crack-comparator ground truth before quoting accuracy.
- Known: NMM can leave long thin crack fragments unmerged when the overlap strip is small (documented by a test).
- Not implemented: Depth-Anything-V2 (needs a metric model; plug in via `DistanceEstimator`), ESRGAN-Lite, ByteTrack / OC-SORT (a simple IoU tracker stands in), TensorRT export, MAVLink / SDK gimbal control (`LoggingGimbal` only records), the Gemini call itself.
- `<80 ms per cycle on Jetson Orin` is a design target, not a result. `summary.json` records per-stage timings of whatever machine ran the batch.
- Temporal gating needs consecutive frames, so video uses `video.stride_s` (default 0.2 s), not the 2 s used by V2.
