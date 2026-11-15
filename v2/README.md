# RODIC V2

Next enhancement of the demo app. Flat modules, run from this folder.

| File | Role |
|---|---|
| `video_edge_processor.py` | Sample video (mp4 / avi / webcam / RTSP) at a set interval, localize, 15-20 % padded auto-zoom crop, analyze, mark repeats |
| `localizer.py` | `OpenCVAnomalyLocalizer` (CPU heuristic, not validated) and `UltralyticsLocalizer` (YOLOv8n / RT-DETR, needs defect-trained weights) |
| `irc_recommender.py` | Severity class + repair guidance replacing `Engineering decision: not_provided` |
| `pdf_exporter.py` | One-click PDF inspection report (reportlab) |
| `pipeline_adapter.py` | Plug point for the existing CODEBRIM + SDNET2018 + Gemini pipeline |
| `app_v2.py` | Streamlit app: Static Image / Live Video Stream (Edge Auto-Zoom) |

```
pip install -r v2/requirements.txt
RODIC_ANALYZER=your_module:analyze_crop streamlit run v2/app_v2.py   # omit the variable to run with the stub
```

Analyzer contract: `analyzer(crop_bgr) -> {"defect_class", "confidence", "scores", "source"}`.

## Recommender: what it is

A lookup table. Crack severity bands (starting values, edit `CRACK_WIDTH_BANDS`): `<0.2 mm` cosmetic, `0.2-0.5` watch, `0.5-2.0` significant (epoxy injection), `>=2.0` urgent (structural audit). Unknown width gives a provisional "watch" and asks for a measurement. Referenced documents: IRC:SP:35, IRC:SP:40, MoRTH Specifications for Road and Bridge Works, with no clause numbers (not verified).

## Limits

- The Streamlit UI was compile-checked only; Streamlit was not run in the build environment.
- The OpenCV localizer finds dark thin structures and will fire on shadows, joints and stains.
- Auto-zoom here is a digital crop. Optical zoom / gimbal control is V3 (command maths only).
- Devanagari text in the PDF is not supported (Latin fonts only).
