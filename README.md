# Rodic-infra-Colab

Versioned Rodic work lives here and only here (`Vx/` folders). Research and production code go in separate repositories.

| Path | What it is |
|---|---|
| `Rodic_Edge_CV.ipynb` | Colab notebook: CPU baseline benchmark, sliding-window boundary fix, localization experiment |
| `v2/` | Video auto-zoom, IRC / MoRTH-grounded recommendations, one-click PDF report, Streamlit app |
| `v3/` | Sensorless long-range pipeline: SAHI + RT-DETR, tracker-gated zoom, crack-width metrology, batch runner, offline VLM queue |

Deploy V2 on Streamlit Cloud: main file `v2/app_v2.py` (steps in `v2/README.md`).

Run the tests: `pip install -r v2/requirements.txt pytest pyyaml scikit-image scipy && pytest`
(the tests do not need torch or the checkpoints).

## Status (read this before presenting anything)

- **Implemented and unit-tested (synthetic data only):** all V2 and V3 modules in this repo, 84 tests.
- **Models:** V2 vendors the hosted POC's inference code (`v2/hosted/`, unchanged) and downloads the two checkpoints on first start with a SHA-256 check. The real checkpoints have not been run in the build environment. V3 takes the same analyzer through `RODIC_ANALYZER=module:function`; without one, reports say `stub_not_evaluated` and carry no model scores.
- **Not done:** a trained RT-DETR / YOLO on infrastructure defects, real-footage validation, real-image crack-width calibration, Jetson Orin benchmarks, and any real gimbal / MAVLink control.
- Every recommendation is an indicative suggestion that needs engineer validation. Standards are named, not quoted, and must be checked against the current edition.
