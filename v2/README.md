# RODIC V2

The next version of the hosted POC (`rodic-infraai-gemini-resilient`): the same CODEBRIM + SDNET2018 + Gemini
flow, plus video edge auto-zoom, repair guidance, and a PDF report.

| File | Role |
|---|---|
| `app_v2.py` | Streamlit app. **Main file path on Streamlit Cloud.** Static Image / Live Video Stream (Edge Auto-Zoom) |
| `app_logic.py` | Everything the app does except drawing widgets (findings, Gemini attach, PDF bytes); unit-tested |
| `rodic_analyzer.py` | Wraps the hosted CODEBRIM + SDNET2018 inference in the analyzer contract |
| `deploy_assets.py` | Downloads the two checkpoints on first start and verifies SHA-256 |
| `hosted/` | Code vendored **unchanged** from the hosted repo (see `hosted/PROVENANCE.md`) |
| `video_edge_processor.py`, `localizer.py` | Sample video, localize, 15-20 % padded auto-zoom crop |
| `irc_recommender.py`, `pdf_exporter.py` | Severity + repair guidance; one-click PDF |
| `pipeline_adapter.py` | Custom analyzer hook: `RODIC_ANALYZER=module:function` |

## Deploy on Streamlit Community Cloud

1. share.streamlit.io -> **Create app** -> repository `Subhash-Pal/Rodic-infra-Colab`, branch `main`.
2. **Main file path:** `v2/app_v2.py`.
3. **Advanced settings:** Python **3.12** (what the torch wheels support best), and Secrets:
   ```
   GEMINI_API_KEY = "your-key"
   ```
   Without the secret the app still works; the Gemini checkbox is disabled and the sidebar says why.
4. Deploy. Dependencies come from `v2/requirements.txt` (Cloud reads the file next to the main file first,
   so keep the full list there). `.streamlit/config.toml` is read from the repo root.

**First start is slow:** installing torch, then downloading the two 45 MB checkpoints (checksum-verified, from a
commit-pinned URL in `hosted/models/MANIFEST.json`). After that they stay cached while the app container lives; a
sleeping app re-downloads on wake. To avoid the download, commit the two `.pt` files into `v2/hosted/models/`
(remove the `.gitignore` line for them); `deploy_assets.py` skips files that are already present and valid.

If the models cannot load, the app does not crash: it runs with a stub analyzer (localizes and zooms, produces no
scores) and shows the error with a **Retry model loading** button.

Local: `pip install -r v2/requirements.txt && streamlit run v2/app_v2.py`
Edge box (webcam / RTSP sources): `RODIC_ENABLE_LIVE_SOURCES=1 streamlit run v2/app_v2.py`
Check the weights without starting the app: `python v2/deploy_assets.py --check`

## Behaviour

- **Static Image:** the hosted flow on the whole image (CODEBRIM + SDNET2018, optional Gemini + evidence
  arbitration), plus severity, repair guidance and PDF. Same 10 MB / decode / size validation as the hosted app.
- **Video:** frames sampled every N seconds, candidate regions localized, 15-20 % padded crop analyzed, repeats of
  the same region skipped. Gemini runs on at most 5 crops per run (highest severity first) to protect quota.
- Results persist across reruns (session state), so clicking download does not wipe them.
- CODEBRIM and SDNET2018 scores stay separate and are never averaged (same policy as the hosted app).

## Recommender

A lookup table, not a structural assessment. Crack severity bands (starting values; edit `CRACK_WIDTH_BANDS`):
`<0.2 mm` cosmetic, `0.2-0.5` watch, `0.5-2.0` significant (epoxy injection), `>=2.0` urgent (structural audit).
Unknown width gives a provisional "watch". Standards are named (IRC:SP:35, IRC:SP:40, MoRTH Specifications for
Road and Bridge Works) without clause numbers; verify against the current editions.

## Not verified

- The real checkpoints were **not run** during development here (torch could not be installed in the build
  sandbox). The vendored inference code is identical to the hosted app's, and the integration is tested with the
  model calls replaced by fakes. First real run happens on your machine or on Cloud.
- The pinned download URL and both checksums were verified from the build sandbox.
- No Streamlit Cloud deployment was performed from here; the steps above are the standard flow.
- The OpenCV localizer is a heuristic (dark thin structures); expect false positives on shadows, joints and stains.
- Auto-zoom is a digital crop; optical zoom / gimbal control is V3 (command maths only).
- The PDF supports Latin text only (no Devanagari font).
