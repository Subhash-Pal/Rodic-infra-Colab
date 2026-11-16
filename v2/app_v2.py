"""app_v2.py - RODIC InfraAI V2, deployable on Streamlit Community Cloud.

    Main file path on Streamlit Cloud:  v2/app_v2.py
    Local:                              streamlit run v2/app_v2.py

Modes: Static Image (the hosted app's flow, plus recommendation and PDF) and Live Video Stream (Edge Auto-Zoom).
Models: the hosted CODEBRIM + SDNET2018 checkpoints, downloaded and checksum-verified on first start.
Gemini visual evidence + evidence arbitration are optional and need the GEMINI_API_KEY secret.
If the models cannot load, the app still runs with a stub analyzer and says so on screen.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.join(HERE, "hosted")):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import streamlit as st

from app_logic import (
    MAX_GEMINI_CALLS,
    analyze_static,
    analyze_video,
    attach_gemini,
    decode_image_bytes,
    make_pdf_bytes,
)
from irc_recommender import DISCLAIMER, SEVERITY_ORDER
from localizer import make_localizer
from pipeline_adapter import load_analyzer, stub_analyzer

st.set_page_config(page_title="RODIC InfraAI V2", page_icon="🏗️", layout="wide")

LIVE_SOURCES = os.environ.get("RODIC_ENABLE_LIVE_SOURCES") == "1"   # webcam / RTSP only make sense on an edge box


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading inspection models (first start downloads about 90 MB)...")
def load_hosted():
    """(pipeline, error). A failure is cached too, so it is not retried on every click."""
    try:
        from rodic_analyzer import HostedPipeline
        return HostedPipeline(), None
    except Exception as exc:  # torch missing, download blocked, checksum mismatch, ...
        return None, f"{type(exc).__name__}: {exc}"


def build_analyzer(threshold: float):
    spec = os.environ.get("RODIC_ANALYZER")
    if spec:
        return load_analyzer(spec), f"custom analyzer ({spec})", None
    pipeline, error = load_hosted()
    if pipeline is None:
        return stub_analyzer, "stub (no model scores)", error
    return functools.partial(pipeline.analyze, threshold=threshold), "hosted CODEBRIM + SDNET2018", None


def gemini_ready():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return False, "Set the GEMINI_API_KEY secret to enable."
    if importlib.util.find_spec("google.genai") is None:
        return False, "google-genai is not installed."
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = str(key)   # the hosted client reads st.secrets, then the environment
    return True, ""


@st.cache_data(show_spinner=False, max_entries=32)
def run_gemini(image_bytes: bytes):
    from rodic_vlm.gemini_vlm import analyze_image_with_gemini
    return analyze_image_with_gemini(image_bytes)


def arbitration_record(cv_result, gemini_result):
    from rodic_inference.evidence_arbitration import build_inspection_evidence_record
    return build_inspection_evidence_record(cv_result, gemini_result)


# --------------------------------------------------------------------------- rendering
def render_gemini(gemini: dict) -> None:
    st.markdown("**Gemini visual evidence**")
    for i, obs in enumerate(gemini.get("observations", []), start=1):
        st.markdown(f"{i}. **{obs.get('evidence_type', 'other')}** - {obs.get('observation', '')}")
        st.caption(f"Location: {obs.get('location_description', 'n/a')} | "
                   f"visual confidence: {obs.get('visual_confidence', 'n/a')}")
    if not gemini.get("observations"):
        st.write("No specific visual observations reported.")
    for item in gemini.get("limitations", []):
        st.caption(f"Limitation: {item}")
    st.caption("Gemini provides visual evidence only; the engineering decision is not provided.")


def render_finding(f: dict) -> None:
    analysis, rec = f.get("analysis") or {}, f["recommendation"]
    if f.get("repeat"):
        st.info("Same region as the previous sampled frame; not re-analyzed.")
    if f.get("overview") is not None:
        st.image(cv2.cvtColor(f["overview"], cv2.COLOR_BGR2RGB), width="stretch", caption="Source frame with box")
    scores = (analysis.get("scores") or {}).get("codebrim")
    if scores:
        left, right = st.columns(2)
        with left:
            st.markdown("**CODEBRIM**")
            for name, score in scores.items():
                st.write(f"{name} - {score:.1%}")
                st.progress(min(max(float(score), 0.0), 1.0))
        with right:
            st.markdown("**SDNET2018**")
            st.metric("Crack probability", f"{analysis['scores']['sdnet_crack_probability']:.1%}")
            st.caption(analysis.get("source", ""))
    if f.get("gemini"):
        render_gemini(f["gemini"])
    if f.get("gemini_error"):
        st.warning(f"Gemini was unavailable for this crop ({f['gemini_error']}). CV results remain available.")
    arb = f.get("arbitration")
    if arb:
        st.markdown(f"**Evidence arbitration:** {arb['evidence_state'].replace('_', ' ').title()}")
        st.caption(arb["rationale"])
    st.markdown(f"**Severity: {rec['severity']}** - {rec['severity_basis']}")
    for action in rec["actions"]:
        st.write("- " + action)
    for note in rec["notes"]:
        st.caption(note)


def render_run(run: dict) -> None:
    findings = run["findings"]
    st.divider()
    st.caption(f"Analyzer: {run['analyzer']}")
    if not findings:
        st.success("No candidate defects localized.")
        return
    counts = {s: sum(1 for f in findings if f["recommendation"]["severity"] == s) for s in SEVERITY_ORDER}
    for col, sev in zip(st.columns(len(SEVERITY_ORDER)), SEVERITY_ORDER):
        col.metric(sev, counts[sev])

    if run["mode"] != "Static Image":
        st.subheader("Detected Defect Timeline")
        cols = st.columns(4)
        for i, f in enumerate(findings):
            with cols[i % 4]:
                st.image(cv2.cvtColor(f["image"], cv2.COLOR_BGR2RGB), width="stretch",
                         caption=f"t={f['t_sec']:.1f}s | {f['recommendation']['severity']}"
                                 + (" | repeat" if f.get("repeat") else ""))
    else:
        st.image(cv2.cvtColor(findings[0]["image"], cv2.COLOR_BGR2RGB), width="stretch")

    for f in findings:
        title = f"Finding {f['id']}: {f['defect_class'] or 'unclassified'} - {f['recommendation']['severity']}"
        with st.expander(title, expanded=(run["mode"] == "Static Image")):
            render_finding(f)

    if run.get("pdf"):
        st.download_button("Download PDF inspection report", run["pdf"],
                           file_name="rodic_inspection_report.pdf", mime="application/pdf")
    st.markdown(f'<div style="padding:1rem;border-left:5px solid #d97706;background:#fff7ed;border-radius:.4rem;">'
                f"<strong>Human validation required</strong><br>{DISCLAIMER}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- page
st.title("RODIC InfraAI - V2")
st.caption("AI-assisted infrastructure inspection: screening evidence, auto-zoom on video, repair guidance, PDF report.")

mode = st.sidebar.radio("Input mode", ["Static Image", "Live Video Stream (Edge Auto-Zoom)"])
threshold = st.sidebar.slider("Detection threshold", 0.10, 0.90, 0.50, 0.05)
width_mm = st.sidebar.number_input(
    "Known crack width, mm (optional)", min_value=0.0, value=0.0, step=0.1,
    help="From a crack comparator or calibrated image. Applied to crack findings. Leave 0 if unknown; "
         "crack severity is then provisional. (V3 estimates width from the image.)")
width_mm = width_mm if width_mm > 0 else None

ready, why_not = gemini_ready()
use_gemini = st.sidebar.checkbox("Gemini visual evidence + arbitration", value=ready, disabled=not ready,
                                 help=why_not or "Calls Gemini once per analyzed image (up to a small cap for video).")
if not ready:
    st.sidebar.caption(why_not)
gemini_calls = (st.sidebar.slider("Gemini calls per video run", 1, MAX_GEMINI_CALLS, 3)
                if use_gemini and mode != "Static Image" else 1)

analyzer, analyzer_label, load_error = build_analyzer(threshold)
st.sidebar.caption(f"Analyzer: {analyzer_label}")
if load_error:
    st.sidebar.warning(f"Models did not load: {load_error}")
    if st.sidebar.button("Retry model loading"):
        load_hosted.clear()
        st.rerun()
st.warning(DISCLAIMER)
if analyzer is stub_analyzer:
    st.info("No model scores are being produced (stub analyzer). Defects are localized and zoomed, but they "
            "are not classified, and recommendations stay provisional.")


def finish(findings, original_bytes=None):
    if original_bytes is not None and findings:
        findings[0]["gemini_bytes"] = original_bytes
    if use_gemini and findings:
        attach_gemini(findings, run_gemini, arbitration_record, gemini_calls)
    meta = {"mode": mode, "analyzer": analyzer_label, "detection threshold": f"{threshold:.2f}"}
    st.session_state["run"] = {"mode": mode, "findings": findings, "analyzer": analyzer_label,
                               "pdf": make_pdf_bytes(findings, meta) if findings else None}


if mode == "Static Image":
    upload = st.file_uploader("Upload an infrastructure image", type=["jpg", "jpeg", "png", "webp"],
                              help="Maximum file size: 10 MB.")
    if upload is not None and st.button("Analyze image", type="primary"):
        data = upload.getvalue()
        try:
            frame = decode_image_bytes(data)
        except ValueError as exc:
            st.error(f"Image validation failed: {exc}")
        else:
            with st.spinner("Running inspection..."):
                finish(analyze_static(frame, analyzer, width_mm), original_bytes=data)
else:
    localizers = ["opencv"] + (["yolo", "rtdetr"] if importlib.util.find_spec("ultralytics") else [])
    kind = st.sidebar.selectbox("Localizer", localizers,
                                help="opencv is a heuristic fallback (not a trained detector). "
                                     "yolo / rtdetr need defect-trained weights.")
    weights = st.sidebar.text_input("Weights path (yolo / rtdetr)") if kind != "opencv" else None
    pad_pct = st.sidebar.slider("Zoom padding (%)", 10, 30, 20)
    interval = st.sidebar.number_input("Sample every N seconds", 0.5, 30.0, 2.0, 0.5)
    max_samples = st.sidebar.slider("Max sampled frames", 5, 200, 60)
    min_score = st.sidebar.slider("Min localizer score", 0.0, 1.0, 0.0, 0.01)

    options = ["Upload mp4 / avi"] + (["Webcam index", "RTSP URL"] if LIVE_SOURCES else [])
    source_kind = st.radio("Source", options, horizontal=True)
    source, temp_path = None, None
    if source_kind == "Upload mp4 / avi":
        video = st.file_uploader("Video", type=["mp4", "avi", "mov"])
        if video is not None:
            source = ("upload", video)
    elif source_kind == "Webcam index":
        source = int(st.number_input("Camera index", 0, 10, 0))
    else:
        source = st.text_input("RTSP URL") or None
    if not LIVE_SOURCES:
        st.caption("Webcam / RTSP sources are available when running on an edge device "
                   "(set RODIC_ENABLE_LIVE_SOURCES=1).")

    if source is not None and st.button("Run edge auto-zoom", type="primary"):
        try:
            if isinstance(source, tuple):
                video = source[1]
                tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(video.name)[1] or ".mp4", delete=False)
                tmp.write(video.getvalue())
                tmp.close()
                temp_path = source = tmp.name
            bar = st.progress(0.0, text="Processing video...")
            localizer = make_localizer(kind, weights or None)
            findings = analyze_video(
                source, localizer, analyzer, interval_s=float(interval), pad_frac=pad_pct / 100,
                max_samples=int(max_samples), min_score=min_score, width_mm=width_mm,
                on_sample=lambda n, m: bar.progress(min(n / m, 1.0), text=f"Sampled frame {n} of up to {m}"))
            bar.empty()
            with st.spinner("Finishing..."):
                finish(findings)
        except (IOError, ValueError) as exc:
            st.error(f"Could not process the video: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if st.session_state.get("run"):
    if st.button("Clear results"):
        del st.session_state["run"]
        st.rerun()
    else:
        render_run(st.session_state["run"])
