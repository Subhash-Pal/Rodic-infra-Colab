"""app_v2.py - RODIC InfraAI V2 Streamlit app.

Run:  streamlit run v2/app_v2.py

Modes (sidebar): Static Image | Live Video Stream (Edge Auto-Zoom).
The heavy models are NOT in this repo: set RODIC_ANALYZER=module:function to plug
in the existing CODEBRIM + SDNET2018 + Gemini pipeline (see pipeline_adapter.py).
Without it the app runs with a stub analyzer and says so on screen.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import streamlit as st

from irc_recommender import DISCLAIMER, recommend
from localizer import make_localizer
from pdf_exporter import export_report
from pipeline_adapter import is_stub, load_analyzer
from video_edge_processor import process_samples, process_video, sample_frames

st.set_page_config(page_title="RODIC InfraAI V2", layout="wide")
st.title("RODIC InfraAI - V2")
st.warning(DISCLAIMER)

analyzer = load_analyzer()
if is_stub(analyzer):
    st.info("Stub analyzer active: defects are localized and zoomed, but no model scores are produced. "
            "Set RODIC_ANALYZER to use the real pipeline.")

mode = st.sidebar.radio("Input mode", ["Static Image", "Live Video Stream (Edge Auto-Zoom)"])
localizer_kind = st.sidebar.selectbox("Localizer", ["opencv", "yolo", "rtdetr"])
weights = st.sidebar.text_input("Weights path (yolo / rtdetr)", "") if localizer_kind != "opencv" else None
pad_pct = st.sidebar.slider("Zoom padding (%)", 10, 30, 20)
min_score = st.sidebar.slider("Min localizer score", 0.0, 1.0, 0.0, 0.01)
crack_width_mm = st.sidebar.number_input(
    "Known crack width mm (optional, from calibrated measurement)", min_value=0.0, value=0.0, step=0.1)


def to_findings(events):
    findings = []
    for i, ev in enumerate(events, start=1):
        a = ev.analysis or {}
        width = crack_width_mm if crack_width_mm > 0 else None
        rec = recommend(a.get("defect_class"), a.get("confidence"), width)
        findings.append({
            "id": i, "frame_index": ev.frame_index, "t_sec": ev.t_sec, "box": ev.box,
            "image": ev.crop, "defect_class": a.get("defect_class"),
            "confidence": a.get("confidence"), "crack_width_mm": width,
            "source": a.get("source", "n/a"), "recommendation": rec.to_dict(),
        })
    return findings


def show(events):
    if not events:
        st.success("No candidate defects localized.")
        return
    st.subheader("Detected Defect Timeline")
    findings = to_findings(events)
    cols = st.columns(4)
    for i, (ev, f) in enumerate(zip(events, findings)):
        with cols[i % 4]:
            st.image(cv2.cvtColor(ev.crop, cv2.COLOR_BGR2RGB),
                     caption=f"t={ev.t_sec:.1f}s | {f['recommendation']['severity']}"
                             + (" | repeat" if ev.repeat else ""))
    for f in findings:
        with st.expander(f"Finding {f['id']}: {f['defect_class'] or 'unclassified'} "
                         f"- {f['recommendation']['severity']}"):
            st.write(f["recommendation"]["severity_basis"])
            for action in f["recommendation"]["actions"]:
                st.write("- " + action)
            for note in f["recommendation"]["notes"]:
                st.caption(note)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        export_report(tmp.name, findings, meta={"mode": mode, "analyzer": findings[0]["source"]})
        st.download_button("Download PDF inspection report", open(tmp.name, "rb").read(),
                           file_name="rodic_inspection_report.pdf", mime="application/pdf")


localizer = make_localizer(localizer_kind, weights or None)

if mode == "Static Image":
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if upload and st.button("Analyze"):
        frame = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
        from video_edge_processor import FrameSample
        show(process_samples([FrameSample(0, 0.0, frame)], localizer, analyzer,
                             pad_frac=pad_pct / 100, min_score=min_score))
else:
    interval = st.sidebar.number_input("Sample every N seconds", 0.5, 30.0, 2.0, 0.5)
    max_samples = st.sidebar.number_input("Max sampled frames", 1, 2000, 60)
    kind = st.radio("Source", ["Upload mp4/avi", "Webcam index", "RTSP URL"], horizontal=True)
    source = None
    if kind == "Upload mp4/avi":
        up = st.file_uploader("Video", type=["mp4", "avi"])
        if up:
            tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False)
            tmp.write(up.read())
            tmp.close()
            source = tmp.name
    elif kind == "Webcam index":
        source = int(st.number_input("Camera index", 0, 10, 0))
    else:
        source = st.text_input("RTSP URL", "") or None
    if source is not None and st.button("Run edge auto-zoom"):
        with st.spinner("Processing..."):
            show(process_video(source, localizer, analyzer, interval_s=interval,
                               pad_frac=pad_pct / 100, max_samples=int(max_samples),
                               min_score=min_score))
