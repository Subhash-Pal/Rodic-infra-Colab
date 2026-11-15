"""batch_runner.py - bulk image / video processing with one official PDF at the end.

Built for the "can you run our 500 km project from tomorrow?" question:

* streams results to ``results.jsonl`` one source at a time, so memory stays flat
  for 10,000+ images and an interrupted run RESUMES (finished sources are skipped);
* one bad file is recorded as an error and never aborts the batch;
* crops are saved to ``crops/`` and only the highest-severity findings are
  embedded in the PDF (the table lists all of them);
* writes ``summary.json`` and ``inspection_report.pdf``.

It is sequential on purpose. Throughput on real hardware has NOT been measured;
the per-stage timings in ``summary.json`` are what to look at first.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, Iterable, List, Optional

import cv2
import numpy as np

from . import _v2
from .pipeline import EdgeDronePipeline

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def iter_inputs(inputs: Iterable[str]) -> List[str]:
    found: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, names in os.walk(item):
                for name in names:
                    if os.path.splitext(name)[1].lower() in IMAGE_EXT | VIDEO_EXT:
                        found.append(os.path.join(root, name))
        elif os.path.isfile(item):
            found.append(item)
    return sorted(found)


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _done_sources(jsonl_path: str) -> set:
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["source"])
                except (ValueError, KeyError):
                    continue          # a half-written last line from an interrupted run
    return done


def _save_findings(findings: List[dict], crops_dir: str, tag: str) -> List[dict]:
    slim = []
    for i, f in enumerate(findings, start=1):
        f = dict(f)
        image = f.pop("image", None)
        if image is not None:
            path = os.path.join(crops_dir, f"{tag}_{i}.jpg")
            cv2.imwrite(path, image)
            f["image_path"] = path
        slim.append(f)
    return slim


def run_batch(
    inputs: Iterable[str],
    pipeline: EdgeDronePipeline,
    out_dir: str,
    resume: bool = True,
    max_sources: Optional[int] = None,
    video_max_samples: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
    jsonl_path = os.path.join(out_dir, "results.jsonl")
    if not resume and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    done = _done_sources(jsonl_path) if resume else set()

    sources = iter_inputs(inputs)
    if max_sources:
        sources = sources[:max_sources]
    stats = {"total": len(sources), "processed": 0, "skipped_resume": 0, "errors": 0, "rejected_quality": 0}
    stage_ms: Dict[str, float] = {}

    with open(jsonl_path, "a", encoding="utf-8") as out:
        for n, source in enumerate(sources, start=1):
            if on_progress:
                on_progress(n, len(sources), source)
            if source in done:
                stats["skipped_resume"] += 1
                continue
            record = {"source": source, "kind": "video" if os.path.splitext(source)[1].lower() in VIDEO_EXT
                      else "image", "status": "ok", "findings": [], "reasons": []}
            tag = f"{n:06d}"
            try:
                if record["kind"] == "image":
                    frame = cv2.imread(source, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise IOError("unreadable image")
                    results = [pipeline.process_image(frame, n)]
                else:
                    results = list(pipeline.process_video(source, max_samples=video_max_samples))
                findings: List[dict] = []
                for r in results:
                    if not r.accepted:
                        record["reasons"].extend(r.reasons)
                    findings.extend(r.findings)
                    for k, v in r.timings_ms.items():
                        stage_ms[k] = stage_ms.get(k, 0.0) + v
                if results and not any(r.accepted for r in results):
                    record["status"] = "rejected_quality"
                    stats["rejected_quality"] += 1
                record["frames"] = len(results)
                record["findings"] = _save_findings(findings, os.path.join(out_dir, "crops"), tag)
                stats["processed"] += 1
            except Exception as exc:  # one bad file must not stop a 10,000-image run
                record["status"], record["error"] = "error", f"{type(exc).__name__}: {exc}"
                stats["errors"] += 1
            out.write(json.dumps(record, default=_json_default) + "\n")
            out.flush()

    summary = {**stats, "stage_ms_total": {k: round(v, 1) for k, v in stage_ms.items()}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def load_findings(out_dir: str) -> List[dict]:
    findings: List[dict] = []
    path = os.path.join(out_dir, "results.jsonl")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            for f in rec.get("findings", []):
                f["source_file"] = rec["source"]
                findings.append(f)
    for i, f in enumerate(findings, start=1):
        f["id"] = i
    return findings


def build_pdf(out_dir: str, meta: Optional[dict] = None, max_detail: int = 100) -> str:
    """Official PDF from a finished (or resumed) batch."""
    findings = load_findings(out_dir)
    order = _v2.irc_recommender.SEVERITY_ORDER
    ranked = sorted(findings, key=lambda f: (-order.index(f["recommendation"]["severity"]),
                                             -(f.get("confidence") or 0.0)))
    for f in ranked[:max_detail]:
        path = f.get("image_path")
        f["image"] = cv2.imread(path) if path and os.path.exists(path) else None
    summary_path = os.path.join(out_dir, "summary.json")
    meta = dict(meta or {})
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as fh:
            s = json.load(fh)
        meta.setdefault("sources processed", f"{s['processed']} of {s['total']}")
        meta.setdefault("errors", s["errors"])
    return _v2.pdf_exporter.export_report(os.path.join(out_dir, "inspection_report.pdf"), findings, meta,
                                          max_detail=max_detail)
