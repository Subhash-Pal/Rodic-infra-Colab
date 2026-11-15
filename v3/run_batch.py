"""CLI: process a folder of images / videos and write results + one PDF report.

    python v3/run_batch.py --input /data/survey --out /data/out --detector rtdetr --weights best.pt
    python v3/run_batch.py --input /data/survey --out /data/out --detector heuristic   # demo only

Set RODIC_ANALYZER=module:function to use the existing CODEBRIM + SDNET2018 + Gemini
pipeline (see v2/pipeline_adapter.py). Re-running the same command resumes.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edge_drone_pipeline import _v2
from edge_drone_pipeline.batch_runner import build_pdf, run_batch
from edge_drone_pipeline.pipeline import EdgeDronePipeline, load_config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="files and/or folders")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--detector", choices=["rtdetr", "heuristic"], default="rtdetr")
    p.add_argument("--weights", default=None, help="RT-DETR weights trained on infrastructure defects")
    p.add_argument("--gsd-mm-per-px", type=float, default=None,
                   help="known scale (e.g. from a reference object); overrides distance-based GSD")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-sources", type=int, default=None)
    p.add_argument("--site", default="", help="site / asset name for the report header")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.detector == "rtdetr":
        if not args.weights:
            p.error("--weights is required for --detector rtdetr")
        from edge_drone_pipeline.sahi_rtdetr_engine import RTDETRDetector
        detector = RTDETRDetector(args.weights, conf=cfg["sahi"]["conf"], imgsz=cfg["sahi"]["tile"])
    else:
        from edge_drone_pipeline.heuristic_detector import HeuristicTileDetector
        print("WARNING: heuristic detector is a demo, not a validated model.")
        detector = HeuristicTileDetector()

    analyzer = _v2.pipeline_adapter.load_analyzer()
    if _v2.pipeline_adapter.is_stub(analyzer):
        print("NOTE: stub analyzer (no CODEBRIM / SDNET2018 / Gemini scores). Set RODIC_ANALYZER to enable.")

    pipeline = EdgeDronePipeline(detector, analyzer, cfg, gsd_override_mm_per_px=args.gsd_mm_per_px)
    summary = run_batch(
        args.input, pipeline, args.out, resume=not args.no_resume, max_sources=args.max_sources,
        on_progress=lambda n, total, src: print(f"[{n}/{total}] {src}", flush=True),
    )
    pdf = build_pdf(args.out, {"site": args.site, "detector": args.detector},
                    max_detail=cfg["batch"]["max_detail_in_pdf"])
    print(summary)
    print("PDF:", pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
