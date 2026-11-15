"""vlm_arbitration_bridge.py - send a zoom crop + context + measurements to a VLM,
and survive having no connectivity (tunnels).

The bridge does not call Gemini itself (no key / network here). You inject a
``sender(request) -> dict`` that does. If the sender raises a connection error
the request is stored in an SQLite queue and ``flush()`` retries it later.
VLM answers are unverified opinions: they are stored next to, never instead of,
the detector scores and the measurement.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


def _jpeg_b64(img: np.ndarray, max_side: Optional[int] = None, quality: int = 85) -> str:
    if max_side:
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class ArbitrationRequest:
    track_id: int
    frame_index: int
    t_sec: float
    crop_jpeg_b64: str
    context_jpeg_b64: str
    crack_width_mm: Optional[float]
    width_resolvable: Optional[bool]
    model_scores: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> "ArbitrationRequest":
        return ArbitrationRequest(**json.loads(text))


def build_request(
    track_id: int,
    frame_index: int,
    t_sec: float,
    crop_bgr: np.ndarray,
    context_bgr: np.ndarray,
    crack_width_mm: Optional[float] = None,
    width_resolvable: Optional[bool] = None,
    model_scores: Optional[dict] = None,
) -> ArbitrationRequest:
    return ArbitrationRequest(
        track_id, frame_index, t_sec,
        _jpeg_b64(crop_bgr, max_side=1024),
        _jpeg_b64(context_bgr, max_side=512),
        crack_width_mm, width_resolvable, dict(model_scores or {}),
    )


def build_prompt(req: ArbitrationRequest) -> str:
    width = ("not measured" if req.crack_width_mm is None else
             f"{req.crack_width_mm:.2f} mm" + ("" if req.width_resolvable else " (upper bound, below image resolution)"))
    return (
        "You assist a civil engineer reviewing a drone image of infrastructure. Image 1 is a zoomed crop of a "
        "candidate defect; image 2 is wider context. Detector scores: "
        f"{json.dumps(req.model_scores)}. Measured crack width: {width}. "
        "Describe only what is visible. Say whether the crop shows a real defect or a false positive "
        "(shadow, joint, stain, vegetation), name the defect type, and state your uncertainty. "
        "Do not give a structural safety verdict. Answer as JSON with keys: "
        "is_defect, defect_type, rationale, uncertainty."
    )


class OfflineQueue:
    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS queue ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',"
            "attempts INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, last_error TEXT, response TEXT)"
        )
        self.conn.commit()

    def enqueue(self, req: ArbitrationRequest) -> int:
        cur = self.conn.execute("INSERT INTO queue (payload, created) VALUES (?, ?)", (req.to_json(), time.time()))
        self.conn.commit()
        return int(cur.lastrowid)

    def pending(self, limit: int = 100) -> List[Tuple[int, ArbitrationRequest]]:
        rows = self.conn.execute(
            "SELECT id, payload FROM queue WHERE status='pending' ORDER BY id LIMIT ?", (limit,)).fetchall()
        return [(rid, ArbitrationRequest.from_json(p)) for rid, p in rows]

    def mark_done(self, rid: int, response: dict) -> None:
        self.conn.execute("UPDATE queue SET status='done', response=?, attempts=attempts+1 WHERE id=?",
                          (json.dumps(response), rid))
        self.conn.commit()

    def mark_failed(self, rid: int, error: str) -> None:
        self.conn.execute("UPDATE queue SET attempts=attempts+1, last_error=? WHERE id=?", (error, rid))
        self.conn.commit()

    def counts(self) -> dict:
        return dict(self.conn.execute("SELECT status, COUNT(*) FROM queue GROUP BY status").fetchall())


class ArbitrationBridge:
    NETWORK_ERRORS = (ConnectionError, TimeoutError, OSError)

    def __init__(self, sender: Callable[[ArbitrationRequest], dict], queue: Optional[OfflineQueue] = None) -> None:
        self.sender = sender
        self.queue = queue or OfflineQueue()

    def submit(self, req: ArbitrationRequest) -> Optional[dict]:
        """Send now; on a network error queue it and return None."""
        try:
            return self.sender(req)
        except self.NETWORK_ERRORS:
            self.queue.enqueue(req)
            return None

    def flush(self, limit: int = 100) -> dict:
        done = failed = 0
        for rid, req in self.queue.pending(limit):
            try:
                self.queue.mark_done(rid, self.sender(req))
                done += 1
            except self.NETWORK_ERRORS as exc:
                self.queue.mark_failed(rid, repr(exc))
                failed += 1
                break          # still offline: stop and keep the rest queued
        return {"sent": done, "still_offline": failed > 0, "queue": self.queue.counts()}
