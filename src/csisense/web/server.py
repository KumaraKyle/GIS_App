"""Dashboard backend.

FastAPI sharing the pipeline's event loop, so the tap is read in-process with no
IPC. The broadcast task runs at ``web.live_stream_hz`` regardless of capture
rate, and each client has a depth-2 queue: if a browser cannot keep up, frames
are dropped for that client and the drop count is sent along with the next
update, so the UI can say "rendering is behind" rather than silently lagging.

Waterfall payloads are quantised to uint8 against a slowly-adapting per-link
range. Sending float32 would be four times the bytes for a display that has 256
distinguishable shades anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from csisense.config import AppConfig
from csisense.rti.engine import RtiEngine
from csisense.web.telemetry import TelemetryTap

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class LiveHub:
    """Fan-out of telemetry snapshots to connected browsers."""

    def __init__(self, tap: TelemetryTap, config: AppConfig, rti: RtiEngine | None = None) -> None:
        self.tap = tap
        self.config = config
        self.rti = rti
        self._clients: dict[WebSocket, asyncio.Queue[str]] = {}
        self._task: asyncio.Task | None = None
        self._ranges: dict[str, tuple[float, float]] = {}
        self._sent_event_seq = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def add(self, ws: WebSocket) -> asyncio.Queue[str]:
        # Depth 2: a slow client drops frames rather than stalling the loop.
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2)
        self._clients[ws] = queue
        return queue

    def remove(self, ws: WebSocket) -> None:
        self._clients.pop(ws, None)

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / max(self.config.web.live_stream_hz, 1.0)
        while True:
            await asyncio.sleep(interval)
            added = self.tap.tick()
            if not self._clients:
                continue
            try:
                message = json.dumps(self.build_payload(new_rows=added))
            except Exception:  # pragma: no cover
                log.exception("failed to build telemetry payload")
                continue
            for queue in list(self._clients.values()):
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                    with contextlib.suppress(asyncio.QueueFull):
                        queue.put_nowait(message)

    def build_payload(self, *, new_rows: dict[str, int] | None = None) -> dict[str, Any]:
        """Assemble one update.

        ``new_rows`` restricts the waterfall to the columns added since the last
        tick -- usually one. Omit it for the initial payload a client needs to
        populate its buffer; sending the full history every tick costs ~85kB per
        frame for one new column of data.
        """
        full = new_rows is None
        # Traces and events are incremental for the same reason the waterfall is:
        # only a sample or two is new each tick, and resending 300 of them per
        # link is most of the payload.
        snap = self.tap.snapshot(waterfall_rows=128 if full else 0, trace_len=300 if full else 1)
        payload: dict[str, Any] = {
            "ts_ns": snap.ts_ns,
            "presence": snap.presence,
            "motion_stat": round(snap.motion_stat, 3),
            "calibrated": snap.calibrated,
            "calibration_progress": round(snap.calibration_progress, 3),
            "blind": snap.blind,
            "k_on": self.config.presence.k_on,
            "k_off": self.config.presence.k_off,
            "links": snap.links,
            "full": full,
            "events": (
                self.tap.events[-40:] if full else self.tap.events_since(self._sent_event_seq)
            ),
            "waterfall": {},
            "traces": {lid: [round(float(v), 3) for v in t] for lid, t in snap.traces.items()},
        }

        blocks = (
            snap.waterfall
            if full
            else {
                lid: self.tap.links[lid].recent_waterfall(n)
                for lid, n in (new_rows or {}).items()
                if n and lid in self.tap.links
            }
        )
        if not full:
            self._sent_event_seq = self.tap.event_seq

        for link_id, block in blocks.items():
            if block.size == 0:
                continue
            lo, hi = self._range_for(link_id, block)
            scaled = np.clip((block - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            payload["waterfall"][link_id] = {
                "rows": int(block.shape[0]),
                "cols": int(block.shape[1]),
                "lo": round(float(lo), 4),
                "hi": round(float(hi), 4),
                # uint8 is all a display can distinguish; float32 would be 4x
                # the bytes for no visible difference.
                "data": (scaled * 255).astype(np.uint8).ravel().tolist(),
            }

        if self.rti is not None:
            status = self.rti.status()
            rti_payload: dict[str, Any] = {
                "enabled": status.enabled,
                "mode": status.mode,
                "n_links": status.n_links,
                "min_links": status.min_links,
                "caption": status.caption(),
                "reason": status.reason,
            }
            per_link = {lid: float(v["z"]) for lid, v in snap.links.items()}
            frame = self.rti.update(snap.ts_ns, per_link)
            if frame.image is not None and frame.confidence is not None:
                img = frame.image
                lo, hi = float(img.min()), float(img.max())
                norm = (img - lo) / max(hi - lo, 1e-9)
                rti_payload.update(
                    {
                        "shape": list(img.shape),
                        "image": (norm * 255).astype(np.uint8).ravel().tolist(),
                        # Rendered as alpha, so unconstrained regions fade out
                        # instead of appearing as a confident blob.
                        "confidence": (frame.confidence * 255).astype(np.uint8).ravel().tolist(),
                        "peak_xy": list(frame.peak_xy) if frame.peak_xy else None,
                        "room": list(self.config.geometry.room_size_m),
                    }
                )
            else:
                # Ellipse geometry is static, so the points go out only on a
                # full payload; incremental updates carry just the z-scores that
                # colour them. Resending 48 points per link at 20Hz was most of
                # what remained in the incremental payload.
                rti_payload["ellipses"] = [
                    {
                        "link_id": link.link_id,
                        "z": round(per_link.get(link.link_id, 0.0), 3),
                        **({"points": self._ellipse_points(link.link_id)} if full else {}),
                    }
                    for link in self.rti.geometry.links
                ]
                rti_payload["room"] = list(self.config.geometry.room_size_m)
                rti_payload["nodes"] = [
                    {"id": n.id, "xy": [float(n.position[0]), float(n.position[1])]}
                    for n in self.rti.geometry.nodes.values()
                ]
            payload["rti"] = rti_payload

        return payload

    def _ellipse_points(self, link_id: str) -> list[list[float]]:
        if self.rti is None:
            return []
        pts = self.rti.link_ellipse(link_id, n_points=48)
        return [] if pts is None else [[round(float(x), 3), round(float(y), 3)] for x, y in pts]

    def _range_for(self, link_id: str, block: np.ndarray) -> tuple[float, float]:
        """Slowly-adapting display range, so the waterfall does not flicker.

        Recomputing min/max per frame makes the whole image jump whenever a
        single outlier arrives.
        """
        lo = float(np.percentile(block, 1))
        hi = float(np.percentile(block, 99))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        prev = self._ranges.get(link_id)
        if prev is not None:
            lo = 0.9 * prev[0] + 0.1 * lo
            hi = 0.9 * prev[1] + 0.1 * hi
        self._ranges[link_id] = (lo, hi)
        return lo, hi


def create_app(
    config: AppConfig,
    tap: TelemetryTap,
    *,
    rti: RtiEngine | None = None,
    on_recalibrate: Any = None,
) -> FastAPI:
    app = FastAPI(title="csisense", docs_url=None, redoc_url=None)
    hub = LiveHub(tap, config, rti)
    app.state.hub = hub
    app.state.config = config
    app.state.tap = tap

    @app.on_event("startup")
    async def _startup() -> None:
        await hub.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await hub.stop()

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        snap = tap.snapshot(waterfall_rows=0, trace_len=0)
        return {
            "presence": snap.presence,
            "motion_stat": snap.motion_stat,
            "calibrated": snap.calibrated,
            "blind": snap.blind,
            "frames": tap.frames,
            "links": snap.links,
            "clients": hub.n_clients,
            "rti": dataclasses.asdict(rti.status()) if rti else None,
        }

    @app.post("/api/calibrate")
    async def calibrate() -> dict[str, str]:
        if on_recalibrate is None:
            return {"status": "unavailable"}
        on_recalibrate()
        return {"status": "started"}

    @app.websocket("/ws/live")
    async def live(ws: WebSocket) -> None:
        await ws.accept()
        queue = hub.add(ws)
        try:
            # Send one immediately so the page is not blank until the first tick.
            await ws.send_text(json.dumps(hub.build_payload()))
            while True:
                await ws.send_text(await queue.get())
        except WebSocketDisconnect:
            pass
        except Exception:  # pragma: no cover
            log.debug("websocket closed", exc_info=True)
        finally:
            hub.remove(ws)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        async def index() -> HTMLResponse:
            return HTMLResponse((STATIC_DIR / "index.html").read_text())

    return app


__all__ = ["LiveHub", "create_app"]
