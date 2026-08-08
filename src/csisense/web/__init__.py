"""The dashboard: FastAPI backend plus a single self-contained page.

Deliberately no build step. A hand-written page with canvas rendering means no
Node toolchain to install on a Raspberry Pi, one file to deploy, and nothing
that can go stale between a source tree and a compiled bundle. The
visualisations are canvas-drawn either way, so a framework would buy little.
"""

from csisense.web.telemetry import TelemetrySnapshot, TelemetryTap

__all__ = ["TelemetrySnapshot", "TelemetryTap"]
