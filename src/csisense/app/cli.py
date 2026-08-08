"""Command line entry point.

Built on argparse rather than a CLI framework so that the Pi install stays
dependency-light -- every megabyte of virtualenv matters on a 1GB board.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from csisense import __version__
from csisense.config import AppConfig
from csisense.events.types import Event, LinkQualityEvent
from csisense.sources.synthetic import SyntheticCsiSource, available_scenarios


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_event(event: Event) -> None:
    if isinstance(event, LinkQualityEvent):
        print(
            f"  [link] {event.link_id:<16} {event.rate_hz:6.1f} Hz  "
            f"loss {event.loss_pct:5.2f}%  rssi {event.rssi:6.1f}  "
            f"gain {'locked' if event.gain_locked else 'FREE'}"
        )
    else:
        print(f"  [event] {event}")


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Run the full pipeline against the synthetic source. No hardware."""
    config = AppConfig.load(args.config)
    source = SyntheticCsiSource(
        seed=args.seed,
        scenario=args.scenario,
        n_links=args.links,
        n_sub=args.subcarriers,
        rate_hz=args.rate,
        duration_s=args.duration,
        realtime=args.realtime,
    )

    from csisense.app.pipeline import Pipeline

    pipeline = Pipeline(config, sinks=[_print_event] if args.verbose else [])

    print(
        f"simulating scenario {args.scenario!r}: {args.links} links, "
        f"{args.subcarriers} subcarriers @ {args.rate:g} Hz"
    )
    stats = asyncio.run(pipeline.run(source))

    print(
        f"\n{stats.frames} frames over {stats.duration_s:.1f}s "
        f"({stats.mean_rate_hz:.1f} Hz/link), {stats.events} events"
    )
    for link_id, n in sorted(stats.per_link_frames.items()):
        print(f"  {link_id:<16} {n:6d} frames")
    return 0


def _build_source(config: AppConfig, args: argparse.Namespace):
    """Construct the configured source. Kept here so `serve` and `run` agree."""
    from csisense.sources.synthetic import SyntheticCsiSource

    kind = getattr(args, "source", None) or config.source.kind
    if kind == "synthetic":
        s = config.source.synthetic
        return SyntheticCsiSource(
            seed=s.seed,
            scenario=getattr(args, "scenario", None) or s.scenario,
            n_links=s.n_links,
            n_sub=s.n_sub,
            rate_hz=s.rate_hz,
            snr_db=s.snr_db,
            carrier_hz=s.carrier_hz,
            duration_s=None,
            realtime=True,
            # The dashboard should keep showing motion, not go quiet after 20s.
            loop=True,
            loss_pct=s.loss_pct,
        )
    if kind == "nexmon":
        from csisense.sources.udp import NexmonCsiSource

        return NexmonCsiSource(config.source.nexmon)
    if kind == "udp":
        from csisense.sources.udp import EspUdpCsiSource

        return EspUdpCsiSource(config.source.udp)
    if kind == "replay":
        from csisense.sources.replay import ReplayCsiSource

        r = config.source.replay
        return ReplayCsiSource(r.path, speed=r.speed, loop=r.loop)
    raise SystemExit(f"source kind {kind!r} is not runnable from the CLI yet")


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the pipeline with the dashboard attached."""
    import uvicorn

    from csisense.app.pipeline import Pipeline
    from csisense.rti.engine import RtiEngine
    from csisense.web.server import create_app
    from csisense.web.telemetry import TelemetryTap

    config = AppConfig.load(args.config)
    if args.port:
        config.web.port = args.port

    source = _build_source(config, args)
    tap = TelemetryTap()
    pipeline = Pipeline(config, source_info=source.info())
    pipeline.tap = tap

    rti = RtiEngine(config)
    app = create_app(config, tap, rti=rti, on_recalibrate=lambda: pipeline.start_calibration())

    if args.calibrate:
        pipeline.start_calibration(args.calibrate)

    async def main() -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=config.web.host,
                port=config.web.port,
                log_level="warning",
                access_log=False,
            )
        )
        # Capture and the web server share one event loop, so the tap is read
        # in-process with no IPC.
        capture = asyncio.create_task(pipeline.run(source))
        try:
            await server.serve()
        finally:
            capture.cancel()

    print(f"dashboard: http://{config.web.host}:{config.web.port}/")
    print(f"source:    {source.info().kind.value}, {source.info().n_sub} subcarriers")
    if args.calibrate:
        print(f"calibrating for {args.calibrate:g}s -- keep the room empty")
    asyncio.run(main())
    return 0


def _cmd_scenarios(_args: argparse.Namespace) -> int:
    from csisense.sources.synthetic import SCENARIOS

    print("available synthetic scenarios:\n")
    for name in available_scenarios():
        s = SCENARIOS[name]
        notes = []
        if s.loss_pct:
            notes.append(f"{s.loss_pct:g}% packet loss")
        if s.agc_glitch_at_s:
            notes.append(f"AGC glitches at {', '.join(f'{t:g}s' for t in s.agc_glitch_at_s)}")
        if s.dropout:
            notes.append(f"link dropout {s.dropout}")
        if s.motion_intervals:
            notes.append(
                "motion during " + ", ".join(f"{a:g}-{b:g}s" for a, b in s.motion_intervals)
            )
        print(f"  {name:<16} {s.duration_s:5.1f}s   {'; '.join(notes) or 'static, no motion'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csisense",
        description="Wi-Fi CSI human sensing: presence detection and gesture automation.",
    )
    parser.add_argument("--version", action="version", version=f"csisense {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="path to config YAML")
    parser.add_argument("--log-level", default="INFO")

    # Repeated on every subcommand so that both `csisense --config X simulate`
    # and `csisense simulate --config X` work. SUPPRESS is what makes that safe:
    # without it, the subparser's default would overwrite a value the main
    # parser already captured.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=argparse.SUPPRESS)
    common.add_argument("--log-level", default=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser(
        "simulate",
        parents=[common],
        help="run the pipeline against synthetic CSI (no hardware)",
    )
    sim.add_argument("--scenario", default="walk_through", choices=list(available_scenarios()))
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--links", type=int, default=2)
    sim.add_argument("--subcarriers", type=int, default=64)
    sim.add_argument("--rate", type=float, default=100.0, help="frames per second per link")
    sim.add_argument("--duration", type=float, default=None, help="seconds (default: scenario)")
    sim.add_argument(
        "--realtime",
        action="store_true",
        help="pace generation at wall-clock speed instead of running flat out",
    )
    sim.add_argument("-v", "--verbose", action="store_true", help="print every event")
    sim.set_defaults(func=_cmd_simulate)

    serve = sub.add_parser(
        "serve", parents=[common], help="run the pipeline with the live dashboard"
    )
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument(
        "--source",
        default=None,
        choices=["synthetic", "nexmon", "udp", "replay"],
        help="override source.kind from config",
    )
    serve.add_argument("--scenario", default=None, help="synthetic scenario to run")
    serve.add_argument(
        "--calibrate",
        type=float,
        nargs="?",
        const=120.0,
        default=None,
        metavar="SECONDS",
        help="start a calibration immediately; the room must be empty",
    )
    serve.set_defaults(func=_cmd_serve)

    scen = sub.add_parser(
        "scenarios", parents=[common], help="list synthetic scenarios and their ground truth"
    )
    scen.set_defaults(func=_cmd_scenarios)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
