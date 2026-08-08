"""The source abstraction.

A source turns *something* -- a UDP socket, a Parquet file, a physics model --
into an async stream of :class:`~csisense.wire.CsiFrame`. Everything downstream
is a pure function of that stream, which is what lets the entire pipeline run
against replayed or synthetic data with no hardware attached.

Adding a new radio means adding a source and nothing else.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from csisense.wire import CsiFrame, SourceKind


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """What a source will produce, known before the first frame arrives.

    The pipeline uses this to preallocate ring buffers and to check that a
    loaded model matches the data it is about to be fed.
    """

    kind: SourceKind
    n_sub: int
    rate_hz: float
    link_ids: tuple[str, ...] = field(default=())

    @property
    def n_links(self) -> int:
        return len(self.link_ids)


class CsiSource(abc.ABC):
    """Base class for anything that yields CSI frames."""

    @abc.abstractmethod
    def info(self) -> SourceInfo:
        """Describe the stream. Must be valid before iteration starts."""

    @abc.abstractmethod
    def frames(self) -> AsyncIterator[CsiFrame]:
        """Yield frames until exhausted or cancelled.

        Implementations must be cancellation-safe: closing sockets and file
        handles on ``asyncio.CancelledError`` rather than leaking them, because
        the pipeline restarts sources on reconfiguration.
        """

    async def aclose(self) -> None:
        """Release resources. Safe to call more than once."""
        return None

    async def __aenter__(self) -> CsiSource:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


__all__ = ["CsiSource", "SourceInfo"]
