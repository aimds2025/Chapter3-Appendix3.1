"""
physio_pipeline.streaming
=========================

Turns the batch pipeline into a continuous, time-paced simulation:

    StreamSource  -> BoundedBus -> StreamRunner (L2 -> L3 -> L4 online) -> L5

Exercises what batch mode can't: continuous flow, mid-stream clinical events,
backpressure under load, multi-patient interleave, and true incremental
(P^2-quantile) streaming statistics for the online poison guard.
"""
from .bus import BoundedBus, BusMetrics
from .clock import StreamClock
from .online_guard import OnlineRobustGuard
from .pquantile import P2Quantile
from .runner import StreamResult, StreamRunner
from .source import StreamSource, onset, poison_next

__all__ = [
    "StreamClock", "BoundedBus", "BusMetrics", "P2Quantile",
    "OnlineRobustGuard", "StreamSource", "onset", "poison_next",
    "StreamRunner", "StreamResult",
]
