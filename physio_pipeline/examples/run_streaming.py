"""
Streaming simulation demo.

Three patients stream continuously. Mid-stream:
  * P002 goes into tachycardia at t=21s  (watch detection latency)
  * a load spike from t=24-36s makes the consumer fall behind -> bus drops
  * P003 emits a fabricated (poisoned) window at t=44s (after the spike)

Run accelerated (default 60x) so 60s of stream takes ~1s of wall-clock:
    python examples/run_streaming.py
    python examples/run_streaming.py --speed 1     # real time
    python examples/run_streaming.py --speed 0     # as fast as possible
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio_pipeline.layer1_edge import EdgeDevice                       # noqa: E402
from physio_pipeline.streaming import (                                  # noqa: E402
    BoundedBus, StreamClock, StreamRunner, StreamSource, onset, poison_next,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=60.0,
                    help="clock speed: 1=real time, 60=accelerated, 0=no sleep")
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    devices = [
        EdgeDevice("P001", "MON-A", seed=1),               # stays normal
        EdgeDevice("P002", "MON-B", seed=2),               # -> tachycardia @21s
        EdgeDevice("P003", "MON-C", seed=3),               # -> poison @44s
    ]
    source = StreamSource(devices, window_s=4.0, duration_s=args.duration,
                          jitter_s=0.3, late_prob=0.1, late_max_s=1.0, seed=7)
    source.schedule(21.0, "P002", onset(140, 170), "tachycardia onset")
    source.schedule(44.0, "P003", poison_next(), "poisoned window")

    clock = StreamClock(speed=args.speed)
    bus = BoundedBus(maxsize=6, policy="drop_oldest")
    runner = StreamRunner(source, clock=clock, bus=bus)

    # consumer drains at 25% during the spike window -> backpressure
    result = runner.run(load_spike=(24.0, 36.0, 0.25))

    print("=" * 66)
    print(f"STREAMING SIMULATION  (speed={args.speed}x, {args.duration:.0f}s stream)")
    print("=" * 66)
    print(f"\n{result.summary()}")
    print(f"real elapsed: {result.real_elapsed_s:.2f}s   "
          f"throughput: {result.throughput_pps:.0f} pkt/s")

    print("\nScheduled events:")
    for at_s, pid, label in result.onsets:
        print(f"  t={at_s:5.1f}s  {pid}  {label}")

    print("\nDetection latency (first alarm after onset):")
    if result.detection_latency_s:
        for pid, lat in result.detection_latency_s.items():
            print(f"  {pid}: {lat:.1f}s after onset")
    else:
        print("  (none)")

    print("\nAlarms over time (first 8):")
    for ts, pid, kind in result.alarms[:8]:
        print(f"  t={ts:5.1f}s  {pid}  {kind}")

    print("\nBackpressure:")
    print(f"  bus maxsize=6, policy=drop_oldest")
    print(f"  admitted={result.consumed}  dropped={result.dropped} "
          f"({result.drop_rate:.1%})  peak depth={result.max_depth}")

    print("\nOnline poison guard (P^2 streaming stats):")
    print(f"  coreset admitted={result.coreset_admitted}  vetoed={result.vetoed}")

    print(f"\nAudit: {result.audit_events} events, chain head "
          f"{result.audit_chain_head[:16]}...")


if __name__ == "__main__":
    main()
