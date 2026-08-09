"""
Streaming tests. All run at speed=0 (no sleep) for speed and determinism.

    python -m pytest tests/test_streaming.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio_pipeline.layer1_edge import EdgeDevice
from physio_pipeline.streaming import (
    BoundedBus, OnlineRobustGuard, P2Quantile, StreamClock, StreamRunner,
    StreamSource, onset, poison_next,
)


def _build(seed: int = 7, maxsize: int = 6, spike=(24.0, 36.0, 0.25)):
    devices = [EdgeDevice("P001", "MON-A", seed=1),
               EdgeDevice("P002", "MON-B", seed=2),
               EdgeDevice("P003", "MON-C", seed=3)]
    src = StreamSource(devices, window_s=4.0, duration_s=60.0,
                       jitter_s=0.3, late_prob=0.1, late_max_s=1.0, seed=seed)
    src.schedule(21.0, "P002", onset(140, 170), "tachycardia onset")
    src.schedule(44.0, "P003", poison_next(), "poisoned window")
    runner = StreamRunner(src, clock=StreamClock(0.0),
                          bus=BoundedBus(maxsize=maxsize, policy="drop_oldest"))
    return runner, spike


def test_p2_quantile_accuracy():
    rng = np.random.default_rng(1)
    data = rng.normal(70, 12, 10000)
    est = P2Quantile(0.5)
    for x in data:
        est.update(x)
    assert abs(est.value - np.percentile(data, 50)) < 1.0


def test_stream_is_deterministic_at_speed_zero():
    r1, s1 = _build(seed=7)
    r2, s2 = _build(seed=7)
    a = r1.run(load_spike=s1)
    b = r2.run(load_spike=s2)
    assert (a.produced, a.consumed, a.dropped, a.vetoed) == \
           (b.produced, b.consumed, b.dropped, b.vetoed)
    assert a.alarms == b.alarms


def test_onset_is_detected_after_it_occurs():
    runner, spike = _build()
    res = runner.run(load_spike=spike)
    tachy = [(ts, pid) for ts, pid, kind in res.alarms
             if kind == "tachycardia" and pid == "P002"]
    assert tachy, "no tachycardia alarm for P002 after onset"
    assert all(ts >= 21.0 for ts, _ in tachy), "alarm fired before onset"
    assert "P002" in res.detection_latency_s
    assert res.detection_latency_s["P002"] >= 0.0


def test_backpressure_drops_and_bounds_depth():
    runner, spike = _build(maxsize=6)
    res = runner.run(load_spike=spike)
    assert res.dropped > 0, "load spike should have caused drops"
    assert res.max_depth <= 6, "bus exceeded its maxsize"
    assert res.consumed + res.dropped == res.produced


def test_no_drops_without_a_load_spike():
    runner, _ = _build(maxsize=6)
    res = runner.run(load_spike=None)
    assert res.dropped == 0, "steady state should not drop"


def test_online_guard_vetoes_injected_poison():
    runner, spike = _build()
    res = runner.run(load_spike=spike)
    assert res.vetoed >= 1, "injected poisoned window was not vetoed"
    assert res.coreset_admitted > 0


def test_guard_warmup_admits_everything_initially():
    g = OnlineRobustGuard(n_features=4, warmup=5)
    for _ in range(5):
        accept, z = g.check(np.array([0.5, 0.03, 0.8, 0.05]))
        assert accept and z == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("all streaming tests passed")
