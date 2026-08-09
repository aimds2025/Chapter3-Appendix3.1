"""
Multi-tenant streaming demo (Section 9 + streaming).

Streams the multi-hospital synthetic cohort as one interleaved live feed, then
serves tenant-scoped DP queries over the resulting per-tenant stores through the
MCP gateway.

    python data/generate_synthetic.py --hospitals 3
    python examples/run_streaming_multitenant.py
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio_pipeline.layer1_edge import load_cohort                       # noqa: E402
from physio_pipeline.streaming import (                                   # noqa: E402
    BoundedBus, StreamClock, StreamRunner, StreamSource)
from physio_pipeline.mcp_platform import AccessTier, MultiTenantPlatform  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
TOKEN = "Bearer demo"


def main() -> None:
    devices = load_cohort(DATA)
    pid2hosp = {d.patient_id: d.hospital_id for d in devices}
    hospitals = sorted(set(pid2hosp.values()))

    print("=" * 68)
    print("MULTI-TENANT STREAMING  (interleaved live feed across hospitals)")
    print("=" * 68)
    print(f"\n{len(devices)} devices across {len(hospitals)} hospitals: {hospitals}")

    # ---- 1) stream everything as one interleaved feed ----
    source = StreamSource(devices, window_s=4.0, duration_s=40.0,
                          jitter_s=0.3, late_prob=0.1, late_max_s=1.0, seed=7)
    runner = StreamRunner(source, clock=StreamClock(0.0),
                          bus=BoundedBus(maxsize=8, policy="drop_oldest"))
    res = runner.run()

    print(f"\nStream: produced={res.produced} consumed={res.consumed} "
          f"dropped={res.dropped} alarms={len(res.alarms)}")
    by_hosp_alarms: dict[str, Counter] = defaultdict(Counter)
    for _ts, pid, kind in res.alarms:
        by_hosp_alarms[pid2hosp.get(pid, "?")][kind] += 1
    print("Alarms by hospital:")
    for h in hospitals:
        c = by_hosp_alarms.get(h)
        print(f"  {h}: {dict(c) if c else '{}'}")

    # ---- 2) serve governed, tenant-scoped MCP queries over the stores ----
    plat = MultiTenantPlatform()
    plat.populate_from_devices(devices, windows_per_device=5)
    # a population-analytics project cleared for all hospitals at DP tier
    plat.register_project_at_tier("spiffe://ml/population", AccessTier.DP_AGGREGATE,
                                  set(hospitals))
    print("\nTenant-scoped DP queries (population project, DP-aggregate tier):")
    for h in hospitals:
        out = plat.call("spiffe://ml/population", TOKEN, "query_vitals_dp",
                        tenant=h, epsilon=0.5)["result"]
        print(f"  {h}: DP hr_mean={out['value']} bpm  (n={out['n']}, "
              f"noise_sigma={out['noise_sigma']}, eps_left={out['budget_remaining']})")

    print("\nNote: DP noise is large on these toy per-hospital cohorts (small n); "
          "with population-scale n and Renyi-DP composition the estimates tighten.")
    print("\nDone.")


if __name__ == "__main__":
    main()
