"""
End-to-end demo: push synthetic 250 Hz vitals from several bedside devices
through all eight layers and print what each layer produced.

    python -m examples.run_demo      (from the package root)
    # or:  python examples/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio_pipeline import EdgeDevice, PhysioPipeline  # noqa: E402


def main() -> None:
    pipe = PhysioPipeline(total_epsilon=5.0)

    devices = [
        EdgeDevice(patient_id="P001", device_id="MON-A"),
        EdgeDevice(patient_id="P002", device_id="MON-B"),
        # P003 is tachycardic -> exercises the Layer 4 alarm path:
        EdgeDevice(patient_id="P003", device_id="MON-C", hr_range=(130, 160)),
    ]

    result = pipe.run_batch(devices, windows_per_device=5)

    print("=" * 68)
    print("PHYSIO PIPELINE - END-TO-END RUN")
    print("=" * 68)

    print("\n[L4] Stream processing")
    print(f"  alarms:        {len(result.batch.alarms)}")
    for a in result.batch.alarms[:5]:
        print(f"    - {a.patient_id} {a.kind:12s} [{a.severity}] {a.detail}")
    print(f"  coreset size:  {len(result.batch.coreset)}")
    print(f"  rejected(guard): {result.batch.rejected}")
    print(f"  DQ metrics:    {result.batch.dq_metrics}")

    print("\n[L5] Storage")
    print(f"  raw WORM uri:  {result.receipt.raw_lake_uri}")
    print(f"  raw sha256:    {result.receipt.raw_sha256[:16]}...")
    print(f"  TSDB points:   {result.receipt.tsdb_points}")
    print(f"  feature keys:  {len(result.receipt.feature_keys)}")

    print("\n[L6] MCP gateway (agent tool call)")
    print(f"  tool:          {result.agent_response['tool']}")
    print(f"  output:        {result.agent_response['output']}")
    print(f"  hmac:          {result.agent_response['hmac'][:16]}...")

    print("\n[L7] Batch training (DP-SGD)")
    print(f"  model:         {result.model.name} v{result.model.version}")
    print(f"  metrics:       {result.model.metrics}")
    print(f"  eps spent:     {result.model.dp_epsilon_spent}")
    print(f"  stage:         {result.model.stage}")

    print("\n[L8] Governance & compliance")
    print(f"  audit events:  {result.compliance.audit_events}")
    print(f"  chain head:    {result.compliance.audit_chain_head[:16]}...")
    print(f"  DP remaining:  {result.compliance.privacy_budget_remaining}")
    for control, ok in result.compliance.controls.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {control}")

    print("\n[cross-cutting] Trace timeline (ms)")
    for name, ms in result.trace:
        print(f"    {name:40s} {ms:8.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
