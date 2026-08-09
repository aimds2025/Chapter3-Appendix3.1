"""
Per-layer walkthrough
======================

Drives one batch through every layer BY HAND (instead of the wired
`run_batch()`), printing the data contract each layer emits so you can see
exactly what L1..L8 produce and how it flows.

    python examples/run_layers_verbose.py

L1/L2/L3 are shown in detail for the FIRST packet only (to keep the output
readable); the remaining packets are processed the same way and then L4..L8
run over the whole set.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio_pipeline import PhysioPipeline                    # noqa: E402
from physio_pipeline.layer1_edge import EdgeDevice            # noqa: E402


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    pipe = PhysioPipeline(total_epsilon=5.0)
    devices = [
        EdgeDevice("P001", "MON-A", seed=1),
        EdgeDevice("P002", "MON-B", seed=2),
        EdgeDevice("P003", "MON-C", seed=3, hr_range=(140, 165)),  # tachycardic
    ]
    windows_per_device = 5

    records = []
    first = True
    for dev in devices:
        pipe.register_device(dev)
        for _ in range(windows_per_device):
            # ---------------- LAYER 1 : EDGE ----------------
            frame = dev.acquire()                 # raw 250 Hz window
            packet = dev.emit(frame)              # on-device DSP -> EdgePacket
            if first:
                hr("LAYER 1 - EDGE (device / bedside)   output: EdgePacket")
                print(f"  patient/device : {packet.patient_id} / {packet.device_id}")
                print(f"  sampling       : {packet.fs_hz} Hz x {len(packet.channels)} ch "
                      f"{packet.channels}")
                print(f"  raw window     : {packet.n_samples} samples/ch "
                      f"({packet.n_samples/packet.fs_hz:.1f}s)")
                print(f"  DWT payload    : {len(packet.compressed)} bytes  "
                      f"(PRD {packet.prd_pct:.2f}%)")
                print(f"  QRS peaks      : {len(packet.qrs_indices)}  -> "
                      f"HR ~{60*len(packet.qrs_indices)/(packet.n_samples/packet.fs_hz):.0f} bpm")
                print(f"  SQI            : {packet.sqi:.3f}")
                print(f"  attestation    : {packet.attestation}")

            # ---------------- LAYER 2 : PERIMETER ----------------
            packet = pipe.perimeter.authorize(packet)
            if first:
                hr("LAYER 2 - ZERO-TRUST PERIMETER   output: EdgePacket (authorized)")
                print(f"  identity       : {packet.attestation['spiffe_id']}")
                print(f"  authorized     : {packet.authorized}")
                print(f"  network zone   : {packet.zone}")
                print(f"  checks passed  : mTLS identity + OPA policy + DLP scan")

            # ---------------- LAYER 3 : INGESTION ----------------
            rec = pipe.ingestion.produce(packet)
            records.append(rec)
            if first:
                hr("LAYER 3 - STREAM INGESTION   output: StreamRecord")
                print(f"  topic          : {rec.topic}")
                print(f"  partition key  : {rec.key}   offset: {rec.offset}")
                print(f"  schema version : v{rec.schema_version}")
                print(f"  (Kafka-style append-only log; 7-day replay window)")
                first = False

    print(f"\n  ... {len(records)} packets ingested from {len(devices)} patients "
          f"({windows_per_device} windows each).")

    # ---------------- LAYER 4 : PROCESSING ----------------
    batch = pipe.processor.process(records)
    hr("LAYER 4 - STREAM PROCESSING   output: ProcessedBatch")
    print(f"  alarms         : {len(batch.alarms)}")
    for a in batch.alarms[:4]:
        print(f"     - {a.patient_id} {a.kind} [{a.severity}] {a.detail}")
    if len(batch.alarms) > 4:
        print(f"     ... (+{len(batch.alarms)-4} more)")
    print(f"  coreset (Job2) : {len(batch.coreset)} points "
          f"(Sieve-Streaming, facility-location)")
    print(f"  poison vetoes  : {batch.rejected}")
    print(f"  DQ-filtered    : {batch.dq_filtered}   flood-capped: {batch.capped}")
    print(f"  DQ metrics     : {batch.dq_metrics}")

    # ---------------- LAYER 5 : STORAGE ----------------
    receipt = pipe.storage.persist("walkthrough", records, batch)
    hr("LAYER 5 - STORAGE (multi-layered)   output: StorageReceipt")
    print(f"  WORM raw lake  : {receipt.raw_lake_uri}")
    print(f"  sha-256 anchor : {receipt.raw_sha256}")
    print(f"  TSDB (hot)     : {receipt.tsdb_points} points")
    print(f"  feature store  : {len(receipt.feature_keys)} coreset keys")
    print(f"  e.g. key       : {receipt.feature_keys[0] if receipt.feature_keys else '-'}")

    # ---------------- LAYER 6 : MCP GATEWAY ----------------
    resp = pipe.gateway.call_tool(
        spiffe_id="spiffe://agents/clinical-copilot", token="Bearer demo",
        tool="summarize_trends",
        prompt="Summarize the last hour of vitals.",
        result=f"{receipt.tsdb_points} points; {len(batch.alarms)} alarms; MRN:12345 present.",
    )
    hr("LAYER 6 - MCP SECURITY GATEWAY   output: gated tool response")
    print(f"  tool           : {resp['tool']}")
    print(f"  output (scrubbed): {resp['output']}")
    print(f"  ^ note the MRN was redacted by the PHI scrubber")
    print(f"  HMAC signature : {resp['hmac'][:32]}...")

    # ---------------- LAYER 7 : TRAINING ----------------
    model = pipe.training.train(receipt.feature_keys)
    hr("LAYER 7 - BATCH ML TRAINING   output: ModelArtifact")
    print(f"  model          : {model.name} v{model.version}")
    print(f"  metrics        : {model.metrics}")
    print(f"  DP-SGD epsilon : {model.dp_epsilon_spent}")
    print(f"  stage          : {model.stage}")
    print(f"  SBOM           : {model.sbom}")
    print(f"  signature      : {model.signature[:32]}...")

    # ---------------- LAYER 8 : GOVERNANCE ----------------
    report = pipe.governance.attest(model=model)
    hr("LAYER 8 - GOVERNANCE & COMPLIANCE   output: ComplianceReport")
    print(f"  audit events   : {report.audit_events}")
    print(f"  audit chain head: {report.audit_chain_head[:32]}...")
    print(f"  DP budget left : {report.privacy_budget_remaining}")
    print(f"  SOUP components: {report.soup_components}")
    print(f"  controls       :")
    for name, ok in report.controls.items():
        print(f"     [{'PASS' if ok else 'FAIL'}] {name}")

    # ---------------- CROSS-CUTTING ----------------
    hr("CROSS-CUTTING   IAM . audit/SIEM . DP ledger . tracing")
    print(f"  audit log      : {len(pipe.audit)} events, chain verified="
          f"{pipe.audit.verify()}")
    print(f"  privacy ledger : {pipe.ledger.summary()}")
    print("\nWalkthrough complete - every layer's output shown above.")


if __name__ == "__main__":
    main()
