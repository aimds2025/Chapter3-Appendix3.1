"""
Run the on-disk synthetic cohort through all eight layers.

    python data/generate_synthetic.py          # 1. write the dataset
    python examples/run_with_synthetic_data.py # 2. replay it through the pipeline

Unlike `run_demo.py` (which synthesizes on the fly), this reads recorded
waveforms via `ReplayDevice`, so runs are reproducible and the cohort labels
let us assert that each layer behaved as intended.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio_pipeline import PhysioPipeline                     # noqa: E402
from physio_pipeline.layer1_edge import load_cohort             # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def main() -> None:
    devices = load_cohort(DATA_DIR)
    cohorts = {d.patient_id: d.cohort for d in devices}
    print(f"Loaded {len(devices)} patients from {DATA_DIR}")
    for d in devices:
        print(f"  {d.patient_id}  {d.cohort:13s} true HR {d.hr_true_bpm:5.1f} bpm "
              f"({d.n_windows} windows available)")

    pipe = PhysioPipeline(total_epsilon=5.0)
    result = pipe.run_batch(devices, windows_per_device=4)

    # ---------------- Layer 4: did each cohort behave as labeled? -------------
    print("\n[L4] Alarms by patient/cohort")
    by_patient: dict[str, Counter] = {}
    for a in result.batch.alarms:
        by_patient.setdefault(a.patient_id, Counter())[a.kind] += 1
    for pid in sorted(cohorts):
        kinds = by_patient.get(pid)
        summary = ", ".join(f"{k}x{v}" for k, v in kinds.items()) if kinds else "-"
        print(f"  {pid}  {cohorts[pid]:13s} {summary}")

    print(f"\n  coreset size:        {len(result.batch.coreset)}")
    print(f"  poison vetoes:       {result.batch.rejected}")
    print(f"  DQ-filtered (low SQI): {result.batch.dq_filtered}")
    print(f"  flood-capped:        {result.batch.capped}")
    print(f"  DQ metrics:          {result.batch.dq_metrics}")

    # ---------------- validation assertions ---------------------------------
    print("\n[validation] expected behaviour per cohort")
    checks: list[tuple[str, bool]] = []

    def kinds_for(cohort: str) -> set[str]:
        out: set[str] = set()
        for pid, c in cohorts.items():
            if c == cohort:
                out |= set(by_patient.get(pid, {}))
        return out

    checks.append(("normal      -> no critical rate alarms",
                   not ({"tachycardia", "bradycardia"} & kinds_for("normal"))))
    checks.append(("tachycardia -> tachycardia alarm",
                   "tachycardia" in kinds_for("tachycardia")))
    checks.append(("bradycardia -> bradycardia alarm",
                   "bradycardia" in kinds_for("bradycardia")))
    checks.append(("signal_loss -> signal_loss warning",
                   "signal_loss" in kinds_for("signal_loss")))
    checks.append(("signal_loss -> rate alarms GATED (no false tachy)",
                   not ({"tachycardia", "bradycardia"} & kinds_for("signal_loss"))))
    checks.append(("poisoned    -> poison veto fired",
                   result.batch.rejected > 0))
    checks.append(("signal_loss -> DQ-filtered from coreset (not called poison)",
                   result.batch.dq_filtered > 0))

    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    # ---------------- downstream layers --------------------------------------
    print("\n[L5] Storage")
    print(f"  WORM uri: {result.receipt.raw_lake_uri}")
    print(f"  sha256:   {result.receipt.raw_sha256[:16]}...")
    print(f"  TSDB pts: {result.receipt.tsdb_points}   features: {len(result.receipt.feature_keys)}")

    print("\n[L6] MCP gateway")
    print(f"  output:   {result.agent_response['output']}")

    print("\n[L7] DP-SGD training")
    print(f"  {result.model.name} v{result.model.version}  "
          f"metrics={result.model.metrics}  eps={result.model.dp_epsilon_spent}  "
          f"stage={result.model.stage}")

    print("\n[L8] Governance")
    for control, ok in result.compliance.controls.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {control}")
    print(f"  DP budget remaining: {result.compliance.privacy_budget_remaining}")

    failed = [l for l, ok in checks if not ok]
    print(f"\n{'ALL COHORT CHECKS PASSED' if not failed else 'FAILURES: ' + str(failed)}")


if __name__ == "__main__":
    main()
