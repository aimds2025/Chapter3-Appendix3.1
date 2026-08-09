"""
Cohort tests: assert each labeled synthetic cohort triggers the intended
layer behaviour. Requires the dataset:

    python data/generate_synthetic.py
    python -m pytest tests/test_synthetic_cohorts.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio_pipeline import PhysioPipeline
from physio_pipeline.layer1_edge import load_cohort
from physio_pipeline.layer4_processing import embed

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def _run():
    devices = load_cohort(DATA)
    cohorts = {d.patient_id: d.cohort for d in devices}
    result = PhysioPipeline().run_batch(devices, windows_per_device=4)
    kinds: dict[str, set[str]] = {}
    for a in result.batch.alarms:
        kinds.setdefault(cohorts[a.patient_id], set()).add(a.kind)
    return result, kinds


def test_dataset_present():
    assert DATA.exists() and list(DATA.glob("*.npz")), \
        "run: python data/generate_synthetic.py"


def test_cohort_alarms():
    _, kinds = _run()
    assert "tachycardia" in kinds.get("tachycardia", set())
    assert "bradycardia" in kinds.get("bradycardia", set())
    assert "signal_loss" in kinds.get("signal_loss", set())
    # normal patients must not raise critical rate alarms
    assert not ({"tachycardia", "bradycardia"} & kinds.get("normal", set()))


def test_rate_alarms_gated_on_signal_quality():
    """A noise-only trace must not produce a false tachycardia alarm."""
    _, kinds = _run()
    assert not ({"tachycardia", "bradycardia"} & kinds.get("signal_loss", set()))


def test_poison_veto_fires_and_dq_filter_is_separate():
    result, _ = _run()
    assert result.batch.rejected > 0, "poisoned cohort should trip the veto"
    assert result.batch.dq_filtered > 0, "low-SQI windows should be DQ-filtered"


def test_amplitude_scaling_is_invisible_to_the_embedding():
    """
    Regression guard for a real finding: every embedding feature is
    scale-invariant, so an amplitude-only attack cannot be a poison.
    """
    dev = load_cohort(DATA)[0]
    frame = dev.acquire()
    e1 = embed(dev.emit(frame))
    frame.samples = frame.samples * 50.0
    e2 = embed(dev.emit(frame))
    assert np.allclose(e1, e2, atol=1e-6), "features unexpectedly scale-dependent"


def test_robust_guard_keeps_rare_pathology():
    """Bradycardia is rare but real: it must NOT be vetoed as adversarial."""
    devices = load_cohort(DATA)
    result = PhysioPipeline().run_batch(devices, windows_per_device=4)
    # 2 poisoned patients contribute 1 fabricated window each
    assert result.batch.rejected <= 2, \
        f"guard vetoed {result.batch.rejected} points; suspect clinical false positives"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("all cohort tests passed")
