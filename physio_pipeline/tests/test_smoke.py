"""
Smoke tests: end-to-end run + a few per-layer invariants.
Run with:  python -m pytest -q     (or)   python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio_pipeline import EdgeDevice, PhysioPipeline
from physio_pipeline.core.exceptions import AuthorizationError, PrivacyBudgetExhausted
from physio_pipeline.crosscutting.privacy_budget import PrivacyBudgetLedger


def _devices():
    return [EdgeDevice("P001", "MON-A"), EdgeDevice("P002", "MON-B")]


def test_end_to_end_runs():
    pipe = PhysioPipeline()
    res = pipe.run_batch(_devices(), windows_per_device=4)
    assert res.receipt.tsdb_points > 0
    assert res.receipt.raw_sha256                       # L5 immutability anchor
    assert res.compliance.controls["21cfr11_audit_integrity"]  # L8


def test_audit_chain_tamper_evident():
    pipe = PhysioPipeline()
    pipe.run_batch(_devices(), windows_per_device=2)
    assert pipe.audit.verify()
    # tamper with an event and confirm verification now fails
    pipe.audit._events[0].meta["x"] = "tampered"
    assert not pipe.audit.verify()


def test_worm_is_immutable():
    pipe = PhysioPipeline()
    pipe.storage.lake.put("k/0.bin", b"first")
    try:
        pipe.storage.lake.put("k/0.bin", b"second")
        raise AssertionError("WORM store allowed an overwrite")
    except PermissionError:
        pass


def test_privacy_budget_enforced():
    led = PrivacyBudgetLedger(total_epsilon=1.0)
    led.debit("a", 0.6)
    try:
        led.debit("b", 0.6)
        raise AssertionError("ledger allowed budget overrun")
    except PrivacyBudgetExhausted:
        pass


def test_perimeter_rejects_bad_attestation():
    pipe = PhysioPipeline()
    dev = EdgeDevice("P001", "MON-A")
    pipe.register_device(dev)
    pkt = dev.emit(dev.acquire())
    pkt.attestation["signed"] = False                   # break the attestation
    try:
        pipe.perimeter.authorize(pkt)
        raise AssertionError("perimeter authorized a bad packet")
    except AuthorizationError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("all smoke tests passed")
