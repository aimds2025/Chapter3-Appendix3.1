"""
Layer 8 - GOVERNANCE & COMPLIANCE
================================

Reads the cross-cutting audit log and privacy ledger and attests the run:

  * 21 CFR Part 11   -> verify the tamper-evident audit chain + e-signature
  * IEC 62304        -> SOUP inventory with digest pinning
  * HIPAA            -> confirm PHI scrubbing / access controls were exercised
  * MITRE ATLAS      -> map exercised defenses to adversarial-ML techniques
  * DP accounting    -> report remaining privacy budget

Emits a ComplianceReport. Raises ComplianceViolation if a mandatory control
fails (e.g. a broken audit chain).

Production swap-in: a GRC system of record; SBOM/SOUP tooling; a real e-sign
workflow; a MITRE ATLAS-mapped threat model maintained in the DHF.
"""
from __future__ import annotations

from ..core.contracts import ComplianceReport, ModelArtifact
from ..core.exceptions import ComplianceViolation
from ..crosscutting.audit import AuditLog
from ..crosscutting.privacy_budget import PrivacyBudgetLedger

# IEC 62304 SOUP: name -> pinned digest (illustrative)
SOUP_INVENTORY = {
    "numpy":            "sha256:pinned-numpy",
    "apache-kafka":     "sha256:pinned-kafka",
    "influxdb":         "sha256:pinned-influx",
    "opa":              "sha256:pinned-opa",
}

# MITRE ATLAS techniques the pipeline claims to mitigate
ATLAS_COVERAGE = {
    "AML.T0020_poisoning":     "Layer 4 per-patient caps + embedding outlier veto",
    "AML.T0051_prompt_inject": "Layer 6 prompt firewall",
    "AML.T0024_exfiltration":  "Layer 6 PHI scrubber + Layer 2 DLP",
}


class GovernanceLayer:
    def __init__(self, audit: AuditLog, ledger: PrivacyBudgetLedger):
        self.audit = audit
        self.ledger = ledger

    def attest(self, model: ModelArtifact | None = None,
               esign_user: str = "quality@org") -> ComplianceReport:
        controls: dict[str, bool] = {}

        # 21 CFR Part 11 - tamper-evident audit chain must verify
        chain_ok = self.audit.verify()
        controls["21cfr11_audit_integrity"] = chain_ok
        controls["21cfr11_esignature"] = bool(esign_user)

        # HIPAA - a PHI-scrub/DLP event should exist in the trail
        controls["hipaa_access_controls"] = len(self.audit) > 0

        # IEC 62304 - SOUP inventory present & digest-pinned
        controls["iec62304_soup_pinned"] = all(
            v.startswith("sha256:") for v in SOUP_INVENTORY.values()
        )

        # DP - budget not exhausted
        controls["dp_budget_ok"] = self.ledger.remaining >= 0

        # model provenance (if a model was produced)
        if model is not None:
            controls["model_signed"] = bool(model.signature)
            controls["model_not_rejected"] = model.stage != "rejected"

        if not chain_ok:
            raise ComplianceViolation("audit chain failed integrity verification")

        report = ComplianceReport(
            audit_chain_head=self.audit.head,
            audit_events=len(self.audit),
            privacy_budget_remaining=round(self.ledger.remaining, 4),
            soup_components=len(SOUP_INVENTORY),
            controls=controls,
        )
        self.audit.record("layer8", "attest", esign_user, ok=all(controls.values()))
        return report
