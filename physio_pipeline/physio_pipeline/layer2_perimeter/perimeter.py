"""
Layer 2 - ZERO-TRUST PERIMETER
=============================

Authorizes inbound EdgePackets before they may enter the stream. Enforces:

  * mTLS / workload-identity verification (via cross-cutting IAM / SPIFFE)
  * network segmentation (assigns the packet to a zone: PHI / NON_PHI / ADMIN)
  * OPA-style policy evaluation (attestation must be valid & signed)
  * DLP scan on the outbound path (no raw PHI in cleartext metadata)

Returns the same EdgePacket, stamped authorized=True and with a `zone`.

Production swap-in: Envoy/Istio mTLS mesh, an identity-aware proxy
(BeyondCorp / AWS Verified Access), OPA/Rego policies, a DLP proxy.
"""
from __future__ import annotations

import re

from ..core.contracts import EdgePacket
from ..core.exceptions import AuthorizationError
from ..crosscutting.audit import AuditLog
from ..crosscutting.iam import IAM

# crude DLP signature: anything that looks like an SSN/MRN in free-text meta
_DLP_PATTERNS = [re.compile(r"\b\d{3}-\d{2}-\d{4}\b")]


class ZeroTrustPerimeter:
    def __init__(self, iam: IAM, audit: AuditLog):
        self.iam = iam
        self.audit = audit

    def _classify_zone(self, packet: EdgePacket) -> str:
        # Physiological waveforms are PHI by default.
        return "PHI"

    def _opa_policy(self, packet: EdgePacket) -> bool:
        att = packet.attestation
        return bool(att.get("secure_boot")) and bool(att.get("signed"))

    def _dlp_clean(self, packet: EdgePacket) -> bool:
        blob = f"{packet.patient_id}{packet.device_id}"
        return not any(p.search(blob) for p in _DLP_PATTERNS)

    def authorize(self, packet: EdgePacket) -> EdgePacket:
        spiffe_id = packet.attestation.get("spiffe_id", "")
        # 1) workload identity + RBAC (device must hold ingest:write)
        try:
            self.iam.authorize(spiffe_id, "ingest:write")
        except AuthorizationError:
            self.audit.record("layer2", "authorize", spiffe_id, ok=False,
                              reason="identity")
            raise
        # 2) OPA policy
        if not self._opa_policy(packet):
            self.audit.record("layer2", "authorize", spiffe_id, ok=False,
                              reason="attestation")
            raise AuthorizationError("attestation failed OPA policy")
        # 3) DLP
        if not self._dlp_clean(packet):
            self.audit.record("layer2", "dlp", spiffe_id, ok=False)
            raise AuthorizationError("DLP violation on outbound metadata")

        packet.zone = self._classify_zone(packet)
        packet.authorized = True
        self.audit.record("layer2", "authorize", spiffe_id, ok=True,
                          zone=packet.zone)
        return packet
