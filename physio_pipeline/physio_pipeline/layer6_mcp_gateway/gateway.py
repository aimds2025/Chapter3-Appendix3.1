"""
Layer 6 - MCP SECURITY GATEWAY (AI agent access layer)
=====================================================

Mediates every AI-agent tool call against the stored data:

  * OAuth 2.1 bearer-token check           (stub verifier)
  * static tool registry                   (allowlist; no dynamic tools)
  * RBAC scope enforcement                 (via cross-cutting IAM)
  * prompt firewall                        (prompt-injection patterns)
  * PHI/PII output scrubber                (regex redaction)
  * HMAC-SHA256 response signing
  * human-in-the-loop (HITL) for writes

Production swap-in: an OAuth 2.1 AS with PKCE, an MCP server exposing a vetted
static tool registry, OPA for authz, an output DLP/scrubber, KMS-backed signing.
"""
from __future__ import annotations

import hmac
import hashlib
import re

from ..core.exceptions import AuthorizationError
from ..crosscutting.audit import AuditLog
from ..crosscutting.iam import IAM

_INJECTION = [
    re.compile(r"ignore (all|previous) instructions", re.I),
    re.compile(r"exfiltrate|dump .*credentials", re.I),
]
_PHI = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),        # SSN-like
    re.compile(r"\bMRN[:#]?\s*\d+\b", re.I),     # medical record number
]


class MCPGateway:
    def __init__(self, iam: IAM, audit: AuditLog, signing_key: bytes = b"kms-managed"):
        self.iam = iam
        self.audit = audit
        self._key = signing_key
        # static allowlist: tool -> (required_scope, is_write)
        self._tools: dict[str, tuple[str, bool]] = {
            "get_recent_vitals": ("tool:read", False),
            "summarize_trends":  ("tool:read", False),
            "annotate_chart":    ("tool:read", True),   # write => needs HITL
        }

    def _verify_token(self, token: str) -> bool:
        return token.startswith("Bearer ")           # stub

    def _prompt_firewall(self, prompt: str) -> None:
        if any(p.search(prompt) for p in _INJECTION):
            raise AuthorizationError("prompt firewall: injection detected")

    def _scrub(self, text: str) -> str:
        for p in _PHI:
            text = p.sub("[REDACTED]", text)
        return text

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def call_tool(self, spiffe_id: str, token: str, tool: str, prompt: str,
                  result: str, hitl_approved: bool = False) -> dict:
        if not self._verify_token(token):
            raise AuthorizationError("invalid OAuth 2.1 token")
        if tool not in self._tools:
            raise AuthorizationError(f"tool '{tool}' not in static registry")
        scope, is_write = self._tools[tool]
        self.iam.authorize(spiffe_id, scope)          # RBAC
        self._prompt_firewall(prompt)
        if is_write and not hitl_approved:
            self.audit.record("layer6", "hitl_block", spiffe_id, tool=tool, ok=False)
            raise AuthorizationError(f"write tool '{tool}' requires HITL approval")
        clean = self._scrub(result)
        sig = self._sign(clean)
        self.audit.record("layer6", "call_tool", spiffe_id, tool=tool, ok=True)
        return {"tool": tool, "output": clean, "hmac": sig}
