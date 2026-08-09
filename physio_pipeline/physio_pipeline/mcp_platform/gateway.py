"""
mcp_platform.gateway
====================

The gateway middleware every tool call passes through (Section 9.6). It applies,
in order: authentication, scope-and-tenant authorization, prompt-injection
firewall, and HITL gating on writes (enforce()); then PHI/PII output scrub and
HMAC signing on the result (finalize()). The generic control catalog
(injection firewall, PHI scrub, signing) is the Layer 6 set of Section 8.5; the
multi-tenant additions here are the tenant filter and the per-(principal x
tenant) scope check.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re

from ..core.exceptions import AuthorizationError
from ..crosscutting.audit import AuditLog
from ..crosscutting.iam import Principal, PrincipalRegistry
from .access import TOOL_REGISTRY, ToolSpec

_INJECTION = [
    re.compile(r"ignore (all|previous) instructions", re.I),
    re.compile(r"exfiltrate|dump .*credentials|reveal .*token", re.I),
]
_PHI = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),        # SSN-like
    re.compile(r"\bMRN[:#]?\s*\d+\b", re.I),     # medical record number
]


class InsufficientScope(AuthorizationError):
    """403-equivalent: valid principal, but missing scope or tenant claim."""


class TenantDenied(AuthorizationError):
    """Requested tenant is outside the principal's allowed set."""


class HITLRequired(AuthorizationError):
    """A write tool was called without human-in-the-loop approval."""


class MCPGatewayMiddleware:
    def __init__(self, principals: PrincipalRegistry, audit: AuditLog,
                 signing_key: bytes = b"kms-managed"):
        self.principals = principals
        self.audit = audit
        self._key = signing_key

    @staticmethod
    def _verify_token(token: str) -> bool:
        return isinstance(token, str) and token.startswith("Bearer ")   # stub

    def _prompt_firewall(self, prompt: str | None) -> None:
        if prompt and any(p.search(prompt) for p in _INJECTION):
            raise AuthorizationError("prompt firewall: injection detected")

    def enforce(self, principal_id: str, token: str, tool_name: str,
                tenant_id: str | None, prompt: str | None,
                write_approved: bool) -> tuple[Principal, ToolSpec]:
        """Run all pre-execution checks; return (principal, spec) or raise."""
        if not self._verify_token(token):
            raise AuthorizationError("invalid OAuth 2.1 token")

        spec = TOOL_REGISTRY.get(tool_name)
        if spec is None:
            raise AuthorizationError(f"tool '{tool_name}' not in static registry")

        principal = self.principals.get(principal_id)          # raises if unknown

        # scope check (native 403 insufficient_scope in real MCP)
        if not principal.has(spec.scope):
            self.audit.record("layer6", "authz", principal_id, ok=False,
                              tool=tool_name, need=spec.scope)
            raise InsufficientScope(
                f"principal '{principal_id}' lacks scope '{spec.scope}' "
                f"(has {sorted(principal.scopes)})")

        # tenant filter (data-plane tools only; admin tools carry no tenant)
        if spec.tier is not None:
            if tenant_id is None:
                raise TenantDenied(f"tool '{tool_name}' requires a tenant")
            if not principal.may_see(tenant_id):
                self.audit.record("layer6", "tenant_filter", principal_id,
                                  ok=False, tool=tool_name, tenant=tenant_id)
                raise TenantDenied(
                    f"principal '{principal_id}' may not access tenant "
                    f"'{tenant_id}' (allowed {sorted(principal.tenants)})")

        self._prompt_firewall(prompt)

        if spec.write and not write_approved:
            self.audit.record("layer6", "hitl_block", principal_id, ok=False,
                              tool=tool_name)
            raise HITLRequired(f"write tool '{tool_name}' requires HITL approval")

        return principal, spec

    # ---- output side ----
    def _scrub(self, obj):
        if isinstance(obj, str):
            for p in _PHI:
                obj = p.sub("[REDACTED]", obj)
            return obj
        if isinstance(obj, dict):
            return {k: self._scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._scrub(v) for v in obj]
        return obj

    def finalize(self, result) -> dict:
        clean = self._scrub(result)
        payload = json.dumps(clean, sort_keys=True, default=str).encode()
        sig = hmac.new(self._key, payload, hashlib.sha256).hexdigest()
        return {"result": clean, "hmac": sig}
