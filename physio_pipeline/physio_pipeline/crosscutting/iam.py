"""
crosscutting.iam
================

Cross-cutting identity & access management. Modeled on SPIFFE/SPIRE workload
identity + RBAC. Used by Layer 2 (perimeter) and Layer 6 (MCP gateway).

Production swap-in: SPIRE server for SVIDs, OPA/Rego for policy, an OAuth 2.1
authorization server for tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.exceptions import AuthorizationError


@dataclass
class Identity:
    """A SPIFFE-style workload identity: spiffe://trust-domain/workload."""
    spiffe_id: str
    roles: frozenset[str]

    @property
    def trust_domain(self) -> str:
        return self.spiffe_id.split("/")[2] if "://" in self.spiffe_id else ""


class IAM:
    """Minimal RBAC engine with a role -> permitted-scopes map."""

    def __init__(self) -> None:
        self._registry: dict[str, Identity] = {}
        self._role_scopes: dict[str, frozenset[str]] = {
            "device":        frozenset({"ingest:write"}),
            "clinician_ro":  frozenset({"tsdb:read", "tool:read"}),
            "ml_trainer":    frozenset({"coreset:read", "model:write"}),
            "auditor":       frozenset({"audit:read", "ledger:read"}),
        }

    def register(self, spiffe_id: str, roles: set[str]) -> Identity:
        ident = Identity(spiffe_id=spiffe_id, roles=frozenset(roles))
        self._registry[spiffe_id] = ident
        return ident

    def authorize(self, spiffe_id: str, scope: str) -> Identity:
        """Return the Identity if it holds `scope`, else raise."""
        ident = self._registry.get(spiffe_id)
        if ident is None:
            raise AuthorizationError(f"unknown identity: {spiffe_id}")
        granted: set[str] = set()
        for role in ident.roles:
            granted |= self._role_scopes.get(role, frozenset())
        if scope not in granted:
            raise AuthorizationError(
                f"{spiffe_id} lacks scope '{scope}' (has {sorted(granted)})"
            )
        return ident


# --------------------------------------------------------------------------- #
# Section 9: multi-tenant principals (ML-project consumers)
# --------------------------------------------------------------------------- #
@dataclass
class Principal:
    """
    An ML-project consumer (Section 9.3). Its grant is the triple
    (allowed tenants x scopes x DP budget). Scopes here are TOOL scopes
    (e.g. 'dp:read', 'feature:read', 'raw:request', 'model:write'); the DP
    budget is tracked separately in a TenantPrivacyLedger.
    """
    principal_id: str                 # e.g. "spiffe://ml/project-alpha"
    scopes: set[str] = field(default_factory=set)
    tenants: set[str] = field(default_factory=set)   # allowed hospital ids

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def may_see(self, tenant_id: str) -> bool:
        return tenant_id in self.tenants


class PrincipalRegistry:
    """
    Deny-by-default registry of ML-project principals, plus the admin grant
    operations exposed by the admin/IAM MCP server (Section 9.6).
    """

    def __init__(self) -> None:
        self._principals: dict[str, Principal] = {}

    def register(self, principal_id: str, scopes: set[str] | None = None,
                 tenants: set[str] | None = None) -> Principal:
        p = Principal(principal_id, set(scopes or set()), set(tenants or set()))
        self._principals[principal_id] = p
        return p

    def get(self, principal_id: str) -> Principal:
        p = self._principals.get(principal_id)
        if p is None:
            raise AuthorizationError(f"unknown principal: {principal_id}")
        return p

    def grant_tenant(self, principal_id: str, tenant_id: str) -> None:
        self.get(principal_id).tenants.add(tenant_id)

    def grant_scope(self, principal_id: str, scope: str) -> None:
        self.get(principal_id).scopes.add(scope)

    def revoke_tenant(self, principal_id: str, tenant_id: str) -> None:
        self.get(principal_id).tenants.discard(tenant_id)

    def list_principals(self) -> list[dict]:
        return [
            {"principal_id": p.principal_id,
             "scopes": sorted(p.scopes), "tenants": sorted(p.tenants)}
            for p in self._principals.values()
        ]
