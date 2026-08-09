"""
mcp_platform.access
===================

The tiered data-access model (Section 9.4) and the static tool registry
(Section 9.6). Each tool is bound to a required scope, an access tier, whether
it is a write (HITL-gated), and which MCP server exposes it (data-plane vs.
admin/IAM). This registry is the single source of truth the gateway consults.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AccessTier(Enum):
    DP_AGGREGATE = "dp-aggregate"   # noised aggregates only; no row-level PHI
    FEATURE = "feature"             # de-identified feature / coreset vectors
    RAW = "raw"                     # raw / PHI -- HITL approval workflow only


@dataclass(frozen=True)
class ToolSpec:
    name: str
    scope: str                      # scope a principal must hold
    tier: AccessTier | None         # None for admin tools (no data tier)
    write: bool                     # write tools require HITL approval
    server: str                     # "data" | "admin"


# Static allowlist -- no dynamic tool registration (Section 9.6 / 8.5).
TOOL_REGISTRY: dict[str, ToolSpec] = {
    # ---- data-plane server ----
    "query_vitals_dp":   ToolSpec("query_vitals_dp",   "dp:read",      AccessTier.DP_AGGREGATE, False, "data"),
    "get_feature_batch": ToolSpec("get_feature_batch", "feature:read", AccessTier.FEATURE,      False, "data"),
    "get_coreset":       ToolSpec("get_coreset",       "feature:read", AccessTier.FEATURE,      False, "data"),
    "request_raw_access":ToolSpec("request_raw_access","raw:request",  AccessTier.RAW,          False, "data"),
    "trigger_retraining":ToolSpec("trigger_retraining","model:write",  AccessTier.FEATURE,      True,  "data"),
    # ---- admin / IAM server (higher privilege, isolated) ----
    "grant_access":      ToolSpec("grant_access",      "admin:iam",    None, False, "admin"),
    "set_budget":        ToolSpec("set_budget",        "admin:iam",    None, False, "admin"),
    "list_principals":   ToolSpec("list_principals",   "admin:iam",    None, False, "admin"),
}

# Convenience: scopes that correspond to each tier, for granting.
TIER_SCOPES: dict[AccessTier, set[str]] = {
    AccessTier.DP_AGGREGATE: {"dp:read"},
    AccessTier.FEATURE: {"dp:read", "feature:read"},        # feature implies dp
    AccessTier.RAW: {"dp:read", "feature:read", "raw:request"},
}
