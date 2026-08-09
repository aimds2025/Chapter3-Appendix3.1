"""
physio_pipeline.mcp_platform
============================

The Section 9 agentic multi-tenant access platform: multiple hospitals
(tenants), multiple ML-project consumers (principals), a tiered data-access
model, per-(project x tenant) differential-privacy budgets, and a gateway-
mediated MCP entry point. `server.py` is an optional real MCP-SDK adapter.
"""
from .access import TOOL_REGISTRY, AccessTier, ToolSpec
from .gateway import (
    HITLRequired,
    InsufficientScope,
    MCPGatewayMiddleware,
    TenantDenied,
)
from .platform import MultiTenantPlatform

__all__ = [
    "MultiTenantPlatform", "AccessTier", "ToolSpec", "TOOL_REGISTRY",
    "MCPGatewayMiddleware", "InsufficientScope", "TenantDenied", "HITLRequired",
]
