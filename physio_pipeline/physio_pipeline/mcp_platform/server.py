"""
mcp_platform.server
===================

OPTIONAL real MCP server adapter. This is the D3 "runnable over stdio" surface
from Section 9: it exposes the same tool handlers through the official Model
Context Protocol Python SDK, so a real MCP client (an LLM agent / host) can
connect and call them.

It is intentionally NOT imported by the package by default and is NOT required
for the in-process demos or the test suite -- those exercise the identical
gateway + handlers directly. Install the SDK to use this adapter:

    pip install "physio-pipeline[mcp]"      # or: pip install mcp
    python -m physio_pipeline.mcp_platform.server

Simplification for the reference: in a production deployment the principal and
tenant claims come from the validated OAuth 2.1 access token. Here they are
passed as explicit tool arguments so the server is runnable without an
authorization server; the gateway enforcement is otherwise identical.
"""
from __future__ import annotations

import sys

from .platform import MultiTenantPlatform


def build_platform() -> MultiTenantPlatform:
    """Build a platform pre-populated from the synthetic cohort, if present."""
    plat = MultiTenantPlatform()
    try:
        from pathlib import Path
        from ..layer1_edge import load_cohort
        data_dir = Path(__file__).resolve().parents[2] / "data" / "synthetic"
        plat.populate_from_devices(load_cohort(data_dir))
    except Exception as exc:                       # pragma: no cover
        print(f"[server] no synthetic data loaded ({exc}); stores are empty",
              file=sys.stderr)
    return plat


def main() -> None:                                # pragma: no cover
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        sys.exit("The 'mcp' package is required: pip install mcp")

    plat = build_platform()
    app = FastMCP("physio-pipeline-dataplane")

    @app.tool()
    def query_vitals_dp(principal_id: str, token: str, tenant: str,
                        epsilon: float = 0.5) -> dict:
        """DP-aggregate vitals query (returns a noised mean)."""
        return plat.call(principal_id, token, "query_vitals_dp",
                         tenant=tenant, epsilon=epsilon)

    @app.tool()
    def get_feature_batch(principal_id: str, token: str, tenant: str,
                          n: int = 8) -> dict:
        """De-identified coreset feature vectors for training."""
        return plat.call(principal_id, token, "get_feature_batch",
                         tenant=tenant, n=n)

    @app.tool()
    def request_raw_access(principal_id: str, token: str, tenant: str,
                           reason: str = "") -> dict:
        """Initiate a HITL-gated raw-access request (returns a ticket only)."""
        return plat.call(principal_id, token, "request_raw_access",
                         tenant=tenant, reason=reason)

    app.run()   # stdio transport


if __name__ == "__main__":
    main()
