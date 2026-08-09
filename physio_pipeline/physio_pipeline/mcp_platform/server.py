"""
mcp_platform.server
===================

OPTIONAL real MCP server adapter. Exposes the same governed tool handlers
through the official Model Context Protocol Python SDK, so a real MCP client
(an LLM agent / host) can connect over stdio and call them.

Supports BOTH SDK generations:
  * mcp >= 2.0 :  from mcp.server import MCPServer      (current)
  * mcp 1.x    :  from mcp.server.fastmcp import FastMCP (legacy)

Not imported by the package by default and NOT required for the in-process
demos or tests -- those exercise the identical gateway + handlers directly.

    pip install "physio-pipeline[mcp]"        # or: pip install mcp
    python -m physio_pipeline.mcp_platform.server            # run over stdio
    python -m physio_pipeline.mcp_platform.server --selftest # verify without a client

Simplification for the reference: principal and tenant are passed as tool
arguments so the server is runnable without an OAuth authorization server; in
production they come from the validated OAuth 2.1 access token.
"""
from __future__ import annotations

import sys

from .platform import MultiTenantPlatform


def _load_server_class():
    """
    Return (ServerClass, is_modern). Distinguishes three cases with an accurate
    message: mcp not installed at all vs. installed-but-unexpected-API.
    """
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The 'mcp' package is not installed. Install it with:\n"
            '    pip install -e ".[mcp]"      (or: pip install mcp)')
    # modern SDK (>= 2.0)
    try:
        from mcp.server import MCPServer
        return MCPServer, True
    except Exception:
        pass
    # legacy SDK (1.x)
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP, False
    except Exception as exc:
        ver = getattr(sys.modules.get("mcp"), "__version__", "unknown")
        raise SystemExit(
            f"'mcp' is installed (version {ver}) but neither MCPServer "
            f"(>=2.0) nor FastMCP (1.x) could be imported: {exc}\n"
            "Your SDK version may be newer than this adapter expects; please "
            "report the version so the import path can be updated.")


def build_platform() -> MultiTenantPlatform:
    """Build a platform pre-populated from the synthetic cohort, if present."""
    plat = MultiTenantPlatform()
    try:
        from pathlib import Path
        from ..layer1_edge import load_cohort
        data_dir = Path(__file__).resolve().parents[2] / "data" / "synthetic"
        plat.populate_from_devices(load_cohort(data_dir))
    except Exception as exc:
        print(f"[server] no synthetic data loaded ({exc}); stores are empty. "
              f"Run: python data/generate_synthetic.py", file=sys.stderr)
    # a default project so calls work out of the box (DP-aggregate on all tenants)
    from .access import AccessTier
    if plat.tenant_ids():
        plat.register_project_at_tier("spiffe://ml/demo", AccessTier.FEATURE,
                                      set(plat.tenant_ids()))
    return plat


def build_server(plat: MultiTenantPlatform):
    ServerClass, is_modern = _load_server_class()
    app = ServerClass(name="physio-pipeline-dataplane", version="0.2.0")

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

    return app, is_modern


def _selftest(plat: MultiTenantPlatform) -> int:
    """Verify the server builds, tools register, and a governed call works."""
    app, is_modern = build_server(plat)
    print("MCP server built OK")
    print(f"  SDK API      : {'MCPServer (>=2.0)' if is_modern else 'FastMCP (1.x)'}")
    print(f"  tenants      : {plat.tenant_ids() or '(none - generate data first)'}")
    print("  tools        : query_vitals_dp, get_feature_batch, request_raw_access")
    if plat.tenant_ids():
        t = plat.tenant_ids()[0]
        out = plat.call("spiffe://ml/demo", "Bearer selftest",
                        "get_feature_batch", tenant=t, n=2)
        print(f"  sample call  : get_feature_batch(tenant={t}) -> "
              f"{out['result']['n']} feature vectors, phi={out['result']['phi']}, "
              f"signed={len(out['hmac'])==64}")
    print("\nSelftest OK. To serve a real MCP client, run without --selftest "
          "(the process then waits on stdio).")
    return 0


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    plat = build_platform()

    if "--selftest" in argv:
        raise SystemExit(_selftest(plat))

    app, is_modern = build_server(plat)
    print("[server] starting on stdio; waiting for an MCP client. "
          "Ctrl+C to stop.", file=sys.stderr)
    if is_modern:
        app.run(transport="stdio")
    else:
        app.run()   # FastMCP 1.x defaults to stdio


if __name__ == "__main__":
    main()
