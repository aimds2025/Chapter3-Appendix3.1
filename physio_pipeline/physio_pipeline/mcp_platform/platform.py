"""
mcp_platform.platform
======================

`MultiTenantPlatform` ties Section 9 together:

  * one ISOLATED Layer 5 store per hospital (tenant), created on demand
  * a PrincipalRegistry of ML-project consumers (deny-by-default)
  * a per-(principal x tenant) differential-privacy ledger
  * a single gateway-mediated entry point, call(), through which every agent
    tool invocation must pass

Typical use:

    plat = MultiTenantPlatform()
    plat.populate_from_devices(load_cohort("data/synthetic"))   # fill stores
    plat.register_project("spiffe://ml/alpha",
                          scopes={"dp:read"}, tenants={"H001"})
    plat.call("spiffe://ml/alpha", "Bearer x", "query_vitals_dp",
              tenant="H001", epsilon=0.5)
"""
from __future__ import annotations

from collections import defaultdict

from ..crosscutting.audit import AuditLog
from ..crosscutting.iam import IAM, PrincipalRegistry
from ..crosscutting.privacy_budget import PrivacyBudgetLedger, TenantPrivacyLedger
from ..layer2_perimeter import ZeroTrustPerimeter
from ..layer3_ingestion import StreamIngestion
from ..layer4_processing import StreamProcessor
from ..layer5_storage import StorageLayer
from .access import TIER_SCOPES, AccessTier
from .gateway import MCPGatewayMiddleware
from .handlers import HANDLERS


class MultiTenantPlatform:
    def __init__(self, dp_default_epsilon: float = 3.0):
        self.audit = AuditLog()
        self.iam = IAM()                              # device/workload identity
        self.principals = PrincipalRegistry()         # ML-project principals
        self.ledger = TenantPrivacyLedger(default_epsilon=dp_default_epsilon)
        self.gateway = MCPGatewayMiddleware(self.principals, self.audit)
        self._stores: dict[str, StorageLayer] = {}    # tenant -> isolated store
        self.raw_requests: list[dict] = []
        self.retraining_jobs: list[dict] = []

    # ---- tenants / stores ----
    def tenant_ids(self) -> list[str]:
        return sorted(self._stores)

    def _ensure_store(self, tenant_id: str) -> StorageLayer:
        if tenant_id not in self._stores:
            # each tenant gets its OWN store and its OWN storage-side DP ledger
            self._stores[tenant_id] = StorageLayer(PrivacyBudgetLedger())
        return self._stores[tenant_id]

    def tenant_store(self, tenant_id: str) -> StorageLayer:
        if tenant_id not in self._stores:
            raise KeyError(f"unknown tenant '{tenant_id}'")
        return self._stores[tenant_id]

    # ---- population: run Layers 1-5 per tenant to fill isolated stores ----
    def populate_from_devices(self, devices, windows_per_device: int = 5) -> None:
        by_tenant: dict[str, list] = defaultdict(list)
        for d in devices:
            by_tenant[d.hospital_id].append(d)
        for tenant_id, devs in by_tenant.items():
            store = self._ensure_store(tenant_id)
            perimeter = ZeroTrustPerimeter(self.iam, self.audit)
            ingestion = StreamIngestion(self.audit)
            processor = StreamProcessor(self.audit)
            records = []
            for d in devs:
                self.iam.register(f"spiffe://edge/{d.device_id}", {"device"})
                for _ in range(windows_per_device):
                    pkt = perimeter.authorize(d.emit(d.acquire()))
                    records.append(ingestion.produce(pkt))
            batch = processor.process(records)
            store.persist(tenant_id, records, batch)

    # ---- principals (ML projects) ----
    def register_project(self, principal_id: str, scopes: set[str] | None = None,
                         tenants: set[str] | None = None):
        return self.principals.register(principal_id, scopes or set(), tenants or set())

    def register_project_at_tier(self, principal_id: str, tier: AccessTier,
                                 tenants: set[str]):
        """Convenience: grant the scope bundle implied by an access tier."""
        return self.principals.register(principal_id, set(TIER_SCOPES[tier]), set(tenants))

    def register_admin(self, principal_id: str):
        return self.principals.register(principal_id, {"admin:iam"}, set())

    # ---- the single gateway-mediated entry point ----
    def call(self, principal_id: str, token: str, tool: str,
             tenant: str | None = None, prompt: str | None = None,
             write_approved: bool = False, **args) -> dict:
        principal, spec = self.gateway.enforce(
            principal_id, token, tool, tenant, prompt, write_approved)
        handler = HANDLERS[tool]
        result = handler(self, principal, tenant, **args)
        out = self.gateway.finalize(result)
        self.audit.record("layer6", "mcp_call", principal_id, ok=True,
                          tool=tool, tenant=tenant or "-")
        return out
