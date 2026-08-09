"""
mcp_platform.handlers
=====================

The tool implementations. Data-plane handlers read from the requesting
principal's permitted tenant store (Layer 5) and enforce the tier semantics of
Section 9.4:

  * query_vitals_dp   -> noised aggregate via the Gaussian mechanism; debits the
                         per-(principal x tenant) DP budget (Section 9.7)
  * get_feature_batch -> de-identified coreset feature vectors (no PHI)
  * get_coreset       -> coreset vectors + weights for training
  * request_raw_access-> creates a HITL approval ticket; returns NO data
  * trigger_retraining-> (write, HITL-gated) queues a retraining job

Admin handlers implement the IAM control surface (Section 9.6).

Handlers never receive DB credentials from the caller; they hold references to
the platform's isolated tenant stores and act with the caller's scoped grant.
"""
from __future__ import annotations

import time
import uuid

import numpy as np

from ..core.exceptions import PrivacyBudgetExhausted
from ..crosscutting.iam import Principal

# heart-rate is bounded to a physiological range; used as DP sensitivity bound.
_HR_MIN, _HR_MAX = 30.0, 220.0   # physiological clamp -> bounded DP sensitivity


# --------------------------------------------------------------------------- #
# data-plane handlers
# --------------------------------------------------------------------------- #
def query_vitals_dp(platform, principal: Principal, tenant_id: str,
                    metric: str = "hr_mean", epsilon: float = 0.5,
                    delta: float = 1e-6) -> dict:
    store = platform.tenant_store(tenant_id)
    values = [hr for (_pid, hr, _sqi, _ts) in store.tsdb._points]
    # debit the per-(principal x tenant) budget BEFORE releasing anything
    platform.ledger.debit(principal.principal_id, tenant_id,
                          f"query:{metric}", epsilon, delta)
    n = len(values)
    if n == 0:
        return {"metric": metric, "tenant": tenant_id, "value": None, "n": 0,
                "epsilon_spent": epsilon}
    clamped = [min(max(v, _HR_MIN), _HR_MAX) for v in values]  # bound contribution
    true_mean = float(np.mean(clamped))
    # Gaussian mechanism: sensitivity of a mean over a bounded range is
    # (range)/n; sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon.
    sensitivity = (_HR_MAX - _HR_MIN) / n
    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon
    noised = true_mean + float(np.random.default_rng().normal(0, sigma))
    # clamp to the physiological range -- DP post-processing is immune, and it
    # avoids releasing impossible values when noise dominates a small cohort.
    noised = min(max(noised, _HR_MIN), _HR_MAX)
    return {"metric": metric, "tenant": tenant_id, "value": round(float(noised), 1),
            "n": n, "epsilon_spent": epsilon, "noise_sigma": round(float(sigma), 1),
            "budget_remaining": round(
                platform.ledger.remaining(principal.principal_id, tenant_id), 3)}


def get_feature_batch(platform, principal: Principal, tenant_id: str,
                      n: int = 8) -> dict:
    store = platform.tenant_store(tenant_id)
    keys = store.features.keys()[:n]
    vectors = [[round(float(x), 4) for x in store.features.get(k)] for k in keys]
    return {"tenant": tenant_id, "n": len(vectors), "dim": len(vectors[0]) if vectors else 0,
            "features": vectors, "phi": False}


def get_coreset(platform, principal: Principal, tenant_id: str) -> dict:
    store = platform.tenant_store(tenant_id)
    keys = store.features.keys()
    vectors = [[round(float(x), 4) for x in store.features.get(k)] for k in keys]
    return {"tenant": tenant_id, "coreset_size": len(vectors), "vectors": vectors}


def request_raw_access(platform, principal: Principal, tenant_id: str,
                       reason: str = "") -> dict:
    ticket = {
        "ticket_id": uuid.uuid4().hex[:12],
        "principal": principal.principal_id, "tenant": tenant_id,
        "reason": reason, "status": "pending_hitl_approval",
        "created": time.time(),
    }
    platform.raw_requests.append(ticket)
    # NB: returns a ticket only -- never raw PHI.
    return {"status": "pending_hitl_approval", "ticket_id": ticket["ticket_id"],
            "note": "raw access requires human sign-off; no data returned"}


def trigger_retraining(platform, principal: Principal, tenant_id: str,
                       dataset: str = "coreset") -> dict:
    job_id = uuid.uuid4().hex[:12]
    platform.retraining_jobs.append(
        {"job_id": job_id, "principal": principal.principal_id,
         "tenant": tenant_id, "dataset": dataset, "status": "queued"})
    return {"status": "queued", "job_id": job_id, "tenant": tenant_id}


# --------------------------------------------------------------------------- #
# admin / IAM handlers
# --------------------------------------------------------------------------- #
def grant_access(platform, principal: Principal, tenant_id: str,
                 target: str, grant_tenant: str, scope: str | None = None) -> dict:
    platform.principals.grant_tenant(target, grant_tenant)
    if scope:
        platform.principals.grant_scope(target, scope)
    p = platform.principals.get(target)
    return {"granted": True, "principal": target,
            "tenants": sorted(p.tenants), "scopes": sorted(p.scopes)}


def set_budget(platform, principal: Principal, tenant_id: str,
               target: str, grant_tenant: str, epsilon: float) -> dict:
    platform.ledger.set_budget(target, grant_tenant, epsilon)
    return {"principal": target, "tenant": grant_tenant, "epsilon": epsilon}


def list_principals(platform, principal: Principal, tenant_id: str) -> dict:
    return {"principals": platform.principals.list_principals()}


HANDLERS = {
    "query_vitals_dp": query_vitals_dp,
    "get_feature_batch": get_feature_batch,
    "get_coreset": get_coreset,
    "request_raw_access": request_raw_access,
    "trigger_retraining": trigger_retraining,
    "grant_access": grant_access,
    "set_budget": set_budget,
    "list_principals": list_principals,
}
