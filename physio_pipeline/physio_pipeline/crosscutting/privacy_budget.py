"""
crosscutting.privacy_budget
===========================

Differential-privacy budget ledger. Tracks cumulative epsilon spent against a
fixed total budget under basic (sequential) composition. Debited by Layer 5's
DP release store and by Layer 7's DP-SGD training.

Production swap-in: a transactional, hash-chained table in Aurora Postgres so
the ledger is both consistent and tamper-evident; consider Renyi-DP accounting
for tighter composition than the basic bound used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.exceptions import PrivacyBudgetExhausted


@dataclass
class LedgerEntry:
    purpose: str
    epsilon: float
    delta: float


class PrivacyBudgetLedger:
    def __init__(self, total_epsilon: float = 5.0, total_delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self._entries: list[LedgerEntry] = []

    @property
    def spent_epsilon(self) -> float:
        return sum(e.epsilon for e in self._entries)

    @property
    def remaining(self) -> float:
        return self.total_epsilon - self.spent_epsilon

    def debit(self, purpose: str, epsilon: float, delta: float = 0.0) -> None:
        """Charge (epsilon, delta) to the budget or raise if it would overrun."""
        if self.spent_epsilon + epsilon > self.total_epsilon + 1e-12:
            raise PrivacyBudgetExhausted(
                f"'{purpose}' needs eps={epsilon:.3f}; only "
                f"{self.remaining:.3f} of {self.total_epsilon} remains"
            )
        self._entries.append(LedgerEntry(purpose, epsilon, delta))

    def summary(self) -> dict[str, float]:
        return {
            "total_epsilon": self.total_epsilon,
            "spent_epsilon": round(self.spent_epsilon, 4),
            "remaining_epsilon": round(self.remaining, 4),
        }


class TenantPrivacyLedger:
    """
    Section 9.7: differential-privacy budget accounted per (principal x tenant)
    pair rather than globally. A cross-tenant query debits each participating
    tenant's ledger independently; exhaustion denies further release for that
    (principal, tenant) only.

    Production swap-in: Renyi-DP / moments-accountant composition backed by a
    transactional, hash-chained table in Aurora Postgres.
    """

    def __init__(self, default_epsilon: float = 3.0, default_delta: float = 1e-6):
        self.default_epsilon = default_epsilon
        self.default_delta = default_delta
        self._ledgers: dict[tuple[str, str], PrivacyBudgetLedger] = {}

    def _ledger(self, principal_id: str, tenant_id: str) -> PrivacyBudgetLedger:
        key = (principal_id, tenant_id)
        if key not in self._ledgers:
            self._ledgers[key] = PrivacyBudgetLedger(
                self.default_epsilon, self.default_delta)
        return self._ledgers[key]

    def set_budget(self, principal_id: str, tenant_id: str,
                   total_epsilon: float, total_delta: float = 1e-6) -> None:
        self._ledgers[(principal_id, tenant_id)] = PrivacyBudgetLedger(
            total_epsilon, total_delta)

    def debit(self, principal_id: str, tenant_id: str, purpose: str,
              epsilon: float, delta: float = 0.0) -> None:
        self._ledger(principal_id, tenant_id).debit(
            f"{principal_id}@{tenant_id}:{purpose}", epsilon, delta)

    def remaining(self, principal_id: str, tenant_id: str) -> float:
        return self._ledger(principal_id, tenant_id).remaining

    def summary(self) -> dict[str, dict[str, float]]:
        return {f"{pid}@{tid}": led.summary()
                for (pid, tid), led in self._ledgers.items()}
