"""
streaming.online_guard
======================

The streaming counterpart to Layer 4's batch poisoning guard. Instead of
computing median/MAD over a materialized batch, it maintains per-feature
quantiles incrementally with P^2 (O(1) memory per feature) and vetoes points
whose robust z-score exceeds a threshold.

Robust scale uses the inter-quartile range: scale = (Q3 - Q1) / 1.349 ~= sigma,
which avoids the recursive "MAD-of-a-moving-median" problem in a stream.

A short warm-up period admits everything (statistics aren't meaningful yet),
which matters clinically: you don't want to veto the very first rare case just
because the estimator is cold.
"""
from __future__ import annotations

import numpy as np

from .pquantile import P2Quantile


class OnlineRobustGuard:
    def __init__(self, n_features: int, z_threshold: float = 8.0,
                 warmup: int = 20, scale_floor: float = 0.02):
        self.z_threshold = z_threshold
        self.warmup = warmup
        self.scale_floor = scale_floor
        self._seen = 0
        self._q1 = [P2Quantile(0.25) for _ in range(n_features)]
        self._med = [P2Quantile(0.50) for _ in range(n_features)]
        self._q3 = [P2Quantile(0.75) for _ in range(n_features)]

    def _robust_z(self, e: np.ndarray) -> float:
        zmax = 0.0
        for j, x in enumerate(e):
            med = self._med[j].value
            iqr = self._q3[j].value - self._q1[j].value
            scale = max(iqr / 1.349, self.scale_floor)
            zmax = max(zmax, abs(x - med) / scale)
        return zmax

    def check(self, e: np.ndarray) -> tuple[bool, float]:
        """
        Returns (accept, max_z). During warm-up always accepts. The point is
        folded into the statistics AFTER the decision, so a point is never
        judged against itself.
        """
        self._seen += 1
        if self._seen <= self.warmup:
            self._update(e)
            return True, 0.0
        z = self._robust_z(e)
        accept = z <= self.z_threshold
        if accept:
            # only trusted points update the statistics, so a flood of
            # outliers can't drag the baseline toward itself
            self._update(e)
        return accept, z

    def _update(self, e: np.ndarray) -> None:
        for j, x in enumerate(e):
            self._q1[j].update(float(x))
            self._med[j].update(float(x))
            self._q3[j].update(float(x))
