"""
streaming.pquantile
===================

The P-square (P^2) algorithm (Jain & Chlamtac, 1985): estimate a quantile of a
stream in a single pass using O(1) memory (five markers), without storing the
samples. This is what lets the online robust guard maintain median/IQR-style
statistics on an unbounded stream instead of buffering everything.

Accuracy is validated against numpy in tests/test_streaming.py.
"""
from __future__ import annotations


class P2Quantile:
    """Single-quantile P^2 estimator. Feed values with update(x); read .value."""

    def __init__(self, p: float):
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0, 1)")
        self.p = p
        self._init: list[float] = []
        self.ready = False
        # marker heights (q), positions (n), desired positions (np_), increments
        self.q: list[float] = []
        self.n: list[float] = []
        self.np_: list[float] = []
        self.dn: list[float] = []

    def update(self, x: float) -> None:
        if not self.ready:
            self._init.append(float(x))
            if len(self._init) == 5:
                self._init.sort()
                self.q = list(self._init)
                self.n = [1.0, 2.0, 3.0, 4.0, 5.0]
                self.np_ = [1.0, 1 + 2 * self.p, 1 + 4 * self.p, 3 + 2 * self.p, 5.0]
                self.dn = [0.0, self.p / 2, self.p, (1 + self.p) / 2, 1.0]
                self.ready = True
            return

        # 1) find cell k and possibly extend the min/max markers
        if x < self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[4]:
            self.q[4] = x
            k = 3
        else:
            k = 3
            for i in range(4):
                if self.q[i] <= x < self.q[i + 1]:
                    k = i
                    break

        # 2) increment positions
        for i in range(k + 1, 5):
            self.n[i] += 1
        for i in range(5):
            self.np_[i] += self.dn[i]

        # 3) adjust the three interior markers
        for i in range(1, 4):
            d = self.np_[i] - self.n[i]
            if (d >= 1 and self.n[i + 1] - self.n[i] > 1) or \
               (d <= -1 and self.n[i - 1] - self.n[i] < -1):
                s = 1 if d >= 0 else -1
                qp = self._parabolic(i, s)
                if self.q[i - 1] < qp < self.q[i + 1]:
                    self.q[i] = qp
                else:
                    self.q[i] = self._linear(i, s)
                self.n[i] += s

    def _parabolic(self, i: int, d: int) -> float:
        q, n = self.q, self.n
        return q[i] + d / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        )

    def _linear(self, i: int, d: int) -> float:
        q, n = self.q, self.n
        return q[i] + d * (q[i + d] - q[i]) / (n[i + d] - n[i])

    @property
    def value(self) -> float:
        if self.ready:
            return self.q[2]
        if not self._init:
            return 0.0
        s = sorted(self._init)
        return s[min(len(s) - 1, int(self.p * len(s)))]
