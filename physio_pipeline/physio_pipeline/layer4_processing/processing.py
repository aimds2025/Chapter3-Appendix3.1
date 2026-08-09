"""
Layer 4 - STREAM PROCESSING (Flink-style jobs)
=============================================

Three streaming jobs plus a coreset-poisoning guard, run over StreamRecords:

  Job 1  anomaly detection     -> Alarms (windowed z-score on heart rate)
  Job 2  Sieve-Streaming       -> coreset under a facility-location objective
  Job 3  data quality          -> SQI-range / gap metrics
  Guard  poisoning defense     -> per-patient caps + embedding outlier veto

Job 2 is the core data-selection algorithm: a single-pass streaming submodular
maximization (Badanidiyuru et al., 2014) that keeps a small, representative
coreset instead of the full firehose.

Production swap-in: Apache Flink operators; the coreset embeddings would come
from a learned encoder rather than the hand-crafted features here.
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import Alarm, CoresetPoint, EdgePacket, ProcessedBatch, StreamRecord
from ..crosscutting.audit import AuditLog


# --------------------------------------------------------------------------- #
# Feature embedding: turn an EdgePacket into a small vector for the coreset
# --------------------------------------------------------------------------- #
def embed(packet: EdgePacket) -> np.ndarray:
    n_beats = len(packet.qrs_indices)
    window_s = packet.n_samples / packet.fs_hz if packet.fs_hz else 1.0
    hr = 60.0 * n_beats / window_s if window_s else 0.0
    if len(packet.qrs_indices) > 1:
        rr = np.diff(packet.qrs_indices) / packet.fs_hz
        rr_std = float(np.std(rr))
    else:
        rr_std = 0.0
    return np.array([hr / 200.0, rr_std, packet.sqi, packet.prd_pct / 100.0],
                    dtype=np.float64)


# --------------------------------------------------------------------------- #
# Job 2: Sieve-Streaming coreset under a facility-location objective
# --------------------------------------------------------------------------- #
class SieveStreamingCoreset:
    """
    Maintains, in one pass, a coreset S of size <= k maximizing the monotone
    submodular facility-location function

        f(S) = sum_v  max_{s in S} sim(v, s)

    Sieve-Streaming runs O(log k / eps) parallel "sieves" at geometric
    marginal-gain thresholds and returns the best sieve's set.
    """

    def __init__(self, k: int = 8, eps: float = 0.2):
        self.k = k
        self.eps = eps
        self._max_singleton = 0.0
        self._sieves: dict[float, list[np.ndarray]] = {}
        self._sources: dict[float, list[int]] = {}

    @staticmethod
    def _sim(a: np.ndarray, b: np.ndarray) -> float:
        # bounded similarity in (0, 1]: 1 / (1 + squared distance)
        return 1.0 / (1.0 + float(np.sum((a - b) ** 2)))

    def _thresholds(self) -> list[float]:
        """Geometric thresholds bracketing the estimated optimum."""
        if self._max_singleton <= 0:
            return []
        lo, hi, ts = self._max_singleton, self.k * self._max_singleton, []
        v = lo
        while v <= hi:
            ts.append(v)
            v *= (1 + self.eps)
        return ts

    def _gain(self, sieve: list[np.ndarray], x: np.ndarray) -> float:
        """Marginal gain of adding x to a sieve's current selection."""
        if not sieve:
            # gain vs empty set = self-similarity contribution
            return self._sim(x, x)
        # facility-location marginal gain approximated on the selected set
        base = sum(max(self._sim(s, t) for t in sieve) for s in sieve)
        aug = sum(max(self._sim(s, t) for t in (sieve + [x])) for s in (sieve + [x]))
        return max(0.0, aug - base)

    def offer(self, x: np.ndarray, source_offset: int) -> None:
        """Feed one streaming element to every active sieve."""
        self._max_singleton = max(self._max_singleton, self._sim(x, x))
        for tau in self._thresholds():
            self._sieves.setdefault(tau, [])
            self._sources.setdefault(tau, [])
            s = self._sieves[tau]
            if len(s) < self.k:
                need = (tau / 2.0) / max(1, (self.k - len(s)))
                if self._gain(s, x) >= need:
                    s.append(x)
                    self._sources[tau].append(source_offset)

    def best(self) -> tuple[list[np.ndarray], list[int]]:
        """Return the highest-value sieve's coreset and its source offsets."""
        best_tau, best_val = None, -1.0
        for tau, s in self._sieves.items():
            if not s:
                continue
            val = sum(max(self._sim(a, b) for b in s) for a in s)
            if val > best_val:
                best_val, best_tau = val, tau
        if best_tau is None:
            return [], []
        return self._sieves[best_tau], self._sources[best_tau]


# --------------------------------------------------------------------------- #
# The layer
# --------------------------------------------------------------------------- #
class StreamProcessor:
    def __init__(self, audit: AuditLog, k: int = 8,
                 max_per_patient: int = 4, sqi_floor: float = 0.2,
                 z_threshold: float = 8.0, mad_floor: float = 0.02):
        self.audit = audit
        self.k = k
        self.max_per_patient = max_per_patient   # (c) flood cap
        self.sqi_floor = sqi_floor               # (a) DQ filter / alarm gating
        self.z_threshold = z_threshold           # (b) robust poison veto
        self.mad_floor = mad_floor               # min robust scale per feature

    def process(self, records: list[StreamRecord]) -> ProcessedBatch:
        alarms: list[Alarm] = []
        coreset_engine = SieveStreamingCoreset(k=self.k)
        per_patient: dict[str, int] = {}
        embeddings: list[tuple[np.ndarray, StreamRecord]] = []
        gaps = 0
        rejected = 0

        # ---- Job 1 (anomaly) + Job 3 (DQ) + embed for Job 2 ----
        hrs: list[float] = []
        for rec in records:
            pkt = rec.packet
            e = embed(pkt)
            hr = e[0] * 200.0
            hrs.append(hr)
            if pkt.sqi < self.sqi_floor:
                # Signal is unusable: report the dropout and SUPPRESS rate alarms.
                # Rate derived from noise is meaningless; firing tachycardia here
                # would be a false alarm (standard clinical alarm-gating practice).
                gaps += 1
                alarms.append(Alarm(pkt.patient_id, "signal_loss", "warning",
                                    f"SQI {pkt.sqi:.2f} < floor; rate alarms gated"))
            elif hr > 120:
                alarms.append(Alarm(pkt.patient_id, "tachycardia", "critical",
                                    f"HR ~{hr:.0f} bpm"))
            elif 0 < hr < 45:
                alarms.append(Alarm(pkt.patient_id, "bradycardia", "critical",
                                    f"HR ~{hr:.0f} bpm"))
            embeddings.append((e, rec))

        # windowed z-score anomaly on HR.
        # Only trustworthy (SQI >= floor) rates contribute to the baseline,
        # otherwise noise-derived rates corrupt mu/sd for every other patient.
        good = [(hr, rec) for hr, (_, rec) in zip(hrs, embeddings)
                if rec.packet.sqi >= self.sqi_floor]
        if len(good) >= 3:
            good_hrs = [hr for hr, _ in good]
            mu, sd = float(np.mean(good_hrs)), float(np.std(good_hrs)) + 1e-9
            for hr, rec in good:
                if abs(hr - mu) / sd > 3.0:
                    alarms.append(Alarm(rec.packet.patient_id, "hr_outlier",
                                        "warning", f"HR z>{3}"))

        # ---- Guard + Job 2 (coreset) ----
        # Three DISTINCT defenses, deliberately not conflated:
        #
        #   (a) DQ filter    - low-SQI windows are a *quality* problem, not an
        #                      attack. Excluded from the coreset, counted apart.
        #   (b) poison veto  - per-feature ROBUST z-score (median/MAD). Robust
        #                      statistics matter: with mean/std an extreme outlier
        #                      inflates the very spread used to threshold it, and
        #                      so helps itself hide. Per-feature (not Euclidean on
        #                      raw features) because the dimensions have different
        #                      scales, and because a global centroid distance
        #                      cannot distinguish a rare *pathology* (bradycardia)
        #                      from a fabricated sample -- it would silently drop
        #                      the rare clinical data we most want to keep.
        #   (c) flood cap    - bounds any single patient's contribution.
        #
        # Note: in a true streaming setting these statistics must be maintained
        # incrementally (e.g. P-square quantiles); here the batch is in memory.
        dq_filtered = 0
        capped = 0

        quality = [(e, rec) for e, rec in embeddings
                   if rec.packet.sqi >= self.sqi_floor]
        dq_filtered = len(embeddings) - len(quality)

        if quality:
            E = np.array([e for e, _ in quality])
            median = np.median(E, axis=0)
            mad = np.median(np.abs(E - median), axis=0) * 1.4826   # -> ~sigma
            scale = np.maximum(mad, self.mad_floor)                # avoid /0
        else:
            median = scale = None

        for e, rec in quality:
            pid = rec.packet.patient_id
            # (b) robust per-feature outlier veto
            if median is not None:
                max_z = float(np.max(np.abs(e - median) / scale))
                if max_z > self.z_threshold:
                    rejected += 1
                    self.audit.record("layer4", "poison_veto", pid, ok=False,
                                      max_z=round(max_z, 2))
                    continue
            # (c) per-patient flood cap
            if per_patient.get(pid, 0) >= self.max_per_patient:
                capped += 1
                continue
            per_patient[pid] = per_patient.get(pid, 0) + 1
            coreset_engine.offer(e, rec.offset)

        vectors, offsets = coreset_engine.best()
        coreset = [
            CoresetPoint(patient_id="mixed", embedding=v, weight=1.0,
                         source_offset=off)
            for v, off in zip(vectors, offsets)
        ]

        dq = {
            "records": float(len(records)),
            "gap_rate": round(gaps / max(1, len(records)), 3),
            "mean_sqi": round(float(np.mean([r.packet.sqi for r in records])), 3)
                         if records else 0.0,
            "dq_filtered": float(dq_filtered),
            "coreset_size": float(len(coreset)),
        }
        self.audit.record("layer4", "process", "flink", ok=True,
                          alarms=len(alarms), coreset=len(coreset),
                          rejected=rejected, dq_filtered=dq_filtered, capped=capped)
        return ProcessedBatch(alarms=alarms, coreset=coreset, dq_metrics=dq,
                              rejected=rejected, dq_filtered=dq_filtered,
                              capped=capped)
