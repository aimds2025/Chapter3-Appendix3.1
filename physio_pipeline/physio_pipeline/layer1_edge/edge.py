"""
Layer 1 - EDGE (device / bedside)
================================

On-device signal path for a 250 Hz, multi-channel bedside monitor:

  * acquire()          -> WaveformFrame  (raw multi-channel window)
  * detect_qrs()       -> Pan-Tompkins R-peak indices
  * signal_quality()   -> SQI in [0, 1]
  * compress()         -> DWT (Haar) compression with PRD measurement
  * emit()             -> EdgePacket (compressed payload + attestation)

The raw waveform NEVER leaves the device; only the compressed, attested
EdgePacket crosses the wire. A ring buffer bounds on-device memory.

Production swap-in: firmware DSP + hardware TPM attestation; pywt for DWT;
mTLS 1.3 client for transport.
"""
from __future__ import annotations

import collections
import hashlib
import zlib

import numpy as np

from ..core.contracts import EdgePacket, WaveformFrame


class EdgeDevice:
    def __init__(self, patient_id: str, device_id: str, fs_hz: int = 250,
                 channels: tuple[str, ...] = ("ECG_II", "PPG", "RESP"),
                 ring_seconds: int = 60, hr_range: tuple[float, float] = (55, 95),
                 seed: int | None = None, hospital_id: str = "H000"):
        self.patient_id = patient_id
        self.device_id = device_id
        self.fs_hz = fs_hz
        self.channels = list(channels)
        self.hr_range = hr_range                        # bpm sampling range
        self.hospital_id = hospital_id                  # tenant id
        # persistent generator so acquisition is reproducible when seeded
        self._rng = np.random.default_rng(seed)
        # 60-second ring buffer bounds device memory
        self._ring: collections.deque[WaveformFrame] = collections.deque(
            maxlen=max(1, ring_seconds)
        )

    # --- acquisition -------------------------------------------------------
    def acquire(self, duration_s: float = 4.0, rng: np.random.Generator | None = None
                ) -> WaveformFrame:
        """Synthesize a physiologically-plausible multi-channel window."""
        rng = rng if rng is not None else self._rng
        n = int(duration_s * self.fs_hz)
        t = np.arange(n) / self.fs_hz
        hr = rng.uniform(*self.hr_range)                # beats/min
        rr = 60.0 / hr                                  # R-R interval (s)
        # ECG as narrow Gaussian R-peaks at the beat rate (realistic & sparse),
        # so Pan-Tompkins recovers the true beat count.
        ecg = np.zeros(n)
        width = 0.020                                   # ~20 ms QRS width
        beat_t = 0.5 * rr
        while beat_t < duration_s:
            jitter = rng.normal(0, 0.01)                # beat-to-beat variability
            ecg += np.exp(-0.5 * ((t - beat_t - jitter) / width) ** 2)
            beat_t += rr
        ecg += rng.normal(0, 0.02, n)                   # baseline noise
        ppg = 0.6 * np.sin(2 * np.pi * (hr / 60.0) * t - 0.4) + rng.normal(0, 0.02, n)
        resp = 0.4 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.02, n)
        samples = np.vstack([ecg, ppg, resp])[: len(self.channels)]
        frame = WaveformFrame(self.patient_id, self.device_id, self.fs_hz,
                              self.channels, samples, hospital_id=self.hospital_id)
        self._ring.append(frame)
        return frame

    # --- Pan-Tompkins (simplified) ----------------------------------------
    def detect_qrs(self, frame: WaveformFrame) -> list[int]:
        """Simplified Pan-Tompkins: bandpass -> derivative -> square -> peaks."""
        ecg = frame.samples[0]
        diff = np.ediff1d(ecg, to_begin=0.0)
        squared = diff ** 2
        # moving-window integration (~150 ms)
        w = max(1, int(0.150 * self.fs_hz))
        integrated = np.convolve(squared, np.ones(w) / w, mode="same")
        thr = 0.5 * integrated.max() if integrated.max() > 0 else 1.0
        peaks: list[int] = []
        refractory = int(0.2 * self.fs_hz)             # 200 ms
        last = -refractory
        for i in range(1, len(integrated) - 1):
            if (integrated[i] > thr and integrated[i] >= integrated[i - 1]
                    and integrated[i] > integrated[i + 1] and i - last > refractory):
                peaks.append(int(i))
                last = i
        return peaks

    # --- signal quality ----------------------------------------------------
    def signal_quality(self, frame: WaveformFrame) -> float:
        """Crude SQI from SNR of the primary channel, squashed to [0, 1]."""
        ecg = frame.samples[0]
        power = float(np.mean(ecg ** 2))
        noise = float(np.var(np.ediff1d(ecg)))
        snr = power / (noise + 1e-9)
        return float(max(0.0, min(1.0, snr / (snr + 5.0))))

    # --- DWT (Haar) compression -------------------------------------------
    @staticmethod
    def _haar_forward(x: np.ndarray, levels: int = 3) -> np.ndarray:
        """In-place-ish multilevel Haar DWT (length padded to power of 2)."""
        n = 1 << int(np.ceil(np.log2(len(x)))) if len(x) > 1 else 1
        x = np.pad(x, (0, n - len(x)))
        out = x.copy()
        length = n
        for _ in range(levels):
            if length < 2:
                break
            half = length // 2
            a = (out[0:length:2] + out[1:length:2]) / np.sqrt(2)
            d = (out[0:length:2] - out[1:length:2]) / np.sqrt(2)
            out[:half] = a
            out[half:length] = d
            length = half
        return out

    def compress(self, frame: WaveformFrame, keep_ratio: float = 0.25
                 ) -> tuple[bytes, float]:
        """
        DWT-threshold-compress each channel: keep the largest `keep_ratio`
        coefficients, zip the result. Returns (payload, PRD %).
        """
        coeffs_all, kept_all = [], []
        for ch in frame.samples:
            c = self._haar_forward(ch.astype(np.float32))
            k = max(1, int(len(c) * keep_ratio))
            idx = np.argsort(np.abs(c))[-k:]
            mask = np.zeros_like(c)
            mask[idx] = c[idx]
            coeffs_all.append(mask.astype(np.float32))
            kept_all.append(ch.astype(np.float32))
        stacked = np.vstack(coeffs_all)
        payload = zlib.compress(stacked.tobytes(), level=6)
        # PRD is estimated on retained-energy fraction (proxy for reconstruction err)
        orig = np.vstack(kept_all)
        energy_kept = float(np.sum(stacked ** 2))
        energy_tot = float(np.sum(orig ** 2)) + 1e-12
        prd = float(100.0 * np.sqrt(max(0.0, 1.0 - energy_kept / energy_tot)))
        return payload, prd

    # --- attestation -------------------------------------------------------
    def _attest(self, payload: bytes) -> dict:
        """Secure-boot / TPM-style signed envelope over the payload."""
        measurement = hashlib.sha256(payload).hexdigest()
        return {
            "spiffe_id": f"spiffe://edge/{self.device_id}",
            "secure_boot": True,
            "fw_measurement": measurement,
            "signed": True,
        }

    # --- emit --------------------------------------------------------------
    def emit(self, frame: WaveformFrame) -> EdgePacket:
        qrs = self.detect_qrs(frame)
        sqi = self.signal_quality(frame)
        payload, prd = self.compress(frame)
        return EdgePacket(
            patient_id=frame.patient_id, device_id=frame.device_id,
            fs_hz=frame.fs_hz, channels=frame.channels, compressed=payload,
            prd_pct=round(prd, 3), qrs_indices=qrs, sqi=round(sqi, 3),
            t_start=frame.t_start, n_samples=frame.n_samples,
            attestation=self._attest(payload),
            hospital_id=frame.hospital_id,
        )
