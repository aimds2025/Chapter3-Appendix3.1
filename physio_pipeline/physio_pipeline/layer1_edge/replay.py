"""
Layer 1 - REPLAY DEVICE
======================

A drop-in `EdgeDevice` that reads recorded waveforms from disk instead of
synthesizing them. Everything downstream (L2-L8) is unchanged, because it still
emits the same `EdgePacket` contract.

This is the seam where a real device recording (WFDB, EDF, vendor export) would
plug in: swap `_load()` for your reader and the whole pipeline runs on it.

    from physio_pipeline.layer1_edge.replay import ReplayDevice, load_cohort

    devices = load_cohort("data/synthetic")
    result  = PhysioPipeline().run_batch(devices, windows_per_device=5)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.contracts import WaveformFrame
from .edge import EdgeDevice


class ReplayDevice(EdgeDevice):
    """Replays a recorded multi-channel waveform in fixed-length windows."""

    def __init__(self, npz_path: str | Path, device_id: str | None = None,
                 window_s: float = 4.0):
        blob = np.load(Path(npz_path), allow_pickle=True)
        self._samples: np.ndarray = blob["samples"]
        patient_id = str(blob["patient_id"])
        fs_hz = int(blob["fs_hz"])
        channels = [str(c) for c in blob["channels"]]
        self.cohort = str(blob["cohort"]) if "cohort" in blob else "unknown"
        self.hr_true_bpm = float(blob["hr_true_bpm"]) if "hr_true_bpm" in blob else float("nan")
        hospital_id = str(blob["hospital_id"]) if "hospital_id" in blob else "H000"

        super().__init__(
            patient_id=patient_id,
            device_id=device_id or f"MON-{patient_id}",
            fs_hz=fs_hz,
            channels=tuple(channels),
            hospital_id=hospital_id,
        )
        self._window_n = int(window_s * fs_hz)
        self._cursor = 0

    @property
    def n_windows(self) -> int:
        return self._samples.shape[1] // self._window_n

    def acquire(self, duration_s: float | None = None,
                rng: np.random.Generator | None = None) -> WaveformFrame:
        """Return the next window from the recording (wraps at end-of-file)."""
        if self._cursor + self._window_n > self._samples.shape[1]:
            self._cursor = 0                       # loop the recording
        win = self._samples[:, self._cursor: self._cursor + self._window_n]
        self._cursor += self._window_n
        frame = WaveformFrame(
            patient_id=self.patient_id, device_id=self.device_id,
            fs_hz=self.fs_hz, channels=self.channels, samples=win.astype(float),
            hospital_id=self.hospital_id,
        )
        self._ring.append(frame)
        return frame


def load_cohort(data_dir: str | Path, window_s: float = 4.0) -> list[ReplayDevice]:
    """Build a ReplayDevice for every .npz in `data_dir` (sorted by patient id)."""
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no .npz files in {data_dir}. Run: python data/generate_synthetic.py"
        )
    return [ReplayDevice(p, window_s=window_s) for p in paths]
