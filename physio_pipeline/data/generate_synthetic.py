"""
Synthetic physiological dataset generator
=========================================

Writes a labeled, reproducible test cohort to disk so the pipeline can be
exercised against *files* rather than the on-the-fly generator in Layer 1.

Each patient gets:
  * data/synthetic/<patient_id>.npz   -- samples (n_channels, n_samples) + metadata
  * a row in data/synthetic/manifest.csv

Cohorts deliberately exercise different layer behaviours:

  | cohort        | HR (bpm) | what it triggers                                |
  |---------------|----------|-------------------------------------------------|
  | normal        | 55-95    | baseline; no alarms                              |
  | tachycardia   | 130-170  | L4 Job1 -> critical alarm                        |
  | bradycardia   | 30-42    | L4 Job1 -> critical alarm                        |
  | signal_loss   | 60-80    | low SQI -> L4 Job3 gap + signal_loss warning     |
  | poisoned      | 60-80    | extreme embedding -> L4 poisoning guard veto     |

Everything is seeded, so two runs produce byte-identical files.

Usage:
    python data/generate_synthetic.py                 # default cohort
    python data/generate_synthetic.py --seconds 120   # longer recordings
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

FS_HZ = 250
CHANNELS = ["ECG_II", "PPG", "RESP"]

# cohort -> (hr_low, hr_high, noise_sigma, description)
COHORTS: dict[str, tuple[float, float, float, str]] = {
    "normal":      (55, 95, 0.02, "healthy sinus rhythm"),
    "tachycardia": (130, 170, 0.02, "sustained tachycardia -> critical alarm"),
    "bradycardia": (30, 42, 0.02, "sustained bradycardia -> critical alarm"),
    "signal_loss": (60, 80, 0.55, "heavy noise -> low SQI, DQ gap"),
    "poisoned":    (60, 80, 0.02, "adversarial outlier -> coreset guard veto"),
}


def synth_ecg(n: int, fs: int, hr: float, noise: float,
              rng: np.random.Generator) -> np.ndarray:
    """ECG as sparse Gaussian R-peaks at the beat rate (Pan-Tompkins friendly)."""
    t = np.arange(n) / fs
    rr = 60.0 / hr                      # R-R interval, seconds
    ecg = np.zeros(n)
    width = 0.020                       # ~20 ms QRS
    beat_t = 0.5 * rr
    while beat_t < n / fs:
        jitter = rng.normal(0, 0.01)    # beat-to-beat variability
        ecg += np.exp(-0.5 * ((t - beat_t - jitter) / width) ** 2)
        beat_t += rr
    return ecg + rng.normal(0, noise, n)


def synth_patient(patient_id: str, cohort: str, seconds: float,
                  seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    hr_lo, hr_hi, noise, _ = COHORTS[cohort]
    hr = float(rng.uniform(hr_lo, hr_hi))
    n = int(seconds * FS_HZ)
    t = np.arange(n) / FS_HZ

    ecg = synth_ecg(n, FS_HZ, hr, noise, rng)
    ppg = 0.6 * np.sin(2 * np.pi * (hr / 60.0) * t - 0.4) + rng.normal(0, noise, n)
    resp = 0.4 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.02, n)

    if cohort == "poisoned":
        # Fabricated (adversarial) window injected in the 3rd 4-second window,
        # placed early so short replays actually exercise Layer 4's guard.
        #
        # NOTE: simply scaling the amplitude (e.g. ecg *= 50) does NOT work as a
        # poison here -- every feature in the Layer 4 embedding (HR from peak
        # positions, RR-std, SQI as a power ratio, PRD as an energy ratio) is
        # SCALE-INVARIANT, so an amplitude attack is invisible. To move the
        # feature vector you must corrupt the *rhythm*: an erratic burst of
        # dense beats punctuated by long gaps drives RR-std far outside any
        # physiological range.
        lo, hi = FS_HZ * 8, FS_HZ * 12
        seg = np.zeros(hi - lo)
        tt = np.arange(hi - lo) / FS_HZ
        bt = 0.1
        while bt < (hi - lo) / FS_HZ:
            seg += np.exp(-0.5 * ((tt - bt) / 0.020) ** 2)
            bt += rng.choice([0.21, 0.22, 1.70])     # dense, then a long gap
        ecg[lo:hi] = seg + rng.normal(0, 0.02, hi - lo)

    samples = np.vstack([ecg, ppg, resp]).astype(np.float32)
    meta = {
        "patient_id": patient_id,
        "cohort": cohort,
        "fs_hz": FS_HZ,
        "channels": CHANNELS,
        "hr_true_bpm": round(hr, 1),
        "seconds": seconds,
        "seed": seed,
    }
    return samples, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic vitals cohort.")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="recording length per patient (default 60)")
    ap.add_argument("--per-cohort", type=int, default=2,
                    help="patients per cohort (default 2)")
    ap.add_argument("--hospitals", type=int, default=3,
                    help="number of hospitals/tenants to spread patients across")
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).parent / "synthetic")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    pid_n = 1
    for cohort in COHORTS:
        for i in range(args.per_cohort):
            pid = f"P{pid_n:03d}"
            hospital_id = f"H{(pid_n - 1) % max(1, args.hospitals) + 1:03d}"
            seed = 1000 + pid_n           # deterministic
            samples, meta = synth_patient(pid, cohort, args.seconds, seed)
            np.savez_compressed(
                args.outdir / f"{pid}.npz",
                samples=samples,
                patient_id=pid, cohort=cohort, fs_hz=FS_HZ,
                channels=np.array(CHANNELS), hr_true_bpm=meta["hr_true_bpm"],
                hospital_id=hospital_id,
            )
            rows.append({
                "patient_id": pid, "device_id": f"MON-{pid}",
                "hospital_id": hospital_id, "cohort": cohort,
                "fs_hz": FS_HZ, "n_channels": len(CHANNELS),
                "n_samples": samples.shape[1], "seconds": args.seconds,
                "hr_true_bpm": meta["hr_true_bpm"],
                "expected_behaviour": COHORTS[cohort][3], "seed": seed,
            })
            pid_n += 1

    manifest = args.outdir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # a small human-readable CSV of one patient's first 2 s, for eyeballing
    sample_npz = np.load(args.outdir / "P001.npz", allow_pickle=True)
    head = sample_npz["samples"][:, : FS_HZ * 2].T
    with (args.outdir / "P001_preview.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + CHANNELS)
        for i, row in enumerate(head):
            w.writerow([round(i / FS_HZ, 4)] + [round(float(v), 5) for v in row])

    total_mb = sum(p.stat().st_size for p in args.outdir.glob("*.npz")) / 1e6
    print(f"Wrote {len(rows)} patients to {args.outdir}  ({total_mb:.2f} MB)")
    print(f"  manifest: {manifest}")
    print(f"  preview:  {args.outdir / 'P001_preview.csv'}")


if __name__ == "__main__":
    main()
