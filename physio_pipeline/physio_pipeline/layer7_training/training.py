"""
Layer 7 - BATCH ML TRAINING
==========================

Trains from the coreset parked in Layer 5, end to end:

  coreset -> ETL -> feature-eng -> DVC-version -> DP-SGD train
          -> MLflow-log -> validate (gate) -> SBOM -> deploy(shadow/prod)

DP-SGD adds calibrated Gaussian noise to clipped gradients and debits the
shared privacy-budget ledger, so training's privacy cost composes with the
DP release store's.

Production swap-in: Spark ETL, DVC, PyTorch + Opacus (DP-SGD), MLflow, a
validation/FDA gate, Triton serving with shadow deployment.
"""
from __future__ import annotations

import hashlib
import hmac

import numpy as np

from ..core.contracts import ModelArtifact
from ..crosscutting.audit import AuditLog
from ..crosscutting.privacy_budget import PrivacyBudgetLedger
from ..layer5_storage.storage import StorageLayer


class TrainingPipeline:
    def __init__(self, storage: StorageLayer, ledger: PrivacyBudgetLedger,
                 audit: AuditLog, signing_key: bytes = b"kms-managed"):
        self.storage = storage
        self.ledger = ledger
        self.audit = audit
        self._key = signing_key

    # ---- ETL + feature engineering ---------------------------------------
    def _etl(self, feature_keys: list[str]) -> np.ndarray:
        rows = [self.storage.features.get(k) for k in feature_keys]
        return np.vstack(rows) if rows else np.empty((0, 4))

    # ---- DP-SGD (logistic regression, one label derived from HR proxy) ----
    def _train_dp_sgd(self, X: np.ndarray, epsilon: float, delta: float,
                      epochs: int = 30, lr: float = 0.1, clip: float = 1.0
                      ) -> tuple[np.ndarray, float]:
        if len(X) == 0:
            return np.zeros(4), 0.0
        # synthetic label: elevated HR proxy (feature 0) => 1
        y = (X[:, 0] > np.median(X[:, 0])).astype(float)
        w = np.zeros(X.shape[1])
        rng = np.random.default_rng(0)
        sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon   # noise multiplier
        for _ in range(epochs):
            preds = 1.0 / (1.0 + np.exp(-X @ w))
            grads = (preds - y)[:, None] * X                    # per-sample grads
            # clip per-sample gradients to bound sensitivity
            norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
            grads = grads * np.minimum(1.0, clip / norms)
            noise = rng.normal(0, sigma * clip, size=w.shape)
            g = (grads.sum(axis=0) + noise) / len(X)
            w -= lr * g
        # accuracy on the (noisy) training set as a smoke metric
        acc = float(np.mean((1.0 / (1.0 + np.exp(-X @ w)) > 0.5) == y))
        self.ledger.debit("dp_sgd_training", epsilon, delta)
        return w, acc

    def _sign(self, w: np.ndarray) -> str:
        return hmac.new(self._key, w.tobytes(), hashlib.sha256).hexdigest()

    def train(self, feature_keys: list[str], name: str = "vitals-risk",
              version: str = "0.1.0", epsilon: float = 1.0, delta: float = 1e-6
              ) -> ModelArtifact:
        X = self._etl(feature_keys)                              # ETL + features
        # (DVC versioning would snapshot X here)
        w, acc = self._train_dp_sgd(X, epsilon, delta)          # DP-SGD
        metrics = {"train_acc": round(acc, 3), "n": float(len(X))}
        # validation gate (stand-in for FDA/clinical validation)
        stage = "shadow" if acc >= 0.5 and len(X) > 0 else "rejected"
        sbom = ["numpy", "physio_pipeline", "opacus(prod)", "torch(prod)"]
        artifact = ModelArtifact(
            name=name, version=version, weights=w, metrics=metrics,
            dp_epsilon_spent=epsilon, sbom=sbom, stage=stage,
            signature=self._sign(w),
        )
        self.audit.record("layer7", "train", "ml_trainer", ok=(stage != "rejected"),
                          acc=metrics["train_acc"], stage=stage)
        return artifact
