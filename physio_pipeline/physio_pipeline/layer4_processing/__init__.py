"""Layer 4 - Stream processing: anomaly, Sieve-Streaming coreset, DQ, poisoning guard."""
from .processing import SieveStreamingCoreset, StreamProcessor, embed

__all__ = ["StreamProcessor", "SieveStreamingCoreset", "embed"]
