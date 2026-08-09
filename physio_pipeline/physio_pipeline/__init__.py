"""
physio_pipeline
===============

An 8-layer, reference-architecture Python package for a secure, privacy-
preserving pipeline over high-frequency physiological time-series data.

Each layer is its own subpackage; `PhysioPipeline` in `pipeline.py` wires them
together. See README.md for the layer-to-module map and data-flow diagram.
"""
from .pipeline import BatchResult, PhysioPipeline
from .layer1_edge import EdgeDevice

__version__ = "0.2.0"
__all__ = ["PhysioPipeline", "BatchResult", "EdgeDevice", "__version__"]
