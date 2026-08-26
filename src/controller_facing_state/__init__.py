"""Public orchestration layer for the manuscript reference implementation."""

from .config import load_experiment_registry, load_manuscript_config, validate_manuscript_config

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "load_experiment_registry",
    "load_manuscript_config",
    "validate_manuscript_config",
]

