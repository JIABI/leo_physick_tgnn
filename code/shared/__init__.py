"""Shared experiment, provenance, metric, and inference utilities.

The satellite and UAV implementations use these contracts so that run IDs,
episode IDs, exogenous streams, metric denominators, and uncertainty estimates
have identical semantics across platforms.
"""

from .schemas import (
    EpisodeManifest,
    MetricObservation,
    ResultRecord,
    RunManifest,
)
from .seeding import SeedBundle, derive_seed, seed_everything
from .statistics import BootstrapMode, ContrastSummary, paired_hierarchical_bootstrap

__all__ = [
    "BootstrapMode",
    "ContrastSummary",
    "EpisodeManifest",
    "MetricObservation",
    "ResultRecord",
    "RunManifest",
    "SeedBundle",
    "derive_seed",
    "paired_hierarchical_bootstrap",
    "seed_everything",
]
