"""Fail-closed parameter identities for the learned systems in Table S11.

The source-data ledger reports exact deployed parameter counts, whereas the
public manuscript describes only selected structural hyperparameters.  This
module keeps those two facts separate.  ``audit_reported_model`` constructs the
available executable reference and reports its real count.
``build_exact_reported_model`` returns a model only when that real count equals
the frozen ledger.  It never allocates an unused parameter tensor to bridge a
gap and never relabels a structural reference as a recovered checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch.nn as nn

from .models import build_paper_model, parameter_count


@dataclass(frozen=True)
class ReportedModelIdentity:
    """One exact model identity read from ``paper_model_ledger``."""

    implementation_id: str
    method: str
    interface: str
    reported_online_parameters: int
    evidence: str

    @classmethod
    def from_mapping(
        cls,
        implementation_id: str,
        value: Mapping[str, Any],
    ) -> "ReportedModelIdentity":
        if not isinstance(implementation_id, str) or not implementation_id.strip():
            raise ValueError("implementation_id must be non-empty")
        if not isinstance(value, Mapping):
            raise TypeError(f"paper_model_ledger.{implementation_id} must be a mapping")
        required = ("method", "interface", "reported_online_parameters", "evidence")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(
                f"paper_model_ledger.{implementation_id} is missing: "
                + ", ".join(missing)
            )
        reported = value["reported_online_parameters"]
        if isinstance(reported, bool) or int(reported) != reported or int(reported) <= 0:
            raise ValueError("reported_online_parameters must be a positive integer")
        method = str(value["method"]).strip()
        interface = str(value["interface"]).strip()
        evidence = str(value["evidence"]).strip()
        if not method or not interface or not evidence:
            raise ValueError("method, interface and evidence must be non-empty")
        return cls(
            implementation_id=implementation_id.strip(),
            method=method,
            interface=interface,
            reported_online_parameters=int(reported),
            evidence=evidence,
        )


@dataclass(frozen=True)
class ModelParameterAudit:
    """Comparison between a real constructor and a frozen paper identity."""

    identity: ReportedModelIdentity
    constructed_online_parameters: int
    exact_match: bool
    difference: int
    constructor_class: str
    status: str

    def as_dict(self) -> dict[str, str | int | bool]:
        return {
            "implementation_id": self.identity.implementation_id,
            "method": self.identity.method,
            "interface": self.identity.interface,
            "reported_online_parameters": self.identity.reported_online_parameters,
            "constructed_online_parameters": self.constructed_online_parameters,
            "difference": self.difference,
            "exact_match": self.exact_match,
            "constructor_class": self.constructor_class,
            "status": self.status,
            "evidence": self.identity.evidence,
        }


class ReportedModelIdentityMismatch(RuntimeError):
    """The public structural reference is not the exact reported model."""


def reported_model_identities(
    config: Mapping[str, Any],
) -> dict[str, ReportedModelIdentity]:
    """Parse the exact model ledger without constructing a network."""

    if not isinstance(config, Mapping):
        raise TypeError("paper configuration must be a mapping")
    ledger = config.get("paper_model_ledger")
    if not isinstance(ledger, Mapping) or not ledger:
        raise ValueError("paper_model_ledger mapping is required")
    return {
        str(implementation_id): ReportedModelIdentity.from_mapping(
            str(implementation_id), value
        )
        for implementation_id, value in ledger.items()
    }


def audit_reported_model(
    config: Mapping[str, Any],
    implementation_id: str,
) -> tuple[nn.Module, ModelParameterAudit]:
    """Construct and count one identity without claiming that it matches."""

    identities = reported_model_identities(config)
    if implementation_id not in identities:
        choices = ", ".join(sorted(identities))
        raise KeyError(
            f"unknown reported implementation {implementation_id!r}; expected {choices}"
        )
    identity = identities[implementation_id]
    model = build_paper_model(config, identity.method)
    actual = parameter_count(model)
    exact = actual == identity.reported_online_parameters
    audit = ModelParameterAudit(
        identity=identity,
        constructed_online_parameters=actual,
        exact_match=exact,
        difference=actual - identity.reported_online_parameters,
        constructor_class=type(model).__name__,
        status=(
            "exact_executable_identity"
            if exact
            else "structural_reference_not_checkpoint_exact"
        ),
    )
    return model, audit


def audit_reported_models(
    config: Mapping[str, Any],
) -> tuple[ModelParameterAudit, ...]:
    """Return deterministic ledger-order audits for every reported identity."""

    identities = reported_model_identities(config)
    return tuple(
        audit_reported_model(config, implementation_id)[1]
        for implementation_id in identities
    )


def build_exact_reported_model(
    config: Mapping[str, Any],
    implementation_id: str,
) -> nn.Module:
    """Return a paper-labelled model only after an exact real-count check."""

    model, audit = audit_reported_model(config, implementation_id)
    if not audit.exact_match:
        raise ReportedModelIdentityMismatch(
            f"{implementation_id} reports "
            f"{audit.identity.reported_online_parameters:,} online parameters, "
            f"but the executable {audit.constructor_class} constructor has "
            f"{audit.constructed_online_parameters:,} ({audit.difference:+,}). "
            "The manuscript/v4 release does not contain enough constructor detail "
            "to close this architecture-level gap; unused parameter padding is "
            "forbidden."
        )
    setattr(model, "paper_implementation_id", implementation_id)
    setattr(model, "reported_online_parameters", audit.constructed_online_parameters)
    setattr(model, "parameter_identity_evidence", audit.identity.evidence)
    return model


__all__ = [
    "ModelParameterAudit",
    "ReportedModelIdentity",
    "ReportedModelIdentityMismatch",
    "audit_reported_model",
    "audit_reported_models",
    "build_exact_reported_model",
    "reported_model_identities",
]
