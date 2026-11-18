"""Independent candidate-model catalog used by the comparison endpoint.

Candidate weights live in their own directory, completely separate from the
production ``MODEL_DIR`` whose weights alone back ``POST /v1/infer``.  The
catalog is a small manifest mapping ``(model_id, version)`` pairs to weight
files; it is re-read on every comparison request so that publishing a new
candidate artifact never requires restarting the service.

Validation is intentionally as strict as :mod:`app.models`: the feature
schema must match the service, every array must be finite and correctly
sized, ``feature_scale`` must be positive and the declared byte budget must
match the arrays that would stay resident.  Any failure surfaces as
:class:`~app.errors.ModelLoadError`, which the cache turns into a clean
release of the reservation — a broken candidate never leaves half a result
or a budget leak behind.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ModelLoadError, UnknownCandidateError
from .features import FEATURE_NAMES
from .models import LinearModel

_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
MANIFEST_NAME = "candidate_manifest.json"


@dataclass(frozen=True)
class CandidateDescriptor:
    """One candidate weight artifact addressable by (model id, version)."""

    model_id: str
    version: str
    file_name: str
    declared_bytes: int


@dataclass(frozen=True)
class CandidateCatalog:
    """Snapshot of the candidate directory at one point in time."""

    candidate_dir: Path
    descriptors: dict[tuple[str, str], CandidateDescriptor]

    def descriptor(self, model_id: str, version: str) -> CandidateDescriptor:
        descriptor = self.descriptors.get((model_id, version))
        if descriptor is None:
            raise UnknownCandidateError(model_id, version)
        return descriptor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_candidate_catalog(candidate_dir: Path) -> CandidateCatalog:
    """Read and validate ``candidate_manifest.json`` from the candidate directory.

    Raises :class:`ValueError` for an unreadable or structurally invalid
    manifest; unknown ``(model_id, version)`` lookups raise
    :class:`UnknownCandidateError` later, at request time.
    """

    manifest_path = candidate_dir / MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"candidate manifest not found: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read candidate manifest {manifest_path}: {exc}") from exc

    _require(isinstance(raw, dict), "candidate manifest must be a JSON object")
    _require(raw.get("schema_version") == 1, "candidate manifest schema_version must be 1")

    entries = raw.get("candidates")
    _require(
        isinstance(entries, list) and entries,
        "candidate manifest candidates must be a non-empty list",
    )

    descriptors: dict[tuple[str, str], CandidateDescriptor] = {}
    for entry in entries:
        _require(isinstance(entry, dict), "each candidate entry must be an object")
        model_id = entry.get("model_id")
        version = entry.get("version")
        file_name = entry.get("file")
        weight_bytes = entry.get("weights_bytes")

        _require(
            isinstance(model_id, str) and bool(_MODEL_ID_PATTERN.fullmatch(model_id)),
            f"invalid candidate model id: {model_id!r}",
        )
        _require(
            isinstance(version, str) and bool(_VERSION_PATTERN.fullmatch(version)),
            f"invalid candidate version: {version!r}",
        )
        _require(
            isinstance(file_name, str)
            and file_name == Path(file_name).name
            and file_name.endswith(".json"),
            f"candidate {model_id}@{version}: file must be a plain .json file name",
        )
        _require(
            isinstance(weight_bytes, int)
            and not isinstance(weight_bytes, bool)
            and weight_bytes > 0,
            f"candidate {model_id}@{version}: weights_bytes must be a positive integer",
        )
        key = (model_id, version)
        _require(key not in descriptors, f"duplicate candidate entry: {model_id}@{version}")

        descriptors[key] = CandidateDescriptor(
            model_id=model_id,
            version=version,
            file_name=file_name,
            declared_bytes=weight_bytes,
        )

    return CandidateCatalog(candidate_dir=candidate_dir, descriptors=descriptors)


def _finite_array(value: Any, length: int, field: str, model_id: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(model_id, f"{field} is not numeric") from exc
    if array.shape != (length,):
        raise ModelLoadError(model_id, f"{field} must contain {length} values")
    if not np.all(np.isfinite(array)):
        raise ModelLoadError(model_id, f"{field} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def read_candidate_file(
    descriptor: CandidateDescriptor, candidate_dir: Path
) -> LinearModel:
    """Read and validate one candidate weight file.

    The file is resolved strictly inside ``candidate_dir``: only the plain
    file name recorded in the candidate manifest is accepted, so a malicious
    or broken manifest cannot point a load outside the mounted directory.
    """

    path = candidate_dir / descriptor.file_name
    # Defence in depth: manifest validation already restricts the file name to
    # a plain basename, but reject anything that resolves outside the catalog.
    if ".." in path.parts:
        raise ModelLoadError(descriptor.model_id, "candidate file escapes the candidate directory")
    base_dir = candidate_dir.resolve()
    if path.resolve().parent != base_dir:
        raise ModelLoadError(descriptor.model_id, "candidate file escapes the candidate directory")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelLoadError(
            descriptor.model_id, f"candidate file not found: {descriptor.file_name}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(
            descriptor.model_id, f"cannot read candidate {descriptor.file_name}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ModelLoadError(descriptor.model_id, "candidate file must contain a JSON object")
    if raw.get("model_id") != descriptor.model_id:
        raise ModelLoadError(descriptor.model_id, "model_id does not match candidate manifest")
    if raw.get("version") != descriptor.version:
        raise ModelLoadError(descriptor.model_id, "version does not match candidate manifest")
    if raw.get("features") != list(FEATURE_NAMES):
        raise ModelLoadError(descriptor.model_id, "feature schema does not match service schema")

    n_features = len(FEATURE_NAMES)
    coefficients = _finite_array(raw.get("coefficients"), n_features, "coefficients", descriptor.model_id)
    feature_mean = _finite_array(raw.get("feature_mean"), n_features, "feature_mean", descriptor.model_id)
    feature_scale = _finite_array(raw.get("feature_scale"), n_features, "feature_scale", descriptor.model_id)
    intercept = _finite_array([raw.get("intercept")], 1, "intercept", descriptor.model_id)[0]

    threshold_raw = raw.get("decision_threshold")
    if not isinstance(threshold_raw, (int, float)) or isinstance(threshold_raw, bool):
        raise ModelLoadError(descriptor.model_id, "decision_threshold must be a number")
    threshold = float(threshold_raw)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ModelLoadError(descriptor.model_id, "decision_threshold must be between 0 and 1")
    if np.any(feature_scale <= 0.0):
        raise ModelLoadError(descriptor.model_id, "feature_scale values must be positive")

    actual_bytes = int(
        coefficients.nbytes + feature_mean.nbytes + feature_scale.nbytes + intercept.nbytes
    )
    if actual_bytes != descriptor.declared_bytes:
        raise ModelLoadError(
            descriptor.model_id,
            f"candidate resident size {actual_bytes} does not match declared "
            f"weights_bytes {descriptor.declared_bytes}",
        )

    name = raw.get("name", descriptor.model_id)
    device_type = raw.get("device_type", descriptor.model_id)
    description = raw.get("description", "")

    return LinearModel(
        model_id=descriptor.model_id,
        name=name if isinstance(name, str) else descriptor.model_id,
        device_type=device_type if isinstance(device_type, str) else descriptor.model_id,
        version=descriptor.version,
        description=description if isinstance(description, str) else "",
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        intercept=float(intercept),
        decision_threshold=threshold,
        weights_bytes=actual_bytes,
    )
