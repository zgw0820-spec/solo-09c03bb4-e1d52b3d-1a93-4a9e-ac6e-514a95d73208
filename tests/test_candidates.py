"""Validation tests for the independent candidate-weight catalog."""

from __future__ import annotations

import json

import pytest

from app.candidates import (
    load_candidate_catalog,
    read_candidate_file,
)
from app.errors import ModelLoadError, UnknownCandidateError
from tests.conftest import CANDIDATE_VERSION, write_candidate


def test_catalog_lists_published_versions(candidate_dir):
    catalog = load_candidate_catalog(candidate_dir)
    descriptor = catalog.descriptor("alpha", CANDIDATE_VERSION)
    assert descriptor.model_id == "alpha"
    assert descriptor.version == CANDIDATE_VERSION
    assert descriptor.declared_bytes == 296


def test_unknown_version_is_a_domain_404_error(candidate_dir):
    catalog = load_candidate_catalog(candidate_dir)
    with pytest.raises(UnknownCandidateError):
        catalog.descriptor("alpha", "9.9.9")
    with pytest.raises(UnknownCandidateError):
        catalog.descriptor("missing", CANDIDATE_VERSION)


def test_missing_candidate_manifest_is_value_error(tmp_path):
    with pytest.raises(ValueError, match="candidate manifest not found"):
        load_candidate_catalog(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "empty", "bad_id", "bad_version", "bad_file", "bad_bytes", "duplicate"],
)
def test_invalid_candidate_manifests_are_rejected(candidate_dir, mutation):
    path = candidate_dir / "candidate_manifest.json"
    raw = json.loads(path.read_text())

    if mutation == "schema":
        raw["schema_version"] = 2
    elif mutation == "empty":
        raw["candidates"] = []
    elif mutation == "bad_id":
        raw["candidates"][0]["model_id"] = "bad/id"
    elif mutation == "bad_version":
        raw["candidates"][0]["version"] = "../escape"
    elif mutation == "bad_file":
        raw["candidates"][0]["file"] = "../outside.json"
    elif mutation == "bad_bytes":
        raw["candidates"][0]["weights_bytes"] = 0
    elif mutation == "duplicate":
        raw["candidates"].append(dict(raw["candidates"][0]))

    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_candidate_catalog(candidate_dir)


def test_candidate_file_round_trip(candidate_dir):
    catalog = load_candidate_catalog(candidate_dir)
    descriptor = catalog.descriptor("alpha", CANDIDATE_VERSION)
    model = read_candidate_file(descriptor, candidate_dir)
    assert model.model_id == "alpha"
    assert model.version == CANDIDATE_VERSION
    assert model.weights_bytes == 296


@pytest.mark.parametrize(
    "mutation",
    ["model_id", "version", "features", "coefficients", "scale", "threshold", "missing"],
)
def test_invalid_candidate_files_are_rejected(candidate_dir, mutation):
    catalog = load_candidate_catalog(candidate_dir)
    descriptor = catalog.descriptor("alpha", CANDIDATE_VERSION)
    path = candidate_dir / descriptor.file_name
    raw = json.loads(path.read_text())

    if mutation == "model_id":
        raw["model_id"] = "other"
    elif mutation == "version":
        raw["version"] = "0.0.1"
    elif mutation == "features":
        raw["features"] = raw["features"][:-1]
    elif mutation == "coefficients":
        raw["coefficients"] = [1.0, 2.0]
    elif mutation == "scale":
        raw["feature_scale"][0] = 0.0
    elif mutation == "threshold":
        raw["decision_threshold"] = 1.5
    elif mutation == "missing":
        path.unlink()

    if mutation != "missing":
        path.write_text(json.dumps(raw))

    with pytest.raises(ModelLoadError):
        read_candidate_file(descriptor, candidate_dir)


def test_size_mismatch_between_manifest_and_arrays_is_rejected(candidate_dir):
    manifest_path = candidate_dir / "candidate_manifest.json"
    raw = json.loads(manifest_path.read_text())
    raw["candidates"][0]["weights_bytes"] = 999
    manifest_path.write_text(json.dumps(raw))

    catalog = load_candidate_catalog(candidate_dir)
    descriptor = catalog.descriptor("alpha", CANDIDATE_VERSION)
    with pytest.raises(ModelLoadError, match="does not match declared"):
        read_candidate_file(descriptor, candidate_dir)


def test_republished_catalog_is_visible_without_reload(candidate_dir):
    catalog = load_candidate_catalog(candidate_dir)
    with pytest.raises(UnknownCandidateError):
        catalog.descriptor("alpha", "2.0.0")

    file_name = write_candidate(candidate_dir, "alpha", version="2.0.0")
    raw = json.loads((candidate_dir / "candidate_manifest.json").read_text())
    raw["candidates"].append(
        {
            "model_id": "alpha",
            "version": "2.0.0",
            "file": file_name,
            "weights_bytes": 296,
        }
    )
    (candidate_dir / "candidate_manifest.json").write_text(json.dumps(raw))

    refreshed = load_candidate_catalog(candidate_dir)
    assert refreshed.descriptor("alpha", "2.0.0").file_name == file_name
