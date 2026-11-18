#!/usr/bin/env python3
"""Generate sample candidate-weight artifacts for the comparison endpoint.

The candidate directory is deliberately independent from ``models/``: it has
its own ``candidate_manifest.json`` and is mounted into the container at
``/srv/candidate_models``.  Nothing here is ever read by ``POST /v1/infer``;
only ``POST /v1/compare`` loads these weights, and only for the explicit
``(model_id, candidate_version)`` pair named in the request.

The generated ``1.1.0-rc1`` weights keep the same 12-feature schema and the
same 296-byte footprint as the production models but shift coefficients and
standardisation statistics slightly, so a comparison on real-looking signals
can produce a visible score difference and, near the decision threshold, a
flipped anomaly verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES = [
    "vibration_mean",
    "vibration_rms",
    "vibration_peak",
    "vibration_std",
    "vibration_crest",
    "vibration_kurtosis",
    "current_mean",
    "current_rms",
    "current_peak",
    "current_std",
    "current_crest",
    "current_kurtosis",
]

CANDIDATE_VERSION = "1.1.0-rc1"

# Same shape as the production weights; slightly retuned values so comparison
# results differ without changing the on-disk byte budget.
CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "pump",
        "name": "离心泵",
        "device_type": "pump",
        "description": "离心泵候选权重：提高振动类特征权重并下调截距",
        "feature_mean": [1.7, 2.5, 6.0, 1.8, 2.40, 3.0, 12.1, 13.1, 19.8, 2.9, 1.74, 3.0],
        "feature_scale": [1.1, 1.4, 2.7, 0.9, 0.55, 1.2, 4.5, 4.8, 7.0, 1.8, 0.40, 1.2],
        "coefficients": [0.12, 1.45, 0.72, 1.05, 0.60, 1.28, -0.08, 0.58, 0.38, 0.48, 0.32, 0.22],
        "intercept": -1.28,
        "decision_threshold": 0.50,
    },
    {
        "id": "fan",
        "name": "引风机",
        "device_type": "fan",
        "description": "引风机候选权重：峭度敏感度上调",
        "feature_mean": [1.2, 1.8, 4.5, 1.25, 2.60, 3.2, 8.5, 9.4, 14.5, 2.2, 1.65, 3.1],
        "feature_scale": [0.8, 1.0, 2.2, 0.7, 0.50, 1.5, 3.2, 3.5, 5.0, 1.4, 0.35, 1.4],
        "coefficients": [0.06, 1.02, 0.44, 1.52, 0.85, 1.72, -0.18, 0.38, 0.27, 0.32, 0.22, 0.17],
        "intercept": -1.10,
        "decision_threshold": 0.50,
    },
    {
        "id": "compressor",
        "name": "螺杆压缩机",
        "device_type": "compressor",
        "description": "螺杆压缩机候选权重：电流特征权重微调",
        "feature_mean": [2.4, 3.3, 8.0, 2.5, 2.35, 3.4, 23.8, 25.9, 37.8, 5.4, 1.55, 3.5],
        "feature_scale": [1.4, 1.8, 3.5, 1.2, 0.60, 1.8, 8.0, 8.5, 12.0, 3.0, 0.45, 1.6],
        "coefficients": [0.17, 1.16, 0.58, 0.90, 0.48, 0.95, 0.14, 0.85, 0.58, 0.68, 0.37, 0.27],
        "intercept": -1.38,
        "decision_threshold": 0.50,
    },
]


def weights_bytes(artifact: dict[str, Any]) -> int:
    total = np.asarray(artifact["coefficients"], dtype=np.float64).nbytes
    total += np.asarray(artifact["feature_mean"], dtype=np.float64).nbytes
    total += np.asarray(artifact["feature_scale"], dtype=np.float64).nbytes
    total += np.asarray(artifact["intercept"], dtype=np.float64).nbytes
    return int(total)


def main() -> int:
    output_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1] / "candidate_models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for definition in CANDIDATES:
        if len(definition["feature_mean"]) != len(FEATURE_NAMES):
            raise SystemExit(f"{definition['id']}: feature_mean length mismatch")
        if len(definition["feature_scale"]) != len(FEATURE_NAMES):
            raise SystemExit(f"{definition['id']}: feature_scale length mismatch")
        if any(scale <= 0 for scale in definition["feature_scale"]):
            raise SystemExit(f"{definition['id']}: feature_scale must be positive")

        artifact = {
            "model_id": definition["id"],
            "name": definition["name"],
            "device_type": definition["device_type"],
            "description": definition["description"],
            "version": CANDIDATE_VERSION,
            "features": FEATURE_NAMES,
            "feature_mean": definition["feature_mean"],
            "feature_scale": definition["feature_scale"],
            "coefficients": definition["coefficients"],
            "intercept": definition["intercept"],
            "decision_threshold": definition["decision_threshold"],
        }
        file_name = f"{definition['id']}-{CANDIDATE_VERSION}.json"
        (output_dir / file_name).write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest_entries.append(
            {
                "model_id": definition["id"],
                "version": CANDIDATE_VERSION,
                "file": file_name,
                "weights_bytes": weights_bytes(artifact),
            }
        )

    manifest = {"schema_version": 1, "candidates": manifest_entries}
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
