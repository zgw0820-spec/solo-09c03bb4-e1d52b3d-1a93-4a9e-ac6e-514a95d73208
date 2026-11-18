"""End-to-end tests for POST /v1/compare via the ASGI test client."""

from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import create_app
from tests.conftest import CANDIDATE_VERSION, MODEL_BYTES, write_candidate


def compare_payload(model_id: str = "alpha", version: str = CANDIDATE_VERSION) -> dict:
    return {
        "model_id": model_id,
        "candidate_version": version,
        "vibration": [1.0 + (index % 3) for index in range(8)],
        "current": [5.0 + (index % 2) for index in range(8)],
    }


async def test_compare_returns_both_versions_and_score_difference(client: AsyncClient):
    response = await client.post("/v1/compare", json=compare_payload())
    assert response.status_code == 200
    body = response.json()

    assert body["model_id"] == "alpha"
    assert body["current"]["model_version"] == "1.0.0"
    assert body["candidate"]["model_version"] == CANDIDATE_VERSION
    assert 0.0 <= body["current"]["anomaly_score"] <= 1.0
    assert 0.0 <= body["candidate"]["anomaly_score"] <= 1.0
    assert isinstance(body["current"]["is_anomaly"], bool)
    assert isinstance(body["candidate"]["is_anomaly"], bool)
    assert body["decision_changed"] == (
        body["current"]["is_anomaly"] != body["candidate"]["is_anomaly"]
    )
    assert body["score_difference"] == pytest.approx(
        body["candidate"]["anomaly_score"] - body["current"]["anomaly_score"]
    )
    assert len(body["features"]) == 12


async def test_compare_is_deterministic(client: AsyncClient):
    first = await client.post("/v1/compare", json=compare_payload())
    second = await client.post("/v1/compare", json=compare_payload())
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


async def test_compare_matches_standalone_inference_for_current_version(
    client: AsyncClient,
):
    infer = await client.post(
        "/v1/infer",
        json={key: value for key, value in compare_payload().items() if key != "candidate_version"},
    )
    comparison = await client.post("/v1/compare", json=compare_payload())
    assert infer.status_code == 200
    assert comparison.status_code == 200
    infer_body = infer.json()
    current = comparison.json()["current"]
    assert current["anomaly_score"] == infer_body["anomaly_score"]
    assert current["is_anomaly"] == infer_body["is_anomaly"]
    assert current["decision_threshold"] == infer_body["decision_threshold"]


async def test_formal_infer_never_uses_candidate_weights(client: AsyncClient):
    # Even after a comparison loaded the candidate into the cache, /v1/infer
    # must still answer with the 1.0.0 production weights.
    await client.post("/v1/compare", json=compare_payload())
    infer = await client.post(
        "/v1/infer",
        json={
            "model_id": "alpha",
            "vibration": compare_payload()["vibration"],
            "current": compare_payload()["current"],
        },
    )
    assert infer.status_code == 200
    assert infer.json()["model_version"] == "1.0.0"


async def test_compare_unknown_candidate_returns_404(client: AsyncClient):
    response = await client.post(
        "/v1/compare", json=compare_payload(version="not-published")
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "unknown_candidate"
    assert error["candidate_version"] == "not-published"
    assert error["model_id"] == "alpha"


async def test_compare_unknown_model_returns_404(client: AsyncClient):
    response = await client.post(
        "/v1/compare", json=compare_payload(model_id="nope")
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_model"


async def test_compare_invalid_measurement_returns_422(client: AsyncClient):
    payload = compare_payload()
    payload["vibration"] = []
    response = await client.post("/v1/compare", json=payload)
    # Empty arrays are rejected by request schema validation, exactly like
    # POST /v1/infer.
    assert response.status_code == 422


async def test_compare_missing_version_field_returns_422(client: AsyncClient):
    payload = compare_payload()
    del payload["candidate_version"]
    response = await client.post("/v1/compare", json=payload)
    assert response.status_code == 422


async def test_compare_busy_returns_503_without_touching_production(
    manifest, settings, candidate_dir
):
    tight = settings.__class__(
        **{**settings.__dict__, "model_budget_bytes": MODEL_BYTES, "inference_delay_ms": 300}
    )
    app = create_app(tight, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(
                client.post("/v1/infer", json={"model_id": "alpha", **_signals()})
            )
            await asyncio.sleep(0.1)

            rejected = await client.post("/v1/compare", json=compare_payload("alpha"))
            assert rejected.status_code == 503
            assert rejected.headers["retry-after"] == "1"
            assert rejected.json()["error"]["code"] == "model_busy"

            assert (await first).status_code == 200
            snapshot = (await client.get("/v1/cache")).json()
            assert snapshot["counters"]["evictions"] == 0
            assert snapshot["used_bytes"] <= tight.model_budget_bytes


async def test_compare_rejects_when_two_production_models_fill_budget(
    manifest, settings, candidate_dir
):
    tight = settings.__class__(
        **{**settings.__dict__, "model_budget_bytes": MODEL_BYTES * 2}
    )
    app = create_app(tight, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/v1/infer", json={"model_id": "alpha", **_signals()})
            await client.post("/v1/infer", json={"model_id": "beta", **_signals()})

            rejected = await client.post("/v1/compare", json=compare_payload("alpha"))
            assert rejected.status_code == 503
            snapshot = (await client.get("/v1/cache")).json()
            assert snapshot["counters"]["candidate_busy_rejections"] == 1
            assert snapshot["counters"]["evictions"] == 0
            assert snapshot["counters"]["evictions"] == 0


async def test_invalid_candidate_file_returns_500_and_releases_budget(
    manifest, settings, candidate_dir
):
    app = create_app(settings, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            (candidate_dir / f"alpha-{CANDIDATE_VERSION}.json").write_text(
                "broken", encoding="utf-8"
            )
            failed = await client.post("/v1/compare", json=compare_payload("alpha"))
            assert failed.status_code == 500
            assert failed.json()["error"]["code"] == "model_load_failed"

            snapshot = (await client.get("/v1/cache")).json()
            assert snapshot["reserved_bytes"] == 0
            # The current/production leg did load and remains resident; only
            # the failed candidate reservation was rolled back.
            assert snapshot["resident_bytes"] == MODEL_BYTES
            assert snapshot["counters"]["candidate_load_failed"] == 1


async def test_compare_uses_republished_catalog_without_restart(
    manifest, settings, candidate_dir
):
    app = create_app(settings, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            missing = await client.post(
                "/v1/compare", json=compare_payload("alpha", version="2.0.0")
            )
            assert missing.status_code == 404

            file_name = write_candidate(candidate_dir, "alpha", version="2.0.0")
            raw = json.loads((candidate_dir / "candidate_manifest.json").read_text())
            raw["candidates"].append(
                {
                    "model_id": "alpha",
                    "version": "2.0.0",
                    "file": file_name,
                    "weights_bytes": MODEL_BYTES,
                }
            )
            (candidate_dir / "candidate_manifest.json").write_text(json.dumps(raw))

            ok = await client.post(
                "/v1/compare", json=compare_payload("alpha", version="2.0.0")
            )
            assert ok.status_code == 200
            assert ok.json()["candidate"]["model_version"] == "2.0.0"


async def test_compare_endpoint_is_published_in_swagger(client: AsyncClient):
    openapi = await client.get("/openapi.json")
    assert openapi.status_code == 200
    paths = openapi.json()["paths"]
    assert "/v1/compare" in paths
    assert "post" in paths["/v1/compare"]
    schema = openapi.json()["components"]["schemas"]
    assert "ComparisonResponse" in schema
    assert "CompareRequest" in schema


async def test_concurrent_compares_load_candidate_once(manifest, settings, candidate_dir):
    slow = settings.__class__(
        **{**settings.__dict__, "model_load_delay_ms": 150, "inference_delay_ms": 10}
    )
    app = create_app(slow, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(
                *(client.post("/v1/compare", json=compare_payload()) for _ in range(8))
            )
            assert {response.status_code for response in responses} == {200}
            snapshot = (await client.get("/v1/cache")).json()
            assert snapshot["counters"]["candidate_load_attempts"] == 1
            assert snapshot["counters"]["candidate_load_succeeded"] == 1
            assert snapshot["counters"]["load_attempts"] == 1


async def test_successful_compare_leaves_no_inflight_and_keeps_budget(
    client: AsyncClient,
):
    response = await client.post("/v1/compare", json=compare_payload())
    assert response.status_code == 200
    snapshot = (await client.get("/v1/cache")).json()
    assert all(entry["inflight"] == 0 for entry in snapshot["models"])
    assert snapshot["used_bytes"] <= snapshot["budget_bytes"]


def _signals() -> dict:
    return {
        "vibration": [1.0 + (index % 3) for index in range(8)],
        "current": [5.0 + (index % 2) for index in range(8)],
    }
