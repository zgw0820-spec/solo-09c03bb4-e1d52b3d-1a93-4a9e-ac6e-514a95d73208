"""Disconnect and interleaving semantics for POST /v1/compare.

These tests inject ``http.disconnect`` at the ASGI layer (httpx always waits
for the app to finish, so a dropped connection cannot be expressed through it).
"""

from __future__ import annotations

import asyncio

import pytest

from app.api import create_app
from app.errors import ModelBusyError
from tests.conftest import CANDIDATE_VERSION, MODEL_BYTES
from tests.helpers import RawASGIRequest


def compare_payload(model_id: str = "alpha", version: str = CANDIDATE_VERSION) -> dict:
    return {
        "model_id": model_id,
        "candidate_version": version,
        "vibration": [1.0 + (index % 4) for index in range(16)],
        "current": [5.0 + (index % 2) for index in range(16)],
    }


def infer_payload(model_id: str = "alpha") -> dict:
    return {
        "model_id": model_id,
        "vibration": compare_payload()["vibration"],
        "current": compare_payload()["current"],
    }


async def test_disconnect_while_waiting_for_candidate_leaves_no_half_result(
    manifest, settings
):
    slow = settings.__class__(**{**settings.__dict__, "model_load_delay_ms": 300})
    app = create_app(slow, manifest=manifest)

    async with app.router.lifespan_context(app):
        dropped = RawASGIRequest(
            app, "/v1/compare", compare_payload(), disconnect_after=0.05
        )
        dropped_task = asyncio.create_task(dropped.run())
        response = await dropped_task
        assert response.status_code == 499

        cache = app.state.cache
        snapshot = cache.snapshot()
        # The production leg had started loading; no candidate lease or
        # reservation exists because the client left before it was acquired.
        assert all(not entry["is_candidate"] for entry in snapshot["models"])
        assert all(entry["inflight"] == 0 for entry in snapshot["models"])

        # A later, intact comparison still succeeds end to end.
        survivor = RawASGIRequest(app, "/v1/compare", compare_payload())
        survivor_response = await survivor.run()
        assert survivor_response.status_code == 200
        body = survivor_response.json
        assert body["current"]["model_version"] == "1.0.0"
        assert body["candidate"]["model_version"] == CANDIDATE_VERSION

        snapshot = cache.snapshot()
        assert all(entry["inflight"] == 0 for entry in snapshot["models"])
        assert snapshot["used_bytes"] <= snapshot["budget_bytes"]


async def test_disconnect_during_dual_compute_holds_both_leases_until_done(
    manifest, settings
):
    slow = settings.__class__(**{**settings.__dict__, "inference_delay_ms": 400})
    app = create_app(slow, manifest=manifest)

    async with app.router.lifespan_context(app):
        request = RawASGIRequest(
            app, "/v1/compare", compare_payload(), disconnect_after=0.1
        )
        started = asyncio.get_running_loop().time()
        await request.run()
        elapsed = asyncio.get_running_loop().time() - started

        cache = app.state.cache
        snapshot = cache.snapshot()
        # Both detached legs ran to completion before the ASGI call returned
        # and every lease was released by the computation tasks themselves.
        assert all(entry["inflight"] == 0 for entry in snapshot["models"])
        assert elapsed >= slow.inference_delay_seconds * 0.9
        assert snapshot["counters"]["candidate_load_succeeded"] == 1


async def test_disconnected_comparison_pins_candidate_against_production_load(
    manifest, settings
):
    """A still-computing candidate cannot be evicted by a formal cold load."""

    tight = settings.__class__(
        **{**settings.__dict__, "model_budget_bytes": MODEL_BYTES * 2,
           "inference_delay_ms": 400}
    )
    app = create_app(tight, manifest=manifest)

    async with app.router.lifespan_context(app):
        request = RawASGIRequest(
            app, "/v1/compare", compare_payload("alpha"), disconnect_after=0.08
        )
        request_task = asyncio.create_task(request.run())

        await asyncio.sleep(0.2)
        cache = app.state.cache
        snapshot = cache.snapshot()
        inflight = [entry for entry in snapshot["models"] if entry["inflight"] > 0]
        assert len(inflight) == 2  # current + candidate both still computing

        # The budget is full and both comparison legs are busy, so a formal
        # inference for another model must be rejected rather than evicting.
        with pytest.raises(ModelBusyError):
            await cache.acquire("beta")

        await request_task

        snapshot = cache.snapshot()
        assert all(entry["inflight"] == 0 for entry in snapshot["models"])

        # Once the comparison is fully over, its idle candidate yields and the
        # formal cold load can be admitted (evicting the candidate first).
        lease = await cache.acquire("beta")
        snapshot = cache.snapshot()
        assert snapshot["counters"]["candidate_evictions"] == 1
        assert {entry["model_id"] for entry in snapshot["models"]} == {"alpha", "beta"}
        await lease.release()


async def test_formal_inference_interleaved_with_comparisons_keeps_budget(
    manifest, settings
):
    slow = settings.__class__(
        **{**settings.__dict__, "model_budget_bytes": MODEL_BYTES * 3,
           "inference_delay_ms": 50, "model_load_delay_ms": 20}
    )
    app = create_app(slow, manifest=manifest)

    async with app.router.lifespan_context(app):
        results = await asyncio.gather(
            *(RawASGIRequest(app, "/v1/compare", compare_payload("alpha")).run()
              for _ in range(4)),
            *(RawASGIRequest(app, "/v1/infer", infer_payload("beta")).run()
              for _ in range(4)),
            return_exceptions=True,
        )
        statuses = {response.status_code for response in results}
        assert statuses <= {200}

        cache = app.state.cache
        snapshot = cache.snapshot()
        assert snapshot["used_bytes"] <= snapshot["budget_bytes"]
        assert all(entry["inflight"] == 0 for entry in snapshot["models"])
        # The production model must always be the resident 1.0.0 weights.
        beta = next(entry for entry in snapshot["models"] if entry["model_id"] == "beta")
        assert beta["is_candidate"] is False


async def test_disconnected_waiter_does_not_cancel_candidate_load(manifest, settings):
    slow = settings.__class__(**{**settings.__dict__, "model_load_delay_ms": 250})
    app = create_app(slow, manifest=manifest)

    async with app.router.lifespan_context(app):
        cache = app.state.cache
        dropped = asyncio.create_task(
            cache.acquire_candidate("alpha", CANDIDATE_VERSION)
        )
        await asyncio.sleep(0.02)
        dropped.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dropped

        snapshot = cache.snapshot()
        assert snapshot["reserved_bytes"] == MODEL_BYTES
        assert snapshot["counters"]["candidate_load_attempts"] == 1

        # The shared candidate load survives and another caller gets it.
        lease = await cache.acquire_candidate("alpha", CANDIDATE_VERSION)
        try:
            assert lease.model.version == CANDIDATE_VERSION
            assert cache.snapshot()["counters"]["candidate_load_succeeded"] == 1
        finally:
            await lease.release()
