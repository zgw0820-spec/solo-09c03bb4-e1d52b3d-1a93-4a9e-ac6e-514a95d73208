"""Budget and concurrency rules for candidate comparison weights."""

from __future__ import annotations

import asyncio

import pytest

from app.cache import ModelCache, candidate_cache_key
from app.candidates import load_candidate_catalog
from app.errors import (
    ModelBusyError,
    ModelLoadError,
    ModelTooLargeError,
    UnknownCandidateError,
)
from tests.conftest import CANDIDATE_VERSION, MODEL_BYTES, write_candidate


def make_cache(
    manifest,
    settings,
    candidate_dir,
    *,
    budget_bytes: int | None = None,
    load_delay_seconds: float = 0.0,
) -> ModelCache:
    def provider():
        return load_candidate_catalog(candidate_dir)

    return ModelCache(
        manifest,
        budget_bytes if budget_bytes is not None else settings.model_budget_bytes,
        load_delay_seconds,
        candidate_catalog_provider=provider,
    )


async def test_candidate_reservation_counts_toward_shared_budget(
    manifest, settings, candidate_dir
):
    cache = make_cache(
        manifest, settings, candidate_dir, budget_bytes=MODEL_BYTES * 2
    )

    production = await cache.acquire("alpha")
    snapshot = cache.snapshot()
    assert snapshot["resident_bytes"] == MODEL_BYTES

    candidate = await cache.acquire_candidate("beta", CANDIDATE_VERSION)
    snapshot = cache.snapshot()
    assert snapshot["used_bytes"] == MODEL_BYTES * 2
    assert snapshot["resident_bytes"] == MODEL_BYTES * 2
    assert snapshot["free_bytes"] == 0
    assert snapshot["counters"]["candidate_load_succeeded"] == 1

    await production.release()
    await candidate.release()
    await cache.aclose()


async def test_candidate_admission_never_evicts_production(manifest, settings, candidate_dir):
    # Budget holds exactly two entries; both slots are taken by production.
    cache = make_cache(
        manifest, settings, candidate_dir, budget_bytes=MODEL_BYTES * 2
    )
    alpha = await cache.acquire("alpha")
    beta = await cache.acquire("beta")

    with pytest.raises(ModelBusyError):
        await cache.acquire_candidate("gamma", CANDIDATE_VERSION)

    snapshot = cache.snapshot()
    assert snapshot["counters"]["candidate_busy_rejections"] == 1
    assert snapshot["counters"]["evictions"] == 0
    assert snapshot["counters"]["candidate_evictions"] == 0
    assert {entry["model_id"] for entry in snapshot["models"]} == {"alpha", "beta"}
    assert all(not entry["is_candidate"] for entry in snapshot["models"])

    await alpha.release()
    await beta.release()
    await cache.aclose()


async def test_candidate_admission_never_evicts_other_candidates(
    manifest, settings, candidate_dir
):
    cache = make_cache(
        manifest, settings, candidate_dir, budget_bytes=MODEL_BYTES * 2
    )
    candidate = await cache.acquire_candidate("alpha", CANDIDATE_VERSION)
    production = await cache.acquire("beta")

    # No room left; the resident candidate must not be displaced by another
    # candidate request.
    with pytest.raises(ModelBusyError):
        await cache.acquire_candidate("gamma", CANDIDATE_VERSION)

    snapshot = cache.snapshot()
    assert snapshot["counters"]["candidate_busy_rejections"] == 1
    assert snapshot["counters"]["candidate_evictions"] == 0
    assert snapshot["used_bytes"] == MODEL_BYTES * 2

    await candidate.release()
    await production.release()
    await cache.aclose()


async def test_idle_candidate_yields_to_production_before_production_lru(
    manifest, settings, candidate_dir
):
    cache = make_cache(
        manifest, settings, candidate_dir, budget_bytes=MODEL_BYTES * 2
    )
    alpha = await cache.acquire("alpha")
    await alpha.release()
    candidate = await cache.acquire_candidate("beta", CANDIDATE_VERSION)
    await candidate.release()

    snapshot = cache.snapshot()
    assert {entry["model_id"] for entry in snapshot["models"]} == {"alpha", "beta"}

    # gamma is a production cold load.  The idle candidate must go first;
    # the idle production alpha stays resident.
    gamma = await cache.acquire("gamma")
    snapshot = cache.snapshot()
    assert snapshot["counters"]["candidate_evictions"] == 1
    assert snapshot["counters"]["evictions"] == 0
    assert snapshot["last_eviction"] == f"beta@{CANDIDATE_VERSION}"
    assert {entry["model_id"] for entry in snapshot["models"]} == {"alpha", "gamma"}
    assert all(not entry["is_candidate"] for entry in snapshot["models"])

    await gamma.release()
    await cache.aclose()


async def test_busy_candidate_is_not_evicted_for_production(manifest, settings, candidate_dir):
    cache = make_cache(
        manifest, settings, candidate_dir, budget_bytes=MODEL_BYTES * 2
    )
    alpha = await cache.acquire("alpha")
    candidate = await cache.acquire_candidate("beta", CANDIDATE_VERSION)
    # Both entries have active leases: the candidate cannot be evicted and
    # neither can the production model.
    snapshot = cache.snapshot()
    assert all(entry["inflight"] == 1 for entry in snapshot["models"])

    with pytest.raises(ModelBusyError):
        await cache.acquire("gamma")

    snapshot = cache.snapshot()
    assert snapshot["counters"]["candidate_evictions"] == 0
    assert snapshot["counters"]["busy_rejections"] == 1
    resident = {entry["model_id"] for entry in snapshot["models"]}
    assert resident == {"alpha", "beta"}

    # Once the candidate computation ends, it becomes idle and yields: the
    # production cold load is then admitted by evicting the candidate first.
    await candidate.release()
    gamma = await cache.acquire("gamma")
    snapshot = cache.snapshot()
    assert snapshot["counters"]["candidate_evictions"] == 1
    assert {entry["model_id"] for entry in snapshot["models"]} == {"alpha", "gamma"}

    await gamma.release()
    await alpha.release()
    await cache.aclose()


async def test_candidate_single_flight_load(manifest, settings, candidate_dir):
    cache = make_cache(
        manifest,
        settings,
        candidate_dir,
        budget_bytes=MODEL_BYTES * 3,
        load_delay_seconds=0.12,
    )
    leases = await asyncio.gather(
        *(
            cache.acquire_candidate("alpha", CANDIDATE_VERSION)
            for _ in range(8)
        )
    )
    try:
        snapshot = cache.snapshot()
        assert snapshot["counters"]["candidate_load_attempts"] == 1
        assert snapshot["counters"]["candidate_load_succeeded"] == 1
        assert all(lease.model.version == CANDIDATE_VERSION for lease in leases)
        entry = next(
            entry
            for entry in snapshot["models"]
            if entry["is_candidate"]
        )
        assert entry["candidate_version"] == CANDIDATE_VERSION
    finally:
        for lease in leases:
            await lease.release()
    await cache.aclose()


async def test_unknown_candidate_version_raises(manifest, settings, candidate_dir):
    cache = make_cache(manifest, settings, candidate_dir)
    with pytest.raises(UnknownCandidateError):
        await cache.acquire_candidate("alpha", "nope")
    assert cache.snapshot()["used_bytes"] == 0
    await cache.aclose()


async def test_oversized_candidate_is_rejected(manifest, settings, candidate_dir):
    cache = make_cache(
        manifest, settings, candidate_dir, budget_bytes=MODEL_BYTES - 1
    )
    with pytest.raises(ModelTooLargeError):
        await cache.acquire_candidate("alpha", CANDIDATE_VERSION)
    assert cache.snapshot()["used_bytes"] == 0
    await cache.aclose()


async def test_invalid_candidate_file_clears_reservation(
    manifest, settings, candidate_dir
):
    cache = make_cache(
        manifest,
        settings,
        candidate_dir,
        budget_bytes=MODEL_BYTES * 2,
        load_delay_seconds=0.05,
    )
    (candidate_dir / f"alpha-{CANDIDATE_VERSION}.json").write_text(
        "{broken", encoding="utf-8"
    )

    with pytest.raises(ModelLoadError):
        await cache.acquire_candidate("alpha", CANDIDATE_VERSION)

    snapshot = cache.snapshot()
    assert snapshot["reserved_bytes"] == 0
    assert snapshot["resident_bytes"] == 0
    assert snapshot["counters"]["candidate_load_failed"] == 1
    assert candidate_cache_key("alpha", CANDIDATE_VERSION) not in [
        f"candidate:{m['model_id']}:{m['candidate_version']}" for m in snapshot["models"]
    ]

    # Other models still serve under the normal budget rules.
    lease = await cache.acquire("beta")
    assert lease.model.model_id == "beta"
    await lease.release()
    await cache.aclose()


async def test_repaired_candidate_can_be_retried(manifest, settings, candidate_dir):
    cache = make_cache(manifest, settings, candidate_dir)
    (candidate_dir / f"alpha-{CANDIDATE_VERSION}.json").write_text(
        "broken", encoding="utf-8"
    )
    with pytest.raises(ModelLoadError):
        await cache.acquire_candidate("alpha", CANDIDATE_VERSION)

    write_candidate(candidate_dir, "alpha", version=CANDIDATE_VERSION)
    lease = await cache.acquire_candidate("alpha", CANDIDATE_VERSION)
    assert lease.model.version == CANDIDATE_VERSION
    snapshot = cache.snapshot()
    assert snapshot["counters"]["candidate_load_failed"] == 1
    assert snapshot["counters"]["candidate_load_succeeded"] == 1
    await lease.release()
    await cache.aclose()


async def test_production_and_candidate_share_entries_without_key_collision(
    manifest, settings, candidate_dir
):
    cache = make_cache(
        manifest, settings, candidate_dir, budget_bytes=MODEL_BYTES * 3
    )
    production = await cache.acquire("alpha")
    candidate = await cache.acquire_candidate("alpha", CANDIDATE_VERSION)

    snapshot = cache.snapshot()
    assert len(snapshot["models"]) == 2
    versions = {
        (entry["model_id"], entry["is_candidate"]) for entry in snapshot["models"]
    }
    assert versions == {("alpha", False), ("alpha", True)}
    assert production.model.version != candidate.model.version

    await production.release()
    await candidate.release()
    await cache.aclose()


async def test_budget_invariant_under_mixed_interleaving(manifest, settings, candidate_dir):
    cache = make_cache(
        manifest,
        settings,
        candidate_dir,
        budget_bytes=MODEL_BYTES * 2,
        load_delay_seconds=0.01,
    )
    leases = []
    for _ in range(4):
        results = await asyncio.gather(
            cache.acquire("alpha"),
            cache.acquire_candidate("beta", CANDIDATE_VERSION),
            cache.acquire("gamma"),
            cache.acquire_candidate("alpha", CANDIDATE_VERSION),
            return_exceptions=True,
        )
        snapshot = cache.snapshot()
        assert snapshot["used_bytes"] <= snapshot["budget_bytes"]
        assert (
            snapshot["resident_bytes"] + snapshot["reserved_bytes"]
            <= snapshot["budget_bytes"]
        )
        for result in results:
            if not isinstance(result, Exception):
                leases.append(result)

    for lease in leases:
        await lease.release()
    await cache.aclose()
