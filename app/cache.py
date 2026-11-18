"""Budget-aware, single-flight model cache with production/candidate tiers.

Concurrency rules implemented here:

* The sum of resident weights and outstanding load reservations never exceeds
  ``budget_bytes``.  Candidate comparison weights are counted in the *same*
  budget as production weights.
* A cold model is loaded exactly once no matter how many requests arrive while
  the load is in progress.  The loader is an independent task, so a request
  that disconnects while waiting does not cancel the shared load.  This holds
  for both production and candidate loads.
* Production requests reclaim space from *idle* models only.  Idle candidate
  versions are evicted before idle production models, so a candidate that has
  finished its comparison always yields to formal inference.  If even the
  whole idle set is too small, the request fails fast with
  :class:`ModelBusyError` instead of evicting a busy model.
* Candidate comparison requests never evict anything — they may only use
  currently free budget.  When the budget cannot hold the candidate alongside
  the resident production models and existing reservations, the comparison is
  rejected with :class:`ModelBusyError` and the production cache is untouched.
* A lease is released only when its holder actually finishes computing, which
  is what keeps a disconnected-but-still-computing request from losing its
  occupancy early — for both versions involved in a comparison.
* A failed load removes the entry and its reservation, leaving other models
  untouched and the artifact eligible for a later retry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .candidates import CandidateCatalog, CandidateDescriptor, read_candidate_file
from .errors import (
    ModelBusyError,
    ModelCacheError,
    ModelLoadError,
    ModelTooLargeError,
    UnknownCandidateError,
    UnknownModelError,
)
from .models import LinearModel, Manifest, ModelDescriptor, read_model_file

logger = logging.getLogger(__name__)


class EntryState(StrEnum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


def candidate_cache_key(model_id: str, version: str) -> str:
    """Cache-key namespace keeping candidate entries apart from production ones."""

    return f"candidate:{model_id}:{version}"


@dataclass
class _Entry:
    descriptor: ModelDescriptor | CandidateDescriptor
    state: EntryState
    reserved_bytes: int
    event: asyncio.Event = field(default_factory=asyncio.Event)
    model: LinearModel | None = None
    error: ModelCacheError | None = None
    inflight: int = 0
    waiters: int = 0
    last_used: float = field(default_factory=time.monotonic)
    loaded_at: float | None = None
    is_candidate: bool = False
    candidate_version: str | None = None

    @property
    def model_id(self) -> str:
        return self.descriptor.model_id

    @property
    def weights_bytes(self) -> int:
        return self.model.weights_bytes if self.model is not None else self.reserved_bytes


@dataclass
class CacheCounters:
    hits: int = 0
    cold_requests: int = 0
    load_attempts: int = 0
    load_succeeded: int = 0
    load_failed: int = 0
    evictions: int = 0
    busy_rejections: int = 0
    waiters_joined: int = 0
    candidate_load_attempts: int = 0
    candidate_load_succeeded: int = 0
    candidate_load_failed: int = 0
    candidate_evictions: int = 0
    candidate_busy_rejections: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "cold_requests": self.cold_requests,
            "load_attempts": self.load_attempts,
            "load_succeeded": self.load_succeeded,
            "load_failed": self.load_failed,
            "evictions": self.evictions,
            "busy_rejections": self.busy_rejections,
            "waiters_joined": self.waiters_joined,
            "candidate_load_attempts": self.candidate_load_attempts,
            "candidate_load_succeeded": self.candidate_load_succeeded,
            "candidate_load_failed": self.candidate_load_failed,
            "candidate_evictions": self.candidate_evictions,
            "candidate_busy_rejections": self.candidate_busy_rejections,
        }


class ModelLease:
    """Exclusive-use marker for one in-flight inference or comparison leg.

    The cache cannot evict an entry while at least one lease exists.  Releasing
    is idempotent.
    """

    __slots__ = ("_cache", "_entry", "_released")

    def __init__(self, cache: ModelCache, entry: _Entry) -> None:
        self._cache = cache
        self._entry = entry
        self._released = False

    @property
    def model(self) -> LinearModel:
        model = self._entry.model
        if model is None:  # pragma: no cover - defensive
            raise RuntimeError("lease has no loaded model")
        return model

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._cache._release(self._entry)


class ModelCache:
    def __init__(
        self,
        manifest: Manifest,
        budget_bytes: int,
        load_delay_seconds: float = 0.0,
        *,
        reader: Callable[[ModelDescriptor, Path], LinearModel] = read_model_file,
        candidate_catalog_provider: Callable[[], CandidateCatalog] | None = None,
        candidate_reader: Callable[
            [CandidateDescriptor, Path], LinearModel
        ] = read_candidate_file,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self._manifest = manifest
        self._budget_bytes = budget_bytes
        self._load_delay_seconds = max(0.0, load_delay_seconds)
        self._reader = reader
        self._candidate_catalog_provider = candidate_catalog_provider
        self._candidate_reader = candidate_reader
        self._lock = asyncio.Lock()
        # Keyed by production model id, or by candidate_cache_key(...) for
        # candidate versions.
        self._entries: dict[str, _Entry] = {}
        self._resident_bytes = 0
        self._reserved_bytes = 0
        self._load_tasks: dict[str, asyncio.Task[None]] = {}
        self._counters = CacheCounters()
        self._last_eviction: str | None = None

    # ------------------------------------------------------------------
    # Public read-only views
    # ------------------------------------------------------------------
    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def used_bytes(self) -> int:
        return self._resident_bytes + self._reserved_bytes

    def descriptor(self, model_id: str) -> ModelDescriptor:
        descriptor = self._manifest.descriptors.get(model_id)
        if descriptor is None:
            raise UnknownModelError(model_id)
        return descriptor

    def list_models(self) -> list[dict[str, object]]:
        return [
            {
                "model_id": descriptor.model_id,
                "name": descriptor.name,
                "device_type": descriptor.device_type,
                "version": descriptor.version,
                "weights_bytes": descriptor.declared_bytes,
            }
            for descriptor in self._manifest.descriptors.values()
        ]

    def snapshot(self) -> dict[str, object]:
        """Consistent-enough view for tests and operations dashboards."""

        models = []
        for entry in self._entries.values():
            models.append(
                {
                    "model_id": entry.model_id,
                    "state": entry.state.value,
                    "weights_bytes": entry.weights_bytes,
                    "inflight": entry.inflight,
                    "waiters": entry.waiters,
                    "last_used": entry.last_used,
                    "loaded_at": entry.loaded_at,
                    "error": entry.error.code if entry.error else None,
                    "is_candidate": entry.is_candidate,
                    "candidate_version": entry.candidate_version,
                }
            )
        models.sort(key=lambda item: (not item["is_candidate"], str(item["model_id"])))
        return {
            "budget_bytes": self._budget_bytes,
            "resident_bytes": self._resident_bytes,
            "reserved_bytes": self._reserved_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self._budget_bytes - self.used_bytes,
            "models": models,
            "counters": self._counters.as_dict(),
            "last_eviction": self._last_eviction,
        }

    # ------------------------------------------------------------------
    # Production acquisition
    # ------------------------------------------------------------------
    async def acquire(self, model_id: str) -> ModelLease:
        """Return a lease once the production model is resident.

        Concurrent cold callers all wait on one loader task.  Waiters count is
        tracked for diagnostics only; a cancelled waiter never touches the
        loader, so the remaining waiters still receive the model.
        """

        descriptor = self.descriptor(model_id)

        async with self._lock:
            entry = self._entries.get(model_id)
            if entry is not None and entry.state is EntryState.READY:
                self._counters.hits += 1
                return self._lease_locked(entry)

            if entry is None or entry.state is EntryState.FAILED:
                if entry is not None:
                    # A previous load failed and its entry is being torn down;
                    # this request starts a fresh attempt.
                    self._entries.pop(model_id, None)
                entry = self._start_load_locked(descriptor, model_id)
            else:
                self._counters.waiters_joined += 1
            entry.waiters += 1
            event = entry.event

        try:
            await event.wait()
        except asyncio.CancelledError:
            await self._drop_waiter(entry)
            raise

        try:
            async with self._lock:
                if entry.error is not None:
                    raise entry.error
                if entry.state is not EntryState.READY or entry.model is None:
                    raise ModelLoadError(model_id, "model is not available")
                self._counters.hits += 1
                return self._lease_locked(entry)
        finally:
            await self._drop_waiter(entry)

    # ------------------------------------------------------------------
    # Candidate acquisition (comparison endpoint only)
    # ------------------------------------------------------------------
    async def acquire_candidate(self, model_id: str, version: str) -> ModelLease:
        """Return a lease for a resident candidate weight version.

        The candidate catalog is re-read for every call, so newly published
        candidate artifacts become visible without a restart.  Admission uses
        only the budget that is currently free: this path never evicts a
        production model (nor any other candidate), and raises
        :class:`ModelBusyError` when the candidate cannot fit.
        """

        descriptor = self._resolve_candidate_descriptor(model_id, version)
        key = candidate_cache_key(model_id, version)

        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.state is EntryState.READY:
                self._counters.hits += 1
                return self._lease_locked(entry)

            if entry is None or entry.state is EntryState.FAILED:
                if entry is not None:
                    self._entries.pop(key, None)
                entry = self._start_candidate_load_locked(descriptor, key)
            else:
                self._counters.waiters_joined += 1
            entry.waiters += 1
            event = entry.event

        try:
            await event.wait()
        except asyncio.CancelledError:
            await self._drop_waiter(entry)
            raise

        try:
            async with self._lock:
                if entry.error is not None:
                    raise entry.error
                if entry.state is not EntryState.READY or entry.model is None:
                    raise ModelLoadError(model_id, "candidate model is not available")
                self._counters.hits += 1
                return self._lease_locked(entry)
        finally:
            await self._drop_waiter(entry)

    def _resolve_candidate_descriptor(
        self, model_id: str, version: str
    ) -> CandidateDescriptor:
        if self._candidate_catalog_provider is None:  # pragma: no cover - defensive
            raise UnknownCandidateError(model_id, version)
        try:
            catalog = self._candidate_catalog_provider()
        except UnknownCandidateError:
            raise
        except ValueError as exc:
            # A broken candidate manifest is an operator-visible load failure,
            # not an unknown-version 404.
            raise ModelLoadError(model_id, f"invalid candidate manifest: {exc}") from exc
        return catalog.descriptor(model_id, version)

    async def _drop_waiter(self, entry: _Entry) -> None:
        async with self._lock:
            entry.waiters = max(0, entry.waiters - 1)

    def _lease_locked(self, entry: _Entry) -> ModelLease:
        entry.inflight += 1
        entry.last_used = time.monotonic()
        return ModelLease(self, entry)

    async def _release(self, entry: _Entry) -> None:
        async with self._lock:
            if entry.inflight > 0:
                entry.inflight -= 1
            entry.last_used = time.monotonic()

    # ------------------------------------------------------------------
    # Load startup: production (may evict idle entries) vs candidate (may not)
    # ------------------------------------------------------------------
    def _start_load_locked(
        self, descriptor: ModelDescriptor, cache_key: str
    ) -> _Entry:
        required = descriptor.declared_bytes
        if required > self._budget_bytes:
            raise ModelTooLargeError(
                descriptor.model_id,
                budget_bytes=self._budget_bytes,
                required_bytes=required,
            )

        self._counters.cold_requests += 1
        self._make_room_locked(required, descriptor.model_id)

        entry = _Entry(
            descriptor=descriptor,
            state=EntryState.LOADING,
            reserved_bytes=required,
        )
        self._entries[cache_key] = entry
        self._reserved_bytes += required
        self._counters.load_attempts += 1

        task = asyncio.create_task(self._finish_load(entry, cache_key))
        self._register_load_task(cache_key, task)
        return entry

    def _start_candidate_load_locked(
        self, descriptor: CandidateDescriptor, cache_key: str
    ) -> _Entry:
        required = descriptor.declared_bytes
        if required > self._budget_bytes:
            raise ModelTooLargeError(
                descriptor.model_id,
                budget_bytes=self._budget_bytes,
                required_bytes=required,
            )

        # Comparison admission is strict: only currently free budget may be
        # used.  Nothing is evicted on behalf of a candidate, so a rejected
        # comparison leaves every resident production model untouched.
        free = self._budget_bytes - self.used_bytes
        if free < required:
            self._counters.candidate_busy_rejections += 1
            raise ModelBusyError(
                descriptor.model_id,
                budget_bytes=self._budget_bytes,
                used_bytes=self.used_bytes,
                required_bytes=required,
            )

        entry = _Entry(
            descriptor=descriptor,
            state=EntryState.LOADING,
            reserved_bytes=required,
            is_candidate=True,
            candidate_version=descriptor.version,
        )
        self._entries[cache_key] = entry
        self._reserved_bytes += required
        self._counters.candidate_load_attempts += 1

        task = asyncio.create_task(self._finish_candidate_load(entry, cache_key))
        self._register_load_task(cache_key, task)
        return entry

    def _register_load_task(self, cache_key: str, task: asyncio.Task[None]) -> None:
        # The task is deliberately not stored per request: it belongs to the
        # cache and keeps running even if every current waiter disconnects.
        self._load_tasks[cache_key] = task

        def _forget(finished: asyncio.Task[None]) -> None:
            # Only drop our own task: a retry may already have registered a
            # newer loader for the same cache key.
            if self._load_tasks.get(cache_key) is finished:
                self._load_tasks.pop(cache_key, None)

        task.add_done_callback(_forget)

    def _make_room_locked(self, required: int, requested_model_id: str) -> None:
        free = self._budget_bytes - self.used_bytes
        if free >= required:
            return

        needed = required - free

        def _idle(entry: _Entry) -> bool:
            return (
                entry.state is EntryState.READY
                and entry.inflight == 0
                and entry.waiters == 0
            )

        idle_candidates = sorted(
            (entry for entry in self._entries.values() if _idle(entry) and entry.is_candidate),
            key=lambda entry: (entry.last_used, entry.model_id),
        )
        idle_production = sorted(
            (entry for entry in self._entries.values() if _idle(entry) and not entry.is_candidate),
            key=lambda entry: (entry.last_used, entry.model_id),
        )
        # An idle candidate always yields to formal inference first; among the
        # same tier, least-recently-used goes first.
        candidates = idle_candidates + idle_production
        idle_bytes = sum(entry.weights_bytes for entry in candidates)

        # Decide before evicting anything: a request that will be rejected must
        # not disturb the models that are already resident and idle.
        if idle_bytes < needed:
            self._counters.busy_rejections += 1
            raise ModelBusyError(
                requested_model_id,
                budget_bytes=self._budget_bytes,
                used_bytes=self.used_bytes,
                required_bytes=required,
            )

        for candidate in candidates:
            if needed <= 0:
                break
            self._evict_locked(candidate)
            needed -= candidate.weights_bytes

    def _evict_locked(self, entry: _Entry) -> None:
        model_id = entry.model_id
        cache_key = (
            candidate_cache_key(model_id, entry.candidate_version)
            if entry.is_candidate
            else model_id
        )
        if self._entries.get(cache_key) is entry:
            del self._entries[cache_key]
        self._resident_bytes -= entry.weights_bytes
        self._resident_bytes = max(0, self._resident_bytes)
        if entry.is_candidate:
            self._counters.candidate_evictions += 1
            logger.info(
                "evicted idle candidate %s@%s to free %s bytes for formal inference",
                model_id,
                entry.candidate_version,
                entry.weights_bytes,
            )
        else:
            self._counters.evictions += 1
            logger.info(
                "evicted idle model %s to free %s bytes", model_id, entry.weights_bytes
            )
        self._last_eviction = (
            f"{model_id}@{entry.candidate_version}" if entry.is_candidate else model_id
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    async def _finish_load(self, entry: _Entry, cache_key: str) -> None:
        model_id = entry.model_id
        try:
            if self._load_delay_seconds > 0:
                await asyncio.sleep(self._load_delay_seconds)
            model = await asyncio.to_thread(
                self._reader, entry.descriptor, self._manifest.model_dir
            )
            if model.model_id != model_id:
                raise ModelLoadError(model_id, "reader returned a different model")
            if model.weights_bytes > entry.reserved_bytes:
                # The reservation is what was promised to the budget.  A larger
                # result cannot be admitted without risking oversubscription.
                raise ModelLoadError(model_id, "loaded weights exceed the reserved size")
        except asyncio.CancelledError:
            await self._fail_load(entry, cache_key, ModelLoadError(model_id, "load cancelled"))
            raise
        except ModelCacheError as exc:
            await self._fail_load(entry, cache_key, exc)
            return
        except Exception as exc:  # noqa: BLE001 - convert to a domain error
            await self._fail_load(entry, cache_key, ModelLoadError(model_id, str(exc)))
            return

        await self._admit_loaded_model(entry, cache_key, model)

    async def _finish_candidate_load(self, entry: _Entry, cache_key: str) -> None:
        model_id = entry.model_id
        version = entry.candidate_version
        try:
            if self._load_delay_seconds > 0:
                await asyncio.sleep(self._load_delay_seconds)
            catalog = self._candidate_catalog_provider()
            # Re-resolve through the fresh catalog: a republished manifest may
            # have changed the file behind this version while we were waiting.
            descriptor = catalog.descriptor(model_id, version or "")
            candidate_dir = catalog.candidate_dir
            model = await asyncio.to_thread(
                self._candidate_reader, descriptor, candidate_dir
            )
            if model.model_id != model_id:
                raise ModelLoadError(model_id, "candidate reader returned a different model")
            if model.version != version:
                raise ModelLoadError(model_id, "candidate version changed during load")
            if model.weights_bytes > entry.reserved_bytes:
                raise ModelLoadError(
                    model_id, "candidate weights exceed the reserved size"
                )
        except asyncio.CancelledError:
            await self._fail_load(
                entry, cache_key, ModelLoadError(model_id, "candidate load cancelled")
            )
            raise
        except ModelCacheError as exc:
            await self._fail_load(entry, cache_key, exc)
            return
        except Exception as exc:  # noqa: BLE001 - convert to a domain error
            await self._fail_load(entry, cache_key, ModelLoadError(model_id, str(exc)))
            return

        await self._admit_loaded_model(entry, cache_key, model)

    async def _admit_loaded_model(
        self, entry: _Entry, cache_key: str, model: LinearModel
    ) -> None:
        model_id = entry.model_id
        async with self._lock:
            # The entry can only be replaced if it was evicted, which is
            # impossible while it is loading, but stay defensive.
            if self._entries.get(cache_key) is not entry:
                self._reserved_bytes -= entry.reserved_bytes
                self._reserved_bytes = max(0, self._reserved_bytes)
                entry.error = ModelLoadError(model_id, "load was superseded")
                entry.event.set()
                return

            self._reserved_bytes -= entry.reserved_bytes
            self._reserved_bytes = max(0, self._reserved_bytes)
            self._resident_bytes += model.weights_bytes
            entry.model = model
            entry.state = EntryState.READY
            entry.loaded_at = time.monotonic()
            entry.last_used = entry.loaded_at
            if entry.is_candidate:
                self._counters.candidate_load_succeeded += 1
            else:
                self._counters.load_succeeded += 1
            entry.event.set()

    async def _fail_load(
        self, entry: _Entry, cache_key: str, error: ModelCacheError
    ) -> None:
        model_id = entry.model_id
        async with self._lock:
            if self._entries.get(cache_key) is entry:
                del self._entries[cache_key]
            if entry.state is EntryState.LOADING:
                self._reserved_bytes -= entry.reserved_bytes
                self._reserved_bytes = max(0, self._reserved_bytes)
            entry.error = error
            entry.state = EntryState.FAILED  # wake all waiters; error is checked first
            if entry.is_candidate:
                self._counters.candidate_load_failed += 1
            else:
                self._counters.load_failed += 1
            entry.event.set()
        tier = "candidate" if entry.is_candidate else "model"
        logger.warning("failed to load %s %s: %s", tier, model_id, error)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        tasks = list(self._load_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            self._load_tasks.clear()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ModelCache(models={len(self._manifest.descriptors)}, "
            f"used={self.used_bytes}/{self._budget_bytes})"
        )
