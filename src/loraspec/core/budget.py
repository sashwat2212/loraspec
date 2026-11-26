"""Atomic GPU spend & runtime circuit breaker (`GpuBudgetGuard`)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from loraspec.core.errors import BudgetExceededError, ConfigError

GpuHardwareProfile = Literal[
    "local_cpu",
    "local_mps",
    "demo_replay",
    "e2e_l4_24gb",
    "e2e_a100_80gb",
]

DEFAULT_RATE_TABLE_INR_PER_HR: dict[str, float] = {
    "local_cpu": 0.0,
    "local_mps": 0.0,
    "demo_replay": 0.0,
    "e2e_l4_24gb": 57.82,
    "e2e_a100_80gb": 223.02,
}


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: int
    hardware: str
    reserved_seconds: float
    reserved_cost_inr: float


class GpuBudgetGuard:
    """Thread/async-safe INR & GPU-time cap enforcer with --smoke support."""

    def __init__(
        self,
        *,
        max_cost_inr: float = 2500.0,
        smoke: bool = False,
        smoke_cap_inr: float = 10.0,
        rate_table_inr_per_hr: dict[str, float] | None = None,
    ) -> None:
        if max_cost_inr < 0.0:
            raise ConfigError("max_cost_inr must be non-negative.")
        if smoke and smoke_cap_inr <= 0.0:
            raise ConfigError("smoke_cap_inr must be positive.")

        self.effective_cap_inr: float = min(max_cost_inr, smoke_cap_inr) if smoke else max_cost_inr
        self.smoke: bool = smoke
        self._rates: dict[str, float] = dict(DEFAULT_RATE_TABLE_INR_PER_HR)
        if rate_table_inr_per_hr is not None:
            self._rates.update(rate_table_inr_per_hr)

        self._lock = asyncio.Lock()
        self._committed_cost_inr: float = 0.0
        self._reserved_cost_inr: float = 0.0
        self._committed_seconds: float = 0.0
        self._next_id: int = 1
        self._active: dict[int, BudgetReservation] = {}

    @property
    def spent_cost_inr(self) -> float:
        return round(self._committed_cost_inr, 6)

    @property
    def total_gpu_hours(self) -> float:
        return round(self._committed_seconds / 3600.0, 6)

    def hourly_rate(self, hardware: str) -> float:
        if hardware not in self._rates:
            raise ConfigError(
                f"Unknown hardware profile '{hardware}'. Known profiles: {sorted(self._rates)}."
            )
        return self._rates[hardware]

    async def reserve_or_raise(
        self,
        hardware: str,
        estimated_seconds: float,
    ) -> BudgetReservation:
        """Atomically check projected GPU spend against cap before launching a GPU step."""
        if estimated_seconds < 0.0:
            raise ConfigError("estimated_seconds must be non-negative.")

        rate = self.hourly_rate(hardware)
        projected = rate * (estimated_seconds / 3600.0)

        async with self._lock:
            total_if_committed = self._committed_cost_inr + self._reserved_cost_inr + projected
            if total_if_committed > self.effective_cap_inr + 1e-9:
                raise BudgetExceededError(
                    f"GPU budget ceiling reached: projected ₹{total_if_committed:.2f} "
                    f"> cap ₹{self.effective_cap_inr:.2f} ({hardware})."
                )
            res_id = self._next_id
            self._next_id += 1
            res = BudgetReservation(
                reservation_id=res_id,
                hardware=hardware,
                reserved_seconds=estimated_seconds,
                reserved_cost_inr=projected,
            )
            self._reserved_cost_inr += projected
            self._active[res_id] = res
            return res

    async def reconcile(self, reservation: BudgetReservation, actual_seconds: float) -> float:
        """Release reservation and record exact actual elapsed GPU cost."""
        if actual_seconds < 0.0:
            raise ConfigError("actual_seconds must be non-negative.")

        rate = self.hourly_rate(reservation.hardware)
        actual_cost = rate * (actual_seconds / 3600.0)

        async with self._lock:
            active = self._active.pop(reservation.reservation_id, None)
            if active is not None:
                self._reserved_cost_inr = max(
                    0.0, self._reserved_cost_inr - active.reserved_cost_inr
                )
            self._committed_cost_inr += actual_cost
            self._committed_seconds += actual_seconds
            return actual_cost

    async def release(self, reservation: BudgetReservation) -> None:
        """Release an unused reservation cleanly on error or cancellation."""
        async with self._lock:
            active = self._active.pop(reservation.reservation_id, None)
            if active is not None:
                self._reserved_cost_inr = max(
                    0.0, self._reserved_cost_inr - active.reserved_cost_inr
                )
