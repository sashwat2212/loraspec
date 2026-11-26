"""Unit tests for atomic GpuBudgetGuard & --smoke circuit breaker."""

from __future__ import annotations

import asyncio

import pytest

from loraspec.core.budget import GpuBudgetGuard
from loraspec.core.errors import EXIT_BUDGET_EXCEEDED, BudgetExceededError, ConfigError


async def test_concurrent_reservations_respect_cap() -> None:
    # A100 @ 223.02/hr: 3600s = Rs 223.02. Cap = Rs 500 -> at most 2 concurrent 1-hr jobs
    guard = GpuBudgetGuard(max_cost_inr=500.0)

    async def try_reserve() -> bool:
        try:
            res = await guard.reserve_or_raise("e2e_a100_80gb", 3600.0)
            await asyncio.sleep(0.01)
            await guard.reconcile(res, 1800.0)  # actual 0.5 hr = Rs 111.51
            return True
        except BudgetExceededError:
            return False

    results = await asyncio.gather(*(try_reserve() for _ in range(5)))
    assert sum(1 for r in results if r) <= 2
    assert guard.spent_cost_inr <= 500.0


async def test_smoke_cap_and_local_zero_cost() -> None:
    smoke_guard = GpuBudgetGuard(max_cost_inr=2500.0, smoke=True, smoke_cap_inr=10.0)
    assert smoke_guard.effective_cap_inr == 10.0

    # Local CPU/MPS is always Rs 0.0
    cpu_res = await smoke_guard.reserve_or_raise("local_cpu", 7200.0)
    await smoke_guard.reconcile(cpu_res, 3600.0)
    assert smoke_guard.spent_cost_inr == 0.0

    # 30 mins of L4 = 28.91 INR > Rs 10 smoke cap -> must raise BudgetExceededError (exit 3)
    with pytest.raises(BudgetExceededError) as exc_info:
        await smoke_guard.reserve_or_raise("e2e_l4_24gb", 1800.0)
    assert exc_info.value.exit_code == EXIT_BUDGET_EXCEEDED

    # Short 60s L4 smoke run = 0.96367 INR <= Rs 10 -> succeeds and reconciles cleanly
    l4_res = await smoke_guard.reserve_or_raise("e2e_l4_24gb", 60.0)
    actual = await smoke_guard.reconcile(l4_res, 30.0)
    assert actual == pytest.approx(57.82 * (30.0 / 3600.0), rel=1e-5)
    assert smoke_guard.total_gpu_hours > 1.0  # includes 1h local_cpu + 30s L4


async def test_release_and_validation_errors() -> None:
    guard = GpuBudgetGuard(max_cost_inr=100.0)
    res = await guard.reserve_or_raise("e2e_l4_24gb", 3600.0)
    await guard.release(res)
    assert guard.spent_cost_inr == 0.0

    # Can reserve again after release
    res2 = await guard.reserve_or_raise("e2e_l4_24gb", 3600.0)
    await guard.reconcile(res2, 3600.0)
    assert guard.spent_cost_inr == pytest.approx(57.82, rel=1e-5)

    with pytest.raises(ConfigError):
        await guard.reserve_or_raise("unknown_gpu", 10.0)
    with pytest.raises(ConfigError):
        await guard.reserve_or_raise("local_cpu", -5.0)
    with pytest.raises(ConfigError):
        await guard.reconcile(res2, 10.0)  # Double reconcile rejected
    with pytest.raises(ConfigError):
        GpuBudgetGuard(max_cost_inr=float("nan"))
