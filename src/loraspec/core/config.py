"""Strict `loraspec.toml` configuration loader using Pydantic v2 (`extra="forbid"`)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from loraspec.core.errors import ConfigError
from loraspec.core.schemas import Severity


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProjectConfig(_StrictModel):
    fail_on: Severity = Severity.HIGH
    default_backend: Literal["auto", "triton_cuda", "torch_mps", "torch_cpu", "pure_python"] = (
        "auto"
    )


class AuditConfig(_StrictModel):
    rank_collapse_ratio: float = Field(default=0.20, gt=0.0, le=1.0)
    top1_energy_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    spectral_spike_multiplier: float = Field(default=5.0, gt=1.0)
    collision_high_chordal: float = Field(default=0.45, ge=0.0, le=1.0)
    collision_crit_chordal: float = Field(default=0.70, ge=0.0, le=1.0)
    sign_conflict_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    require_all_linear: bool = True

    @model_validator(mode="after")
    def _validate_collision_thresholds(self) -> AuditConfig:
        if self.collision_crit_chordal < self.collision_high_chordal:
            raise ValueError(
                "collision_crit_chordal must be >= collision_high_chordal "
                f"({self.collision_crit_chordal} < {self.collision_high_chordal})."
            )
        return self


class KnotsConfig(_StrictModel):
    default_method: Literal["knots_ties", "knots_dare_ties", "knots_linear"] = "knots_ties"
    ties_density: float = Field(default=0.50, gt=0.0, le=1.0)
    dare_drop_rate: float = Field(default=0.20, ge=0.0, lt=1.0)
    min_energy_retention: float = Field(default=0.90, gt=0.0, le=1.0)
    rng_seed: int = 42


class MoloraConfig(_StrictModel):
    top_k: int = Field(default=2, ge=1, le=16)
    router_temperature: float = Field(default=1.0, gt=0.0)
    use_gks: bool = False


class BudgetConfig(_StrictModel):
    max_cost_inr: float = Field(default=2500.0, ge=0.0, le=3000.0)
    smoke_cap_inr: float = Field(default=10.0, gt=0.0, le=50.0)
    l4_rate_inr_per_hr: float = Field(default=57.82, ge=0.0)
    a100_rate_inr_per_hr: float = Field(default=223.02, ge=0.0)


class StudioConfig(_StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    demo_replay: bool = True


class LoraspecConfig(_StrictModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    knots: KnotsConfig = Field(default_factory=KnotsConfig)
    molora: MoloraConfig = Field(default_factory=MoloraConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    studio: StudioConfig = Field(default_factory=StudioConfig)


def load_config(path: Path | None = None) -> LoraspecConfig:
    """Load `loraspec.toml` from an explicit path or current directory, or return defaults."""
    target = path
    if target is None:
        candidate = Path.cwd() / "loraspec.toml"
        if candidate.is_file():
            target = candidate

    if target is None:
        return LoraspecConfig()

    if not target.is_file():
        raise ConfigError(f"Configuration file not found: {target}")

    try:
        raw_data = tomllib.loads(target.read_text(encoding="utf-8"))
        return LoraspecConfig.model_validate(raw_data)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Malformed TOML in {target}: {exc}") from exc
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {target}: {exc}") from exc
