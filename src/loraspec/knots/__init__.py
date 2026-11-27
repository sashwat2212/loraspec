"""Factorized memory-safe KnOTS joint-SVD subspace aligner, operators, and PEFT serializer."""

from __future__ import annotations

from loraspec.knots.aligner import (
    FactorizedKnOTSResult,
    LayerPolicy,
    align_subspaces_knots,
    harmonize_heterogeneous_ranks,
)
from loraspec.knots.operators import (
    KnotsMethod,
    apply_genknowsub,
    knots_dare_ties,
    knots_linear,
    knots_ties,
    merge_aligned_factors,
)
from loraspec.knots.serializer import (
    compute_merge_diagnostics,
    export_peft_merged_adapter,
    factorized_frobenius_rmse,
    load_verified_knots_summary,
    merge_knots_adapters,
    reconstruct_merged_lora,
)

__all__ = [
    "FactorizedKnOTSResult",
    "KnotsMethod",
    "LayerPolicy",
    "align_subspaces_knots",
    "apply_genknowsub",
    "compute_merge_diagnostics",
    "export_peft_merged_adapter",
    "factorized_frobenius_rmse",
    "harmonize_heterogeneous_ranks",
    "knots_dare_ties",
    "knots_linear",
    "knots_ties",
    "load_verified_knots_summary",
    "merge_aligned_factors",
    "merge_knots_adapters",
    "reconstruct_merged_lora",
]
