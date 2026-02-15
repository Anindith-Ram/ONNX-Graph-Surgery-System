#!/usr/bin/env python3
"""
Evaluation models for the graph surgery pipeline.

Provides ``PipelineEvaluation`` -- a structured, serialisable record of
how well a single pipeline run performed across multiple dimensions:
compilation, efficiency, model quality, ground-truth comparison, and
knowledge-base contribution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PipelineEvaluation(BaseModel):
    """Structured evaluation metrics for a single pipeline run."""

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------
    blockers_before: int = Field(
        description="Number of compilation blockers detected by the diagnosis agent."
    )
    blockers_after: int = Field(
        description="Number of blockers remaining after all surgery iterations."
    )
    blocker_resolution_rate: float = Field(
        description="Fraction of original blockers resolved (0.0 -- 1.0)."
    )

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------
    iterations_used: int = Field(
        description="Number of execute-validate cycles performed."
    )
    max_iterations: int = Field(
        description="Configured iteration cap."
    )
    llm_calls: int = Field(
        description="Total number of LLM invocations across all nodes."
    )
    total_time_seconds: float = Field(
        description="Wall-clock time for the entire pipeline run."
    )

    # ------------------------------------------------------------------
    # Model quality
    # ------------------------------------------------------------------
    model_valid: bool = Field(
        description="True if the modified model passes onnx.checker.check_model."
    )
    compilation_passes: bool = Field(
        description="True if CompilationSimulator reports zero blockers."
    )

    # ------------------------------------------------------------------
    # Ground-truth comparison (optional -- only when GT is provided)
    # ------------------------------------------------------------------
    gt_similarity: Optional[float] = Field(
        default=None,
        description="Jaccard similarity of op-type sets between modified and GT model.",
    )
    gt_op_match_rate: Optional[float] = Field(
        default=None,
        description="Fraction of GT op-type counts matched in the modified model.",
    )

    # ------------------------------------------------------------------
    # Knowledge contribution
    # ------------------------------------------------------------------
    new_patterns_learned: int = Field(
        default=0,
        description="Number of novel transformation patterns discovered.",
    )
    kb_records_added: int = Field(
        default=0,
        description="Number of records written back to SurgeryDatabase.",
    )

    model_config = {"extra": "allow"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_state(cls, evaluation_dict: Dict[str, Any]) -> "PipelineEvaluation":
        """Construct from the dict stored in ``PipelineState.evaluation``."""
        return cls.model_validate(evaluation_dict)

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        return (
            f"resolution={self.blocker_resolution_rate:.0%}  "
            f"iters={self.iterations_used}/{self.max_iterations}  "
            f"valid={self.model_valid}  compiles={self.compilation_passes}  "
            f"kb+={self.kb_records_added}"
        )
