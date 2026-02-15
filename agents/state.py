#!/usr/bin/env python3
"""
Pipeline state definition for the LangGraph-based surgery pipeline.

Defines the shared PipelineState TypedDict that flows through all graph nodes.
Each node receives the full state and returns a partial dict of updates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    """
    Shared state for the LangGraph surgery pipeline.

    Every node receives this state and returns a partial update dict.
    Fields marked ``total=False`` are optional so nodes only need to
    set the keys they actually modify.
    """

    # ------------------------------------------------------------------
    # Inputs (set once at pipeline start)
    # ------------------------------------------------------------------
    model_path: str
    ground_truth_path: Optional[str]
    api_key: str
    config: Dict[str, Any]  # serialised PipelineConfig

    # ------------------------------------------------------------------
    # Diagnosis output
    # ------------------------------------------------------------------
    diagnosis: Optional[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Strategy / plan output
    # ------------------------------------------------------------------
    plan: Optional[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Surgery execution loop
    # ------------------------------------------------------------------
    current_model_bytes: Optional[bytes]  # serialised onnx.ModelProto
    iteration: int
    max_iterations: int
    surgery_history: List[Dict[str, Any]]  # [{code, success, error, delta, …}]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    compilation_report: Optional[Dict[str, Any]]
    remaining_blockers: List[str]  # op_type strings still blocking

    # ------------------------------------------------------------------
    # KB enrichment tracking
    # ------------------------------------------------------------------
    transformations_applied: List[Dict[str, Any]]
    kb_records_added: int

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    evaluation: Optional[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    phase_times: Dict[str, float]
