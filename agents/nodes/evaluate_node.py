#!/usr/bin/env python3
"""
Evaluate node -- computes structured metrics for the pipeline run.

Produces a ``PipelineEvaluation`` dict that summarises blocker resolution,
efficiency, model quality, ground-truth comparison, and KB contribution.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import onnx

from agents.config import PipelineConfig
from agents.state import PipelineState


def _compute_gt_similarity(
    model_bytes: bytes,
    gt_path: str,
) -> Dict[str, Optional[float]]:
    """
    Compare the modified model to the ground-truth model.

    Returns Jaccard similarity of op-type multisets and an op-match rate.
    """
    try:
        modified = onnx.load_from_string(model_bytes)
        gt = onnx.load(gt_path)

        mod_ops = [n.op_type for n in modified.graph.node]
        gt_ops = [n.op_type for n in gt.graph.node]

        mod_set = set(mod_ops)
        gt_set = set(gt_ops)

        if not mod_set and not gt_set:
            return {"gt_similarity": 1.0, "gt_op_match_rate": 1.0}

        jaccard = len(mod_set & gt_set) / len(mod_set | gt_set) if (mod_set | gt_set) else 0.0

        # Op-count match rate (how many GT op counts are matched)
        from collections import Counter

        mod_counts = Counter(mod_ops)
        gt_counts = Counter(gt_ops)
        matched = sum(min(mod_counts[k], gt_counts[k]) for k in gt_counts)
        total_gt = sum(gt_counts.values())
        match_rate = matched / total_gt if total_gt else 0.0

        return {"gt_similarity": round(jaccard, 4), "gt_op_match_rate": round(match_rate, 4)}
    except Exception:
        return {"gt_similarity": None, "gt_op_match_rate": None}


def make_evaluate_node(
    config: PipelineConfig,
) -> Callable[[PipelineState], Dict[str, Any]]:
    """Return a LangGraph node function for pipeline evaluation."""

    def evaluate_node(state: PipelineState) -> Dict[str, Any]:
        start = time.time()

        diagnosis = state.get("diagnosis", {})
        remaining = state.get("remaining_blockers", [])
        compilation = state.get("compilation_report", {})
        history = state.get("surgery_history", [])
        model_bytes = state.get("current_model_bytes", b"")
        gt_path = state.get("ground_truth_path")
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)
        kb_added = state.get("kb_records_added", 0)
        phase_times = dict(state.get("phase_times", {}))

        blockers_before = len(diagnosis.get("blockers", []))
        blockers_after = len(remaining)

        resolution_rate = (
            (blockers_before - blockers_after) / blockers_before
            if blockers_before > 0
            else 1.0
        )

        # Count LLM calls (one per phase per iteration, plus diagnosis + plan)
        llm_calls = 2  # diagnosis + plan
        for entry in history:
            llm_calls += len(entry.get("suggestions", []))

        # ONNX checker validation
        model_valid = False
        if model_bytes:
            try:
                m = onnx.load_from_string(model_bytes)
                onnx.checker.check_model(m)
                model_valid = True
            except Exception:
                model_valid = False

        compilation_passes = compilation.get("will_compile", False)

        # Ground-truth comparison
        gt_metrics: Dict[str, Optional[float]] = {
            "gt_similarity": None,
            "gt_op_match_rate": None,
        }
        if gt_path and model_bytes:
            gt_metrics = _compute_gt_similarity(model_bytes, gt_path)

        total_time = sum(phase_times.values())

        evaluation: Dict[str, Any] = {
            "blockers_before": blockers_before,
            "blockers_after": blockers_after,
            "blocker_resolution_rate": round(resolution_rate, 4),
            "iterations_used": iteration,
            "max_iterations": max_iter,
            "llm_calls": llm_calls,
            "total_time_seconds": round(total_time, 2),
            "model_valid": model_valid,
            "compilation_passes": compilation_passes,
            "gt_similarity": gt_metrics.get("gt_similarity"),
            "gt_op_match_rate": gt_metrics.get("gt_op_match_rate"),
            "new_patterns_learned": 0,
            "kb_records_added": kb_added,
        }

        elapsed = time.time() - start
        phase_times["evaluation"] = elapsed

        if config.verbose:
            print(f"  [evaluate] resolution={resolution_rate:.1%}  "
                  f"valid={model_valid}  compiles={compilation_passes}")

        return {
            "evaluation": evaluation,
            "phase_times": phase_times,
        }

    return evaluate_node
