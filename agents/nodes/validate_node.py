#!/usr/bin/env python3
"""
Validate node -- runs CompilationSimulator on the current model.

After each surgery round, this node checks whether compilation blockers
remain. The result feeds into the ``should_retry`` conditional edge.
"""

from __future__ import annotations

import io
import tempfile
import time
from typing import Any, Callable, Dict, List

import onnx

from agents.config import PipelineConfig
from agents.state import PipelineState


def make_validate_node(
    config: PipelineConfig,
) -> Callable[[PipelineState], Dict[str, Any]]:
    """Return a LangGraph node function for compilation validation."""

    from core_analysis.compilation_simulator import CompilationSimulator

    simulator = CompilationSimulator(verbose=config.verbose)

    def validate_node(state: PipelineState) -> Dict[str, Any]:
        model_bytes: bytes = state["current_model_bytes"]
        iteration: int = state.get("iteration", 0)

        start = time.time()

        # Write model to a temp file so CompilationSimulator.simulate() can
        # load it (it expects a file path).
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name

        report = simulator.simulate(tmp_path)
        elapsed = time.time() - start

        remaining: List[str] = list(report.blocker_ops.keys())

        phase_times = dict(state.get("phase_times", {}))
        phase_times["validation"] = phase_times.get("validation", 0.0) + elapsed

        if config.verbose:
            status = "PASS" if report.will_compile else "FAIL"
            print(
                f"  [validate] iteration={iteration}  "
                f"blockers={report.blocker_count}  status={status}"
            )

        return {
            "compilation_report": report.to_dict(),
            "remaining_blockers": remaining,
            "iteration": iteration + 1,
            "phase_times": phase_times,
        }

    return validate_node
