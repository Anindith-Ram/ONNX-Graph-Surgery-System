#!/usr/bin/env python3
"""
Execute node -- applies surgery code to the ONNX model.

For each phase in the current plan the SurgeryAgent generates executable
GraphSurgeon / ONNX-helper code. The code is executed in a restricted
``exec()`` sandbox against the live ``onnx.ModelProto``.

Graph snapshots are captured before / after each phase so that the
diagnostics module can compute deltas.
"""

from __future__ import annotations

import io
import time
import traceback
from typing import Any, Callable, Dict, List

import numpy as np
import onnx

from agents.config import PipelineConfig
from agents.diagnostics import GraphSnapshot, TransformationDelta
from agents.state import PipelineState


# ---------------------------------------------------------------------------
# Sandboxed code execution helpers
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}


def _execute_surgery_code(
    code: str,
    model: onnx.ModelProto,
    timeout_seconds: float = 30.0,
) -> onnx.ModelProto:
    """
    Execute *code* against *model* in a restricted namespace.

    The code has access to ``onnx``, ``numpy`` (as ``np``), the ``model``
    variable, and a limited set of builtins.  After execution the (possibly
    mutated) ``model`` is returned.
    """
    namespace: Dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "onnx": onnx,
        "np": np,
        "numpy": np,
        "model": model,
        "helper": onnx.helper,
        "numpy_helper": onnx.numpy_helper,
        "TensorProto": onnx.TensorProto,
    }

    try:
        compile(code, "<surgery>", "exec")  # syntax check
    except SyntaxError as exc:
        raise SyntaxError(f"Surgery code has a syntax error: {exc}") from exc

    exec(code, namespace)  # noqa: S102 -- intentional sandboxed exec

    return namespace.get("model", model)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def make_execute_node(
    api_key: str,
    config: PipelineConfig,
) -> Callable[[PipelineState], Dict[str, Any]]:
    """Return a LangGraph node function for surgery execution."""

    from agents.specialized.strategy_agent import TransformationPhase
    from agents.specialized.surgery_agent import SurgeryAgent

    agent = SurgeryAgent(
        api_key=api_key,
        pipeline_config=config,
        config=config.surgery_agent_config,
    )

    sandbox_timeout = getattr(config, "sandbox_timeout_seconds", 30.0)

    def execute_node(state: PipelineState) -> Dict[str, Any]:
        plan_dict = state["plan"]
        phases_raw: List[Dict[str, Any]] = plan_dict.get("phases", [])
        model_bytes: bytes = state["current_model_bytes"]
        history: List[Dict[str, Any]] = list(state.get("surgery_history", []))
        applied: List[Dict[str, Any]] = list(state.get("transformations_applied", []))
        diagnosis = state.get("diagnosis", {})
        model_category = diagnosis.get("architecture_type", "Unknown")

        model = onnx.load_from_string(model_bytes)

        start = time.time()

        for phase_raw in phases_raw:
            phase = TransformationPhase.model_validate(phase_raw)

            # Capture snapshot before
            snap_before = GraphSnapshot.capture(model)

            # Ask the SurgeryAgent for executable suggestions
            suggestion_set = agent.generate_suggestions(
                phase, model, model_category=model_category,
            )

            phase_entry: Dict[str, Any] = {
                "phase_id": phase.phase_id,
                "phase_name": phase.name,
                "suggestions": [],
                "success": True,
            }

            for suggestion in suggestion_set.suggestions:
                code = suggestion.code_snippet
                entry: Dict[str, Any] = {
                    "suggestion_id": suggestion.suggestion_id,
                    "code": code,
                    "success": False,
                    "error": None,
                    "delta": None,
                }

                if not code or not code.strip():
                    entry["error"] = "Empty code snippet"
                    phase_entry["suggestions"].append(entry)
                    continue

                try:
                    model = _execute_surgery_code(
                        code, model, timeout_seconds=sandbox_timeout,
                    )
                    snap_after = GraphSnapshot.capture(model)
                    delta = TransformationDelta.compute(snap_before, snap_after)
                    entry["success"] = True
                    entry["delta"] = delta.to_dict()
                    snap_before = snap_after  # chain snapshots

                    applied.append({
                        "phase_id": phase.phase_id,
                        "suggestion_id": suggestion.suggestion_id,
                        "code": code,
                        "target_ops": suggestion.target_ops,
                        "delta": delta.to_dict(),
                    })
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    phase_entry["success"] = False

                phase_entry["suggestions"].append(entry)

            history.append(phase_entry)

        elapsed = time.time() - start
        phase_times = dict(state.get("phase_times", {}))
        phase_times["execution"] = phase_times.get("execution", 0.0) + elapsed

        return {
            "current_model_bytes": model.SerializeToString(),
            "surgery_history": history,
            "transformations_applied": applied,
            "phase_times": phase_times,
        }

    return execute_node
