#!/usr/bin/env python3
"""
Plan node -- wraps StrategyAgent.plan().

Takes the diagnosis report from state, produces a multi-phase
TransformationPlan, and loads the original ONNX model into
``current_model_bytes`` for subsequent surgery execution.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import onnx

from agents.config import PipelineConfig
from agents.state import PipelineState


def make_plan_node(
    api_key: str,
    config: PipelineConfig,
) -> Callable[[PipelineState], Dict[str, Any]]:
    """Return a LangGraph node function for strategy planning."""

    from agents.specialized.diagnosis_agent import DiagnosisReport
    from agents.specialized.strategy_agent import StrategyAgent

    agent = StrategyAgent(
        api_key=api_key,
        pipeline_config=config,
        config=config.strategy_agent_config,
    )

    def plan_node(state: PipelineState) -> Dict[str, Any]:
        diagnosis_dict = state["diagnosis"]
        diagnosis = DiagnosisReport.model_validate(diagnosis_dict)

        start = time.time()
        plan = agent.plan(diagnosis)
        elapsed = time.time() - start

        # Load model bytes so the execute node can work on them
        model_path: str = state["model_path"]
        model_proto = onnx.load(model_path)
        model_bytes = model_proto.SerializeToString()

        phase_times = dict(state.get("phase_times", {}))
        phase_times["planning"] = elapsed

        return {
            "plan": plan.model_dump(),
            "current_model_bytes": model_bytes,
            "iteration": 0,
            "surgery_history": [],
            "transformations_applied": [],
            "remaining_blockers": [],
            "phase_times": phase_times,
        }

    return plan_node
