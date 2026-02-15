#!/usr/bin/env python3
"""
Refine-plan node -- feeds compilation errors back to the StrategyAgent.

When the validate node reports remaining blockers, this node constructs
a rich error context and asks the StrategyAgent to produce a revised plan
that targets only the unsolved blockers.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from agents.config import PipelineConfig
from agents.state import PipelineState


def make_refine_plan_node(
    api_key: str,
    config: PipelineConfig,
) -> Callable[[PipelineState], Dict[str, Any]]:
    """Return a LangGraph node function for strategy refinement."""

    from agents.specialized.diagnosis_agent import DiagnosisReport
    from agents.specialized.strategy_agent import StrategyAgent

    agent = StrategyAgent(
        api_key=api_key,
        pipeline_config=config,
        config=config.strategy_agent_config,
    )

    def refine_plan_node(state: PipelineState) -> Dict[str, Any]:
        diagnosis_dict = state["diagnosis"]
        diagnosis = DiagnosisReport.model_validate(diagnosis_dict)
        remaining = state.get("remaining_blockers", [])
        history = state.get("surgery_history", [])
        compilation = state.get("compilation_report", {})
        iteration = state.get("iteration", 0)

        start = time.time()

        # Use the refine_plan method if available; fall back to plan()
        if hasattr(agent, "refine_plan"):
            plan = agent.refine_plan(
                diagnosis=diagnosis,
                remaining_blockers=remaining,
                surgery_history=history,
                compilation_report=compilation,
                iteration=iteration,
            )
        else:
            plan = agent.plan(diagnosis)

        elapsed = time.time() - start

        phase_times = dict(state.get("phase_times", {}))
        phase_times["refine_plan"] = phase_times.get("refine_plan", 0.0) + elapsed

        return {
            "plan": plan.model_dump(),
            "phase_times": phase_times,
        }

    return refine_plan_node
