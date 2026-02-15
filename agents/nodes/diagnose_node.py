#!/usr/bin/env python3
"""
Diagnose node -- wraps DiagnosisAgent.analyze().

Reads *model_path* from state, runs architecture analysis + compilation
simulation via the DiagnosisAgent, and writes the report into state.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from agents.config import PipelineConfig
from agents.state import PipelineState


def make_diagnose_node(
    api_key: str,
    config: PipelineConfig,
) -> Callable[[PipelineState], Dict[str, Any]]:
    """Return a LangGraph node function for diagnosis."""

    from agents.specialized.diagnosis_agent import DiagnosisAgent

    agent = DiagnosisAgent(
        api_key=api_key,
        pipeline_config=config,
        config=config.diagnosis_agent_config,
    )

    def diagnose_node(state: PipelineState) -> Dict[str, Any]:
        model_path: str = state["model_path"]

        start = time.time()
        report = agent.analyze(model_path)
        elapsed = time.time() - start

        phase_times = dict(state.get("phase_times", {}))
        phase_times["diagnosis"] = elapsed

        return {
            "diagnosis": report.model_dump(),
            "phase_times": phase_times,
        }

    return diagnose_node
