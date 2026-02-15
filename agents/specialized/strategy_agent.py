#!/usr/bin/env python3
"""
Strategy agent for transformation planning.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents.base.agent_base import BaseAgent
from agents.base import prompts
from agents.config import PipelineConfig, StrategyAgentConfig
from agents.specialized.diagnosis_agent import DiagnosisReport
from agents.base.unified_retriever import RetrievalProfile


class TransformationPhase(BaseModel):
    phase_id: str
    name: str
    objective: str
    target_op_types: List[str] = Field(default_factory=list)
    transformation_type: str
    validation: str
    dependencies: List[str] = Field(default_factory=list)
    fallback: Optional[str] = None


class TransformationPlan(BaseModel):
    strategy_id: str
    strategy_reasoning: str
    phases: List[TransformationPhase] = Field(default_factory=list)
    risk_assessment: str
    expected_success_rate: float
    divide_and_conquer: bool = False

    model_config = {"extra": "allow"}


class StrategyAgent(BaseAgent):
    """LLM-powered planning agent."""

    def __init__(
        self,
        api_key: str,
        pipeline_config: PipelineConfig,
        config: StrategyAgentConfig,
    ) -> None:
        super().__init__(
            api_key=api_key,
            pipeline_config=pipeline_config,
            agent_model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.config = config

    def plan(self, diagnosis: DiagnosisReport) -> TransformationPlan:
        """Generate an initial transformation plan from diagnosis."""
        architecture = diagnosis.architecture_type
        blockers = [b.op_type for b in diagnosis.blockers]
        detected_patterns = diagnosis.detected_patterns

        # Unified retrieval context (KB + Surgery DB + Strategy DB)
        rag_query = f"{architecture} strategy for blockers {', '.join(blockers)}"
        combined = self.retrieve_combined(
            profile=RetrievalProfile.strategy(),
            query=rag_query,
            op_types=blockers,
            model_category=architecture,
            architecture=architecture,
            detected_patterns=detected_patterns,
            blocker_ops=blockers,
        )
        strategy_candidate = combined.strategy_candidate

        prompt = prompts.STRATEGY_PLANNING_PROMPT.format(
            diagnosis_report=diagnosis.model_dump(),
            strategy_candidate=strategy_candidate or "None",
            rag_context=combined.as_text() or "None",
        )

        plan = self.call_llm(
            prompt=prompt,
            response_model=TransformationPlan,
            system_prompt=prompts.STRATEGY_SYSTEM_PROMPT,
        )

        return plan

    # ------------------------------------------------------------------
    # Refinement (feedback loop)
    # ------------------------------------------------------------------

    def refine_plan(
        self,
        diagnosis: DiagnosisReport,
        remaining_blockers: List[str],
        surgery_history: List[Dict[str, Any]],
        compilation_report: Dict[str, Any],
        iteration: int,
    ) -> TransformationPlan:
        """
        Produce a revised plan that targets only the *remaining_blockers*.

        Includes the previous surgery history and compilation errors so
        the LLM can avoid repeating failed strategies.
        """
        architecture = diagnosis.architecture_type
        detected_patterns = diagnosis.detected_patterns

        rag_query = f"{architecture} fix remaining blockers {', '.join(remaining_blockers)}"
        combined = self.retrieve_combined(
            profile=RetrievalProfile.strategy(),
            query=rag_query,
            op_types=remaining_blockers,
            model_category=architecture,
            architecture=architecture,
            detected_patterns=detected_patterns,
            blocker_ops=remaining_blockers,
        )

        # Build a concise history summary (last 3 phases max)
        history_summary = []
        for entry in surgery_history[-3:]:
            phase_summary = {
                "phase_id": entry.get("phase_id", ""),
                "phase_name": entry.get("phase_name", ""),
                "success": entry.get("success", False),
                "suggestion_count": len(entry.get("suggestions", [])),
                "errors": [
                    s.get("error", "")
                    for s in entry.get("suggestions", [])
                    if s.get("error")
                ],
            }
            history_summary.append(phase_summary)

        # Blocker details from compilation report
        blocker_details = compilation_report.get("blocker_ops", {})

        refine_prompt = getattr(prompts, "REFINE_STRATEGY_PROMPT", None)
        if refine_prompt is None:
            # Fallback: augment the standard prompt with error context
            refine_prompt = prompts.STRATEGY_PLANNING_PROMPT

        prompt = refine_prompt.format(
            diagnosis_report=json.dumps(diagnosis.model_dump(), indent=2),
            remaining_blockers=json.dumps(remaining_blockers),
            blocker_details=json.dumps(blocker_details, indent=2),
            surgery_history=json.dumps(history_summary, indent=2),
            iteration=iteration,
            strategy_candidate=combined.strategy_candidate or "None",
            rag_context=combined.as_text() or "None",
        )

        return self.call_llm(
            prompt=prompt,
            response_model=TransformationPlan,
            system_prompt=prompts.STRATEGY_SYSTEM_PROMPT,
        )
