#!/usr/bin/env python3
"""
Surgery agent for LLM-driven code generation and execution.

Generates executable GraphSurgeon / ONNX-helper code for each
transformation phase, with syntax validation and error-corrective
re-generation capabilities.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import onnx
from pydantic import BaseModel, Field

from agents.base.agent_base import BaseAgent
from agents.base import prompts
from agents.config import PipelineConfig, SurgeryAgentConfig
from agents.specialized.strategy_agent import TransformationPhase
from agents.base.unified_retriever import RetrievalProfile


class SurgerySuggestion(BaseModel):
    suggestion_id: str
    summary: str
    rationale: str
    target_ops: List[str] = Field(default_factory=list)
    code_snippet: str
    expected_effect: str
    validation_steps: List[str] = Field(default_factory=list)
    manual_checks: List[str] = Field(default_factory=list)
    risk_level: str = "medium"
    confidence: float = 0.5

    model_config = {"extra": "allow"}


class SurgerySuggestionSet(BaseModel):
    phase_id: str
    phase_name: str
    suggestions: List[SurgerySuggestion] = Field(default_factory=list)
    overall_risk: str = "medium"
    notes: Optional[str] = None

    model_config = {"extra": "allow"}


class SurgeryAgent(BaseAgent):
    """LLM-powered executable code generator for ONNX graph surgery."""

    def __init__(
        self,
        api_key: str,
        pipeline_config: PipelineConfig,
        config: SurgeryAgentConfig,
    ) -> None:
        super().__init__(
            api_key=api_key,
            pipeline_config=pipeline_config,
            agent_model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.config = config

    # ------------------------------------------------------------------
    # Primary generation
    # ------------------------------------------------------------------

    def generate_suggestions(
        self,
        phase: TransformationPhase,
        model: onnx.ModelProto,
        model_category: Optional[str] = None,
    ) -> SurgerySuggestionSet:
        """
        Generate executable surgery suggestions for a transformation phase.

        Each suggestion includes a ``code_snippet`` that can be ``exec()``-ed
        against the ONNX model in the execute node's sandbox.
        """
        region_context = self._build_region_context(model, phase)
        rag_query = f"{phase.transformation_type} for {', '.join(phase.target_op_types)}"
        combined = self.retrieve_combined(
            profile=RetrievalProfile.surgery(),
            query=rag_query,
            op_types=phase.target_op_types,
            model_category=model_category,
        )
        examples = combined.surgery_examples

        prompt = prompts.SURGERY_CODE_PROMPT.format(
            phase=phase.model_dump(),
            region_context=json.dumps(region_context, indent=2),
            transformation_examples=json.dumps(examples, indent=2),
            rag_context=combined.kb_text or "None",
        )

        result = self.call_llm(
            prompt=prompt,
            response_model=SurgerySuggestionSet,
            system_prompt=prompts.SURGERY_SYSTEM_PROMPT,
        )

        # Validate syntax of every snippet before returning
        for suggestion in result.suggestions:
            is_valid, error = self.validate_code(suggestion.code_snippet)
            if not is_valid:
                suggestion.code_snippet = (
                    f"# SYNTAX ERROR -- could not compile:\n"
                    f"# {error}\n"
                    f"# Original code follows:\n"
                    f"# {suggestion.code_snippet}"
                )

        return result

    # ------------------------------------------------------------------
    # Error-corrective re-generation
    # ------------------------------------------------------------------

    def generate_fix_for_error(
        self,
        error_context: str,
        previous_code: str,
        model: onnx.ModelProto,
        phase: TransformationPhase,
        model_category: Optional[str] = None,
    ) -> SurgerySuggestionSet:
        """
        Generate corrected surgery code after a previous attempt failed.

        Uses the ``SURGERY_FIX_PROMPT`` to include the compilation error
        and the code that was tried, so the LLM can produce an amended version.
        """
        region_context = self._build_region_context(model, phase)

        fix_prompt = getattr(prompts, "SURGERY_FIX_PROMPT", None)
        if fix_prompt is None:
            # Fallback: re-use the standard prompt with error prepended
            fix_prompt = (
                "The previous surgery attempt FAILED with the following error:\n"
                "{error_context}\n\n"
                "Previous code that was tried:\n```python\n{previous_code}\n```\n\n"
                + prompts.SURGERY_CODE_PROMPT
            )

        prompt = fix_prompt.format(
            error_context=error_context,
            previous_code=previous_code,
            phase=phase.model_dump(),
            region_context=json.dumps(region_context, indent=2),
            transformation_examples="[]",
            rag_context="None",
        )

        return self.call_llm(
            prompt=prompt,
            response_model=SurgerySuggestionSet,
            system_prompt=prompts.SURGERY_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Syntax validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_code(code: str) -> tuple:
        """
        Check whether *code* compiles without syntax errors.

        Returns ``(True, None)`` if valid, ``(False, error_msg)`` otherwise.
        """
        if not code or not code.strip():
            return (False, "Empty code snippet")
        # Strip comment-only snippets that start with "# SYNTAX ERROR"
        if code.lstrip().startswith("# SYNTAX ERROR"):
            return (False, "Previously flagged syntax error")
        try:
            compile(code, "<surgery_validate>", "exec")
            return (True, None)
        except SyntaxError as exc:
            return (False, str(exc))

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _build_region_context(
        self,
        model: onnx.ModelProto,
        phase: TransformationPhase,
    ) -> Dict[str, Any]:
        """Build a minimal context payload describing the target region."""
        nodes_info: List[Dict[str, Any]] = []
        target_ops = set(phase.target_op_types)

        for node in model.graph.node:
            if not target_ops or node.op_type in target_ops:
                nodes_info.append({
                    "name": node.name,
                    "op_type": node.op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                })
            if len(nodes_info) >= self.config.max_nodes_context:
                break

        return {
            "phase_id": phase.phase_id,
            "target_ops": list(target_ops),
            "nodes": nodes_info,
        }
