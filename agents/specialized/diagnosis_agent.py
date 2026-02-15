#!/usr/bin/env python3
"""
Diagnosis agent for architecture-level analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnx
from pydantic import BaseModel, Field

from agents.base.agent_base import BaseAgent
from agents.base import prompts
from agents.config import PipelineConfig, DiagnosisAgentConfig
from agents.base.unified_retriever import RetrievalProfile
from core_analysis.architecture_analyzer import ArchitectureAnalyzer
from core_analysis.compilation_simulator import CompilationSimulator


class BlockerAnalysis(BaseModel):
    node_name: str
    op_type: str
    reason: str
    severity: str
    region: Optional[str] = None


class DiagnosisReport(BaseModel):
    model_name: str
    architecture_type: str
    architecture_reasoning: str
    detected_patterns: List[str] = Field(default_factory=list)
    blockers: List[BlockerAnalysis] = Field(default_factory=list)
    recommended_approach: str
    confidence: float = 0.0
    analysis_payload: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class DiagnosisAgent(BaseAgent):
    """LLM-powered model analysis agent."""

    def __init__(
        self,
        api_key: str,
        pipeline_config: PipelineConfig,
        config: DiagnosisAgentConfig,
    ) -> None:
        super().__init__(
            api_key=api_key,
            pipeline_config=pipeline_config,
            agent_model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.config = config
        self.arch_analyzer = ArchitectureAnalyzer()
        self.compilation_sim = CompilationSimulator(verbose=pipeline_config.verbose)

    def analyze(self, model_path: str) -> DiagnosisReport:
        model_name = Path(model_path).stem

        # Quick model summary
        model = onnx.load(model_path)
        op_counts: Dict[str, int] = {}
        for node in model.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        top_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        model_summary = {
            "model_name": model_name,
            "node_count": len(model.graph.node),
            "input_count": len(model.graph.input),
            "output_count": len(model.graph.output),
            "top_ops": top_ops,
        }

        # Architecture analysis
        architecture = self.arch_analyzer.analyze(model_path)
        architecture_summary = architecture.to_dict() if hasattr(architecture, "to_dict") else {
            "architecture_type": getattr(architecture.architecture_type, "value", "Unknown"),
            "blocks": len(getattr(architecture, "blocks", [])),
            "block_coverage": getattr(architecture, "block_coverage", 0.0),
        }

        # Compilation simulation
        compilation = self.compilation_sim.simulate(model_path)
        compilation_summary = compilation.to_dict() if hasattr(compilation, "to_dict") else {
            "total_nodes": compilation.total_nodes,
            "blocker_count": compilation.blocker_count,
            "blocker_ops": list({b.op_type for b in compilation.blocker_nodes}),
            "will_compile": compilation.will_compile,
        }

        # Unified retrieval context (KB + Surgery DB)
        blocker_ops = compilation_summary.get("blocker_ops", [])
        arch_value = architecture_summary.get("architecture_type", "Unknown")
        rag_query = f"{arch_value} blockers {', '.join(blocker_ops)} compilation fixes"
        combined = self.retrieve_combined(
            profile=RetrievalProfile.diagnosis(),
            query=rag_query,
            op_types=blocker_ops,
            model_category=arch_value,
            architecture=arch_value,
            detected_patterns=[b.block_type.value for b in getattr(architecture, "blocks", [])],
            blocker_ops=blocker_ops,
        )

        # Build dynamic few-shot examples from surgery DB
        few_shot = ""
        if combined.surgery_examples:
            few_shot_lines = ["Here are similar models and how they were transformed:"]
            for ex in combined.surgery_examples[:3]:
                few_shot_lines.append(
                    f"- {ex.get('source_model', '?')}: {ex.get('op_type', '?')} "
                    f"-> {ex.get('action', '?')} (confidence {ex.get('confidence', 0):.0%})"
                )
            few_shot = "\n".join(few_shot_lines)

        # Prompt and LLM call
        prompt = prompts.DIAGNOSIS_ANALYSIS_PROMPT.format(
            model_summary=json.dumps(model_summary, indent=2),
            architecture_summary=json.dumps(architecture_summary, indent=2),
            compilation_summary=json.dumps(compilation_summary, indent=2),
            rag_context=combined.as_text() or "None",
            few_shot_examples=few_shot or "None available.",
        )

        report = self.call_llm(
            prompt=prompt,
            response_model=DiagnosisReport,
            system_prompt=prompts.DIAGNOSIS_SYSTEM_PROMPT,
        )

        # Enrich report with non-LLM payload
        report.model_name = report.model_name or model_name
        report.analysis_payload = {
            "model_summary": model_summary,
            "architecture_summary": architecture_summary,
            "compilation_summary": compilation_summary,
        }

        return report
