#!/usr/bin/env python3
"""
Agents module for the LangGraph-based graph surgery pipeline.
"""

from agents.config import (
    AgentConfig,
    StrategyConfig,
    ToolConfig,
    PipelineConfig,
    DiagnosisAgentConfig,
    StrategyAgentConfig,
    SurgeryAgentConfig,
    DEFAULT_AGENT_CONFIG,
    DEFAULT_STRATEGY_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    ToTConfig,  # Backward compatibility
    DEFAULT_TOT_CONFIG,  # Backward compatibility
)

from agents.llm_client import LLMClient, call_llm
from agents.base import BaseAgent, prompts, UnifiedRetriever, RetrievalProfile, CombinedContext
from agents.specialized import (
    DiagnosisAgent,
    DiagnosisReport,
    StrategyAgent,
    TransformationPlan,
    SurgeryAgent,
    SurgerySuggestionSet,
    SurgerySuggestion,
)
from agents.orchestrator import build_graph, should_retry
from agents.pipeline import PipelineResult, GraphSurgeryPipeline, run_pipeline, ReActToTPipeline
from agents.state import PipelineState
from agents.evaluation import PipelineEvaluation

__all__ = [
    # Core pipeline
    "GraphSurgeryPipeline",
    "PipelineResult",
    "run_pipeline",
    "ReActToTPipeline",
    "build_graph",
    "should_retry",
    "PipelineState",
    "PipelineEvaluation",

    # Agents
    "BaseAgent",
    "UnifiedRetriever",
    "RetrievalProfile",
    "CombinedContext",
    "DiagnosisAgent",
    "StrategyAgent",
    "SurgeryAgent",
    "DiagnosisReport",
    "TransformationPlan",
    "SurgerySuggestionSet",
    "SurgerySuggestion",

    # LLM
    "LLMClient",
    "call_llm",

    # Prompts
    "prompts",

    # Config
    "AgentConfig",
    "StrategyConfig",
    "ToolConfig",
    "PipelineConfig",
    "DiagnosisAgentConfig",
    "StrategyAgentConfig",
    "SurgeryAgentConfig",
    "DEFAULT_AGENT_CONFIG",
    "DEFAULT_STRATEGY_CONFIG",
    "DEFAULT_PIPELINE_CONFIG",
    "ToTConfig",
    "DEFAULT_TOT_CONFIG",
]
