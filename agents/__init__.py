#!/usr/bin/env python3
"""
Agents module for Graph Surgery Pipeline.

This module provides:
- State machine executor for graph surgery transformations
- Strategy planner for transformation planning (simplified from ToT)
- Enhanced diagnostics for feedback collection
- Unified pipeline combining planning with execution
- LLM client for structured outputs via LiteLLM + Instructor

Key components (new architecture):
- GraphSurgeryExecutor: State machine for executing transformations
- StrategyPlanner: Simplified strategy generation (replaces complex ToT)
- LLMClient: Unified LLM interface with structured outputs

Backward compatibility aliases are provided for:
- ToTPlanner (use StrategyPlanner instead)
- GraphSurgeryReActAgent (use GraphSurgeryExecutor instead)
- ReActToTPipeline (use GraphSurgeryPipeline instead)
"""

# Diagnostics
from agents.diagnostics import (
    ErrorCategory,
    GraphSnapshot,
    TransformationDelta,
    TransformationResult,
    FeedbackCollector,
)

# Configuration (Pydantic models)
from agents.config import (
    AgentConfig,
    StrategyConfig,
    ToolConfig,
    PipelineConfig,
    DEFAULT_AGENT_CONFIG,
    DEFAULT_STRATEGY_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    # Backward compatibility aliases
    ToTConfig,  # -> StrategyConfig
    DEFAULT_TOT_CONFIG,  # -> DEFAULT_STRATEGY_CONFIG
)

# State management (Pydantic models)
from agents.state import (
    AgentState,
    TransformationStrategy,
    StateManager,
)

# New components
from agents.llm_client import (
    LLMClient,
    call_llm,
)

from agents.strategy_planner import (
    StrategyPlanner,
    StrategyEvaluation,
    StrategyList,
    # Backward compatibility alias
    ToTPlanner,
)

from agents.executor import (
    PipelineState,
    ExecutionContext,
    ExecutionResult,
    GraphSurgeryExecutor,
    GraphSurgeryStateMachine,
    AgentResult,
)

from agents.pipeline import (
    PipelineResult,
    GraphSurgeryPipeline,
    run_pipeline,
    # Backward compatibility alias
    ReActToTPipeline,
)

# Note: react_agent.py and tot_planner.py have been removed
# Use GraphSurgeryExecutor and StrategyPlanner instead
# Backward compatibility aliases are provided in executor.py and strategy_planner.py

# Tools (plain functions)
from agents.tools import (
    ToolContext,
    AgentContext,  # Alias for ToolContext
    analyze_model,
    retrieve_patterns,
    apply_suggestion,
    validate_model,
    compare_with_ground_truth,
    get_feedback,
)

__all__ = [
    # === NEW COMPONENTS ===
    # Executor (replaces ReAct agent)
    "GraphSurgeryExecutor",
    "GraphSurgeryStateMachine",
    "PipelineState",
    "ExecutionContext",
    "ExecutionResult",
    
    # Strategy Planner (replaces ToT)
    "StrategyPlanner",
    "StrategyEvaluation",
    "StrategyList",
    "TransformationStrategy",
    
    # Pipeline
    "GraphSurgeryPipeline",
    "PipelineResult",
    "run_pipeline",
    
    # LLM Client
    "LLMClient",
    "call_llm",
    
    # Configuration
    "AgentConfig",
    "StrategyConfig",
    "ToolConfig",
    "PipelineConfig",
    "DEFAULT_AGENT_CONFIG",
    "DEFAULT_STRATEGY_CONFIG",
    "DEFAULT_PIPELINE_CONFIG",
    
    # State Management
    "AgentState",
    "StateManager",
    
    # Diagnostics
    "ErrorCategory",
    "GraphSnapshot",
    "TransformationDelta",
    "TransformationResult",
    "FeedbackCollector",
    
    # Tools
    "ToolContext",
    "analyze_model",
    "retrieve_patterns",
    "apply_suggestion",
    "validate_model",
    "compare_with_ground_truth",
    "get_feedback",
    
    # === BACKWARD COMPATIBILITY ===
    "ToTConfig",  # -> StrategyConfig
    "ToTPlanner",  # -> StrategyPlanner
    "DEFAULT_TOT_CONFIG",  # -> DEFAULT_STRATEGY_CONFIG
    "ReActToTPipeline",  # -> GraphSurgeryPipeline
    "AgentResult",
    "AgentContext",  # -> ToolContext
]
