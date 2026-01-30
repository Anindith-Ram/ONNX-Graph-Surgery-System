#!/usr/bin/env python3
"""
State Machine Executor for Graph Surgery.

Replaces LangChain ReAct agent with a simple, debuggable state machine that:
- Has explicit states: ANALYZE → PLAN → TRANSFORM → VALIDATE → EVALUATE → DONE/FAILED
- Provides full visibility into state transitions
- Uses direct function calls instead of tool abstractions
"""

import copy
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from agents.config import AgentConfig
from agents.diagnostics import FeedbackCollector, TransformationResult
from agents.strategy_planner import TransformationStrategy


# =============================================================================
# State Machine States
# =============================================================================

class PipelineState(Enum):
    """States for the graph surgery pipeline."""
    
    INIT = "init"
    ANALYZE = "analyze"
    PLAN = "plan"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    EVALUATE = "evaluate"
    DONE = "done"
    FAILED = "failed"
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# Execution Context
# =============================================================================

class ExecutionContext(BaseModel):
    """
    Context maintained during state machine execution.
    
    Replaces LangChain's implicit agent context with explicit state.
    """
    
    # Model info
    model_path: str
    model_name: str = ""
    
    # Suggestions
    all_suggestions: List[Dict] = Field(default_factory=list)
    pending_suggestions: List[Dict] = Field(default_factory=list)
    applied_suggestions: List[Dict] = Field(default_factory=list)
    failed_suggestions: List[Dict] = Field(default_factory=list)
    skipped_suggestions: List[Dict] = Field(default_factory=list)
    
    # Strategy
    strategy: Optional[TransformationStrategy] = None
    
    # Execution tracking
    iteration: int = 0
    max_iterations: int = 15
    start_time: float = Field(default_factory=time.time)
    
    # Ground truth
    ground_truth_path: Optional[str] = None
    
    # Results
    evaluation: Optional[Dict] = None
    
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def success_rate(self, feedback: FeedbackCollector) -> float:
        """Calculate success rate from feedback."""
        return feedback.success_rate()
    
    def all_done(self) -> bool:
        """Check if all suggestions have been processed."""
        return len(self.pending_suggestions) == 0


class ExecutionResult(BaseModel):
    """Result from state machine execution."""
    
    success: bool
    message: str
    iterations: int
    final_state: str
    
    # Feedback (stored as dict for serialization)
    feedback_summary: Dict = Field(default_factory=dict)
    
    # Evaluation
    evaluation: Optional[Dict] = None
    
    # Timing
    elapsed_seconds: float = 0.0
    
    # State history for debugging
    state_history: List[str] = Field(default_factory=list)
    
    model_config = {"extra": "allow"}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.model_dump()


# =============================================================================
# State Machine Executor
# =============================================================================

class GraphSurgeryExecutor:
    """
    State machine executor for ONNX graph surgery.
    
    Provides:
    - Explicit state transitions (no hidden agent loops)
    - Full debugging visibility
    - Direct function calls (no tool abstractions)
    - Checkpointing support
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[AgentConfig] = None,
        strategy_change_callback: Optional[Callable] = None,
    ):
        """
        Initialize executor.
        
        Args:
            api_key: API key for LLM provider
            config: Executor configuration
            strategy_change_callback: Callback to request strategy change
        """
        self.api_key = api_key
        self.config = config or AgentConfig()
        self.strategy_change_callback = strategy_change_callback
        
        # State
        self.state = PipelineState.INIT
        self.state_history: List[str] = []
        
        # Model references (not serializable, set during execution)
        self._original_model: Optional[Any] = None
        self._current_model: Optional[Any] = None
        
        # Components (lazy loaded)
        self._applicator = None
        self._comparator = None
        self._analyzer = None
        
        # Feedback collector
        self.feedback_collector = FeedbackCollector()
    
    def run(
        self,
        model_path: str,
        suggestions: List[Dict],
        strategy: Optional[TransformationStrategy] = None,
        ground_truth_path: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Run the state machine to process a model.
        
        Args:
            model_path: Path to ONNX model
            suggestions: List of suggestion dictionaries
            strategy: Optional transformation strategy
            ground_truth_path: Optional path to ground truth model
            
        Returns:
            ExecutionResult with transformation outcomes
        """
        if not ONNX_AVAILABLE:
            return ExecutionResult(
                success=False,
                message="ONNX is not available",
                iterations=0,
                final_state=PipelineState.FAILED.value,
            )
        
        # Initialize context
        ctx = ExecutionContext(
            model_path=model_path,
            model_name=Path(model_path).stem,
            all_suggestions=suggestions,
            pending_suggestions=self._sort_suggestions(suggestions),
            strategy=strategy,
            max_iterations=self.config.max_iterations,
            ground_truth_path=ground_truth_path,
        )
        
        # Reset state
        self.state = PipelineState.INIT
        self.state_history = []
        self.feedback_collector = FeedbackCollector(model_name=ctx.model_name)
        
        # Load model
        try:
            self._original_model = onnx.load(model_path)
            self._current_model = copy.deepcopy(self._original_model)
        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Failed to load model: {e}",
                iterations=0,
                final_state=PipelineState.FAILED.value,
            )
        
        # Run state machine
        self._transition_to(PipelineState.ANALYZE)
        
        while self.state not in (PipelineState.DONE, PipelineState.FAILED):
            if self.config.verbose:
                print(f"[{self.state}] Iteration {ctx.iteration}")
            
            try:
                self._execute_state(ctx)
            except Exception as e:
                if self.config.verbose:
                    print(f"  Error in state {self.state}: {e}")
                self._transition_to(PipelineState.FAILED)
                return ExecutionResult(
                    success=False,
                    message=f"Execution failed in state {self.state}: {e}",
                    iterations=ctx.iteration,
                    final_state=self.state.value,
                    state_history=self.state_history,
                    elapsed_seconds=ctx.elapsed_seconds(),
                )
            
            ctx.iteration += 1
            
            # Safety check for infinite loops
            if ctx.iteration > ctx.max_iterations * 2:
                self._transition_to(PipelineState.FAILED)
                break
        
        # Build result
        success = (
            self.state == PipelineState.DONE and
            self.feedback_collector.success_rate() >= self.config.low_success_threshold
        )
        
        return ExecutionResult(
            success=success,
            message=self._build_result_message(ctx),
            iterations=ctx.iteration,
            final_state=self.state.value,
            feedback_summary=self.feedback_collector.get_summary(),
            evaluation=ctx.evaluation,
            elapsed_seconds=ctx.elapsed_seconds(),
            state_history=self.state_history,
        )
    
    def _execute_state(self, ctx: ExecutionContext) -> None:
        """Execute current state and transition to next."""
        
        if self.state == PipelineState.ANALYZE:
            self._do_analyze(ctx)
            
        elif self.state == PipelineState.PLAN:
            self._do_plan(ctx)
            
        elif self.state == PipelineState.TRANSFORM:
            self._do_transform(ctx)
            
        elif self.state == PipelineState.VALIDATE:
            self._do_validate(ctx)
            
        elif self.state == PipelineState.EVALUATE:
            self._do_evaluate(ctx)
    
    def _do_analyze(self, ctx: ExecutionContext) -> None:
        """Analyze phase: log model info and move to planning."""
        if self.config.verbose:
            print(f"  Model: {ctx.model_name}")
            print(f"  Suggestions: {len(ctx.all_suggestions)}")
            print(f"  Strategy: {ctx.strategy.name if ctx.strategy else 'None'}")
        
        self._transition_to(PipelineState.PLAN)
    
    def _do_plan(self, ctx: ExecutionContext) -> None:
        """Plan phase: prepare for transformation."""
        # If we have pending suggestions, move to transform
        if ctx.pending_suggestions:
            self._transition_to(PipelineState.TRANSFORM)
        else:
            # No suggestions to apply
            self._transition_to(PipelineState.EVALUATE)
    
    def _do_transform(self, ctx: ExecutionContext) -> None:
        """Transform phase: apply the next suggestion."""
        if not ctx.pending_suggestions:
            # Done with all suggestions
            self._transition_to(PipelineState.VALIDATE)
            return
        
        # Get next suggestion
        suggestion = ctx.pending_suggestions[0]
        
        if self.config.verbose:
            sid = suggestion.get('id', '?')
            op_type = suggestion.get('location', {}).get('op_type', '?')
            print(f"  Applying suggestion {sid}: {op_type}")
        
        # Apply suggestion
        result = self._apply_suggestion(suggestion)
        self.feedback_collector.add(result)
        
        # Update context
        ctx.pending_suggestions.pop(0)
        
        if result.effective:
            ctx.applied_suggestions.append(suggestion)
        elif result.success and not result.was_transformed:
            ctx.skipped_suggestions.append(suggestion)
        else:
            ctx.failed_suggestions.append(suggestion)
        
        if self.config.verbose:
            status = "SUCCESS" if result.success else "FAILED"
            transformed = "transformed" if result.was_transformed else "no change"
            print(f"    Result: {status} ({transformed})")
        
        # Check if we should continue
        if self._should_continue_transforming(ctx):
            # Stay in transform state
            pass
        else:
            # Move to validate
            self._transition_to(PipelineState.VALIDATE)
    
    def _do_validate(self, ctx: ExecutionContext) -> None:
        """Validate phase: check model structure."""
        try:
            onnx.checker.check_model(self._current_model)
            if self.config.verbose:
                print("  Model validation: PASSED")
            
            # Check if we have more suggestions
            if ctx.pending_suggestions and ctx.iteration < ctx.max_iterations:
                # More work to do, go back to transform
                self._transition_to(PipelineState.TRANSFORM)
            else:
                # Done, move to evaluate
                self._transition_to(PipelineState.EVALUATE)
                
        except Exception as e:
            if self.config.verbose:
                print(f"  Model validation: FAILED - {e}")
            
            # Validation failed, try to continue or fail
            if ctx.iteration < ctx.max_iterations and ctx.pending_suggestions:
                # Try to recover by continuing
                self._transition_to(PipelineState.TRANSFORM)
            else:
                self._transition_to(PipelineState.EVALUATE)
    
    def _do_evaluate(self, ctx: ExecutionContext) -> None:
        """Evaluate phase: compare with ground truth if available."""
        if ctx.ground_truth_path:
            try:
                gt_model = onnx.load(ctx.ground_truth_path)
                comparator = self._get_comparator()
                ctx.evaluation = comparator.compare_models(
                    self._current_model,
                    gt_model,
                    self._original_model
                )
                
                if self.config.verbose:
                    similarity = ctx.evaluation.get('overall_similarity', 0)
                    print(f"  Ground truth similarity: {similarity:.1%}")
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"  Evaluation error: {e}")
                ctx.evaluation = {"error": str(e)}
        
        self._transition_to(PipelineState.DONE)
    
    def _apply_suggestion(self, suggestion: Dict) -> TransformationResult:
        """Apply a single suggestion to the model."""
        applicator = self._get_applicator()
        return applicator.apply_single(self._current_model, suggestion)
    
    def _should_continue_transforming(self, ctx: ExecutionContext) -> bool:
        """Check if we should continue applying transformations."""
        # No more suggestions
        if not ctx.pending_suggestions:
            return False
        
        # Max iterations reached
        if ctx.iteration >= ctx.max_iterations:
            return False
        
        # Check success rate for early stopping
        if (ctx.iteration >= self.config.min_iterations_before_stop and
            self.feedback_collector.success_rate() < self.config.early_stop_threshold):
            if self.config.verbose:
                print(f"  Early stopping: success rate {self.feedback_collector.success_rate():.1%} below threshold")
            return False
        
        # Check if we should request strategy change
        if (self.strategy_change_callback and
            ctx.iteration >= 3 and
            self.feedback_collector.success_rate() < self.config.low_success_threshold):
            if self.config.verbose:
                print("  Requesting strategy change...")
            new_strategy = self.strategy_change_callback(
                f"Success rate {self.feedback_collector.success_rate():.1%} is too low"
            )
            if new_strategy:
                ctx.strategy = new_strategy
        
        return True
    
    def _transition_to(self, new_state: PipelineState) -> None:
        """Transition to a new state."""
        self.state_history.append(f"{self.state.value} -> {new_state.value}")
        self.state = new_state
    
    def _sort_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Sort suggestions by priority."""
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        return sorted(
            suggestions,
            key=lambda s: priority_order.get(s.get('priority', 'info'), 4)
        )
    
    def _build_result_message(self, ctx: ExecutionContext) -> str:
        """Build result message."""
        parts = [
            f"Processed {ctx.model_name}",
            f"Applied: {len(ctx.applied_suggestions)}",
            f"Failed: {len(ctx.failed_suggestions)}",
            f"Skipped: {len(ctx.skipped_suggestions)}",
            f"Success rate: {self.feedback_collector.success_rate():.1%}",
        ]
        return " | ".join(parts)
    
    def _get_applicator(self):
        """Lazy-load suggestion applicator."""
        if self._applicator is None:
            from suggestion_pipeline.suggestion_applicator import SuggestionApplicator
            self._applicator = SuggestionApplicator()
        return self._applicator
    
    def _get_comparator(self):
        """Lazy-load model comparator."""
        if self._comparator is None:
            from evaluation.model_comparator import ModelComparator
            self._comparator = ModelComparator()
        return self._comparator
    
    @property
    def current_model(self):
        """Get current model (for external access)."""
        return self._current_model
    
    @property
    def original_model(self):
        """Get original model (for external access)."""
        return self._original_model


# =============================================================================
# Backward Compatibility
# =============================================================================

# AgentResult for backward compatibility
class AgentResult(BaseModel):
    """Result from executor (backward compatible with old AgentResult)."""
    
    success: bool
    message: str
    iterations: int
    final_model_path: Optional[str] = None
    evaluation: Optional[Dict] = None
    
    # Keep feedback collector reference for compatibility
    _feedback: Optional[FeedbackCollector] = None
    
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    
    @property
    def feedback(self) -> FeedbackCollector:
        """Get feedback collector."""
        if self._feedback is None:
            self._feedback = FeedbackCollector()
        return self._feedback
    
    @feedback.setter
    def feedback(self, value: FeedbackCollector):
        """Set feedback collector."""
        self._feedback = value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "feedback": self.feedback.to_dict() if self._feedback else {},
            "final_model_path": self.final_model_path,
            "evaluation": self.evaluation,
        }


# Backward compatibility wrapper
class GraphSurgeryStateMachine(GraphSurgeryExecutor):
    """Alias for GraphSurgeryExecutor (backward compatibility)."""
    pass


# Export
__all__ = [
    "PipelineState",
    "ExecutionContext",
    "ExecutionResult",
    "GraphSurgeryExecutor",
    "GraphSurgeryStateMachine",
    "AgentResult",
]
