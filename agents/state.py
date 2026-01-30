#!/usr/bin/env python3
"""
State Management for Graph Surgery Pipeline.

Provides state management:
- AgentState: Core state maintained across iterations
- StateManager: Utilities for state transitions and persistence

Note: TransformationStrategy is now defined in strategy_planner.py
"""

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from agents.diagnostics import FeedbackCollector, TransformationResult
from agents.config import AgentConfig

# Re-export TransformationStrategy from strategy_planner
from agents.strategy_planner import TransformationStrategy


class AgentState(BaseModel):
    """
    State maintained across pipeline iterations.
    
    Tracks:
    - Model state (paths, references)
    - Suggestion queues (pending, applied, failed)
    - Feedback collection
    - Strategy
    - Iteration tracking
    """
    
    # Model paths and state
    model_path: str
    model_name: str = ""
    
    # Suggestion management
    all_suggestions: List[Dict] = Field(default_factory=list)
    pending_suggestions: List[Dict] = Field(default_factory=list)
    applied_suggestions: List[Dict] = Field(default_factory=list)
    failed_suggestions: List[Dict] = Field(default_factory=list)
    skipped_suggestions: List[Dict] = Field(default_factory=list)
    
    # Strategy
    strategy: Optional[TransformationStrategy] = None
    strategy_changed: bool = False
    
    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 15
    start_time: float = Field(default_factory=time.time)
    
    # Ground truth (for evaluation)
    ground_truth_path: Optional[str] = None
    
    # Config reference (not serialized)
    _config: Optional[AgentConfig] = None
    
    # Model references (not serializable)
    _original_model: Optional[Any] = None
    _current_model: Optional[Any] = None
    _feedback_collector: Optional[FeedbackCollector] = None
    
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    
    def model_post_init(self, __context):
        """Initialize after model creation."""
        if not self.model_name:
            self.model_name = Path(self.model_path).stem
        
        if self._feedback_collector is None:
            self._feedback_collector = FeedbackCollector(model_name=self.model_name)
    
    @property
    def feedback_collector(self) -> FeedbackCollector:
        """Get feedback collector."""
        if self._feedback_collector is None:
            self._feedback_collector = FeedbackCollector(model_name=self.model_name)
        return self._feedback_collector
    
    @property
    def original_model(self):
        """Get original model."""
        return self._original_model
    
    @property
    def current_model(self):
        """Get current model."""
        return self._current_model
    
    @property
    def config(self) -> AgentConfig:
        """Get config."""
        if self._config is None:
            self._config = AgentConfig()
        return self._config
    
    @classmethod
    def create(
        cls,
        model_path: str,
        suggestions: List[Dict],
        config: Optional[AgentConfig] = None,
        strategy: Optional[TransformationStrategy] = None,
        ground_truth_path: Optional[str] = None,
    ) -> "AgentState":
        """
        Create initial agent state.
        
        Args:
            model_path: Path to ONNX model
            suggestions: List of suggestion dictionaries
            config: Agent configuration
            strategy: Optional strategy from planner
            ground_truth_path: Optional path to ground truth model
            
        Returns:
            Initialized AgentState
        """
        model_name = Path(model_path).stem
        
        # Sort suggestions by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: priority_order.get(s.get('priority', 'info'), 4)
        )
        
        state = cls(
            model_path=model_path,
            model_name=model_name,
            all_suggestions=sorted_suggestions,
            pending_suggestions=list(sorted_suggestions),
            strategy=strategy,
            max_iterations=config.max_iterations if config else 15,
            ground_truth_path=ground_truth_path,
        )
        state._config = config or AgentConfig()
        state._feedback_collector = FeedbackCollector(model_name=model_name)
        
        # Load model if ONNX available
        if ONNX_AVAILABLE:
            try:
                state._original_model = onnx.load(model_path)
                state._current_model = copy.deepcopy(state._original_model)
            except Exception:
                pass
        
        return state
    
    def should_continue(self) -> bool:
        """Determine if agent should continue iterating."""
        # No more suggestions
        if not self.pending_suggestions:
            return False
        
        # Max iterations reached
        if self.iteration >= self.max_iterations:
            return False
        
        # Check for low success rate
        if (self.iteration >= self.config.min_iterations_before_stop and
            self.feedback_collector.success_rate() < self.config.early_stop_threshold):
            return False
        
        return True
    
    def needs_strategy_change(self) -> bool:
        """Determine if strategy should be changed."""
        if self.iteration < 3:
            return False
        
        success_rate = self.feedback_collector.success_rate()
        if success_rate < self.config.low_success_threshold:
            return True
        
        # Check for dominant error categories
        error_dist = self.feedback_collector.error_distribution()
        total_errors = sum(error_dist.values())
        if total_errors > 0:
            max_error_count = max(error_dist.values()) if error_dist else 0
            if max_error_count / total_errors > 0.5:
                return True
        
        return False
    
    def get_next_suggestion(self) -> Optional[Dict]:
        """Get next pending suggestion based on strategy."""
        if not self.pending_suggestions:
            return None
        
        # If strategy specifies priority order, use it
        if self.strategy and self.strategy.priority_order:
            return self._get_suggestion_by_strategy()
        
        # Default: return first pending (already sorted by priority)
        return self.pending_suggestions[0]
    
    def _get_suggestion_by_strategy(self) -> Optional[Dict]:
        """Get next suggestion based on strategy priority."""
        if not self.pending_suggestions:
            return None
        
        priority_map = {
            "critical_blockers": ["critical", "blocker"],
            "shape_issues": ["tensor_format", "non_4d", "shape"],
            "optimizations": ["optimization", "low"],
            "dynamic_shapes": ["dynamic_shape"],
            "control_flow": ["control_flow", "loop", "if"],
        }
        
        for priority_type in self.strategy.priority_order:
            categories = priority_map.get(priority_type, [priority_type])
            for suggestion in self.pending_suggestions:
                suggestion_category = suggestion.get("category", "").lower()
                suggestion_priority = suggestion.get("priority", "").lower()
                if (suggestion_category in categories or
                    suggestion_priority in categories):
                    return suggestion
        
        return self.pending_suggestions[0]
    
    def mark_applied(self, suggestion: Dict, result: TransformationResult):
        """Mark suggestion as applied and update state."""
        if suggestion in self.pending_suggestions:
            self.pending_suggestions.remove(suggestion)
        
        if result.effective:
            self.applied_suggestions.append(suggestion)
        elif result.success and not result.was_transformed:
            self.skipped_suggestions.append(suggestion)
        else:
            self.failed_suggestions.append(suggestion)
    
    def increment_iteration(self):
        """Increment iteration counter."""
        self.iteration += 1
    
    def update_strategy(self, new_strategy: TransformationStrategy):
        """Update transformation strategy."""
        self.strategy = new_strategy
        self.strategy_changed = True
    
    def get_progress_summary(self) -> Dict:
        """Get progress summary for logging."""
        elapsed = time.time() - self.start_time
        return {
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "elapsed_seconds": elapsed,
            "pending": len(self.pending_suggestions),
            "applied": len(self.applied_suggestions),
            "failed": len(self.failed_suggestions),
            "skipped": len(self.skipped_suggestions),
            "success_rate": self.feedback_collector.success_rate(),
            "transformation_rate": self.feedback_collector.transformation_rate(),
            "strategy": self.strategy.name if self.strategy else None,
            "strategy_changed": self.strategy_changed,
        }
    
    def to_observation_string(self) -> str:
        """Format state for observation."""
        progress = self.get_progress_summary()
        
        return f"""
=== Agent State (Iteration {progress['iteration']}/{progress['max_iterations']}) ===
Model: {self.model_name}
Elapsed: {progress['elapsed_seconds']:.1f}s
Strategy: {progress['strategy'] or 'None'}

Progress:
- Pending: {progress['pending']}
- Applied: {progress['applied']}
- Failed: {progress['failed']}
- Skipped: {progress['skipped']}

Metrics:
- Success Rate: {progress['success_rate']:.1%}
- Transformation Rate: {progress['transformation_rate']:.1%}

{self.feedback_collector.to_observation_string()}
"""
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for persistence."""
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "pending_suggestions": self.pending_suggestions,
            "applied_suggestions": self.applied_suggestions,
            "failed_suggestions": self.failed_suggestions,
            "skipped_suggestions": self.skipped_suggestions,
            "feedback": self.feedback_collector.to_dict(),
            "strategy": self.strategy.to_dict() if self.strategy else None,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "start_time": self.start_time,
            "ground_truth_path": self.ground_truth_path,
        }
    
    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """Save state checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{self.model_name}_state.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return str(checkpoint_file)
    
    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_file: str,
        config: Optional[AgentConfig] = None,
    ) -> "AgentState":
        """Load state from checkpoint."""
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        # Recreate strategy if present
        strategy = None
        if data.get("strategy"):
            strategy = TransformationStrategy.from_dict(data["strategy"])
        
        state = cls(
            model_path=data.get("model_path", ""),
            model_name=data.get("model_name", ""),
            all_suggestions=(
                data.get("pending_suggestions", []) + 
                data.get("applied_suggestions", []) +
                data.get("failed_suggestions", []) +
                data.get("skipped_suggestions", [])
            ),
            pending_suggestions=data.get("pending_suggestions", []),
            applied_suggestions=data.get("applied_suggestions", []),
            failed_suggestions=data.get("failed_suggestions", []),
            skipped_suggestions=data.get("skipped_suggestions", []),
            strategy=strategy,
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 15),
            start_time=data.get("start_time", time.time()),
            ground_truth_path=data.get("ground_truth_path"),
        )
        state._config = config or AgentConfig()
        state._feedback_collector = FeedbackCollector(
            model_name=data.get("model_name", ""),
            session_id=data.get("feedback", {}).get("session_id", ""),
        )
        
        # Load model if path exists
        if ONNX_AVAILABLE and Path(state.model_path).exists():
            try:
                state._original_model = onnx.load(state.model_path)
                state._current_model = copy.deepcopy(state._original_model)
            except Exception:
                pass
        
        return state


class StateManager:
    """
    Utility class for managing agent state transitions.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_state(
        self,
        model_path: str,
        suggestions: List[Dict],
        config: Optional[AgentConfig] = None,
        strategy: Optional[TransformationStrategy] = None,
        ground_truth_path: Optional[str] = None,
    ) -> AgentState:
        """Create new agent state."""
        return AgentState.create(
            model_path=model_path,
            suggestions=suggestions,
            config=config,
            strategy=strategy,
            ground_truth_path=ground_truth_path,
        )
    
    def save_state(self, state: AgentState) -> str:
        """Save state to checkpoint."""
        return state.save_checkpoint(str(self.checkpoint_dir))
    
    def load_state(
        self,
        model_name: str,
        config: Optional[AgentConfig] = None,
    ) -> Optional[AgentState]:
        """Load state from checkpoint if exists."""
        checkpoint_file = self.checkpoint_dir / f"{model_name}_state.json"
        
        if not checkpoint_file.exists():
            return None
        
        return AgentState.load_checkpoint(str(checkpoint_file), config)
    
    def has_checkpoint(self, model_name: str) -> bool:
        """Check if checkpoint exists for model."""
        checkpoint_file = self.checkpoint_dir / f"{model_name}_state.json"
        return checkpoint_file.exists()
    
    def clear_checkpoint(self, model_name: str) -> bool:
        """Clear checkpoint for model."""
        checkpoint_file = self.checkpoint_dir / f"{model_name}_state.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False


# Export
__all__ = [
    "TransformationStrategy",
    "AgentState",
    "StateManager",
]
