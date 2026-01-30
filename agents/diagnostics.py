#!/usr/bin/env python3
"""
Enhanced Diagnostics for ONNX Graph Surgery.

Provides rich feedback collection for ReAct agent observation loops:
- TransformationResult: Per-suggestion result tracking
- ErrorCategory: Categorized failure types
- GraphSnapshot: Model state capture before/after transformations
- TransformationDelta: What changed between snapshots
- FeedbackCollector: Aggregated session feedback
"""

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ErrorCategory(Enum):
    """Categorized failure types for transformation errors."""
    
    # Node location errors
    NODE_NOT_FOUND = "node_not_found"
    INVALID_LOCATION = "invalid_location"
    AMBIGUOUS_LOCATION = "ambiguous_location"
    
    # Graph structure errors
    GRAPH_STRUCTURE_ERROR = "graph_structure"
    SHAPE_MISMATCH = "shape_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    
    # Handler errors
    UNSUPPORTED_PATTERN = "unsupported_pattern"
    HANDLER_NOT_IMPLEMENTED = "no_handler"
    HANDLER_FAILED = "handler_failed"
    
    # Validation errors
    VALIDATION_FAILED = "validation_failed"
    ONNX_CHECKER_FAILED = "onnx_checker_failed"
    
    # Rewiring errors
    REWIRING_FAILED = "rewiring_failed"
    MISSING_INPUT = "missing_input"
    MISSING_OUTPUT = "missing_output"
    
    # Other
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    
    @classmethod
    def from_exception(cls, e: Exception) -> "ErrorCategory":
        """Categorize an exception into an ErrorCategory."""
        error_msg = str(e).lower()
        
        if "not found" in error_msg or "does not exist" in error_msg:
            return cls.NODE_NOT_FOUND
        elif "shape" in error_msg:
            return cls.SHAPE_MISMATCH
        elif "type" in error_msg and "mismatch" in error_msg:
            return cls.TYPE_MISMATCH
        elif "circular" in error_msg or "cycle" in error_msg:
            return cls.CIRCULAR_DEPENDENCY
        elif "validation" in error_msg or "invalid" in error_msg:
            return cls.VALIDATION_FAILED
        elif "rewire" in error_msg or "connect" in error_msg:
            return cls.REWIRING_FAILED
        elif "input" in error_msg and ("missing" in error_msg or "not found" in error_msg):
            return cls.MISSING_INPUT
        elif "output" in error_msg and ("missing" in error_msg or "not found" in error_msg):
            return cls.MISSING_OUTPUT
        elif "timeout" in error_msg:
            return cls.TIMEOUT
        else:
            return cls.UNKNOWN


@dataclass
class GraphSnapshot:
    """
    Snapshot of model graph state.
    
    Captures key metrics for comparison before/after transformations.
    """
    
    node_count: int
    op_type_counts: Dict[str, int]
    node_names: Set[str]
    input_names: Set[str]
    output_names: Set[str]
    edge_count: int
    initializer_count: int
    timestamp: float = field(default_factory=time.time)
    
    @classmethod
    def capture(cls, model: Any) -> "GraphSnapshot":
        """Capture snapshot from an ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required for GraphSnapshot")
        
        nodes = list(model.graph.node)
        op_counts = Counter(n.op_type for n in nodes)
        node_names = {n.name for n in nodes}
        
        # Count edges (inputs to nodes)
        edge_count = sum(len(n.input) for n in nodes)
        
        # Get graph inputs and outputs
        input_names = {inp.name for inp in model.graph.input}
        output_names = {out.name for out in model.graph.output}
        
        return cls(
            node_count=len(nodes),
            op_type_counts=dict(op_counts),
            node_names=node_names,
            input_names=input_names,
            output_names=output_names,
            edge_count=edge_count,
            initializer_count=len(model.graph.initializer),
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "node_count": self.node_count,
            "op_type_counts": self.op_type_counts,
            "node_names": list(self.node_names),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "edge_count": self.edge_count,
            "initializer_count": self.initializer_count,
            "timestamp": self.timestamp,
        }


@dataclass
class TransformationDelta:
    """
    What changed between two graph snapshots.
    
    Provides detailed breakdown of modifications for ReAct observation.
    """
    
    nodes_added: Set[str]
    nodes_removed: Set[str]
    ops_added: Dict[str, int]
    ops_removed: Dict[str, int]
    node_count_delta: int
    edge_count_delta: int
    initializer_delta: int
    
    @classmethod
    def compute(
        cls,
        before: GraphSnapshot,
        after: Optional[GraphSnapshot]
    ) -> "TransformationDelta":
        """Compute delta between two snapshots."""
        if after is None:
            # No transformation occurred
            return cls(
                nodes_added=set(),
                nodes_removed=set(),
                ops_added={},
                ops_removed={},
                node_count_delta=0,
                edge_count_delta=0,
                initializer_delta=0,
            )
        
        nodes_added = after.node_names - before.node_names
        nodes_removed = before.node_names - after.node_names
        
        # Compute op type deltas
        ops_added = {}
        ops_removed = {}
        
        all_ops = set(before.op_type_counts.keys()) | set(after.op_type_counts.keys())
        for op in all_ops:
            before_count = before.op_type_counts.get(op, 0)
            after_count = after.op_type_counts.get(op, 0)
            if after_count > before_count:
                ops_added[op] = after_count - before_count
            elif before_count > after_count:
                ops_removed[op] = before_count - after_count
        
        return cls(
            nodes_added=nodes_added,
            nodes_removed=nodes_removed,
            ops_added=ops_added,
            ops_removed=ops_removed,
            node_count_delta=after.node_count - before.node_count,
            edge_count_delta=after.edge_count - before.edge_count,
            initializer_delta=after.initializer_count - before.initializer_count,
        )
    
    @property
    def has_changes(self) -> bool:
        """Check if any changes occurred."""
        return (
            len(self.nodes_added) > 0 or
            len(self.nodes_removed) > 0 or
            self.node_count_delta != 0 or
            self.edge_count_delta != 0
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "nodes_added": list(self.nodes_added),
            "nodes_removed": list(self.nodes_removed),
            "ops_added": self.ops_added,
            "ops_removed": self.ops_removed,
            "node_count_delta": self.node_count_delta,
            "edge_count_delta": self.edge_count_delta,
            "initializer_delta": self.initializer_delta,
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        parts = []
        if self.nodes_removed:
            parts.append(f"removed {len(self.nodes_removed)} nodes")
        if self.nodes_added:
            parts.append(f"added {len(self.nodes_added)} nodes")
        if self.ops_removed:
            ops_str = ", ".join(f"{k}:{v}" for k, v in self.ops_removed.items())
            parts.append(f"removed ops: {ops_str}")
        if self.ops_added:
            ops_str = ", ".join(f"{k}:{v}" for k, v in self.ops_added.items())
            parts.append(f"added ops: {ops_str}")
        
        return "; ".join(parts) if parts else "no changes"


@dataclass
class TransformationResult:
    """
    Per-suggestion transformation result.
    
    Captures comprehensive information about a single suggestion application
    for ReAct observation and feedback loops.
    """
    
    # Suggestion identification
    suggestion_id: int
    node_name: str
    op_type: str
    action_type: str  # "remove", "add", "replace", "reshape", "rewire"
    
    # Result status
    success: bool
    was_transformed: bool  # Model actually changed (not just attempted)
    
    # Error information (if failed)
    error_category: Optional[ErrorCategory] = None
    error_message: Optional[str] = None
    
    # Snapshots and delta
    before_snapshot: Optional[GraphSnapshot] = None
    after_snapshot: Optional[GraphSnapshot] = None
    transformation_delta: Optional[TransformationDelta] = None
    
    # Timing and retry info
    duration_ms: float = 0.0
    retry_count: int = 0
    
    # Additional context
    handler_name: Optional[str] = None
    validation_passed: Optional[bool] = None
    
    @property
    def effective(self) -> bool:
        """True if transformation succeeded AND made changes."""
        return self.success and self.was_transformed
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "suggestion_id": self.suggestion_id,
            "node_name": self.node_name,
            "op_type": self.op_type,
            "action_type": self.action_type,
            "success": self.success,
            "was_transformed": self.was_transformed,
            "error_category": self.error_category.value if self.error_category else None,
            "error_message": self.error_message,
            "before_snapshot": self.before_snapshot.to_dict() if self.before_snapshot else None,
            "after_snapshot": self.after_snapshot.to_dict() if self.after_snapshot else None,
            "transformation_delta": self.transformation_delta.to_dict() if self.transformation_delta else None,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "handler_name": self.handler_name,
            "validation_passed": self.validation_passed,
        }
    
    def to_observation_string(self) -> str:
        """Format as observation string for ReAct agent."""
        status = "SUCCESS" if self.success else "FAILED"
        transformed = "transformed" if self.was_transformed else "no changes"
        
        lines = [
            f"Suggestion {self.suggestion_id}: {status} ({transformed})",
            f"  Node: {self.node_name} ({self.op_type})",
            f"  Action: {self.action_type}",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        
        if self.error_category:
            lines.append(f"  Error: {self.error_category.value}")
            if self.error_message:
                lines.append(f"  Message: {self.error_message[:100]}")
        
        if self.transformation_delta and self.transformation_delta.has_changes:
            lines.append(f"  Changes: {self.transformation_delta}")
        
        return "\n".join(lines)


@dataclass
class FeedbackCollector:
    """
    Aggregated feedback from transformation session.
    
    Collects results from multiple transformations and provides
    computed statistics for ReAct agent observation and decision-making.
    """
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = ""
    results: List[TransformationResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    # Compatibility with existing SuggestionApplicator counters
    @property
    def applied_count(self) -> int:
        """Backward compatible: count of successfully applied suggestions."""
        return sum(1 for r in self.results if r.success)
    
    @property
    def failed_count(self) -> int:
        """Backward compatible: count of failed suggestions."""
        return sum(1 for r in self.results if not r.success)
    
    @property
    def transformed_count(self) -> int:
        """Backward compatible: count of suggestions that changed the model."""
        return sum(1 for r in self.results if r.was_transformed)
    
    @property
    def attempted_count(self) -> int:
        """Backward compatible: count of attempted suggestions."""
        return len(self.results)
    
    @property
    def skipped_count(self) -> int:
        """Backward compatible: count of skipped suggestions (success but no transform)."""
        return sum(1 for r in self.results if r.success and not r.was_transformed)
    
    def add(self, result: TransformationResult) -> None:
        """Add a transformation result."""
        self.results.append(result)
    
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)
    
    def transformation_rate(self) -> float:
        """Calculate rate of transformations that actually changed the model."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.was_transformed) / len(self.results)
    
    def effectiveness_rate(self) -> float:
        """Calculate rate of effective transformations (success + changed)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.effective) / len(self.results)
    
    def error_distribution(self) -> Dict[ErrorCategory, int]:
        """Get distribution of error categories."""
        dist: Dict[ErrorCategory, int] = {}
        for r in self.results:
            if r.error_category:
                dist[r.error_category] = dist.get(r.error_category, 0) + 1
        return dist
    
    def error_distribution_str(self) -> Dict[str, int]:
        """Get error distribution with string keys for serialization."""
        return {k.value: v for k, v in self.error_distribution().items()}
    
    def most_common_failures(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get most common failure error categories."""
        dist = self.error_distribution()
        sorted_errors = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return [(e.value, c) for e, c in sorted_errors[:n]]
    
    def effective_transformations(self) -> List[TransformationResult]:
        """Get list of effective transformations."""
        return [r for r in self.results if r.effective]
    
    def failed_transformations(self) -> List[TransformationResult]:
        """Get list of failed transformations."""
        return [r for r in self.results if not r.success]
    
    def suggestions_needing_retry(self) -> List[int]:
        """Get IDs of suggestions that might benefit from retry."""
        retry_categories = {
            ErrorCategory.TIMEOUT,
            ErrorCategory.VALIDATION_FAILED,
            ErrorCategory.HANDLER_FAILED,
        }
        return [
            r.suggestion_id for r in self.results
            if r.error_category in retry_categories
        ]
    
    def ops_removed(self) -> Dict[str, int]:
        """Aggregate of all operations removed across transformations."""
        total: Dict[str, int] = {}
        for r in self.results:
            if r.transformation_delta:
                for op, count in r.transformation_delta.ops_removed.items():
                    total[op] = total.get(op, 0) + count
        return total
    
    def ops_added(self) -> Dict[str, int]:
        """Aggregate of all operations added across transformations."""
        total: Dict[str, int] = {}
        for r in self.results:
            if r.transformation_delta:
                for op, count in r.transformation_delta.ops_added.items():
                    total[op] = total.get(op, 0) + count
        return total
    
    def total_nodes_removed(self) -> int:
        """Total nodes removed across all transformations."""
        return sum(
            len(r.transformation_delta.nodes_removed)
            for r in self.results
            if r.transformation_delta
        )
    
    def total_nodes_added(self) -> int:
        """Total nodes added across all transformations."""
        return sum(
            len(r.transformation_delta.nodes_added)
            for r in self.results
            if r.transformation_delta
        )
    
    def average_duration_ms(self) -> float:
        """Average transformation duration in milliseconds."""
        if not self.results:
            return 0.0
        return sum(r.duration_ms for r in self.results) / len(self.results)
    
    def total_duration_ms(self) -> float:
        """Total time spent on transformations."""
        return sum(r.duration_ms for r in self.results)
    
    def _format_failures(self, max_failures: int = 5) -> str:
        """Format failure details for observation string."""
        failures = self.failed_transformations()[:max_failures]
        if not failures:
            return "  None"
        
        lines = []
        for f in failures:
            error = f.error_category.value if f.error_category else "unknown"
            lines.append(f"  - {f.node_name} ({f.op_type}): {error}")
        
        if len(self.failed_transformations()) > max_failures:
            lines.append(f"  ... and {len(self.failed_transformations()) - max_failures} more")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on feedback patterns."""
        recommendations = []
        
        # Check error patterns
        error_dist = self.error_distribution()
        
        if error_dist.get(ErrorCategory.NODE_NOT_FOUND, 0) > 2:
            recommendations.append(
                "- Multiple nodes not found: Consider re-analyzing model or checking node name generation"
            )
        
        if error_dist.get(ErrorCategory.SHAPE_MISMATCH, 0) > 2:
            recommendations.append(
                "- Shape mismatches detected: May need shape inference or explicit reshape operations"
            )
        
        if error_dist.get(ErrorCategory.HANDLER_NOT_IMPLEMENTED, 0) > 0:
            recommendations.append(
                "- Missing handlers: Some operation types don't have transformation handlers"
            )
        
        if self.success_rate() < 0.3 and len(self.results) > 5:
            recommendations.append(
                "- Low success rate: Consider changing transformation strategy"
            )
        
        if self.transformation_rate() < self.success_rate() * 0.5:
            recommendations.append(
                "- Many no-op transformations: Suggestions may not match actual model structure"
            )
        
        if not recommendations:
            if self.effectiveness_rate() > 0.7:
                recommendations.append("- Good progress: Continue with current strategy")
            else:
                recommendations.append("- Consider reviewing suggestion quality")
        
        return "\n".join(recommendations)
    
    def to_observation_string(self) -> str:
        """
        Format feedback for ReAct agent consumption.
        
        This is the primary interface for ReAct observation loops.
        """
        elapsed = time.time() - self.start_time
        
        return f"""
=== Transformation Session Feedback ===
Session: {self.session_id} | Model: {self.model_name}
Elapsed: {elapsed:.1f}s | Avg Duration: {self.average_duration_ms():.1f}ms/suggestion

Results Summary:
- Success Rate: {self.success_rate():.1%}
- Transformation Rate: {self.transformation_rate():.1%}
- Effectiveness Rate: {self.effectiveness_rate():.1%}
- Attempted: {self.attempted_count} | Transformed: {self.transformed_count} | Failed: {self.failed_count}

Operations Changed:
- Removed: {self.ops_removed()}
- Added: {self.ops_added()}
- Net Node Change: {self.total_nodes_added() - self.total_nodes_removed()}

Error Distribution:
{self._format_error_distribution()}

Failed Transformations:
{self._format_failures()}

Recommendations:
{self._generate_recommendations()}
"""
    
    def _format_error_distribution(self) -> str:
        """Format error distribution for display."""
        dist = self.error_distribution_str()
        if not dist:
            return "  None"
        return "\n".join(f"  - {k}: {v}" for k, v in sorted(dist.items(), key=lambda x: -x[1]))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "results": [r.to_dict() for r in self.results],
            "start_time": self.start_time,
            "summary": {
                "success_rate": self.success_rate(),
                "transformation_rate": self.transformation_rate(),
                "effectiveness_rate": self.effectiveness_rate(),
                "attempted": self.attempted_count,
                "transformed": self.transformed_count,
                "failed": self.failed_count,
                "skipped": self.skipped_count,
                "total_duration_ms": self.total_duration_ms(),
                "error_distribution": self.error_distribution_str(),
                "ops_removed": self.ops_removed(),
                "ops_added": self.ops_added(),
            },
        }
    
    def reset(self) -> None:
        """Reset collector for new session."""
        self.session_id = str(uuid.uuid4())[:8]
        self.results = []
        self.start_time = time.time()
    
    def get_summary(self) -> Dict:
        """Get summary compatible with existing SuggestionApplicator interface."""
        return {
            "applied": self.applied_count,
            "failed": self.failed_count,
            "transformed": self.transformed_count,
            "attempted": self.attempted_count,
            "skipped": self.skipped_count,
            "success_rate": self.success_rate(),
            "transformation_rate": self.transformation_rate(),
        }
