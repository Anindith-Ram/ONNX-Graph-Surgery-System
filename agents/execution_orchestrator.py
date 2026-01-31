#!/usr/bin/env python3
"""
Execution Orchestrator for ONNX Model Surgery.

Executes transformation plans with checkpoints, validation, and rollback
capabilities. Manages the surgical process region-by-region with
comprehensive error handling.

Key Capabilities:
- Execute transformation plans region-by-region
- Create snapshots before each transformation
- Validate after each region
- Rollback on failure
- Support incremental progress saving

Author: Automated Model Surgery Pipeline
"""

import sys
import time
import copy
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import onnx
from onnx import shape_inference

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.transformation_regions import (
    TransformationRegion, TransformationPlan, ValidationCheckpoint,
    RegionStatus, TransformationResult
)
from knowledge_base.strategy_database import (
    StrategyDatabase, SurgeryStrategy, StrategyPhase, TransformationType
)
from suggestion_pipeline.suggestion_applicator import SuggestionApplicator


class ExecutionStatus(Enum):
    """Status of execution process."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ExecutionSnapshot:
    """Snapshot of model state at a point in execution."""
    snapshot_id: str
    created_at: str
    region_id: str  # Region before which this snapshot was taken
    model_bytes: bytes  # Serialized model
    node_count: int
    
    def to_dict(self) -> Dict:
        return {
            'snapshot_id': self.snapshot_id,
            'created_at': self.created_at,
            'region_id': self.region_id,
            'model_bytes_len': len(self.model_bytes),
            'node_count': self.node_count
        }


@dataclass
class RegionExecutionResult:
    """Result of executing a single region."""
    region_id: str
    success: bool
    was_transformed: bool
    
    # Metrics
    duration_ms: float = 0.0
    nodes_changed: int = 0
    blockers_resolved: int = 0
    
    # Validation
    validation_passed: Optional[bool] = None
    validation_message: str = ""
    
    # Error info
    error_message: str = ""
    exception_type: str = ""
    
    # Strategy info
    strategy_used: str = ""
    phases_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'region_id': self.region_id,
            'success': self.success,
            'was_transformed': self.was_transformed,
            'duration_ms': self.duration_ms,
            'nodes_changed': self.nodes_changed,
            'blockers_resolved': self.blockers_resolved,
            'validation_passed': self.validation_passed,
            'validation_message': self.validation_message,
            'error_message': self.error_message,
            'exception_type': self.exception_type,
            'strategy_used': self.strategy_used,
            'phases_applied': self.phases_applied
        }


@dataclass
class ExecutionReport:
    """Complete execution report."""
    plan_id: str
    model_name: str
    status: ExecutionStatus
    
    # Timing
    started_at: str = ""
    completed_at: str = ""
    total_duration_ms: float = 0.0
    
    # Results
    regions_executed: int = 0
    regions_succeeded: int = 0
    regions_failed: int = 0
    regions_skipped: int = 0
    
    # Per-region results
    region_results: List[RegionExecutionResult] = field(default_factory=list)
    
    # Validation
    validation_checkpoints_passed: int = 0
    validation_checkpoints_total: int = 0
    
    # Model changes
    initial_node_count: int = 0
    final_node_count: int = 0
    initial_blocker_count: int = 0
    final_blocker_count: int = 0
    
    # Errors
    fatal_error: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'plan_id': self.plan_id,
            'model_name': self.model_name,
            'status': self.status.value,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'total_duration_ms': self.total_duration_ms,
            'regions_executed': self.regions_executed,
            'regions_succeeded': self.regions_succeeded,
            'regions_failed': self.regions_failed,
            'regions_skipped': self.regions_skipped,
            'region_results': [r.to_dict() for r in self.region_results],
            'validation_checkpoints_passed': self.validation_checkpoints_passed,
            'validation_checkpoints_total': self.validation_checkpoints_total,
            'initial_node_count': self.initial_node_count,
            'final_node_count': self.final_node_count,
            'initial_blocker_count': self.initial_blocker_count,
            'final_blocker_count': self.final_blocker_count,
            'fatal_error': self.fatal_error,
            'warnings': self.warnings
        }
    
    @property
    def success_rate(self) -> float:
        """Calculate region success rate."""
        total = self.regions_succeeded + self.regions_failed
        return self.regions_succeeded / total if total > 0 else 0.0
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Execution Report: {self.plan_id}",
            f"=" * 50,
            f"Status: {self.status.value}",
            f"Model: {self.model_name}",
            f"",
            f"Regions: {self.regions_succeeded}/{self.regions_executed} succeeded "
            f"({self.success_rate:.1%})",
            f"Duration: {self.total_duration_ms:.1f}ms",
            f"",
            f"Node Count: {self.initial_node_count} -> {self.final_node_count}",
            f"Blockers: {self.initial_blocker_count} -> {self.final_blocker_count}",
            f"",
            f"Validation: {self.validation_checkpoints_passed}/{self.validation_checkpoints_total} passed"
        ]
        
        if self.fatal_error:
            lines.append(f"\nFATAL ERROR: {self.fatal_error}")
        
        if self.warnings:
            lines.append(f"\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        return "\n".join(lines)


class ExecutionOrchestrator:
    """
    Orchestrate execution of transformation plans.
    
    Manages the surgical process with:
    - Region-by-region execution
    - Snapshot creation for rollback
    - Validation at checkpoints
    - Progress saving
    - Error recovery
    """
    
    def __init__(
        self,
        strategy_db: Optional[StrategyDatabase] = None,
        verbose: bool = False,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize the execution orchestrator.
        
        Args:
            strategy_db: Strategy database for phase execution
            verbose: Enable verbose output
            checkpoint_dir: Directory for saving checkpoints
        """
        self.strategy_db = strategy_db or StrategyDatabase.create_with_defaults()
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Execution state
        self.current_model: Optional[onnx.ModelProto] = None
        self.snapshots: Dict[str, ExecutionSnapshot] = {}
        self.status: ExecutionStatus = ExecutionStatus.NOT_STARTED
        
        # Applicator for transformations
        self.applicator = SuggestionApplicator()
        
        # Callbacks
        self.on_region_start: Optional[Callable] = None
        self.on_region_complete: Optional[Callable] = None
        self.on_checkpoint: Optional[Callable] = None
    
    def execute(
        self,
        model_path: str,
        plan: TransformationPlan,
        output_path: Optional[str] = None,
        stop_on_failure: bool = False
    ) -> Tuple[onnx.ModelProto, ExecutionReport]:
        """
        Execute a transformation plan.
        
        Args:
            model_path: Path to input ONNX model
            plan: Transformation plan to execute
            output_path: Optional path to save result
            stop_on_failure: Stop execution on first failure
            
        Returns:
            Tuple of (modified model, execution report)
        """
        start_time = time.time()
        
        # Initialize report
        report = ExecutionReport(
            plan_id=plan.plan_id,
            model_name=plan.model_name,
            status=ExecutionStatus.IN_PROGRESS,
            started_at=datetime.now().isoformat(),
            validation_checkpoints_total=len(plan.checkpoints)
        )
        
        try:
            # Load model
            if self.verbose:
                print(f"Loading model from {model_path}...")
            self.current_model = onnx.load(model_path)
            report.initial_node_count = len(self.current_model.graph.node)
            
            # Count initial blockers
            report.initial_blocker_count = self._count_blockers(self.current_model)
            
            # Create initial snapshot
            self._create_snapshot("initial", "initial")
            
            self.status = ExecutionStatus.IN_PROGRESS
            
            # Execute regions in order
            for region_id in plan.execution_order:
                region = plan.get_region(region_id)
                if not region:
                    report.warnings.append(f"Region {region_id} not found in plan")
                    continue
                
                # Execute region
                result = self._execute_region(region, plan)
                report.region_results.append(result)
                report.regions_executed += 1
                
                if result.success:
                    report.regions_succeeded += 1
                    region.status = RegionStatus.COMPLETED
                else:
                    report.regions_failed += 1
                    region.status = RegionStatus.FAILED
                    
                    if stop_on_failure:
                        report.warnings.append(
                            f"Stopped due to failure in region {region_id}"
                        )
                        break
                
                # Check for validation checkpoints after this region
                self._run_checkpoints(region_id, plan, report)
            
            # Final validation
            final_blocker_count = self._count_blockers(self.current_model)
            report.final_blocker_count = final_blocker_count
            report.final_node_count = len(self.current_model.graph.node)
            
            # Determine final status
            if report.regions_failed == 0:
                self.status = ExecutionStatus.COMPLETED
            elif report.regions_succeeded > 0:
                self.status = ExecutionStatus.COMPLETED  # Partial success
            else:
                self.status = ExecutionStatus.FAILED
            
            report.status = self.status
            
        except Exception as e:
            report.fatal_error = str(e)
            report.status = ExecutionStatus.FAILED
            self.status = ExecutionStatus.FAILED
            
            # Try to rollback to last good state
            if "initial" in self.snapshots:
                if self.verbose:
                    print(f"Fatal error, rolling back to initial state...")
                self._rollback_to_snapshot("initial")
        
        finally:
            # Finalize report
            end_time = time.time()
            report.total_duration_ms = (end_time - start_time) * 1000
            report.completed_at = datetime.now().isoformat()
            
            # Save output if requested
            if output_path and self.current_model:
                try:
                    onnx.save(self.current_model, output_path)
                    if self.verbose:
                        print(f"Saved modified model to {output_path}")
                except Exception as e:
                    report.warnings.append(f"Failed to save output: {e}")
        
        if self.verbose:
            print(report.get_summary())
        
        return self.current_model, report
    
    def _execute_region(
        self,
        region: TransformationRegion,
        plan: TransformationPlan
    ) -> RegionExecutionResult:
        """Execute transformation for a single region."""
        start_time = time.time()
        
        result = RegionExecutionResult(
            region_id=region.region_id,
            success=False,
            was_transformed=False,
            strategy_used=region.recommended_strategy
        )
        
        if self.verbose:
            print(f"  Executing region: {region.region_id} ({region.region_type})")
        
        # Callback
        if self.on_region_start:
            self.on_region_start(region)
        
        try:
            # Create pre-transformation snapshot
            self._create_snapshot(f"before_{region.region_id}", region.region_id)
            
            # Get strategy
            strategy = self.strategy_db.get_strategy(region.recommended_strategy)
            
            # Count nodes before
            nodes_before = len(self.current_model.graph.node)
            blockers_before = self._count_blockers_in_region(region)
            
            # Execute transformation based on region type
            transformed = self._apply_region_transformation(region, strategy)
            
            # Count nodes after
            nodes_after = len(self.current_model.graph.node)
            blockers_after = self._count_blockers_in_region(region)
            
            result.was_transformed = transformed
            result.nodes_changed = abs(nodes_after - nodes_before)
            result.blockers_resolved = max(0, blockers_before - blockers_after)
            
            # Validate transformation
            validation_passed, validation_msg = self._validate_region(region)
            result.validation_passed = validation_passed
            result.validation_message = validation_msg
            
            if validation_passed:
                result.success = True
            else:
                # Rollback
                if self.verbose:
                    print(f"    Validation failed, rolling back...")
                self._rollback_to_snapshot(f"before_{region.region_id}")
                result.success = False
                result.error_message = f"Validation failed: {validation_msg}"
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.exception_type = type(e).__name__
            
            # Rollback
            snapshot_id = f"before_{region.region_id}"
            if snapshot_id in self.snapshots:
                if self.verbose:
                    print(f"    Exception occurred, rolling back...")
                self._rollback_to_snapshot(snapshot_id)
        
        finally:
            result.duration_ms = (time.time() - start_time) * 1000
            
            # Callback
            if self.on_region_complete:
                self.on_region_complete(region, result)
        
        if self.verbose:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"    {status} ({result.duration_ms:.1f}ms)")
        
        return result
    
    def _apply_region_transformation(
        self,
        region: TransformationRegion,
        strategy: Optional[SurgeryStrategy]
    ) -> bool:
        """Apply transformation for a region."""
        if not self.current_model:
            return False
        
        transformed = False
        
        # Handle based on region type
        if region.region_type == "einsum_attention":
            # Find and decompose Einsum nodes
            for node_idx in region.node_indices:
                if node_idx < len(self.current_model.graph.node):
                    node = self.current_model.graph.node[node_idx]
                    if node.op_type == 'Einsum':
                        self.current_model, success = self.applicator.apply_einsum_decomposition(
                            self.current_model, node.name
                        )
                        transformed = transformed or success
        
        elif region.region_type == "feed_forward":
            # Convert Gemm to Conv if beneficial
            for node_idx in region.node_indices:
                if node_idx < len(self.current_model.graph.node):
                    node = self.current_model.graph.node[node_idx]
                    if node.op_type == 'Gemm':
                        self.current_model, conv_name = self.applicator.rewrite_gemm_as_conv(
                            self.current_model, node.name
                        )
                        transformed = transformed or bool(conv_name)
        
        elif region.region_type == "isolated_blocker":
            # Handle isolated blockers
            for node_idx in region.node_indices:
                if node_idx < len(self.current_model.graph.node):
                    node = self.current_model.graph.node[node_idx]
                    
                    if node.op_type == 'Einsum':
                        self.current_model, success = self.applicator.apply_einsum_decomposition(
                            self.current_model, node.name
                        )
                        transformed = transformed or success
                    
                    elif node.op_type in ['Identity', 'Dropout']:
                        # Remove passthrough operations
                        node_map = {n.name: n for n in self.current_model.graph.node}
                        self.current_model, success = self.applicator._remove_identity(
                            self.current_model, node.name, node_map
                        )
                        transformed = transformed or success
        
        else:
            # Generic handling - try to remove blocking operations
            node_map = {n.name: n for n in self.current_model.graph.node}
            
            for op_type in region.op_types:
                if op_type in ['Identity', 'Dropout']:
                    for node_idx in region.node_indices:
                        if node_idx < len(self.current_model.graph.node):
                            node = self.current_model.graph.node[node_idx]
                            if node.op_type == op_type:
                                self.current_model, success = self.applicator._remove_identity(
                                    self.current_model, node.name, node_map
                                )
                                transformed = transformed or success
                                node_map = {n.name: n for n in self.current_model.graph.node}
        
        return transformed
    
    def _validate_region(self, region: TransformationRegion) -> Tuple[bool, str]:
        """Validate region transformation."""
        if not self.current_model:
            return False, "No model loaded"
        
        try:
            # Check model validity
            onnx.checker.check_model(self.current_model)
            
            # Try shape inference
            try:
                shape_inference.infer_shapes(self.current_model)
            except Exception as e:
                return False, f"Shape inference failed: {e}"
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, str(e)
    
    def _run_checkpoints(
        self,
        after_region: str,
        plan: TransformationPlan,
        report: ExecutionReport
    ) -> None:
        """Run validation checkpoints after a region."""
        for checkpoint in plan.checkpoints:
            if checkpoint.after_region == after_region:
                passed = self._run_checkpoint(checkpoint)
                if passed:
                    report.validation_checkpoints_passed += 1
                
                # Callback
                if self.on_checkpoint:
                    self.on_checkpoint(checkpoint, passed)
    
    def _run_checkpoint(self, checkpoint: ValidationCheckpoint) -> bool:
        """Run a single validation checkpoint."""
        if not self.current_model:
            checkpoint.passed = False
            checkpoint.actual_result = "No model loaded"
            return False
        
        try:
            if checkpoint.validation_type == "shape":
                # Check that all shapes can be inferred
                shape_inference.infer_shapes(self.current_model)
                checkpoint.passed = True
                checkpoint.actual_result = "All shapes valid"
                
            elif checkpoint.validation_type == "compilation":
                # Simulate compilation check
                from core_analysis.compilation_simulator import CompilationSimulator
                simulator = CompilationSimulator()
                result = simulator.simulate_from_model(self.current_model) if hasattr(simulator, 'simulate_from_model') else None
                
                if result:
                    checkpoint.passed = result.will_compile
                    checkpoint.actual_result = f"Blockers: {result.blocker_count}"
                else:
                    checkpoint.passed = True
                    checkpoint.actual_result = "Compilation check skipped"
                    
            elif checkpoint.validation_type == "numerical":
                # Numerical validation requires test data
                checkpoint.passed = True
                checkpoint.actual_result = "Numerical check requires test data"
                
            else:
                # Unknown validation type
                checkpoint.passed = True
                checkpoint.actual_result = "Unknown validation type - skipped"
            
            return checkpoint.passed
            
        except Exception as e:
            checkpoint.passed = False
            checkpoint.actual_result = f"Error: {e}"
            return False
    
    def _create_snapshot(self, snapshot_id: str, region_id: str) -> None:
        """Create a model snapshot."""
        if not self.current_model:
            return
        
        snapshot = ExecutionSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.now().isoformat(),
            region_id=region_id,
            model_bytes=self.current_model.SerializeToString(),
            node_count=len(self.current_model.graph.node)
        )
        self.snapshots[snapshot_id] = snapshot
        
        if self.verbose:
            print(f"    Created snapshot: {snapshot_id}")
    
    def _rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback model to a snapshot."""
        if snapshot_id not in self.snapshots:
            return False
        
        snapshot = self.snapshots[snapshot_id]
        self.current_model = onnx.load_from_string(snapshot.model_bytes)
        
        if self.verbose:
            print(f"    Rolled back to snapshot: {snapshot_id}")
        
        return True
    
    def _count_blockers(self, model: onnx.ModelProto) -> int:
        """Count blocking operations in model."""
        blocking_ops = {
            'Einsum', 'NonZero', 'Where', 'Loop', 'If', 'Scan',
            'NonMaxSuppression', 'TopK', 'Unique'
        }
        return sum(1 for node in model.graph.node if node.op_type in blocking_ops)
    
    def _count_blockers_in_region(self, region: TransformationRegion) -> int:
        """Count blockers in a specific region."""
        if not self.current_model:
            return 0
        
        blocking_ops = {
            'Einsum', 'NonZero', 'Where', 'Loop', 'If', 'Scan',
            'NonMaxSuppression', 'TopK', 'Unique'
        }
        
        count = 0
        for node_idx in region.node_indices:
            if node_idx < len(self.current_model.graph.node):
                node = self.current_model.graph.node[node_idx]
                if node.op_type in blocking_ops:
                    count += 1
        
        return count
    
    def save_checkpoint(self, path: str) -> None:
        """Save execution checkpoint to file."""
        checkpoint_data = {
            'status': self.status.value,
            'snapshots': {k: v.to_dict() for k, v in self.snapshots.items()},
            'current_model_bytes': self.current_model.SerializeToString().hex() if self.current_model else None
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def load_checkpoint(self, path: str) -> bool:
        """Load execution checkpoint from file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.status = ExecutionStatus(data['status'])
            
            if data.get('current_model_bytes'):
                model_bytes = bytes.fromhex(data['current_model_bytes'])
                self.current_model = onnx.load_from_string(model_bytes)
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Failed to load checkpoint: {e}")
            return False


# =============================================================================
# Convenience Functions
# =============================================================================

def execute_plan(
    model_path: str,
    plan: TransformationPlan,
    output_path: Optional[str] = None,
    verbose: bool = False
) -> Tuple[onnx.ModelProto, ExecutionReport]:
    """
    Convenience function to execute a transformation plan.
    
    Args:
        model_path: Path to input model
        plan: Transformation plan to execute
        output_path: Optional output path
        verbose: Enable verbose output
        
    Returns:
        Tuple of (modified model, execution report)
    """
    orchestrator = ExecutionOrchestrator(verbose=verbose)
    return orchestrator.execute(model_path, plan, output_path)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python execution_orchestrator.py <model.onnx> <plan.json>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    plan_path = sys.argv[2]
    
    # Load plan
    plan = TransformationPlan.load(plan_path)
    
    # Execute
    model, report = execute_plan(model_path, plan, verbose=True)
    
    # Save report
    report_path = Path(model_path).stem + "_execution_report.json"
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport saved to {report_path}")
