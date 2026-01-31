#!/usr/bin/env python3
"""
Strategic Planner for ONNX Model Surgery.

Generates comprehensive transformation plans (not just suggestion lists).
This planner operates at the architectural level, deciding whether to use
divide-and-conquer, identifying transformation regions, and ordering
transformations based on dependencies.

Key Capabilities:
- Decide whether to split model (divide-and-conquer)
- Identify coherent transformation regions
- Select appropriate strategy from strategy database
- Order transformations by dependencies
- Generate fallback strategies

Author: Automated Model Surgery Pipeline
"""

import sys
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_analysis.architecture_analyzer import (
    ArchitectureAnalyzer, ModelArchitecture, ArchitecturalBlock, BlockType, ArchitectureType
)
from core_analysis.compilation_simulator import (
    CompilationSimulator, CompilationReport, NodeCompilationResult
)
from knowledge_base.strategy_database import (
    StrategyDatabase, SurgeryStrategy, StrategyPhase, TransformationType
)


class PlanningMode(Enum):
    """Mode for transformation planning."""
    IN_PLACE = "in_place"           # Transform model without splitting
    DIVIDE_AND_CONQUER = "divide_and_conquer"  # Split, transform parts, merge
    HYBRID = "hybrid"                # Combination based on regions


@dataclass
class ValidationCheckpoint:
    """Checkpoint for validating transformation progress."""
    checkpoint_id: str
    name: str
    after_phase: str  # Phase ID
    validation_type: str  # "shape", "numerical", "compilation"
    expected_result: str
    tolerance: float = 1e-6
    
    def to_dict(self) -> Dict:
        return {
            'checkpoint_id': self.checkpoint_id,
            'name': self.name,
            'after_phase': self.after_phase,
            'validation_type': self.validation_type,
            'expected_result': self.expected_result,
            'tolerance': self.tolerance
        }


@dataclass
class TransformationRegion:
    """
    A coherent region of the model to transform together.
    
    Regions represent architectural blocks that should be transformed
    as a unit (e.g., all attention blocks, detection heads).
    """
    region_id: str
    region_type: str  # "attention_block", "detection_head", "normalization"
    
    # Structural info
    start_node_idx: int
    end_node_idx: int
    node_indices: List[int]
    op_types: List[str]
    
    # Purpose and issues
    original_purpose: str
    architectural_issue: str
    
    # Strategy
    transformation_strategy: str  # Strategy ID to use
    transformation_phases: List[str]  # Phase IDs in order
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Other region IDs
    required_by: List[str] = field(default_factory=list)  # Regions that depend on this
    
    # Status
    has_blockers: bool = False
    blocker_count: int = 0
    estimated_difficulty: str = "medium"  # "easy", "medium", "hard"
    
    def to_dict(self) -> Dict:
        return {
            'region_id': self.region_id,
            'region_type': self.region_type,
            'start_node_idx': self.start_node_idx,
            'end_node_idx': self.end_node_idx,
            'node_indices': self.node_indices,
            'op_types': self.op_types,
            'original_purpose': self.original_purpose,
            'architectural_issue': self.architectural_issue,
            'transformation_strategy': self.transformation_strategy,
            'transformation_phases': self.transformation_phases,
            'depends_on': self.depends_on,
            'required_by': self.required_by,
            'has_blockers': self.has_blockers,
            'blocker_count': self.blocker_count,
            'estimated_difficulty': self.estimated_difficulty
        }


@dataclass
class SplitPoint:
    """Point where model should be split for divide-and-conquer."""
    split_id: str
    name: str
    node_idx: int
    tensor_name: str
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            'split_id': self.split_id,
            'name': self.name,
            'node_idx': self.node_idx,
            'tensor_name': self.tensor_name,
            'reason': self.reason
        }


@dataclass
class TransformationPlan:
    """
    Complete transformation plan for a model.
    
    This is the output of strategic planning, containing all information
    needed to execute transformations systematically.
    """
    plan_id: str
    model_name: str
    architecture_type: str
    
    # Planning mode
    mode: PlanningMode
    
    # Split points (if divide-and-conquer)
    split_points: List[SplitPoint] = field(default_factory=list)
    
    # Regions to transform
    regions: List[TransformationRegion] = field(default_factory=list)
    
    # Execution order
    execution_order: List[str] = field(default_factory=list)  # Region IDs in order
    
    # Selected strategy
    strategy_id: str = ""
    strategy_name: str = ""
    
    # Fallback strategy
    fallback_strategy_id: Optional[str] = None
    
    # Validation checkpoints
    checkpoints: List[ValidationCheckpoint] = field(default_factory=list)
    
    # Expected outcomes
    expected_tolerance: float = 1e-4
    expected_lm_files: int = 1
    
    # Metadata
    total_phases: int = 0
    estimated_complexity: str = "medium"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'plan_id': self.plan_id,
            'model_name': self.model_name,
            'architecture_type': self.architecture_type,
            'mode': self.mode.value,
            'split_points': [s.to_dict() for s in self.split_points],
            'regions': [r.to_dict() for r in self.regions],
            'execution_order': self.execution_order,
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'fallback_strategy_id': self.fallback_strategy_id,
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'expected_tolerance': self.expected_tolerance,
            'expected_lm_files': self.expected_lm_files,
            'total_phases': self.total_phases,
            'estimated_complexity': self.estimated_complexity,
            'confidence': self.confidence,
            'warnings': self.warnings
        }
    
    def get_region(self, region_id: str) -> Optional[TransformationRegion]:
        """Get region by ID."""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None
    
    def get_summary(self) -> str:
        """Get human-readable plan summary."""
        lines = [
            f"Transformation Plan: {self.plan_id}",
            f"=" * 50,
            f"Model: {self.model_name}",
            f"Architecture: {self.architecture_type}",
            f"Mode: {self.mode.value}",
            f"Strategy: {self.strategy_name}",
            f"Confidence: {self.confidence:.2f}",
            "",
            f"Regions: {len(self.regions)}",
            f"Total Phases: {self.total_phases}",
            f"Checkpoints: {len(self.checkpoints)}",
        ]
        
        if self.split_points:
            lines.append(f"\nSplit Points: {len(self.split_points)}")
            for sp in self.split_points:
                lines.append(f"  - {sp.name}: node {sp.node_idx}")
        
        lines.append(f"\nExecution Order:")
        for i, region_id in enumerate(self.execution_order, 1):
            region = self.get_region(region_id)
            if region:
                lines.append(f"  {i}. {region.region_type} ({len(region.node_indices)} nodes)")
        
        if self.warnings:
            lines.append(f"\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        return "\n".join(lines)


class StrategicPlanner:
    """
    Strategic planner for ONNX model surgery.
    
    This planner analyzes models at the architectural level and generates
    comprehensive transformation plans, not just lists of suggestions.
    """
    
    def __init__(
        self,
        strategy_db: Optional[StrategyDatabase] = None,
        verbose: bool = False
    ):
        """
        Initialize the strategic planner.
        
        Args:
            strategy_db: Strategy database (created with defaults if None)
            verbose: Enable verbose output
        """
        self.strategy_db = strategy_db or StrategyDatabase.create_with_defaults()
        self.verbose = verbose
        
        # Analysis components
        self.arch_analyzer = ArchitectureAnalyzer()
        self.compilation_sim = CompilationSimulator(verbose=verbose)
        
        # Plan counter for unique IDs
        self._plan_counter = 0
    
    def create_plan(
        self,
        model_path: str,
        force_mode: Optional[PlanningMode] = None
    ) -> TransformationPlan:
        """
        Create a comprehensive transformation plan for a model.
        
        Args:
            model_path: Path to ONNX model
            force_mode: Force a specific planning mode (optional)
            
        Returns:
            TransformationPlan with all transformation details
        """
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter}"
        model_name = Path(model_path).stem
        
        if self.verbose:
            print(f"Creating transformation plan for {model_name}...")
        
        # Step 1: Analyze architecture
        if self.verbose:
            print("  Step 1: Analyzing architecture...")
        architecture = self.arch_analyzer.analyze(model_path)
        
        # Step 2: Simulate compilation
        if self.verbose:
            print("  Step 2: Simulating compilation...")
        compilation = self.compilation_sim.simulate(model_path)
        
        # Step 3: Decide planning mode
        if force_mode:
            mode = force_mode
        else:
            mode = self._decide_planning_mode(architecture, compilation)
        
        if self.verbose:
            print(f"  Step 3: Planning mode: {mode.value}")
        
        # Step 4: Select strategy
        strategy = self._select_strategy(architecture, compilation)
        
        if self.verbose:
            print(f"  Step 4: Selected strategy: {strategy.name if strategy else 'None'}")
        
        # Step 5: Identify transformation regions
        if self.verbose:
            print("  Step 5: Identifying transformation regions...")
        regions = self._identify_regions(architecture, compilation, strategy)
        
        # Step 6: Determine split points (if divide-and-conquer)
        split_points = []
        if mode == PlanningMode.DIVIDE_AND_CONQUER:
            if self.verbose:
                print("  Step 6: Identifying split points...")
            split_points = self._identify_split_points(architecture, regions)
        
        # Step 7: Order regions by dependencies
        if self.verbose:
            print("  Step 7: Ordering regions by dependencies...")
        execution_order = self._compute_execution_order(regions)
        
        # Step 8: Create validation checkpoints
        if self.verbose:
            print("  Step 8: Creating validation checkpoints...")
        checkpoints = self._create_checkpoints(regions, strategy)
        
        # Step 9: Estimate outcomes
        expected_tolerance = strategy.expected_accuracy_tolerance if strategy else 1e-4
        expected_lm_files = 1 if compilation.will_compile else compilation.predicted_lm_files
        
        # Compute total phases
        total_phases = sum(len(r.transformation_phases) for r in regions)
        
        # Estimate complexity
        complexity = self._estimate_complexity(regions, compilation)
        
        # Compute confidence
        confidence = self._compute_confidence(strategy, regions, compilation)
        
        # Generate warnings
        warnings = self._generate_warnings(architecture, compilation, regions)
        
        # Create plan
        plan = TransformationPlan(
            plan_id=plan_id,
            model_name=model_name,
            architecture_type=architecture.architecture_type.value,
            mode=mode,
            split_points=split_points,
            regions=regions,
            execution_order=execution_order,
            strategy_id=strategy.strategy_id if strategy else "",
            strategy_name=strategy.name if strategy else "Generic",
            fallback_strategy_id=strategy.fallback_strategy_id if strategy else None,
            checkpoints=checkpoints,
            expected_tolerance=expected_tolerance,
            expected_lm_files=expected_lm_files,
            total_phases=total_phases,
            estimated_complexity=complexity,
            confidence=confidence,
            warnings=warnings
        )
        
        if self.verbose:
            print(f"\n{plan.get_summary()}")
        
        return plan
    
    def _decide_planning_mode(
        self,
        architecture: ModelArchitecture,
        compilation: CompilationReport
    ) -> PlanningMode:
        """
        Decide which planning mode to use.
        
        Divide-and-conquer is recommended when:
        - Blockers are spread throughout the model
        - Architecture analysis recommends it
        - Model is very large
        """
        # Use architecture recommendation
        if architecture.divide_and_conquer_recommended:
            return PlanningMode.DIVIDE_AND_CONQUER
        
        # Check blocker distribution
        if compilation.blocker_count > 0:
            blocker_positions = [
                n.node_index / compilation.total_nodes 
                for n in compilation.blocker_nodes
            ]
            if blocker_positions:
                spread = max(blocker_positions) - min(blocker_positions)
                if spread > 0.5 and len(blocker_positions) > 5:
                    return PlanningMode.DIVIDE_AND_CONQUER
        
        # Check model size
        if compilation.total_nodes > 500:
            return PlanningMode.HYBRID
        
        return PlanningMode.IN_PLACE
    
    def _select_strategy(
        self,
        architecture: ModelArchitecture,
        compilation: CompilationReport
    ) -> Optional[SurgeryStrategy]:
        """Select the best strategy for the model."""
        # Extract blocker ops
        blocker_ops = list(compilation.blocker_ops.keys())
        
        # Extract detected patterns
        detected_patterns = []
        for block in architecture.blocks:
            if block.block_type == BlockType.EINSUM_ATTENTION:
                detected_patterns.append("einsum_attention")
            elif block.block_type == BlockType.ATTENTION:
                detected_patterns.append("standard_attention")
            elif block.block_type == BlockType.DETECTION_HEAD:
                detected_patterns.append("detection_head")
            elif block.block_type == BlockType.FEED_FORWARD:
                detected_patterns.append("feed_forward")
        
        # Find best strategy
        strategy = self.strategy_db.find_best_strategy(
            architecture=architecture.architecture_type.value,
            detected_patterns=detected_patterns,
            blocker_ops=blocker_ops
        )
        
        return strategy
    
    def _identify_regions(
        self,
        architecture: ModelArchitecture,
        compilation: CompilationReport,
        strategy: Optional[SurgeryStrategy]
    ) -> List[TransformationRegion]:
        """Identify coherent transformation regions."""
        regions = []
        region_counter = 0
        
        # Create regions from architectural blocks with blockers
        for block in architecture.blocks:
            if block.has_blockers:
                region_counter += 1
                
                # Determine transformation strategy and phases
                trans_strategy = ""
                trans_phases = []
                
                if strategy:
                    # Map block type to strategy phases
                    if block.block_type == BlockType.EINSUM_ATTENTION:
                        trans_strategy = strategy.strategy_id
                        trans_phases = [p.phase_id for p in strategy.phases 
                                       if "einsum" in p.target_pattern.lower()]
                    elif block.block_type == BlockType.FEED_FORWARD:
                        trans_strategy = strategy.strategy_id
                        trans_phases = [p.phase_id for p in strategy.phases
                                       if "gemm" in p.target_pattern.lower() or "ffn" in p.target_pattern.lower()]
                    else:
                        trans_strategy = strategy.strategy_id
                        trans_phases = [strategy.phases[0].phase_id] if strategy.phases else []
                
                region = TransformationRegion(
                    region_id=f"region_{region_counter}",
                    region_type=block.block_type.value,
                    start_node_idx=block.start_node_id,
                    end_node_idx=block.end_node_id,
                    node_indices=block.node_ids,
                    op_types=block.op_sequence,
                    original_purpose=block.description,
                    architectural_issue="; ".join(block.blocker_reasons),
                    transformation_strategy=trans_strategy,
                    transformation_phases=trans_phases,
                    has_blockers=True,
                    blocker_count=len(block.blocker_nodes),
                    estimated_difficulty=self._estimate_block_difficulty(block)
                )
                regions.append(region)
        
        # Add regions for isolated blockers not in architectural blocks
        covered_nodes = set()
        for region in regions:
            covered_nodes.update(region.node_indices)
        
        for blocker in compilation.blocker_nodes:
            if blocker.node_index not in covered_nodes:
                region_counter += 1
                region = TransformationRegion(
                    region_id=f"region_{region_counter}",
                    region_type="isolated_blocker",
                    start_node_idx=blocker.node_index,
                    end_node_idx=blocker.node_index,
                    node_indices=[blocker.node_index],
                    op_types=[blocker.op_type],
                    original_purpose=f"Isolated {blocker.op_type} operation",
                    architectural_issue="; ".join(blocker.blocker_reasons),
                    transformation_strategy=strategy.strategy_id if strategy else "",
                    transformation_phases=[],
                    has_blockers=True,
                    blocker_count=1,
                    estimated_difficulty="easy" if blocker.op_type in ["Identity", "Dropout"] else "medium"
                )
                regions.append(region)
                covered_nodes.add(blocker.node_index)
        
        # Compute dependencies between regions
        self._compute_region_dependencies(regions)
        
        return regions
    
    def _estimate_block_difficulty(self, block: ArchitecturalBlock) -> str:
        """Estimate difficulty of transforming a block."""
        if block.block_type in [BlockType.EINSUM_ATTENTION, BlockType.ATTENTION]:
            return "hard"
        elif block.block_type in [BlockType.DETECTION_HEAD]:
            return "medium"
        elif len(block.blocker_nodes) > 3:
            return "hard"
        elif len(block.blocker_nodes) > 1:
            return "medium"
        return "easy"
    
    def _compute_region_dependencies(self, regions: List[TransformationRegion]) -> None:
        """Compute dependencies between regions based on node positions."""
        # Sort by start position
        sorted_regions = sorted(regions, key=lambda r: r.start_node_idx)
        
        for i, region in enumerate(sorted_regions):
            # Regions that start before this one may produce inputs
            for j in range(i):
                other = sorted_regions[j]
                # If other ends close to where this starts, there may be a dependency
                if other.end_node_idx < region.start_node_idx:
                    # Check if there's a gap
                    gap = region.start_node_idx - other.end_node_idx
                    if gap < 10:  # Close enough to likely be connected
                        region.depends_on.append(other.region_id)
                        other.required_by.append(region.region_id)
    
    def _identify_split_points(
        self,
        architecture: ModelArchitecture,
        regions: List[TransformationRegion]
    ) -> List[SplitPoint]:
        """Identify good points to split the model."""
        split_points = []
        
        # Use architecture-suggested split points
        if architecture.suggested_split_points:
            for i, name in enumerate(architecture.suggested_split_points):
                split_points.append(SplitPoint(
                    split_id=f"split_{i}",
                    name=name,
                    node_idx=0,  # Would need to map to actual node
                    tensor_name="",
                    reason=f"Architecture-suggested split point: {name}"
                ))
        
        # Add split points between regions with large gaps
        sorted_regions = sorted(regions, key=lambda r: r.start_node_idx)
        for i in range(len(sorted_regions) - 1):
            current = sorted_regions[i]
            next_region = sorted_regions[i + 1]
            gap = next_region.start_node_idx - current.end_node_idx
            
            if gap > 20:  # Significant gap
                split_points.append(SplitPoint(
                    split_id=f"split_gap_{i}",
                    name=f"Between {current.region_type} and {next_region.region_type}",
                    node_idx=current.end_node_idx + gap // 2,
                    tensor_name="",
                    reason=f"Large gap ({gap} nodes) between regions"
                ))
        
        return split_points
    
    def _compute_execution_order(
        self,
        regions: List[TransformationRegion]
    ) -> List[str]:
        """Compute execution order respecting dependencies."""
        # Topological sort based on dependencies
        executed = set()
        order = []
        remaining = {r.region_id for r in regions}
        region_map = {r.region_id: r for r in regions}
        
        while remaining:
            # Find regions with no unexecuted dependencies
            ready = []
            for region_id in remaining:
                region = region_map[region_id]
                deps_satisfied = all(d in executed for d in region.depends_on)
                if deps_satisfied:
                    ready.append(region_id)
            
            if not ready:
                # Circular dependency or error - just add remaining
                order.extend(sorted(remaining))
                break
            
            # Sort ready regions by difficulty (easy first) and position
            ready.sort(key=lambda rid: (
                {"easy": 0, "medium": 1, "hard": 2}.get(
                    region_map[rid].estimated_difficulty, 1
                ),
                region_map[rid].start_node_idx
            ))
            
            # Execute first ready region
            region_id = ready[0]
            order.append(region_id)
            executed.add(region_id)
            remaining.remove(region_id)
        
        return order
    
    def _create_checkpoints(
        self,
        regions: List[TransformationRegion],
        strategy: Optional[SurgeryStrategy]
    ) -> List[ValidationCheckpoint]:
        """Create validation checkpoints."""
        checkpoints = []
        checkpoint_counter = 0
        
        for region in regions:
            if region.has_blockers:
                checkpoint_counter += 1
                checkpoints.append(ValidationCheckpoint(
                    checkpoint_id=f"checkpoint_{checkpoint_counter}",
                    name=f"After {region.region_type} transformation",
                    after_phase=region.region_id,
                    validation_type="shape",
                    expected_result="All shapes valid",
                    tolerance=0.0
                ))
        
        # Add final compilation check
        checkpoints.append(ValidationCheckpoint(
            checkpoint_id=f"checkpoint_final",
            name="Final compilation check",
            after_phase="final",
            validation_type="compilation",
            expected_result="MLA compatible",
            tolerance=strategy.expected_accuracy_tolerance if strategy else 1e-4
        ))
        
        return checkpoints
    
    def _estimate_complexity(
        self,
        regions: List[TransformationRegion],
        compilation: CompilationReport
    ) -> str:
        """Estimate overall transformation complexity."""
        total_blockers = sum(r.blocker_count for r in regions)
        hard_regions = sum(1 for r in regions if r.estimated_difficulty == "hard")
        
        if total_blockers > 50 or hard_regions > 5:
            return "very_hard"
        elif total_blockers > 20 or hard_regions > 2:
            return "hard"
        elif total_blockers > 5 or hard_regions > 0:
            return "medium"
        return "easy"
    
    def _compute_confidence(
        self,
        strategy: Optional[SurgeryStrategy],
        regions: List[TransformationRegion],
        compilation: CompilationReport
    ) -> float:
        """Compute confidence in the plan's success."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on strategy
        if strategy:
            confidence = strategy.confidence * 0.5 + 0.5
        
        # Adjust based on complexity
        total_blockers = sum(r.blocker_count for r in regions)
        if total_blockers == 0:
            confidence = min(1.0, confidence + 0.3)
        elif total_blockers > 20:
            confidence = max(0.1, confidence - 0.3)
        elif total_blockers > 5:
            confidence = max(0.2, confidence - 0.1)
        
        # Adjust based on known blocker types
        easy_blockers = {'Identity', 'Dropout', 'Cast'}
        hard_blockers = {'Einsum', 'Loop', 'If', 'NonMaxSuppression'}
        
        for op, count in compilation.blocker_ops.items():
            if op in easy_blockers:
                confidence = min(1.0, confidence + 0.05)
            elif op in hard_blockers:
                confidence = max(0.1, confidence - 0.1 * count)
        
        return round(confidence, 2)
    
    def _generate_warnings(
        self,
        architecture: ModelArchitecture,
        compilation: CompilationReport,
        regions: List[TransformationRegion]
    ) -> List[str]:
        """Generate warnings about the plan."""
        warnings = []
        
        # Warn about hard transformations
        hard_regions = [r for r in regions if r.estimated_difficulty == "hard"]
        if hard_regions:
            warnings.append(
                f"{len(hard_regions)} regions require complex transformations"
            )
        
        # Warn about unknown blockers
        known_blockers = {
            'Einsum', 'Identity', 'Dropout', 'Cast', 'NonZero', 'Where',
            'Loop', 'If', 'NonMaxSuppression', 'TopK'
        }
        unknown = [op for op in compilation.blocker_ops.keys() if op not in known_blockers]
        if unknown:
            warnings.append(
                f"Unknown blocker types: {', '.join(unknown)}"
            )
        
        # Warn about low block coverage
        if architecture.block_coverage < 0.3:
            warnings.append(
                f"Low architectural block coverage ({architecture.block_coverage:.1%}) - "
                "model structure may not be well understood"
            )
        
        # Warn about predicted multiple LM files
        if compilation.predicted_lm_files > 1:
            warnings.append(
                f"Model may compile to {compilation.predicted_lm_files} LM files"
            )
        
        return warnings
    
    def update_plan_after_failure(
        self,
        plan: TransformationPlan,
        failed_region_id: str,
        failure_reason: str
    ) -> TransformationPlan:
        """
        Update plan after a region transformation fails.
        
        Args:
            plan: Current plan
            failed_region_id: ID of region that failed
            failure_reason: Why it failed
            
        Returns:
            Updated plan with fallback strategy
        """
        new_plan = copy.deepcopy(plan)
        
        # Add warning about failure
        new_plan.warnings.append(
            f"Region {failed_region_id} failed: {failure_reason}"
        )
        
        # Try fallback strategy
        if new_plan.fallback_strategy_id:
            fallback = self.strategy_db.get_strategy(new_plan.fallback_strategy_id)
            if fallback:
                new_plan.strategy_id = fallback.strategy_id
                new_plan.strategy_name = fallback.name
                new_plan.fallback_strategy_id = fallback.fallback_strategy_id
                
                # Update region with fallback phases
                region = new_plan.get_region(failed_region_id)
                if region and fallback.phases:
                    region.transformation_phases = [p.phase_id for p in fallback.phases]
        
        # Reduce confidence
        new_plan.confidence = max(0.1, new_plan.confidence - 0.2)
        
        return new_plan


# =============================================================================
# Convenience Functions
# =============================================================================

def create_transformation_plan(
    model_path: str,
    strategy_db_path: Optional[str] = None,
    verbose: bool = False
) -> TransformationPlan:
    """
    Convenience function to create a transformation plan.
    
    Args:
        model_path: Path to ONNX model
        strategy_db_path: Optional path to strategy database JSON
        verbose: Enable verbose output
        
    Returns:
        TransformationPlan
    """
    # Load or create strategy database
    if strategy_db_path and Path(strategy_db_path).exists():
        strategy_db = StrategyDatabase.load(strategy_db_path)
    else:
        strategy_db = StrategyDatabase.create_with_defaults()
    
    # Create planner and generate plan
    planner = StrategicPlanner(strategy_db=strategy_db, verbose=verbose)
    return planner.create_plan(model_path)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python strategic_planner.py <model.onnx> [--verbose]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    plan = create_transformation_plan(model_path, verbose=verbose)
    
    print("\n" + plan.get_summary())
    
    # Save plan as JSON
    output_path = Path(model_path).stem + "_plan.json"
    with open(output_path, 'w') as f:
        json.dump(plan.to_dict(), f, indent=2)
    print(f"\nPlan saved to {output_path}")
