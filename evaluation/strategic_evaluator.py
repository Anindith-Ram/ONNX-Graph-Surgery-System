#!/usr/bin/env python3
"""
Strategic Evaluation Module for ONNX Model Surgery.

Evaluates strategic decisions (not just node-level accuracy) including:
- Architecture understanding (correct detection)
- Strategy selection appropriateness
- Transformation completeness
- Blocker resolution effectiveness
- Overall efficiency

Author: Automated Model Surgery Pipeline
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import onnx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_analysis.architecture_analyzer import (
    ArchitectureAnalyzer, ModelArchitecture, ArchitectureType
)
from core_analysis.compilation_simulator import CompilationSimulator
from knowledge_base.strategy_database import StrategyDatabase, SurgeryStrategy


class EvaluationGrade(Enum):
    """Grades for evaluation metrics."""
    EXCELLENT = "excellent"  # > 95%
    GOOD = "good"            # 80-95%
    ACCEPTABLE = "acceptable"  # 60-80%
    POOR = "poor"            # 40-60%
    FAILING = "failing"      # < 40%


@dataclass
class ArchitectureUnderstandingScore:
    """Score for architecture understanding."""
    architecture_correctly_identified: bool = False
    patterns_correctly_detected: int = 0
    patterns_total: int = 0
    patterns_missed: int = 0
    block_coverage: float = 0.0
    
    @property
    def pattern_detection_rate(self) -> float:
        return self.patterns_correctly_detected / self.patterns_total if self.patterns_total > 0 else 0.0
    
    @property
    def score(self) -> float:
        """Compute overall architecture understanding score (0-1)."""
        score = 0.0
        if self.architecture_correctly_identified:
            score += 0.4
        score += 0.4 * self.pattern_detection_rate
        score += 0.2 * self.block_coverage
        return min(1.0, score)
    
    def to_dict(self) -> Dict:
        return {
            'architecture_correctly_identified': self.architecture_correctly_identified,
            'patterns_correctly_detected': self.patterns_correctly_detected,
            'patterns_total': self.patterns_total,
            'patterns_missed': self.patterns_missed,
            'block_coverage': self.block_coverage,
            'pattern_detection_rate': self.pattern_detection_rate,
            'score': self.score
        }


@dataclass
class StrategySelectionScore:
    """Score for strategy selection."""
    strategy_appropriate: bool = False
    divide_and_conquer_correct: bool = False
    strategy_confidence: float = 0.0
    strategy_success_rate: float = 0.0
    fallback_available: bool = False
    
    @property
    def score(self) -> float:
        """Compute overall strategy selection score (0-1)."""
        score = 0.0
        if self.strategy_appropriate:
            score += 0.4
        if self.divide_and_conquer_correct:
            score += 0.2
        score += 0.2 * self.strategy_confidence
        score += 0.1 * self.strategy_success_rate
        if self.fallback_available:
            score += 0.1
        return min(1.0, score)
    
    def to_dict(self) -> Dict:
        return {
            'strategy_appropriate': self.strategy_appropriate,
            'divide_and_conquer_correct': self.divide_and_conquer_correct,
            'strategy_confidence': self.strategy_confidence,
            'strategy_success_rate': self.strategy_success_rate,
            'fallback_available': self.fallback_available,
            'score': self.score
        }


@dataclass
class TransformationCompletenessScore:
    """Score for transformation completeness."""
    regions_total: int = 0
    regions_fully_transformed: int = 0
    regions_partially_transformed: int = 0
    regions_untransformed: int = 0
    
    @property
    def completion_rate(self) -> float:
        return self.regions_fully_transformed / self.regions_total if self.regions_total > 0 else 0.0
    
    @property
    def score(self) -> float:
        """Compute transformation completeness score (0-1)."""
        if self.regions_total == 0:
            return 1.0
        full_weight = 1.0
        partial_weight = 0.5
        score = (
            self.regions_fully_transformed * full_weight +
            self.regions_partially_transformed * partial_weight
        ) / self.regions_total
        return min(1.0, score)
    
    def to_dict(self) -> Dict:
        return {
            'regions_total': self.regions_total,
            'regions_fully_transformed': self.regions_fully_transformed,
            'regions_partially_transformed': self.regions_partially_transformed,
            'regions_untransformed': self.regions_untransformed,
            'completion_rate': self.completion_rate,
            'score': self.score
        }


@dataclass
class BlockerResolutionScore:
    """Score for blocker resolution."""
    blockers_original: int = 0
    blockers_resolved: int = 0
    blockers_remaining: int = 0
    new_blockers_introduced: int = 0
    
    @property
    def resolution_rate(self) -> float:
        return self.blockers_resolved / self.blockers_original if self.blockers_original > 0 else 1.0
    
    @property
    def score(self) -> float:
        """Compute blocker resolution score (0-1)."""
        if self.blockers_original == 0:
            return 1.0 if self.new_blockers_introduced == 0 else 0.8
        
        resolution_score = self.resolution_rate
        # Penalty for new blockers
        if self.new_blockers_introduced > 0:
            penalty = min(0.3, self.new_blockers_introduced * 0.1)
            resolution_score -= penalty
        
        return max(0.0, min(1.0, resolution_score))
    
    def to_dict(self) -> Dict:
        return {
            'blockers_original': self.blockers_original,
            'blockers_resolved': self.blockers_resolved,
            'blockers_remaining': self.blockers_remaining,
            'new_blockers_introduced': self.new_blockers_introduced,
            'resolution_rate': self.resolution_rate,
            'score': self.score
        }


@dataclass
class EfficiencyScore:
    """Score for transformation efficiency."""
    transformations_applied: int = 0
    transformations_successful: int = 0
    unnecessary_transformations: int = 0
    rollbacks_needed: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.transformations_successful / self.transformations_applied if self.transformations_applied > 0 else 0.0
    
    @property
    def efficiency_rate(self) -> float:
        if self.transformations_applied == 0:
            return 1.0
        useful = self.transformations_applied - self.unnecessary_transformations
        return useful / self.transformations_applied
    
    @property
    def score(self) -> float:
        """Compute efficiency score (0-1)."""
        score = 0.5 * self.success_rate + 0.3 * self.efficiency_rate
        # Penalty for rollbacks
        if self.rollbacks_needed > 0:
            penalty = min(0.2, self.rollbacks_needed * 0.05)
            score -= penalty
        return max(0.0, min(1.0, score + 0.2))  # Base score boost
    
    def to_dict(self) -> Dict:
        return {
            'transformations_applied': self.transformations_applied,
            'transformations_successful': self.transformations_successful,
            'unnecessary_transformations': self.unnecessary_transformations,
            'rollbacks_needed': self.rollbacks_needed,
            'success_rate': self.success_rate,
            'efficiency_rate': self.efficiency_rate,
            'score': self.score
        }


@dataclass
class StrategicEvaluationResult:
    """Complete strategic evaluation result."""
    model_name: str
    
    # Component scores
    architecture_understanding: ArchitectureUnderstandingScore = field(
        default_factory=ArchitectureUnderstandingScore
    )
    strategy_selection: StrategySelectionScore = field(
        default_factory=StrategySelectionScore
    )
    transformation_completeness: TransformationCompletenessScore = field(
        default_factory=TransformationCompletenessScore
    )
    blocker_resolution: BlockerResolutionScore = field(
        default_factory=BlockerResolutionScore
    )
    efficiency: EfficiencyScore = field(
        default_factory=EfficiencyScore
    )
    
    # Overall metrics
    overall_score: float = 0.0
    overall_grade: EvaluationGrade = EvaluationGrade.FAILING
    
    # Comparison with ground truth
    ground_truth_available: bool = False
    ground_truth_match_rate: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def compute_overall_score(self) -> None:
        """Compute overall strategic evaluation score."""
        weights = {
            'architecture': 0.15,
            'strategy': 0.20,
            'completeness': 0.25,
            'blockers': 0.25,
            'efficiency': 0.15
        }
        
        self.overall_score = (
            weights['architecture'] * self.architecture_understanding.score +
            weights['strategy'] * self.strategy_selection.score +
            weights['completeness'] * self.transformation_completeness.score +
            weights['blockers'] * self.blocker_resolution.score +
            weights['efficiency'] * self.efficiency.score
        )
        
        # Determine grade
        if self.overall_score >= 0.95:
            self.overall_grade = EvaluationGrade.EXCELLENT
        elif self.overall_score >= 0.80:
            self.overall_grade = EvaluationGrade.GOOD
        elif self.overall_score >= 0.60:
            self.overall_grade = EvaluationGrade.ACCEPTABLE
        elif self.overall_score >= 0.40:
            self.overall_grade = EvaluationGrade.POOR
        else:
            self.overall_grade = EvaluationGrade.FAILING
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'architecture_understanding': self.architecture_understanding.to_dict(),
            'strategy_selection': self.strategy_selection.to_dict(),
            'transformation_completeness': self.transformation_completeness.to_dict(),
            'blocker_resolution': self.blocker_resolution.to_dict(),
            'efficiency': self.efficiency.to_dict(),
            'overall_score': self.overall_score,
            'overall_grade': self.overall_grade.value,
            'ground_truth_available': self.ground_truth_available,
            'ground_truth_match_rate': self.ground_truth_match_rate,
            'recommendations': self.recommendations
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Strategic Evaluation: {self.model_name}",
            f"=" * 50,
            f"Overall Grade: {self.overall_grade.value.upper()} ({self.overall_score:.1%})",
            f"",
            f"Component Scores:",
            f"  Architecture Understanding: {self.architecture_understanding.score:.1%}",
            f"  Strategy Selection: {self.strategy_selection.score:.1%}",
            f"  Transformation Completeness: {self.transformation_completeness.score:.1%}",
            f"  Blocker Resolution: {self.blocker_resolution.score:.1%}",
            f"  Efficiency: {self.efficiency.score:.1%}",
        ]
        
        if self.ground_truth_available:
            lines.append(f"\nGround Truth Match: {self.ground_truth_match_rate:.1%}")
        
        if self.recommendations:
            lines.append(f"\nRecommendations:")
            for rec in self.recommendations[:5]:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)


class StrategicEvaluator:
    """
    Evaluate strategic decisions in model surgery.
    
    Provides comprehensive evaluation of:
    - Architecture understanding
    - Strategy selection
    - Transformation completeness
    - Blocker resolution
    - Overall efficiency
    """
    
    def __init__(
        self,
        strategy_db: Optional[StrategyDatabase] = None,
        verbose: bool = False
    ):
        """
        Initialize the strategic evaluator.
        
        Args:
            strategy_db: Strategy database for validation
            verbose: Enable verbose output
        """
        self.strategy_db = strategy_db or StrategyDatabase.create_with_defaults()
        self.verbose = verbose
        
        self.arch_analyzer = ArchitectureAnalyzer()
        self.compilation_sim = CompilationSimulator(verbose=verbose)
    
    def evaluate(
        self,
        original_model_path: str,
        modified_model_path: str,
        strategy_used: Optional[SurgeryStrategy] = None,
        execution_report: Optional[Dict] = None,
        ground_truth_path: Optional[str] = None
    ) -> StrategicEvaluationResult:
        """
        Evaluate strategic decisions for a model transformation.
        
        Args:
            original_model_path: Path to original model
            modified_model_path: Path to modified model
            strategy_used: Strategy that was used (optional)
            execution_report: Execution report from orchestrator (optional)
            ground_truth_path: Path to ground truth modified model (optional)
            
        Returns:
            StrategicEvaluationResult
        """
        model_name = Path(original_model_path).stem
        result = StrategicEvaluationResult(model_name=model_name)
        
        # Analyze architectures
        original_arch = self.arch_analyzer.analyze(original_model_path)
        modified_arch = self.arch_analyzer.analyze(modified_model_path)
        
        # Simulate compilations
        original_comp = self.compilation_sim.simulate(original_model_path)
        modified_comp = self.compilation_sim.simulate(modified_model_path)
        
        # Evaluate architecture understanding
        result.architecture_understanding = self._evaluate_architecture_understanding(
            original_arch, strategy_used
        )
        
        # Evaluate strategy selection
        result.strategy_selection = self._evaluate_strategy_selection(
            original_arch, original_comp, strategy_used
        )
        
        # Evaluate transformation completeness
        result.transformation_completeness = self._evaluate_transformation_completeness(
            original_arch, modified_arch, execution_report
        )
        
        # Evaluate blocker resolution
        result.blocker_resolution = self._evaluate_blocker_resolution(
            original_comp, modified_comp
        )
        
        # Evaluate efficiency
        result.efficiency = self._evaluate_efficiency(execution_report)
        
        # Compare with ground truth if available
        if ground_truth_path:
            result.ground_truth_available = True
            result.ground_truth_match_rate = self._compare_with_ground_truth(
                modified_model_path, ground_truth_path
            )
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        # Compute overall score
        result.compute_overall_score()
        
        return result
    
    def _evaluate_architecture_understanding(
        self,
        architecture: ModelArchitecture,
        strategy: Optional[SurgeryStrategy]
    ) -> ArchitectureUnderstandingScore:
        """Evaluate architecture understanding."""
        score = ArchitectureUnderstandingScore()
        
        # Check if architecture was correctly identified
        score.architecture_correctly_identified = (
            architecture.architecture_confidence >= 0.7
        )
        
        # Count patterns
        score.patterns_total = len(architecture.blocks)
        score.patterns_correctly_detected = sum(
            1 for b in architecture.blocks if b.has_blockers or b.node_ids
        )
        score.patterns_missed = max(0, 
            score.patterns_total - score.patterns_correctly_detected
        )
        
        # Block coverage
        score.block_coverage = architecture.block_coverage
        
        return score
    
    def _evaluate_strategy_selection(
        self,
        architecture: ModelArchitecture,
        compilation: Any,
        strategy: Optional[SurgeryStrategy]
    ) -> StrategySelectionScore:
        """Evaluate strategy selection."""
        score = StrategySelectionScore()
        
        if strategy:
            # Check if strategy matches architecture
            score.strategy_appropriate = (
                strategy.target_architecture == architecture.architecture_type.value or
                strategy.target_architecture == "Generic"
            )
            
            # Check divide-and-conquer decision
            should_divide = architecture.divide_and_conquer_recommended
            did_divide = strategy.divide_and_conquer
            score.divide_and_conquer_correct = (should_divide == did_divide)
            
            # Strategy confidence
            score.strategy_confidence = strategy.confidence
            score.strategy_success_rate = strategy.success_rate
            
            # Fallback available
            score.fallback_available = strategy.fallback_strategy_id is not None
        else:
            # No strategy - use defaults
            score.strategy_appropriate = True  # Assume generic is appropriate
            score.divide_and_conquer_correct = True
        
        return score
    
    def _evaluate_transformation_completeness(
        self,
        original_arch: ModelArchitecture,
        modified_arch: ModelArchitecture,
        execution_report: Optional[Dict]
    ) -> TransformationCompletenessScore:
        """Evaluate transformation completeness."""
        score = TransformationCompletenessScore()
        
        # Count blocker blocks in original
        blocker_blocks = [b for b in original_arch.blocks if b.has_blockers]
        score.regions_total = len(blocker_blocks)
        
        # Check which were resolved in modified
        modified_blockers = [b for b in modified_arch.blocks if b.has_blockers]
        
        # Estimate transformed regions
        if execution_report:
            regions_succeeded = execution_report.get('regions_succeeded', 0)
            regions_failed = execution_report.get('regions_failed', 0)
            score.regions_fully_transformed = regions_succeeded
            score.regions_partially_transformed = 0
            score.regions_untransformed = regions_failed
        else:
            # Estimate from architecture comparison
            original_blocker_types = {b.block_type for b in blocker_blocks}
            modified_blocker_types = {b.block_type for b in modified_blockers}
            resolved_types = original_blocker_types - modified_blocker_types
            score.regions_fully_transformed = len(resolved_types)
            score.regions_untransformed = len(modified_blocker_types)
        
        return score
    
    def _evaluate_blocker_resolution(
        self,
        original_comp: Any,
        modified_comp: Any
    ) -> BlockerResolutionScore:
        """Evaluate blocker resolution."""
        score = BlockerResolutionScore()
        
        score.blockers_original = original_comp.blocker_count
        score.blockers_remaining = modified_comp.blocker_count
        score.blockers_resolved = max(0, 
            original_comp.blocker_count - modified_comp.blocker_count
        )
        
        # Check for new blockers (ops that weren't blockers before)
        original_ops = set(original_comp.blocker_ops.keys())
        modified_ops = set(modified_comp.blocker_ops.keys())
        new_ops = modified_ops - original_ops
        score.new_blockers_introduced = sum(
            modified_comp.blocker_ops.get(op, 0) for op in new_ops
        )
        
        return score
    
    def _evaluate_efficiency(
        self,
        execution_report: Optional[Dict]
    ) -> EfficiencyScore:
        """Evaluate transformation efficiency."""
        score = EfficiencyScore()
        
        if execution_report:
            score.transformations_applied = execution_report.get('regions_executed', 0)
            score.transformations_successful = execution_report.get('regions_succeeded', 0)
            # Estimate unnecessary transformations (simplified)
            score.unnecessary_transformations = 0
            score.rollbacks_needed = execution_report.get('rollbacks', 0)
        else:
            # Default values when no report available
            score.transformations_applied = 1
            score.transformations_successful = 1
        
        return score
    
    def _compare_with_ground_truth(
        self,
        modified_path: str,
        ground_truth_path: str
    ) -> float:
        """Compare modified model with ground truth."""
        try:
            modified = onnx.load(modified_path)
            ground_truth = onnx.load(ground_truth_path)
            
            # Compare node counts
            mod_nodes = len(modified.graph.node)
            gt_nodes = len(ground_truth.graph.node)
            
            # Compare operation distributions
            mod_ops = {}
            gt_ops = {}
            
            for node in modified.graph.node:
                mod_ops[node.op_type] = mod_ops.get(node.op_type, 0) + 1
            
            for node in ground_truth.graph.node:
                gt_ops[node.op_type] = gt_ops.get(node.op_type, 0) + 1
            
            # Compute similarity
            all_ops = set(mod_ops.keys()) | set(gt_ops.keys())
            if not all_ops:
                return 1.0
            
            match_score = 0.0
            for op in all_ops:
                mod_count = mod_ops.get(op, 0)
                gt_count = gt_ops.get(op, 0)
                if max(mod_count, gt_count) > 0:
                    match_score += min(mod_count, gt_count) / max(mod_count, gt_count)
            
            return match_score / len(all_ops)
            
        except Exception:
            return 0.0
    
    def _generate_recommendations(
        self,
        result: StrategicEvaluationResult
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Architecture understanding recommendations
        if result.architecture_understanding.score < 0.7:
            if not result.architecture_understanding.architecture_correctly_identified:
                recommendations.append(
                    "Improve architecture detection - consider adding more patterns"
                )
            if result.architecture_understanding.block_coverage < 0.5:
                recommendations.append(
                    "Increase block coverage - many nodes not in identified patterns"
                )
        
        # Strategy selection recommendations
        if result.strategy_selection.score < 0.7:
            if not result.strategy_selection.strategy_appropriate:
                recommendations.append(
                    "Consider using architecture-specific strategy"
                )
            if not result.strategy_selection.fallback_available:
                recommendations.append(
                    "Add fallback strategy for robustness"
                )
        
        # Blocker resolution recommendations
        if result.blocker_resolution.score < 0.8:
            if result.blocker_resolution.remaining_blockers > 0:
                recommendations.append(
                    f"Address {result.blocker_resolution.remaining_blockers} remaining blockers"
                )
            if result.blocker_resolution.new_blockers_introduced > 0:
                recommendations.append(
                    "Review transformations - new blockers were introduced"
                )
        
        # Efficiency recommendations
        if result.efficiency.score < 0.7:
            if result.efficiency.rollbacks_needed > 0:
                recommendations.append(
                    "Improve transformation validation to reduce rollbacks"
                )
        
        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_strategy(
    original_path: str,
    modified_path: str,
    verbose: bool = False
) -> StrategicEvaluationResult:
    """Convenience function to evaluate strategic decisions."""
    evaluator = StrategicEvaluator(verbose=verbose)
    return evaluator.evaluate(original_path, modified_path)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python strategic_evaluator.py <original.onnx> <modified.onnx> [ground_truth.onnx]")
        sys.exit(1)
    
    original_path = sys.argv[1]
    modified_path = sys.argv[2]
    ground_truth_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    evaluator = StrategicEvaluator(verbose=True)
    result = evaluator.evaluate(
        original_path, modified_path,
        ground_truth_path=ground_truth_path
    )
    
    print(result.get_summary())
