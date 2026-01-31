#!/usr/bin/env python3
"""
Strategy Database for ONNX Model Surgery.

Stores and retrieves surgery strategies at the architectural level,
not just individual node transformations. This enables the pipeline
to select the right overall approach for different model architectures.

Key concepts:
- SurgeryStrategy: High-level transformation approach for an architecture
- StrategyPhase: Individual phase within a strategy
- ValidationCriteria: How to verify transformation success

Author: Automated Model Surgery Pipeline
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class TransformationType(Enum):
    """Types of transformations in a strategy phase."""
    GEMM_TO_CONV = "gemm_to_conv"
    EINSUM_DECOMPOSITION = "einsum_decomposition"
    LAYERNORM_FUSION = "layernorm_fusion"
    OUTPUT_SEPARATION = "output_separation"
    TENSOR_RESHAPE = "tensor_reshape"
    WEIGHT_TRANSFORM = "weight_transform"
    NODE_REMOVAL = "node_removal"
    NODE_REPLACEMENT = "node_replacement"
    SUBGRAPH_REPLACEMENT = "subgraph_replacement"
    SPLIT_MODEL = "split_model"
    MERGE_OUTPUTS = "merge_outputs"
    SHAPE_INFERENCE_FIX = "shape_inference_fix"
    LAYOUT_TRANSFORM = "layout_transform"  # NCHW <-> NHWC
    DYNAMIC_TO_STATIC = "dynamic_to_static"


class ToleranceLevel(Enum):
    """Expected numerical tolerance levels."""
    IDENTICAL = "identical"  # 0.0 - data reshuffling only
    CLOSE = "close"  # ~1e-6 - math order changes
    ACCEPTABLE = "acceptable"  # ~1e-4 - complex rewrites
    RELAXED = "relaxed"  # ~1e-2 - approximations


@dataclass
class StrategyStep:
    """A single step within a strategy phase."""
    step_number: int
    action: str  # "remove", "replace", "rewrite", "add", "transform"
    description: str
    target_pattern: Optional[str] = None  # Node or pattern to target
    graphsurgeon_code: str = ""  # Code snippet for this step
    validation: str = ""  # How to validate this step
    
    def to_dict(self) -> Dict:
        return {
            'step_number': self.step_number,
            'action': self.action,
            'description': self.description,
            'target_pattern': self.target_pattern,
            'graphsurgeon_code': self.graphsurgeon_code,
            'validation': self.validation
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyStep':
        return cls(**data)


@dataclass
class WeightTransformation:
    """Describes how to transform weights for a conversion."""
    source_shape: str  # e.g., "[K, C]"
    target_shape: str  # e.g., "[K, C, 1, 1]"
    transformation_code: str  # Python code for transformation
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'source_shape': self.source_shape,
            'target_shape': self.target_shape,
            'transformation_code': self.transformation_code,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WeightTransformation':
        return cls(**data)


@dataclass
class StrategyPhase:
    """A phase within a surgery strategy."""
    phase_id: str
    name: str
    objective: str
    target_pattern: str  # e.g., "einsum_attention", "dynamic_layernorm"
    transformation_type: TransformationType
    
    # Steps to execute
    steps: List[StrategyStep] = field(default_factory=list)
    
    # Weight handling
    weight_transformation: Optional[WeightTransformation] = None
    
    # Validation
    validation_method: str = "shape_check"  # "shape_check", "numerical", "compilation"
    expected_tolerance: ToleranceLevel = ToleranceLevel.CLOSE
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Other phase IDs
    
    # Metadata
    difficulty: str = "medium"  # "easy", "medium", "hard"
    estimated_node_changes: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'phase_id': self.phase_id,
            'name': self.name,
            'objective': self.objective,
            'target_pattern': self.target_pattern,
            'transformation_type': self.transformation_type.value,
            'steps': [s.to_dict() for s in self.steps],
            'weight_transformation': self.weight_transformation.to_dict() if self.weight_transformation else None,
            'validation_method': self.validation_method,
            'expected_tolerance': self.expected_tolerance.value,
            'depends_on': self.depends_on,
            'difficulty': self.difficulty,
            'estimated_node_changes': self.estimated_node_changes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyPhase':
        steps = [StrategyStep.from_dict(s) for s in data.get('steps', [])]
        weight_trans = None
        if data.get('weight_transformation'):
            weight_trans = WeightTransformation.from_dict(data['weight_transformation'])
        
        return cls(
            phase_id=data['phase_id'],
            name=data['name'],
            objective=data['objective'],
            target_pattern=data['target_pattern'],
            transformation_type=TransformationType(data['transformation_type']),
            steps=steps,
            weight_transformation=weight_trans,
            validation_method=data.get('validation_method', 'shape_check'),
            expected_tolerance=ToleranceLevel(data.get('expected_tolerance', 'close')),
            depends_on=data.get('depends_on', []),
            difficulty=data.get('difficulty', 'medium'),
            estimated_node_changes=data.get('estimated_node_changes', 0)
        )


@dataclass
class ValidationCriteria:
    """Criteria for validating a strategy's success."""
    numerical_tolerance: float = 1e-4
    require_single_lm_file: bool = True
    max_cvu_nodes: int = 0
    max_apu_nodes: int = 0
    allowed_remaining_blockers: List[str] = field(default_factory=list)
    custom_checks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'numerical_tolerance': self.numerical_tolerance,
            'require_single_lm_file': self.require_single_lm_file,
            'max_cvu_nodes': self.max_cvu_nodes,
            'max_apu_nodes': self.max_apu_nodes,
            'allowed_remaining_blockers': self.allowed_remaining_blockers,
            'custom_checks': self.custom_checks
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ValidationCriteria':
        return cls(**data)


@dataclass
class SurgeryStrategy:
    """
    Complete surgery strategy for a model architecture.
    
    This represents the high-level approach for transforming a model,
    including all phases, their order, and validation criteria.
    """
    strategy_id: str
    name: str  # e.g., "Transformer Attention Decomposition"
    description: str
    
    # Target architecture
    target_architecture: str  # "Transformer", "YOLO", "ViT", "CNN"
    target_patterns: List[str]  # ["einsum_attention", "dynamic_layernorm"]
    
    # Strategy configuration
    divide_and_conquer: bool = False
    split_points: Optional[List[str]] = None  # Where to split if needed
    
    # Phases
    phases: List[StrategyPhase] = field(default_factory=list)
    phase_order: List[str] = field(default_factory=list)  # Explicit execution order
    phase_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Validation
    validation_criteria: ValidationCriteria = field(default_factory=ValidationCriteria)
    expected_accuracy_tolerance: float = 1e-4
    
    # Fallback
    fallback_strategy_id: Optional[str] = None
    
    # Confidence and statistics
    confidence: float = 0.0  # 0.0 to 1.0
    success_count: int = 0
    failure_count: int = 0
    average_execution_time: float = 0.0
    
    # Metadata
    author: str = "system"
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'description': self.description,
            'target_architecture': self.target_architecture,
            'target_patterns': self.target_patterns,
            'divide_and_conquer': self.divide_and_conquer,
            'split_points': self.split_points,
            'phases': [p.to_dict() for p in self.phases],
            'phase_order': self.phase_order,
            'phase_dependencies': self.phase_dependencies,
            'validation_criteria': self.validation_criteria.to_dict(),
            'expected_accuracy_tolerance': self.expected_accuracy_tolerance,
            'fallback_strategy_id': self.fallback_strategy_id,
            'confidence': self.confidence,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'average_execution_time': self.average_execution_time,
            'author': self.author,
            'version': self.version,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SurgeryStrategy':
        phases = [StrategyPhase.from_dict(p) for p in data.get('phases', [])]
        validation = ValidationCriteria.from_dict(data.get('validation_criteria', {}))
        
        return cls(
            strategy_id=data['strategy_id'],
            name=data['name'],
            description=data.get('description', ''),
            target_architecture=data['target_architecture'],
            target_patterns=data.get('target_patterns', []),
            divide_and_conquer=data.get('divide_and_conquer', False),
            split_points=data.get('split_points'),
            phases=phases,
            phase_order=data.get('phase_order', []),
            phase_dependencies=data.get('phase_dependencies', {}),
            validation_criteria=validation,
            expected_accuracy_tolerance=data.get('expected_accuracy_tolerance', 1e-4),
            fallback_strategy_id=data.get('fallback_strategy_id'),
            confidence=data.get('confidence', 0.0),
            success_count=data.get('success_count', 0),
            failure_count=data.get('failure_count', 0),
            average_execution_time=data.get('average_execution_time', 0.0),
            author=data.get('author', 'system'),
            version=data.get('version', '1.0'),
            tags=data.get('tags', [])
        )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def get_phase(self, phase_id: str) -> Optional[StrategyPhase]:
        """Get a phase by ID."""
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        return None
    
    def get_execution_order(self) -> List[StrategyPhase]:
        """Get phases in execution order."""
        if self.phase_order:
            ordered = []
            for phase_id in self.phase_order:
                phase = self.get_phase(phase_id)
                if phase:
                    ordered.append(phase)
            return ordered
        return self.phases
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Strategy: {self.name} ({self.strategy_id})",
            f"Target: {self.target_architecture}",
            f"Patterns: {', '.join(self.target_patterns)}",
            f"Phases: {len(self.phases)}",
            f"Divide & Conquer: {self.divide_and_conquer}",
            f"Confidence: {self.confidence:.2f}",
            f"Success Rate: {self.success_rate:.1%} ({self.success_count}/{self.success_count + self.failure_count})"
        ]
        return "\n".join(lines)


class StrategyDatabase:
    """
    Database for storing and retrieving surgery strategies.
    
    Provides:
    - Strategy storage with JSON persistence
    - Query by architecture, patterns, and blockers
    - Strategy ranking by confidence and success rate
    - Learning from execution outcomes
    """
    
    def __init__(self):
        self.strategies: List[SurgeryStrategy] = []
        
        # Indexes for fast lookup
        self._arch_index: Dict[str, List[str]] = defaultdict(list)  # arch -> strategy_ids
        self._pattern_index: Dict[str, List[str]] = defaultdict(list)  # pattern -> strategy_ids
        self._id_index: Dict[str, SurgeryStrategy] = {}  # strategy_id -> strategy
    
    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes from strategies list."""
        self._arch_index.clear()
        self._pattern_index.clear()
        self._id_index.clear()
        
        for strategy in self.strategies:
            self._id_index[strategy.strategy_id] = strategy
            self._arch_index[strategy.target_architecture].append(strategy.strategy_id)
            for pattern in strategy.target_patterns:
                self._pattern_index[pattern].append(strategy.strategy_id)
    
    def add_strategy(self, strategy: SurgeryStrategy) -> None:
        """Add a strategy to the database."""
        # Remove existing if same ID
        self.strategies = [s for s in self.strategies if s.strategy_id != strategy.strategy_id]
        self.strategies.append(strategy)
        self._rebuild_indexes()
    
    def get_strategy(self, strategy_id: str) -> Optional[SurgeryStrategy]:
        """Get a strategy by ID."""
        return self._id_index.get(strategy_id)
    
    def find_strategies_for_architecture(
        self, 
        architecture: str,
        min_confidence: float = 0.0
    ) -> List[SurgeryStrategy]:
        """Find strategies for a specific architecture."""
        strategy_ids = self._arch_index.get(architecture, [])
        strategies = [self._id_index[sid] for sid in strategy_ids if sid in self._id_index]
        
        # Filter by confidence
        strategies = [s for s in strategies if s.confidence >= min_confidence]
        
        # Sort by confidence descending
        strategies.sort(key=lambda s: s.confidence, reverse=True)
        
        return strategies
    
    def find_strategies_for_patterns(
        self, 
        patterns: List[str],
        min_confidence: float = 0.0
    ) -> List[SurgeryStrategy]:
        """Find strategies that address specific patterns."""
        # Collect all strategy IDs that match any pattern
        matching_ids = set()
        for pattern in patterns:
            matching_ids.update(self._pattern_index.get(pattern, []))
        
        strategies = [self._id_index[sid] for sid in matching_ids if sid in self._id_index]
        
        # Score by how many patterns each strategy covers
        def pattern_coverage(s: SurgeryStrategy) -> int:
            return len(set(s.target_patterns) & set(patterns))
        
        # Filter and sort
        strategies = [s for s in strategies if s.confidence >= min_confidence]
        strategies.sort(key=lambda s: (pattern_coverage(s), s.confidence), reverse=True)
        
        return strategies
    
    def find_best_strategy(
        self, 
        architecture: str,
        detected_patterns: List[str],
        blocker_ops: List[str]
    ) -> Optional[SurgeryStrategy]:
        """
        Find the best strategy for a given model.
        
        Args:
            architecture: Detected architecture type
            detected_patterns: Detected architectural patterns
            blocker_ops: List of blocking operation types
            
        Returns:
            Best matching strategy or None
        """
        # Get strategies for architecture
        arch_strategies = self.find_strategies_for_architecture(architecture)
        
        if not arch_strategies:
            # Try generic strategies
            arch_strategies = self.find_strategies_for_architecture("Generic")
        
        if not arch_strategies:
            return None
        
        # Score each strategy
        scored = []
        for strategy in arch_strategies:
            score = 0.0
            
            # Pattern match score
            pattern_overlap = len(set(strategy.target_patterns) & set(detected_patterns))
            score += pattern_overlap * 0.3
            
            # Confidence score
            score += strategy.confidence * 0.3
            
            # Success rate score
            score += strategy.success_rate * 0.2
            
            # Blocker coverage (does strategy address the blockers?)
            blocker_patterns = set()
            for op in blocker_ops:
                if op == 'Einsum':
                    blocker_patterns.add('einsum_attention')
                elif op in ['NonZero', 'Where']:
                    blocker_patterns.add('dynamic_ops')
                elif op == 'Gemm':
                    blocker_patterns.add('gemm_conversion')
            
            blocker_coverage = len(set(strategy.target_patterns) & blocker_patterns)
            score += blocker_coverage * 0.2
            
            scored.append((strategy, score))
        
        # Return best scoring strategy
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None
    
    def record_execution(
        self, 
        strategy_id: str, 
        success: bool,
        execution_time: float = 0.0
    ) -> None:
        """Record the outcome of a strategy execution."""
        strategy = self.get_strategy(strategy_id)
        if strategy:
            if success:
                strategy.success_count += 1
            else:
                strategy.failure_count += 1
            
            # Update average execution time
            total_runs = strategy.success_count + strategy.failure_count
            strategy.average_execution_time = (
                (strategy.average_execution_time * (total_runs - 1) + execution_time) / total_runs
            )
            
            # Update confidence based on success rate
            if total_runs >= 5:  # Need minimum runs for confidence
                strategy.confidence = strategy.success_rate * 0.7 + 0.3 * strategy.confidence
    
    def to_dict(self) -> Dict:
        """Convert database to dictionary."""
        return {
            'strategies': [s.to_dict() for s in self.strategies],
            'version': '1.0'
        }
    
    def save(self, path: str) -> None:
        """Save database to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Strategy database saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'StrategyDatabase':
        """Load database from JSON file."""
        db = cls()
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            for strategy_data in data.get('strategies', []):
                strategy = SurgeryStrategy.from_dict(strategy_data)
                db.strategies.append(strategy)
            
            db._rebuild_indexes()
            print(f"Loaded {len(db.strategies)} strategies from {path}")
        
        return db
    
    @classmethod
    def create_with_defaults(cls) -> 'StrategyDatabase':
        """Create database with default strategies."""
        db = cls()
        
        # Add default strategies
        db.add_strategy(cls._create_transformer_einsum_strategy())
        db.add_strategy(cls._create_transformer_gemm_strategy())
        db.add_strategy(cls._create_vit_strategy())
        db.add_strategy(cls._create_yolo_strategy())
        db.add_strategy(cls._create_cnn_strategy())
        db.add_strategy(cls._create_generic_strategy())
        
        return db
    
    @staticmethod
    def _create_transformer_einsum_strategy() -> SurgeryStrategy:
        """Create strategy for Transformer models with Einsum operations."""
        phases = [
            StrategyPhase(
                phase_id="einsum_decompose",
                name="Einsum Decomposition",
                objective="Replace Einsum operations with equivalent MatMul chains",
                target_pattern="einsum_attention",
                transformation_type=TransformationType.EINSUM_DECOMPOSITION,
                steps=[
                    StrategyStep(
                        step_number=1,
                        action="analyze",
                        description="Parse Einsum equation to determine operation type",
                        validation="Equation parsed successfully"
                    ),
                    StrategyStep(
                        step_number=2,
                        action="replace",
                        description="Replace QK^T Einsum with Transpose + MatMul",
                        target_pattern="bhid,bhjd->bhij",
                        graphsurgeon_code="""
# Transpose K: [B, H, J, D] -> [B, H, D, J]
transpose_k = gs.Variable("transpose_k", dtype=np.float32)
transpose_node = gs.Node("Transpose", inputs=[k_input], outputs=[transpose_k], attrs={"perm": [0, 1, 3, 2]})

# MatMul: Q @ K^T
qkt_output = gs.Variable("qkt", dtype=np.float32)
matmul_node = gs.Node("MatMul", inputs=[q_input, transpose_k], outputs=[qkt_output])
""",
                        validation="Output shape matches original"
                    ),
                    StrategyStep(
                        step_number=3,
                        action="replace",
                        description="Replace Attention@V Einsum with MatMul",
                        target_pattern="bhij,bhjd->bhid",
                        graphsurgeon_code="""
# MatMul: Attention @ V
av_output = gs.Variable("attention_v", dtype=np.float32)
matmul_node = gs.Node("MatMul", inputs=[attention_weights, v_input], outputs=[av_output])
""",
                        validation="Output shape matches original"
                    )
                ],
                validation_method="numerical",
                expected_tolerance=ToleranceLevel.CLOSE,
                difficulty="hard",
                estimated_node_changes=4
            ),
            StrategyPhase(
                phase_id="shape_fix",
                name="Shape Inference Fix",
                objective="Fix any shape inference issues from Einsum replacement",
                target_pattern="dynamic_shapes",
                transformation_type=TransformationType.SHAPE_INFERENCE_FIX,
                steps=[
                    StrategyStep(
                        step_number=1,
                        action="analyze",
                        description="Run shape inference on modified graph",
                        validation="No shape errors"
                    ),
                    StrategyStep(
                        step_number=2,
                        action="fix",
                        description="Add explicit shape annotations where needed",
                        validation="All shapes inferred"
                    )
                ],
                depends_on=["einsum_decompose"],
                validation_method="shape_check",
                difficulty="medium",
                estimated_node_changes=0
            )
        ]
        
        return SurgeryStrategy(
            strategy_id="transformer_einsum_decomposition",
            name="Transformer Einsum Decomposition",
            description="Decompose Einsum attention patterns into MLA-compatible MatMul operations",
            target_architecture="Transformer",
            target_patterns=["einsum_attention", "dynamic_shapes"],
            divide_and_conquer=False,
            phases=phases,
            phase_order=["einsum_decompose", "shape_fix"],
            validation_criteria=ValidationCriteria(
                numerical_tolerance=1e-6,
                require_single_lm_file=True
            ),
            expected_accuracy_tolerance=1e-6,
            confidence=0.8,
            tags=["transformer", "einsum", "attention"]
        )
    
    @staticmethod
    def _create_transformer_gemm_strategy() -> SurgeryStrategy:
        """Create strategy for Transformer models with Gemm operations."""
        phases = [
            StrategyPhase(
                phase_id="gemm_to_conv",
                name="Gemm to Conv Conversion",
                objective="Convert Gemm/MatMul operations to Conv for MLA efficiency",
                target_pattern="gemm_dense",
                transformation_type=TransformationType.GEMM_TO_CONV,
                steps=[
                    StrategyStep(
                        step_number=1,
                        action="reshape",
                        description="Reshape input to 4D tensor for Conv",
                        graphsurgeon_code="""
# Reshape input: [B, N, C] -> [B, N, C, 1]
reshape_in = gs.Node("Reshape", inputs=[input_tensor, shape_4d], outputs=[input_4d])
""",
                        validation="Input is 4D"
                    ),
                    StrategyStep(
                        step_number=2,
                        action="transform_weights",
                        description="Transform Gemm weights to Conv format",
                        graphsurgeon_code="""
# Gemm weights [K, C] -> Conv weights [K, C, 1, 1]
conv_weights = weights.reshape(K, C, 1, 1)
""",
                        validation="Weight shape is [K, C, 1, 1]"
                    ),
                    StrategyStep(
                        step_number=3,
                        action="replace",
                        description="Replace Gemm with Conv",
                        graphsurgeon_code="""
conv_node = gs.Node("Conv", 
    inputs=[input_4d, conv_weights, bias],
    outputs=[conv_output],
    attrs={"kernel_shape": [1, 1]})
""",
                        validation="Conv node created"
                    ),
                    StrategyStep(
                        step_number=4,
                        action="reshape",
                        description="Reshape output back to original dimensions",
                        graphsurgeon_code="""
# Reshape output: [B, K, N, 1] -> [B, N, K]
reshape_out = gs.Node("Reshape", inputs=[conv_output, original_shape], outputs=[final_output])
""",
                        validation="Output shape matches original"
                    )
                ],
                weight_transformation=WeightTransformation(
                    source_shape="[K, C]",
                    target_shape="[K, C, 1, 1]",
                    transformation_code="weights.reshape(K, C, 1, 1)",
                    description="Reshape 2D Gemm weights to 4D Conv weights"
                ),
                validation_method="numerical",
                expected_tolerance=ToleranceLevel.IDENTICAL,
                difficulty="medium",
                estimated_node_changes=4
            )
        ]
        
        return SurgeryStrategy(
            strategy_id="transformer_gemm_conversion",
            name="Transformer Gemm to Conv Conversion",
            description="Convert Gemm operations to Conv for better MLA mapping",
            target_architecture="Transformer",
            target_patterns=["gemm_dense", "linear_layers"],
            divide_and_conquer=False,
            phases=phases,
            phase_order=["gemm_to_conv"],
            validation_criteria=ValidationCriteria(
                numerical_tolerance=0.0,  # Should be identical
                require_single_lm_file=True
            ),
            expected_accuracy_tolerance=0.0,
            confidence=0.9,
            tags=["transformer", "gemm", "conv", "optimization"]
        )
    
    @staticmethod
    def _create_vit_strategy() -> SurgeryStrategy:
        """Create strategy for Vision Transformer models."""
        phases = [
            StrategyPhase(
                phase_id="patch_embed_fix",
                name="Patch Embedding Fix",
                objective="Ensure patch embedding produces 4D tensors",
                target_pattern="patch_embedding",
                transformation_type=TransformationType.TENSOR_RESHAPE,
                steps=[
                    StrategyStep(
                        step_number=1,
                        action="analyze",
                        description="Identify patch embedding layer structure"
                    )
                ],
                difficulty="easy"
            ),
            StrategyPhase(
                phase_id="attention_fix",
                name="Attention Mechanism Fix",
                objective="Fix attention patterns for MLA compatibility",
                target_pattern="einsum_attention",
                transformation_type=TransformationType.EINSUM_DECOMPOSITION,
                depends_on=["patch_embed_fix"],
                difficulty="hard"
            ),
            StrategyPhase(
                phase_id="head_fix",
                name="Classification Head Fix",
                objective="Convert classification head Gemm to Conv",
                target_pattern="classification_head",
                transformation_type=TransformationType.GEMM_TO_CONV,
                depends_on=["attention_fix"],
                difficulty="medium"
            )
        ]
        
        return SurgeryStrategy(
            strategy_id="vit_full_rewrite",
            name="ViT Full Rewrite",
            description="Complete rewrite of Vision Transformer for MLA compatibility",
            target_architecture="ViT",
            target_patterns=["patch_embedding", "einsum_attention", "classification_head"],
            divide_and_conquer=True,
            split_points=["patch_embed", "transformer_blocks", "head"],
            phases=phases,
            phase_order=["patch_embed_fix", "attention_fix", "head_fix"],
            validation_criteria=ValidationCriteria(
                numerical_tolerance=1e-4,
                require_single_lm_file=True
            ),
            expected_accuracy_tolerance=1e-4,
            confidence=0.7,
            tags=["vit", "vision", "transformer"]
        )
    
    @staticmethod
    def _create_yolo_strategy() -> SurgeryStrategy:
        """Create strategy for YOLO detection models."""
        phases = [
            StrategyPhase(
                phase_id="backbone_opt",
                name="Backbone Optimization",
                objective="Optimize backbone convolutions for MLA",
                target_pattern="conv_backbone",
                transformation_type=TransformationType.LAYOUT_TRANSFORM,
                difficulty="easy"
            ),
            StrategyPhase(
                phase_id="detection_head_sep",
                name="Detection Head Separation",
                objective="Separate multi-output detection heads",
                target_pattern="detection_head",
                transformation_type=TransformationType.OUTPUT_SEPARATION,
                depends_on=["backbone_opt"],
                steps=[
                    StrategyStep(
                        step_number=1,
                        action="identify",
                        description="Find detection head branches"
                    ),
                    StrategyStep(
                        step_number=2,
                        action="separate",
                        description="Create separate outputs for each branch"
                    )
                ],
                difficulty="medium"
            )
        ]
        
        return SurgeryStrategy(
            strategy_id="yolo_detection_opt",
            name="YOLO Detection Optimization",
            description="Optimize YOLO models for MLA compilation",
            target_architecture="YOLO",
            target_patterns=["conv_backbone", "detection_head", "multi_scale"],
            divide_and_conquer=False,
            phases=phases,
            phase_order=["backbone_opt", "detection_head_sep"],
            validation_criteria=ValidationCriteria(
                numerical_tolerance=1e-4,
                require_single_lm_file=True
            ),
            expected_accuracy_tolerance=1e-4,
            confidence=0.85,
            tags=["yolo", "detection", "cnn"]
        )
    
    @staticmethod
    def _create_cnn_strategy() -> SurgeryStrategy:
        """Create strategy for generic CNN models."""
        phases = [
            StrategyPhase(
                phase_id="blocker_removal",
                name="Blocker Removal",
                objective="Remove or replace blocking operations",
                target_pattern="blocking_ops",
                transformation_type=TransformationType.NODE_REMOVAL,
                difficulty="easy"
            )
        ]
        
        return SurgeryStrategy(
            strategy_id="cnn_standard_opt",
            name="CNN Standard Optimization",
            description="Standard optimization for CNN models",
            target_architecture="CNN",
            target_patterns=["blocking_ops"],
            phases=phases,
            validation_criteria=ValidationCriteria(
                numerical_tolerance=1e-5,
                require_single_lm_file=True
            ),
            expected_accuracy_tolerance=1e-5,
            confidence=0.9,
            tags=["cnn", "standard"]
        )
    
    @staticmethod
    def _create_generic_strategy() -> SurgeryStrategy:
        """Create generic fallback strategy."""
        phases = [
            StrategyPhase(
                phase_id="identify_blockers",
                name="Identify Blockers",
                objective="Identify all compilation blocking operations",
                target_pattern="any",
                transformation_type=TransformationType.NODE_REMOVAL,
                difficulty="easy"
            ),
            StrategyPhase(
                phase_id="remove_blockers",
                name="Remove Blockers",
                objective="Remove or replace blocking operations one by one",
                target_pattern="blocking_ops",
                transformation_type=TransformationType.NODE_REPLACEMENT,
                depends_on=["identify_blockers"],
                difficulty="medium"
            )
        ]
        
        return SurgeryStrategy(
            strategy_id="generic_blocker_removal",
            name="Generic Blocker Removal",
            description="Generic strategy for removing compilation blockers",
            target_architecture="Generic",
            target_patterns=["blocking_ops", "dynamic_shapes", "unsupported_ops"],
            phases=phases,
            fallback_strategy_id=None,
            validation_criteria=ValidationCriteria(
                numerical_tolerance=1e-4
            ),
            confidence=0.5,
            tags=["generic", "fallback"]
        )
    
    def get_summary(self) -> str:
        """Get database summary."""
        lines = [
            f"Strategy Database Summary",
            f"========================",
            f"Total Strategies: {len(self.strategies)}",
            "",
            "Strategies by Architecture:"
        ]
        
        for arch, strategy_ids in sorted(self._arch_index.items()):
            lines.append(f"  {arch}: {len(strategy_ids)} strategies")
        
        lines.append("")
        lines.append("Top Strategies by Confidence:")
        top_strategies = sorted(self.strategies, key=lambda s: s.confidence, reverse=True)[:5]
        for s in top_strategies:
            lines.append(f"  {s.name}: {s.confidence:.2f} ({s.success_rate:.1%} success)")
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_strategy_database(path: str = "rag_data/strategy_database.json") -> StrategyDatabase:
    """Load or create the strategy database."""
    if os.path.exists(path):
        return StrategyDatabase.load(path)
    else:
        db = StrategyDatabase.create_with_defaults()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        db.save(path)
        return db


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Create database with defaults
    db = StrategyDatabase.create_with_defaults()
    print(db.get_summary())
    
    print("\n" + "=" * 60)
    
    # Test strategy lookup
    transformer_strategies = db.find_strategies_for_architecture("Transformer")
    print(f"\nTransformer Strategies: {len(transformer_strategies)}")
    for s in transformer_strategies:
        print(f"  - {s.name}")
    
    # Test best strategy finder
    best = db.find_best_strategy(
        architecture="Transformer",
        detected_patterns=["einsum_attention"],
        blocker_ops=["Einsum"]
    )
    if best:
        print(f"\nBest Strategy for Transformer + Einsum:")
        print(best.get_summary())
    
    # Save to test file
    db.save("/tmp/test_strategy_db.json")
