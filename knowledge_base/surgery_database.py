#!/usr/bin/env python3
"""
Unified Surgery Database for ONNX Model Compilation.

This module provides a comprehensive, unified database structure for storing
and retrieving surgery knowledge learned from model transformations. It replaces
the fragmented KnowledgeBase and PatternDatabase with a single, rich data store.

Key Features:
- Precise node-level location data (no holes)
- Full tensor and attribute information
- WHY context (blocker reasons, compilation errors)
- HOW context (surgery steps, code snippets)
- JSON format for human readability

Author: Automated Model Surgery Pipeline
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class TensorInfo:
    """Complete information about a tensor in the graph."""
    name: str
    shape: Optional[List[int]]  # None if dynamic/unknown
    dtype: str  # e.g., "float32", "int64"
    is_initializer: bool = False  # True if this is a weight/constant
    is_dynamic: bool = False  # True if any dimension is dynamic
    dynamic_dims: List[int] = field(default_factory=list)  # Indices of dynamic dimensions
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'shape': self.shape,
            'dtype': self.dtype,
            'is_initializer': self.is_initializer,
            'is_dynamic': self.is_dynamic,
            'dynamic_dims': self.dynamic_dims
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TensorInfo':
        return cls(
            name=data.get('name', ''),
            shape=data.get('shape'),
            dtype=data.get('dtype', 'float32'),
            is_initializer=data.get('is_initializer', False),
            is_dynamic=data.get('is_dynamic', False),
            dynamic_dims=data.get('dynamic_dims', [])
        )


@dataclass
class NodeContext:
    """Context about a node's position and neighbors in the graph."""
    name: str
    op_type: str
    output_shapes: List[Optional[List[int]]] = field(default_factory=list)
    input_shapes: List[Optional[List[int]]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'op_type': self.op_type,
            'output_shapes': self.output_shapes,
            'input_shapes': self.input_shapes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeContext':
        return cls(
            name=data.get('name', ''),
            op_type=data.get('op_type', ''),
            output_shapes=data.get('output_shapes', []),
            input_shapes=data.get('input_shapes', [])
        )


@dataclass
class NodeTransformation:
    """
    Precise record of a single node transformation.
    
    This is the core data structure that captures EVERYTHING about a 
    transformation at a specific node - location, context, tensors,
    attributes, and the transformation itself.
    """
    # =========================================================================
    # Exact Location Identifiers
    # =========================================================================
    original_node_id: int  # Index in the original graph
    original_node_name: str  # Unique name of the node
    original_op_type: str  # Operation type (e.g., "Einsum", "Reshape")
    
    # =========================================================================
    # Graph Context (Precise)
    # =========================================================================
    graph_position: float  # 0.0 (near inputs) to 1.0 (near outputs)
    total_nodes_in_graph: int  # Total nodes for context
    
    # Predecessor nodes (nodes that produce inputs to this node)
    predecessor_nodes: List[Dict] = field(default_factory=list)
    # Format: [{name, op_type, output_shapes}]
    
    # Successor nodes (nodes that consume outputs from this node)
    successor_nodes: List[Dict] = field(default_factory=list)
    # Format: [{name, op_type, input_shapes}]
    
    # =========================================================================
    # Tensor Information (Critical - fills the "holes")
    # =========================================================================
    input_tensors: List[Dict] = field(default_factory=list)
    # Format: [{name, shape, dtype, is_initializer, is_dynamic, dynamic_dims}]
    
    output_tensors: List[Dict] = field(default_factory=list)
    # Format: [{name, shape, dtype}]
    
    # =========================================================================
    # Node Attributes (Critical for understanding)
    # =========================================================================
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Examples:
    # - Einsum: {"equation": "bhid,bhjd->bhij"}
    # - Reshape: {"allowzero": 0}
    # - Conv: {"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]}
    # - Transpose: {"perm": [0, 2, 1, 3]}
    
    # =========================================================================
    # Transformation Details
    # =========================================================================
    action: str = "remove"  # "remove", "replace", "add", "reshape", "rewire", "modify"
    
    # New node details if replaced/added (None for removal)
    result_node: Optional[Dict] = None
    # Format: {name, op_type, attributes, input_tensors, output_tensors}
    
    # For replacements: what the node was replaced with
    replacement_ops: List[str] = field(default_factory=list)
    # e.g., ["Transpose", "MatMul"] for Einsum decomposition
    
    # =========================================================================
    # WHY Context (Explains the reason for transformation)
    # =========================================================================
    blocker_reason: Optional[str] = None
    # e.g., "Einsum tensor contractions not supported on MLA hardware"
    
    compilation_error: Optional[str] = None
    # Actual compiler error message if available
    
    is_compilation_blocker: bool = False
    # True if this node was identified as blocking compilation
    
    # =========================================================================
    # HOW to Fix (Actionable guidance)
    # =========================================================================
    surgery_steps: List[str] = field(default_factory=list)
    # Step-by-step transformation instructions
    # e.g., ["1. Identify Einsum equation pattern", "2. Transpose second input", ...]
    
    code_snippet: Optional[str] = None
    # ONNX GraphSurgeon code example
    
    # =========================================================================
    # Metadata
    # =========================================================================
    confidence: float = 1.0  # 0.0 to 1.0
    source_model: str = ""  # Which model this came from
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            # Location
            'original_node_id': self.original_node_id,
            'original_node_name': self.original_node_name,
            'original_op_type': self.original_op_type,
            
            # Graph context
            'graph_position': self.graph_position,
            'total_nodes_in_graph': self.total_nodes_in_graph,
            'predecessor_nodes': self.predecessor_nodes,
            'successor_nodes': self.successor_nodes,
            
            # Tensors
            'input_tensors': self.input_tensors,
            'output_tensors': self.output_tensors,
            
            # Attributes
            'attributes': self.attributes,
            
            # Transformation
            'action': self.action,
            'result_node': self.result_node,
            'replacement_ops': self.replacement_ops,
            
            # WHY
            'blocker_reason': self.blocker_reason,
            'compilation_error': self.compilation_error,
            'is_compilation_blocker': self.is_compilation_blocker,
            
            # HOW
            'surgery_steps': self.surgery_steps,
            'code_snippet': self.code_snippet,
            
            # Metadata
            'confidence': self.confidence,
            'source_model': self.source_model
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeTransformation':
        """Create from dictionary."""
        return cls(
            original_node_id=data.get('original_node_id', -1),
            original_node_name=data.get('original_node_name', ''),
            original_op_type=data.get('original_op_type', ''),
            graph_position=data.get('graph_position', 0.5),
            total_nodes_in_graph=data.get('total_nodes_in_graph', 0),
            predecessor_nodes=data.get('predecessor_nodes', []),
            successor_nodes=data.get('successor_nodes', []),
            input_tensors=data.get('input_tensors', []),
            output_tensors=data.get('output_tensors', []),
            attributes=data.get('attributes', {}),
            action=data.get('action', 'remove'),
            result_node=data.get('result_node'),
            replacement_ops=data.get('replacement_ops', []),
            blocker_reason=data.get('blocker_reason'),
            compilation_error=data.get('compilation_error'),
            is_compilation_blocker=data.get('is_compilation_blocker', False),
            surgery_steps=data.get('surgery_steps', []),
            code_snippet=data.get('code_snippet'),
            confidence=data.get('confidence', 1.0),
            source_model=data.get('source_model', '')
        )
    
    def get_pattern_key(self) -> str:
        """Generate a key for pattern matching."""
        # Key based on op_type, action, and context
        pred_ops = sorted([p.get('op_type', '') for p in self.predecessor_nodes])
        succ_ops = sorted([s.get('op_type', '') for s in self.successor_nodes])
        return f"{self.original_op_type}:{self.action}:pred={','.join(pred_ops)}:succ={','.join(succ_ops)}"


@dataclass
class TransformationRecord:
    """
    Complete record of a model transformation.
    
    Captures all transformations applied to convert an original model
    to a compilable modified version.
    """
    # =========================================================================
    # Model Identification
    # =========================================================================
    model_name: str
    model_category: str  # "YOLO", "Transformer", "ViT", "CNN", "Other"
    
    # =========================================================================
    # Model-Level Context
    # =========================================================================
    original_node_count: int = 0
    modified_node_count: int = 0
    
    # Why the original model didn't compile
    original_blockers: List[str] = field(default_factory=list)
    # e.g., ["Einsum (3 nodes)", "Dynamic shapes (12 nodes)"]
    
    # Did the modified model compile successfully?
    compilation_success: bool = True
    
    # =========================================================================
    # All Node Transformations (The Core Data)
    # =========================================================================
    transformations: List[NodeTransformation] = field(default_factory=list)
    
    # =========================================================================
    # Transformation Ordering (Dependencies)
    # =========================================================================
    transformation_order: List[int] = field(default_factory=list)
    # Indices into transformations list, in order of application
    # Order matters! Some transformations depend on others
    
    # =========================================================================
    # Statistics
    # =========================================================================
    nodes_removed: int = 0
    nodes_added: int = 0
    nodes_modified: int = 0
    shape_changes: int = 0
    
    # =========================================================================
    # Source Paths
    # =========================================================================
    source_original_path: str = ""
    source_modified_path: str = ""
    
    # =========================================================================
    # Metadata
    # =========================================================================
    extraction_timestamp: str = ""
    extraction_version: str = "1.0.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'model_category': self.model_category,
            'original_node_count': self.original_node_count,
            'modified_node_count': self.modified_node_count,
            'original_blockers': self.original_blockers,
            'compilation_success': self.compilation_success,
            'transformations': [t.to_dict() for t in self.transformations],
            'transformation_order': self.transformation_order,
            'nodes_removed': self.nodes_removed,
            'nodes_added': self.nodes_added,
            'nodes_modified': self.nodes_modified,
            'shape_changes': self.shape_changes,
            'source_original_path': self.source_original_path,
            'source_modified_path': self.source_modified_path,
            'extraction_timestamp': self.extraction_timestamp,
            'extraction_version': self.extraction_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransformationRecord':
        """Create from dictionary."""
        transformations = [
            NodeTransformation.from_dict(t) 
            for t in data.get('transformations', [])
        ]
        return cls(
            model_name=data.get('model_name', ''),
            model_category=data.get('model_category', 'Other'),
            original_node_count=data.get('original_node_count', 0),
            modified_node_count=data.get('modified_node_count', 0),
            original_blockers=data.get('original_blockers', []),
            compilation_success=data.get('compilation_success', True),
            transformations=transformations,
            transformation_order=data.get('transformation_order', []),
            nodes_removed=data.get('nodes_removed', 0),
            nodes_added=data.get('nodes_added', 0),
            nodes_modified=data.get('nodes_modified', 0),
            shape_changes=data.get('shape_changes', 0),
            source_original_path=data.get('source_original_path', ''),
            source_modified_path=data.get('source_modified_path', ''),
            extraction_timestamp=data.get('extraction_timestamp', ''),
            extraction_version=data.get('extraction_version', '1.0.0')
        )
    
    def get_transformations_by_action(self, action: str) -> List[NodeTransformation]:
        """Get all transformations with a specific action."""
        return [t for t in self.transformations if t.action == action]
    
    def get_transformations_by_op_type(self, op_type: str) -> List[NodeTransformation]:
        """Get all transformations for a specific operation type."""
        return [t for t in self.transformations if t.original_op_type == op_type]
    
    def get_blocker_transformations(self) -> List[NodeTransformation]:
        """Get transformations that addressed compilation blockers."""
        return [t for t in self.transformations if t.is_compilation_blocker]


# =============================================================================
# Surgery Templates (Generalized Patterns)
# =============================================================================

@dataclass
class SurgeryStep:
    """Single step in a surgery procedure."""
    step_number: int
    action: str  # "identify", "create_node", "rewire", "remove", "validate"
    description: str
    
    # Precise instructions
    target_pattern: str = ""  # How to find the target (e.g., "node.op_type == 'Einsum'")
    operation: str = ""  # What to do (e.g., "Create Transpose node with perm=[0,1,3,2]")
    validation: str = ""  # How to verify success (e.g., "Output shape matches original")
    
    # Code snippet for this step
    code: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'step_number': self.step_number,
            'action': self.action,
            'description': self.description,
            'target_pattern': self.target_pattern,
            'operation': self.operation,
            'validation': self.validation,
            'code': self.code
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SurgeryStep':
        return cls(
            step_number=data.get('step_number', 0),
            action=data.get('action', ''),
            description=data.get('description', ''),
            target_pattern=data.get('target_pattern', ''),
            operation=data.get('operation', ''),
            validation=data.get('validation', ''),
            code=data.get('code')
        )


@dataclass
class SurgeryTemplate:
    """
    Reusable surgery pattern with conditions and steps.
    
    Templates are generalized from multiple similar transformations
    and can be applied to new models.
    """
    template_id: str
    name: str  # e.g., "Einsum to MatMul Decomposition"
    description: str = ""
    
    # =========================================================================
    # When to Apply (Trigger Conditions)
    # =========================================================================
    trigger_op_type: str = ""  # Primary op type that triggers this template
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"op_type": "Einsum", "model_categories": ["Transformer", "ViT"],
    #        "context_patterns": {"has_attention": True}}
    
    # Model categories this applies to (empty = all)
    applicable_categories: List[str] = field(default_factory=list)
    
    # =========================================================================
    # Confidence (Based on Success Rate)
    # =========================================================================
    confidence: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    
    # =========================================================================
    # Detailed Steps
    # =========================================================================
    steps: List[SurgeryStep] = field(default_factory=list)
    
    # =========================================================================
    # Example Code
    # =========================================================================
    graphsurgeon_code: str = ""  # Complete GraphSurgeon code example
    
    # =========================================================================
    # Warnings and Edge Cases
    # =========================================================================
    warnings: List[str] = field(default_factory=list)
    # e.g., ["Ensure batch dimension is preserved", "Check for broadcasting"]
    
    contraindications: List[str] = field(default_factory=list)
    # When NOT to apply this template
    # e.g., ["If Einsum equation has more than 2 inputs", "If output feeds into another Einsum"]
    
    # =========================================================================
    # Source Examples
    # =========================================================================
    example_models: List[str] = field(default_factory=list)
    # Models where this template was successfully applied
    
    def to_dict(self) -> Dict:
        return {
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'trigger_op_type': self.trigger_op_type,
            'trigger_conditions': self.trigger_conditions,
            'applicable_categories': self.applicable_categories,
            'confidence': self.confidence,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'steps': [s.to_dict() for s in self.steps],
            'graphsurgeon_code': self.graphsurgeon_code,
            'warnings': self.warnings,
            'contraindications': self.contraindications,
            'example_models': self.example_models
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SurgeryTemplate':
        steps = [SurgeryStep.from_dict(s) for s in data.get('steps', [])]
        return cls(
            template_id=data.get('template_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            trigger_op_type=data.get('trigger_op_type', ''),
            trigger_conditions=data.get('trigger_conditions', {}),
            applicable_categories=data.get('applicable_categories', []),
            confidence=data.get('confidence', 0.0),
            success_count=data.get('success_count', 0),
            failure_count=data.get('failure_count', 0),
            steps=steps,
            graphsurgeon_code=data.get('graphsurgeon_code', ''),
            warnings=data.get('warnings', []),
            contraindications=data.get('contraindications', []),
            example_models=data.get('example_models', [])
        )
    
    def matches(self, op_type: str, model_category: str, context: Dict = None) -> bool:
        """Check if this template applies to the given context."""
        # Check op type
        if self.trigger_op_type and self.trigger_op_type != op_type:
            return False
        
        # Check model category
        if self.applicable_categories and model_category not in self.applicable_categories:
            return False
        
        # Check additional conditions
        if context and self.trigger_conditions:
            for key, value in self.trigger_conditions.items():
                if key in context and context[key] != value:
                    return False
        
        return True


# =============================================================================
# Compilation Blocker Reference
# =============================================================================

@dataclass
class CompilationBlocker:
    """
    Reference information about a compilation blocker.
    
    Maps operation types to their blocker reasons and solution templates.
    """
    op_type: str
    reason: str  # Why this blocks compilation
    hardware_limitation: str = ""  # Specific hardware limitation
    solution_templates: List[str] = field(default_factory=list)  # Template IDs
    affected_categories: List[str] = field(default_factory=list)  # Model categories affected
    severity: str = "high"  # "critical", "high", "medium", "low"
    occurrence_count: int = 0  # How often this blocker appears in training data
    
    def to_dict(self) -> Dict:
        return {
            'op_type': self.op_type,
            'reason': self.reason,
            'hardware_limitation': self.hardware_limitation,
            'solution_templates': self.solution_templates,
            'affected_categories': self.affected_categories,
            'severity': self.severity,
            'occurrence_count': self.occurrence_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CompilationBlocker':
        return cls(
            op_type=data.get('op_type', ''),
            reason=data.get('reason', ''),
            hardware_limitation=data.get('hardware_limitation', ''),
            solution_templates=data.get('solution_templates', []),
            affected_categories=data.get('affected_categories', []),
            severity=data.get('severity', 'high'),
            occurrence_count=data.get('occurrence_count', 0)
        )


# =============================================================================
# Compilation Error Database
# =============================================================================

@dataclass
class CompilationErrorEntry:
    """
    Mapping from compilation error message patterns to causes and solutions.
    
    This helps engineers understand WHY their model failed to compile
    and HOW to fix it based on the error message.
    """
    error_id: str  # Unique identifier
    error_pattern: str  # Regex pattern to match error messages
    error_category: str  # "shape", "op_support", "memory", "dtype", "graph_structure"
    
    # Human-readable explanation
    description: str
    cause: str  # Why this error occurs
    
    # Linked solutions
    related_op_types: List[str] = field(default_factory=list)  # Op types that commonly cause this
    solution_templates: List[str] = field(default_factory=list)  # Template IDs that fix this
    
    # Step-by-step fix instructions
    fix_steps: List[str] = field(default_factory=list)
    
    # Example error messages that match this pattern
    example_messages: List[str] = field(default_factory=list)
    
    # Statistics
    occurrence_count: int = 0
    fix_success_rate: float = 0.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict:
        return {
            'error_id': self.error_id,
            'error_pattern': self.error_pattern,
            'error_category': self.error_category,
            'description': self.description,
            'cause': self.cause,
            'related_op_types': self.related_op_types,
            'solution_templates': self.solution_templates,
            'fix_steps': self.fix_steps,
            'example_messages': self.example_messages,
            'occurrence_count': self.occurrence_count,
            'fix_success_rate': self.fix_success_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CompilationErrorEntry':
        return cls(
            error_id=data.get('error_id', ''),
            error_pattern=data.get('error_pattern', ''),
            error_category=data.get('error_category', 'unknown'),
            description=data.get('description', ''),
            cause=data.get('cause', ''),
            related_op_types=data.get('related_op_types', []),
            solution_templates=data.get('solution_templates', []),
            fix_steps=data.get('fix_steps', []),
            example_messages=data.get('example_messages', []),
            occurrence_count=data.get('occurrence_count', 0),
            fix_success_rate=data.get('fix_success_rate', 0.0)
        )
    
    def matches(self, error_message: str) -> bool:
        """Check if this entry matches the given error message."""
        import re
        try:
            return bool(re.search(self.error_pattern, error_message, re.IGNORECASE))
        except:
            return self.error_pattern.lower() in error_message.lower()


# Default compilation error entries based on common MLA compilation issues
DEFAULT_COMPILATION_ERRORS = [
    CompilationErrorEntry(
        error_id="einsum_not_supported",
        error_pattern=r"(Einsum|einsum).*(not supported|unsupported|cannot compile)",
        error_category="op_support",
        description="Einsum operation not supported by MLA hardware",
        cause="Einsum performs arbitrary tensor contractions that cannot be mapped to fixed hardware matrix units",
        related_op_types=["Einsum"],
        solution_templates=["einsum_to_matmul", "einsum_decomposition"],
        fix_steps=[
            "1. Identify the Einsum equation pattern (e.g., 'bhid,bhjd->bhij')",
            "2. Decompose into supported operations (Transpose + MatMul)",
            "3. Create Transpose node for the second input",
            "4. Replace Einsum with MatMul operation",
            "5. Verify output shapes match original"
        ],
        example_messages=[
            "Einsum operation not supported on MLA",
            "Cannot compile model: einsum tensor contraction not supported"
        ]
    ),
    CompilationErrorEntry(
        error_id="dynamic_shape",
        error_pattern=r"(dynamic|unknown|variable).*(shape|dimension|size)",
        error_category="shape",
        description="Dynamic or unknown tensor shapes cannot be compiled",
        cause="MLA requires all tensor shapes to be known at compile time for buffer allocation",
        related_op_types=["Reshape", "Expand", "Tile", "Shape", "Size"],
        solution_templates=["static_shape_enforcement"],
        fix_steps=[
            "1. Identify tensors with dynamic dimensions",
            "2. Replace dynamic dimensions with fixed batch size (e.g., 1)",
            "3. Update Reshape nodes to use constant shape inputs",
            "4. Ensure Shape/Size operations are not feeding into other operations",
            "5. Run shape inference to verify all shapes are static"
        ],
        example_messages=[
            "Dynamic shape not supported",
            "Unknown dimension in tensor shape",
            "Variable batch size not supported"
        ]
    ),
    CompilationErrorEntry(
        error_id="non_4d_tensor",
        error_pattern=r"(non-?4D|[235]D|dimension).*(tensor|not supported|mismatch)",
        error_category="shape",
        description="Non-4D tensor shapes not supported by MLA",
        cause="MLA hardware operates on 4D tensors (NCHW format); other dimensions require reshaping",
        related_op_types=["Reshape", "Squeeze", "Unsqueeze"],
        solution_templates=["reshape_to_4d"],
        fix_steps=[
            "1. Identify non-4D tensors in the model",
            "2. Add Reshape/Unsqueeze nodes to convert to 4D",
            "3. For 3D: Add batch or channel dimension",
            "4. For 5D: Merge dimensions to fit 4D",
            "5. Ensure output matches expected format"
        ],
        example_messages=[
            "3D tensor not supported, expected 4D",
            "Non-4D tensor dimension mismatch",
            "Tensor must be 4-dimensional"
        ]
    ),
    CompilationErrorEntry(
        error_id="control_flow",
        error_pattern=r"(control flow|Loop|If|conditional|branch).*(not supported|unsupported)",
        error_category="graph_structure",
        description="Control flow operations not supported in static graph",
        cause="MLA requires static computation graphs without conditional branching or loops",
        related_op_types=["Loop", "If", "Scan"],
        solution_templates=["loop_unroll", "if_removal"],
        fix_steps=[
            "1. Identify Loop/If/Scan nodes in the graph",
            "2. For loops with known iterations: Unroll the loop body",
            "3. For conditionals: Remove condition and keep the true branch",
            "4. Inline subgraph content into main graph",
            "5. Verify graph connectivity after removal"
        ],
        example_messages=[
            "Loop operation not supported",
            "Conditional branching not supported on MLA",
            "Control flow not allowed in static graph"
        ]
    ),
    CompilationErrorEntry(
        error_id="where_conditional",
        error_pattern=r"(Where|where|conditional select).*(not supported|unsupported)",
        error_category="op_support",
        description="Where (conditional select) operation not supported",
        cause="Where requires data-dependent execution which MLA cannot perform",
        related_op_types=["Where"],
        solution_templates=["where_removal", "where_to_select"],
        fix_steps=[
            "1. Identify Where nodes in the graph",
            "2. Analyze the condition: if constant, simplify to direct selection",
            "3. If mask is constant: Replace with Mul+Add pattern",
            "4. Remove the Where node and rewire connections",
            "5. Verify logical equivalence with test inputs"
        ],
        example_messages=[
            "Where operation not supported",
            "Conditional select cannot be compiled"
        ]
    ),
    CompilationErrorEntry(
        error_id="nonzero_dynamic",
        error_pattern=r"(NonZero|nonzero).*(dynamic|variable|not supported)",
        error_category="op_support",
        description="NonZero operation produces dynamic output shapes",
        cause="NonZero output size depends on input data values, not shape",
        related_op_types=["NonZero"],
        solution_templates=["nonzero_removal"],
        fix_steps=[
            "1. Identify NonZero nodes and their consumers",
            "2. Analyze if NonZero is critical (often used for sparse operations)",
            "3. Consider alternative approaches: fixed-size output, mask-based",
            "4. Remove NonZero and rewire to use dense operations",
            "5. May require algorithm redesign"
        ],
        example_messages=[
            "NonZero has dynamic output size",
            "NonZero operation not supported"
        ]
    ),
    CompilationErrorEntry(
        error_id="nms_dynamic",
        error_pattern=r"(NonMaxSuppression|NMS).*(dynamic|variable|not supported)",
        error_category="op_support",
        description="NonMaxSuppression produces variable output size",
        cause="NMS output depends on detection results which vary per input",
        related_op_types=["NonMaxSuppression"],
        solution_templates=["nms_postprocess"],
        fix_steps=[
            "1. Move NMS to post-processing (outside ONNX model)",
            "2. End model at raw detection outputs (boxes, scores)",
            "3. Implement NMS in application code",
            "4. Alternative: Use TopK with fixed K as approximation",
            "5. Validate detection quality after modification"
        ],
        example_messages=[
            "NonMaxSuppression has dynamic output",
            "NMS not supported on MLA"
        ]
    ),
    CompilationErrorEntry(
        error_id="memory_limit",
        error_pattern=r"(memory|buffer|allocation).*(exceed|limit|overflow|too large)",
        error_category="memory",
        description="Model exceeds available memory on target hardware",
        cause="Total activation memory or weight size exceeds MLA capacity",
        related_op_types=[],
        solution_templates=[],
        fix_steps=[
            "1. Analyze model memory usage",
            "2. Reduce batch size if possible",
            "3. Consider model quantization (FP32 -> INT8)",
            "4. Split model into multiple stages",
            "5. Reduce intermediate tensor sizes by restructuring"
        ],
        example_messages=[
            "Memory allocation exceeds limit",
            "Buffer size too large for hardware"
        ]
    ),
    CompilationErrorEntry(
        error_id="dtype_unsupported",
        error_pattern=r"(dtype|data type|type).*(not supported|unsupported|complex|bfloat)",
        error_category="dtype",
        description="Unsupported data type in model",
        cause="MLA supports limited data types (float32, float16, int8)",
        related_op_types=["Cast"],
        solution_templates=[],
        fix_steps=[
            "1. Identify tensors with unsupported dtypes",
            "2. Add Cast nodes to convert to supported types",
            "3. Complex types: Separate real/imaginary components",
            "4. String types: Move to pre/post-processing",
            "5. Validate numerical accuracy after conversion"
        ],
        example_messages=[
            "Complex64 dtype not supported",
            "BFloat16 not supported on target",
            "String tensor cannot be compiled"
        ]
    ),
    CompilationErrorEntry(
        error_id="custom_op",
        error_pattern=r"(custom|unknown|unrecognized).*(operator|operation|op)",
        error_category="op_support",
        description="Custom or unknown operator not supported",
        cause="Operator is not in the supported ONNX opset for MLA",
        related_op_types=[],
        solution_templates=[],
        fix_steps=[
            "1. Identify the custom operator name and functionality",
            "2. Check if equivalent standard ONNX ops exist",
            "3. Decompose custom op into standard operations",
            "4. If from framework: Export with native ONNX ops instead",
            "5. Contact hardware vendor for custom op support"
        ],
        example_messages=[
            "Unknown operator: CustomOp",
            "Unrecognized operation type"
        ]
    )
]


# =============================================================================
# Unified Surgery Database
# =============================================================================

class SurgeryDatabase:
    """
    Unified database for surgery knowledge.
    
    This is the main interface for storing and retrieving transformation
    knowledge. It provides indexed lookups and rich query capabilities.
    
    All data is stored in JSON format for human readability.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        # Core storage
        self.transformation_records: List[TransformationRecord] = []
        self.surgery_templates: List[SurgeryTemplate] = []
        self.compilation_blockers: List[CompilationBlocker] = []
        self.compilation_errors: List[CompilationErrorEntry] = []  # Error message mappings
        
        # Indexed lookups (built on load/add)
        self._blocker_index: Dict[str, List[NodeTransformation]] = defaultdict(list)
        self._pattern_index: Dict[str, List[NodeTransformation]] = defaultdict(list)
        self._model_category_index: Dict[str, List[TransformationRecord]] = defaultdict(list)
        self._op_type_index: Dict[str, List[NodeTransformation]] = defaultdict(list)
        self._template_index: Dict[str, SurgeryTemplate] = {}  # template_id -> template
        self._error_index: Dict[str, CompilationErrorEntry] = {}  # error_id -> entry
        
        # Metadata
        self.version = self.VERSION
        self.created_at: str = ""
        self.last_updated: str = ""
        self.total_models: int = 0
        self.total_transformations: int = 0
    
    # =========================================================================
    # Add Data
    # =========================================================================
    
    def add_transformation_record(self, record: TransformationRecord) -> None:
        """Add a transformation record and update indices."""
        self.transformation_records.append(record)
        
        # Update model category index
        self._model_category_index[record.model_category].append(record)
        
        # Update transformation indices
        for transformation in record.transformations:
            # Blocker index
            if transformation.is_compilation_blocker:
                self._blocker_index[transformation.original_op_type].append(transformation)
            
            # Op type index
            self._op_type_index[transformation.original_op_type].append(transformation)
            
            # Pattern index
            pattern_key = transformation.get_pattern_key()
            self._pattern_index[pattern_key].append(transformation)
        
        # Update stats
        self.total_models = len(self.transformation_records)
        self.total_transformations = sum(
            len(r.transformations) for r in self.transformation_records
        )
        self.last_updated = datetime.now().isoformat()
    
    def add_surgery_template(self, template: SurgeryTemplate) -> None:
        """Add a surgery template."""
        self.surgery_templates.append(template)
        self._template_index[template.template_id] = template
    
    def add_compilation_blocker(self, blocker: CompilationBlocker) -> None:
        """Add a compilation blocker reference."""
        self.compilation_blockers.append(blocker)
    
    def add_compilation_error(self, error: CompilationErrorEntry) -> None:
        """Add a compilation error entry."""
        self.compilation_errors.append(error)
        self._error_index[error.error_id] = error
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def find_similar_blocker(
        self, 
        op_type: str, 
        model_category: Optional[str] = None,
        context: Optional[Dict] = None,
        top_k: int = 5
    ) -> List[NodeTransformation]:
        """
        Find transformations for similar blockers.
        
        Args:
            op_type: Operation type to find (e.g., "Einsum")
            model_category: Optional filter by model category
            context: Optional additional context for matching
            top_k: Maximum number of results
            
        Returns:
            List of matching NodeTransformation objects
        """
        candidates = self._blocker_index.get(op_type, [])
        
        if model_category:
            candidates = [
                t for t in candidates 
                if t.source_model in self._get_models_for_category(model_category)
            ]
        
        # Sort by confidence
        candidates.sort(key=lambda t: t.confidence, reverse=True)
        
        return candidates[:top_k]
    
    def find_transformations_by_op_type(
        self, 
        op_type: str,
        action: Optional[str] = None,
        model_category: Optional[str] = None
    ) -> List[NodeTransformation]:
        """Find all transformations for a specific operation type."""
        candidates = self._op_type_index.get(op_type, [])
        
        if action:
            candidates = [t for t in candidates if t.action == action]
        
        if model_category:
            model_names = set(
                r.model_name for r in self._model_category_index.get(model_category, [])
            )
            candidates = [t for t in candidates if t.source_model in model_names]
        
        return candidates
    
    def find_similar_patterns(
        self, 
        transformation: NodeTransformation,
        top_k: int = 5
    ) -> List[NodeTransformation]:
        """Find transformations with similar patterns."""
        pattern_key = transformation.get_pattern_key()
        exact_matches = self._pattern_index.get(pattern_key, [])
        
        if len(exact_matches) >= top_k:
            return exact_matches[:top_k]
        
        # Also search by op_type if not enough exact matches
        op_matches = self._op_type_index.get(transformation.original_op_type, [])
        
        # Combine and deduplicate
        all_matches = list(exact_matches)
        seen_ids = {(t.original_node_name, t.source_model) for t in exact_matches}
        
        for t in op_matches:
            key = (t.original_node_name, t.source_model)
            if key not in seen_ids:
                all_matches.append(t)
                seen_ids.add(key)
                if len(all_matches) >= top_k:
                    break
        
        return all_matches[:top_k]
    
    def get_surgery_recommendation(
        self, 
        op_type: str, 
        model_category: str,
        context: Optional[Dict] = None
    ) -> Optional[SurgeryTemplate]:
        """
        Get the best surgery template for a blocker.
        
        Args:
            op_type: Operation type that is blocking
            model_category: Category of the model
            context: Additional context (attributes, shapes, etc.)
            
        Returns:
            Best matching SurgeryTemplate or None
        """
        matching_templates = []
        
        for template in self.surgery_templates:
            if template.matches(op_type, model_category, context):
                matching_templates.append(template)
        
        if not matching_templates:
            return None
        
        # Sort by confidence and return best
        matching_templates.sort(key=lambda t: t.confidence, reverse=True)
        return matching_templates[0]
    
    def get_templates_for_op_type(self, op_type: str) -> List[SurgeryTemplate]:
        """Get all templates that handle a specific operation type."""
        return [t for t in self.surgery_templates if t.trigger_op_type == op_type]
    
    def get_blocker_info(self, op_type: str) -> Optional[CompilationBlocker]:
        """Get compilation blocker information for an operation type."""
        for blocker in self.compilation_blockers:
            if blocker.op_type == op_type:
                return blocker
        return None
    
    def get_records_for_category(self, category: str) -> List[TransformationRecord]:
        """Get all transformation records for a model category."""
        return self._model_category_index.get(category, [])
    
    def _get_models_for_category(self, category: str) -> set:
        """Get set of model names for a category."""
        return {r.model_name for r in self._model_category_index.get(category, [])}
    
    # =========================================================================
    # Compilation Error Queries
    # =========================================================================
    
    def find_matching_error(self, error_message: str) -> Optional[CompilationErrorEntry]:
        """
        Find the best matching compilation error entry for an error message.
        
        Args:
            error_message: The actual compilation error message
            
        Returns:
            Best matching CompilationErrorEntry or None
        """
        best_match = None
        best_score = 0
        
        for entry in self.compilation_errors:
            if entry.matches(error_message):
                # Score based on pattern specificity
                score = len(entry.error_pattern)
                if score > best_score:
                    best_score = score
                    best_match = entry
        
        return best_match
    
    def get_error_by_id(self, error_id: str) -> Optional[CompilationErrorEntry]:
        """Get a specific compilation error entry by ID."""
        return self._error_index.get(error_id)
    
    def get_errors_for_op_type(self, op_type: str) -> List[CompilationErrorEntry]:
        """Get all error entries related to a specific operation type."""
        return [e for e in self.compilation_errors if op_type in e.related_op_types]
    
    def diagnose_compilation_error(
        self, 
        error_message: str
    ) -> Dict[str, Any]:
        """
        Diagnose a compilation error and provide actionable guidance.
        
        Args:
            error_message: The compilation error message
            
        Returns:
            Dict with diagnosis including:
            - matched_error: CompilationErrorEntry or None
            - cause: Explanation of why this error occurs
            - fix_steps: List of steps to fix the issue
            - related_templates: List of relevant surgery templates
            - related_blockers: List of related blocker types
        """
        result = {
            'matched_error': None,
            'cause': 'Unknown error cause',
            'fix_steps': [],
            'related_templates': [],
            'related_blockers': []
        }
        
        # Find matching error entry
        error_entry = self.find_matching_error(error_message)
        
        if error_entry:
            result['matched_error'] = error_entry.to_dict()
            result['cause'] = error_entry.cause
            result['fix_steps'] = error_entry.fix_steps
            
            # Get related templates
            for template_id in error_entry.solution_templates:
                template = self._template_index.get(template_id)
                if template:
                    result['related_templates'].append(template.to_dict())
            
            # Get related blocker info
            for op_type in error_entry.related_op_types:
                blocker = self.get_blocker_info(op_type)
                if blocker:
                    result['related_blockers'].append(blocker.to_dict())
        else:
            # Try to provide generic guidance based on keywords
            error_lower = error_message.lower()
            
            if 'shape' in error_lower:
                result['cause'] = 'Shape-related issue (likely dynamic or non-4D tensors)'
                result['fix_steps'] = ['Check for dynamic dimensions', 'Ensure all tensors are 4D']
            elif 'support' in error_lower:
                result['cause'] = 'Operation not supported by target hardware'
                result['fix_steps'] = ['Identify unsupported operations', 'Replace with supported alternatives']
            elif 'memory' in error_lower:
                result['cause'] = 'Memory constraint exceeded'
                result['fix_steps'] = ['Reduce batch size', 'Consider model quantization']
        
        return result
    
    def export_error_context_for_llm(self, error_message: str) -> str:
        """
        Export error diagnosis as formatted context for LLM prompts.
        
        Args:
            error_message: The compilation error message
            
        Returns:
            Formatted string for LLM context
        """
        diagnosis = self.diagnose_compilation_error(error_message)
        
        lines = []
        lines.append("COMPILATION ERROR DIAGNOSIS")
        lines.append("=" * 50)
        lines.append(f"Error: {error_message[:200]}...")
        lines.append("")
        
        if diagnosis['matched_error']:
            entry = diagnosis['matched_error']
            lines.append(f"Matched Pattern: {entry['error_id']}")
            lines.append(f"Category: {entry['error_category']}")
            lines.append(f"Description: {entry['description']}")
        
        lines.append("")
        lines.append(f"CAUSE: {diagnosis['cause']}")
        lines.append("")
        
        if diagnosis['fix_steps']:
            lines.append("HOW TO FIX:")
            for step in diagnosis['fix_steps']:
                lines.append(f"  {step}")
        
        if diagnosis['related_templates']:
            lines.append("")
            lines.append("RECOMMENDED TEMPLATES:")
            for t in diagnosis['related_templates'][:3]:
                lines.append(f"  - {t['name']}: {t.get('description', '')[:100]}")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Export for LLM
    # =========================================================================
    
    def export_for_llm(
        self, 
        op_type: str, 
        model_category: str,
        include_code: bool = True,
        max_examples: int = 3
    ) -> str:
        """
        Export rich context for LLM prompts.
        
        Args:
            op_type: Operation type to get context for
            model_category: Model category
            include_code: Whether to include code snippets
            max_examples: Maximum number of examples to include
            
        Returns:
            Formatted string for LLM context
        """
        lines = []
        
        # Get blocker info
        blocker = self.get_blocker_info(op_type)
        if blocker:
            lines.append(f"BLOCKER: {op_type}")
            lines.append(f"  Reason: {blocker.reason}")
            if blocker.hardware_limitation:
                lines.append(f"  Hardware Limitation: {blocker.hardware_limitation}")
            lines.append("")
        
        # Get similar transformations
        transformations = self.find_similar_blocker(op_type, model_category, top_k=max_examples)
        
        if transformations:
            lines.append(f"SIMILAR FIXES (from {len(transformations)} examples):")
            
            for i, t in enumerate(transformations, 1):
                lines.append(f"\n  Example {i}: {t.source_model}")
                lines.append(f"    Node: {t.original_node_name} ({t.original_op_type})")
                lines.append(f"    Position: {t.graph_position:.2f} in graph")
                
                if t.input_tensors:
                    shapes = [str(tensor.get('shape', '?')) for tensor in t.input_tensors[:3]]
                    lines.append(f"    Input shapes: {', '.join(shapes)}")
                
                if t.attributes:
                    attr_str = ', '.join(f"{k}={v}" for k, v in list(t.attributes.items())[:3])
                    lines.append(f"    Attributes: {attr_str}")
                
                lines.append(f"    Action: {t.action}")
                if t.replacement_ops:
                    lines.append(f"    Replaced with: {' -> '.join(t.replacement_ops)}")
                
                if t.surgery_steps:
                    lines.append("    Steps:")
                    for step in t.surgery_steps[:4]:
                        lines.append(f"      - {step}")
        
        # Get template
        template = self.get_surgery_recommendation(op_type, model_category)
        if template and include_code:
            lines.append(f"\nRECOMMENDED TEMPLATE: {template.name}")
            lines.append(f"  Confidence: {template.confidence:.2f} ({template.success_count}/{template.success_count + template.failure_count} successes)")
            
            if template.steps:
                lines.append("  Steps:")
                for step in template.steps:
                    lines.append(f"    {step.step_number}. {step.description}")
            
            if template.graphsurgeon_code:
                lines.append("\n  Code:")
                lines.append("  ```python")
                for line in template.graphsurgeon_code.split('\n')[:15]:
                    lines.append(f"  {line}")
                lines.append("  ```")
            
            if template.warnings:
                lines.append("  Warnings:")
                for warning in template.warnings:
                    lines.append(f"    - {warning}")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = {
            'version': self.version,
            'total_models': self.total_models,
            'total_transformations': self.total_transformations,
            'total_templates': len(self.surgery_templates),
            'total_blockers_cataloged': len(self.compilation_blockers),
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'models_by_category': {
                cat: len(records) for cat, records in self._model_category_index.items()
            },
            'transformations_by_action': {},
            'transformations_by_op_type': {
                op: len(trans) for op, trans in self._op_type_index.items()
            },
            'blockers_by_op_type': {
                op: len(trans) for op, trans in self._blocker_index.items()
            }
        }
        
        # Count by action
        action_counts = defaultdict(int)
        for record in self.transformation_records:
            for t in record.transformations:
                action_counts[t.action] += 1
        stats['transformations_by_action'] = dict(action_counts)
        
        return stats
    
    # =========================================================================
    # Persistence (JSON Format)
    # =========================================================================
    
    def to_dict(self) -> Dict:
        """Convert entire database to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'total_models': self.total_models,
            'total_transformations': self.total_transformations,
            'transformation_records': [r.to_dict() for r in self.transformation_records],
            'surgery_templates': [t.to_dict() for t in self.surgery_templates],
            'compilation_blockers': [b.to_dict() for b in self.compilation_blockers],
            'compilation_errors': [e.to_dict() for e in self.compilation_errors]
        }
    
    def save(self, path: str) -> None:
        """
        Save database to JSON file.
        
        Args:
            path: Path to save the JSON file
        """
        self.last_updated = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = self.last_updated
        
        data = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved surgery database to {path}")
        print(f"  - {self.total_models} models")
        print(f"  - {self.total_transformations} transformations")
        print(f"  - {len(self.surgery_templates)} templates")
        print(f"  - {len(self.compilation_errors)} error patterns")
    
    @classmethod
    def load(cls, path: str) -> 'SurgeryDatabase':
        """
        Load database from JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Loaded SurgeryDatabase instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        db = cls()
        db.version = data.get('version', cls.VERSION)
        db.created_at = data.get('created_at', '')
        db.last_updated = data.get('last_updated', '')
        
        # Load transformation records
        for record_data in data.get('transformation_records', []):
            record = TransformationRecord.from_dict(record_data)
            db.add_transformation_record(record)
        
        # Load templates
        for template_data in data.get('surgery_templates', []):
            template = SurgeryTemplate.from_dict(template_data)
            db.add_surgery_template(template)
        
        # Load blockers
        for blocker_data in data.get('compilation_blockers', []):
            blocker = CompilationBlocker.from_dict(blocker_data)
            db.add_compilation_blocker(blocker)
        
        # Load compilation errors
        for error_data in data.get('compilation_errors', []):
            error = CompilationErrorEntry.from_dict(error_data)
            db.add_compilation_error(error)
        
        print(f"Loaded surgery database from {path}")
        print(f"  - {db.total_models} models")
        print(f"  - {db.total_transformations} transformations")
        print(f"  - {len(db.surgery_templates)} templates")
        print(f"  - {len(db.compilation_errors)} error patterns")
        
        return db
    
    def save_templates(self, path: str) -> None:
        """Save only templates to a separate JSON file."""
        data = {
            'version': self.version,
            'templates': [t.to_dict() for t in self.surgery_templates]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_blocker_index(self, path: str) -> None:
        """Save blocker index to a separate JSON file for quick lookups."""
        data = {
            'version': self.version,
            'blockers': [b.to_dict() for b in self.compilation_blockers],
            'blocker_transformations': {
                op_type: [t.to_dict() for t in transformations]
                for op_type, transformations in self._blocker_index.items()
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # =========================================================================
    # Index Rebuilding
    # =========================================================================
    
    def rebuild_indices(self) -> None:
        """Rebuild all indices from scratch."""
        # Clear indices
        self._blocker_index.clear()
        self._pattern_index.clear()
        self._model_category_index.clear()
        self._op_type_index.clear()
        self._template_index.clear()
        
        # Rebuild from records
        for record in self.transformation_records:
            self._model_category_index[record.model_category].append(record)
            
            for transformation in record.transformations:
                if transformation.is_compilation_blocker:
                    self._blocker_index[transformation.original_op_type].append(transformation)
                
                self._op_type_index[transformation.original_op_type].append(transformation)
                
                pattern_key = transformation.get_pattern_key()
                self._pattern_index[pattern_key].append(transformation)
        
        # Rebuild template index
        for template in self.surgery_templates:
            self._template_index[template.template_id] = template
        
        # Update stats
        self.total_models = len(self.transformation_records)
        self.total_transformations = sum(
            len(r.transformations) for r in self.transformation_records
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_empty_database() -> SurgeryDatabase:
    """Create a new empty surgery database."""
    db = SurgeryDatabase()
    db.created_at = datetime.now().isoformat()
    db.last_updated = db.created_at
    return db


# =============================================================================
# Default Compilation Blockers (Built-in Knowledge)
# =============================================================================

DEFAULT_COMPILATION_BLOCKERS = [
    CompilationBlocker(
        op_type="Einsum",
        reason="Einsum performs arbitrary tensor contractions that cannot be mapped to fixed hardware operations",
        hardware_limitation="MLA hardware has fixed matrix multiply units without support for arbitrary index permutations",
        solution_templates=["einsum_to_matmul", "einsum_decomposition"],
        affected_categories=["Transformer", "ViT"],
        severity="critical"
    ),
    CompilationBlocker(
        op_type="Where",
        reason="Conditional selection requires data-dependent branching",
        hardware_limitation="MLA executes fixed computation graphs without conditional branching",
        solution_templates=["where_removal", "where_to_select"],
        affected_categories=["Transformer", "CNN"],
        severity="high"
    ),
    CompilationBlocker(
        op_type="NonZero",
        reason="Output shape depends on input data values, not just shape",
        hardware_limitation="Cannot pre-allocate output buffers for dynamic output sizes",
        solution_templates=["nonzero_removal"],
        affected_categories=["all"],
        severity="critical"
    ),
    CompilationBlocker(
        op_type="Loop",
        reason="Dynamic iteration count prevents static graph compilation",
        hardware_limitation="Hardware requires fixed computation graph structure",
        solution_templates=["loop_unroll"],
        affected_categories=["Transformer"],
        severity="critical"
    ),
    CompilationBlocker(
        op_type="If",
        reason="Conditional branching not supported in static computation graphs",
        hardware_limitation="Hardware executes all branches, cannot skip based on condition",
        solution_templates=["if_removal", "if_to_where"],
        affected_categories=["all"],
        severity="critical"
    ),
    CompilationBlocker(
        op_type="NonMaxSuppression",
        reason="Output size depends on detection results (data-dependent)",
        hardware_limitation="Cannot pre-allocate buffers for variable number of detections",
        solution_templates=["nms_postprocess"],
        affected_categories=["YOLO"],
        severity="high"
    ),
    CompilationBlocker(
        op_type="DynamicShape",
        reason="Dynamic dimensions prevent static memory allocation",
        hardware_limitation="All tensor shapes must be known at compile time for buffer allocation",
        solution_templates=["static_shape_enforcement"],
        affected_categories=["all"],
        severity="critical"
    ),
    CompilationBlocker(
        op_type="GatherND",
        reason="Complex N-dimensional indexing may produce dynamic shapes",
        hardware_limitation="Limited support for indirect memory access patterns",
        solution_templates=["gathernd_simplification"],
        affected_categories=["Transformer"],
        severity="medium"
    ),
    CompilationBlocker(
        op_type="ScatterND",
        reason="N-dimensional scatter operations with complex indexing",
        hardware_limitation="Limited support for indirect memory write patterns",
        solution_templates=["scatternd_removal"],
        affected_categories=["Transformer"],
        severity="medium"
    ),
    CompilationBlocker(
        op_type="Scan",
        reason="Sequential processing with state not supported",
        hardware_limitation="Hardware designed for parallel, not sequential, computation",
        solution_templates=["scan_unroll"],
        affected_categories=["Transformer"],
        severity="critical"
    )
]


def create_database_with_defaults() -> SurgeryDatabase:
    """Create a new surgery database with default blocker and error definitions."""
    db = create_empty_database()
    
    # Add default blockers
    for blocker in DEFAULT_COMPILATION_BLOCKERS:
        db.add_compilation_blocker(blocker)
    
    # Add default compilation error patterns
    for error in DEFAULT_COMPILATION_ERRORS:
        db.add_compilation_error(error)
    
    return db


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Create a test database
    db = create_database_with_defaults()
    
    # Print statistics
    stats = db.get_statistics()
    print("Surgery Database Statistics:")
    print(f"  Version: {stats['version']}")
    print(f"  Total models: {stats['total_models']}")
    print(f"  Total transformations: {stats['total_transformations']}")
    print(f"  Blockers cataloged: {stats['total_blockers_cataloged']}")
    
    # Test save/load
    test_path = "rag_data/test_surgery_database.json"
    db.save(test_path)
    
    # Reload
    db2 = SurgeryDatabase.load(test_path)
    print(f"\nReloaded database has {len(db2.compilation_blockers)} blockers")
    
    # Test LLM export
    print("\n" + "="*60)
    print("Example LLM Export for Einsum:")
    print("="*60)
    print(db.export_for_llm("Einsum", "Transformer"))
    
    # Cleanup test file
    import os
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"\nCleaned up test file: {test_path}")
