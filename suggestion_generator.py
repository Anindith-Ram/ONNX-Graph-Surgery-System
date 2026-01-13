#!/usr/bin/env python3
"""
Suggestion Generator for ONNX Model Compilation.

Analyzes ONNX models and generates actionable suggestions with confidence
scores for engineers to manually implement changes.

This is an advisory system - it suggests but does not modify.

Features:
- Comprehensive issue detection
- Confidence scoring based on pattern matching and complexity
- Priority-based suggestion ordering
- Detailed implementation steps for each suggestion
- References to documentation
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import onnx

from onnx_analyzer import ONNXAnalyzer, ModelAnalysis, NodeAnalysis


class Priority(Enum):
    """Suggestion priority levels."""
    CRITICAL = "critical"  # Blocks compilation entirely
    HIGH = "high"          # Affects major portion of model
    MEDIUM = "medium"      # Affects some nodes
    LOW = "low"            # Optimization opportunity
    INFO = "info"          # Observation, no action needed


@dataclass
class SuggestionLocation:
    """Location of an issue in the model."""
    node_id: int
    node_name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)  # NEW: Nodes that produce inputs
    successors: List[str] = field(default_factory=list)  # NEW: Nodes that consume outputs
    graph_position: Optional[float] = None  # NEW: Position in graph (0.0 = inputs, 1.0 = outputs)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Suggestion:
    """A single suggestion for model modification."""
    id: int
    priority: Priority
    confidence: float  # 0.0 to 1.0
    issue: str
    location: SuggestionLocation
    suggestion: str
    implementation_steps: List[str]
    impact: str
    reference: str = ""
    category: str = ""
    estimated_effort: str = ""  # "trivial", "simple", "moderate", "complex"
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['priority'] = self.priority.value
        d['location'] = self.location.to_dict()
        return d


@dataclass
class SuggestionReport:
    """Complete suggestion report for a model."""
    model_name: str
    model_path: str
    compilation_status: str  # "compilable", "blocked", "partially_blocked"
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    suggestions: List[Suggestion]
    summary: str
    analysis_timestamp: str = ""
    analyzer_version: str = "1.0.0"
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'compilation_status': self.compilation_status,
            'total_issues': self.total_issues,
            'issue_counts': {
                'critical': self.critical_count,
                'high': self.high_count,
                'medium': self.medium_count,
                'low': self.low_count
            },
            'suggestions': [s.to_dict() for s in self.suggestions],
            'summary': self.summary,
            'analysis_timestamp': self.analysis_timestamp,
            'analyzer_version': self.analyzer_version
        }


class ConfidenceScorer:
    """
    Computes confidence scores for suggestions.
    
    Confidence based on:
    - Pattern match: How well the issue matches known patterns
    - Complexity: Simple fixes get higher confidence
    - Documentation: Referenced in official docs
    - Evidence: Seen in training dataset
    """
    
    CONFIDENCE_FACTORS = {
        'pattern_match': 0.30,
        'complexity': 0.25,
        'documentation': 0.20,
        'evidence': 0.25
    }
    
    # Known patterns with high confidence
    HIGH_CONFIDENCE_PATTERNS = {
        'Einsum': 0.95,           # Well-documented replacement
        'Identity': 0.99,         # Trivial to remove
        'Dropout': 0.99,          # Training artifact
        'Squeeze': 0.90,          # Simple rewiring
        'Unsqueeze': 0.90,        # Simple rewiring
        'NonZero': 0.70,          # Complex, may need redesign
        'Where': 0.75,            # Conditional, needs analysis
        'Loop': 0.50,             # Very complex
        'If': 0.50,               # Very complex
        'dynamic_shape': 0.85,    # Usually fixable
        'non_4d_tensor': 0.90,    # Add reshape
    }
    
    # Complexity scores (higher = simpler = more confident)
    COMPLEXITY_SCORES = {
        'trivial': 1.0,    # Remove node, no side effects
        'simple': 0.85,    # Single node replacement
        'moderate': 0.65,  # Multiple node changes
        'complex': 0.40,   # Architectural change needed
    }
    
    def compute_confidence(
        self,
        issue_type: str,
        op_type: str,
        complexity: str,
        has_documentation: bool,
        seen_in_dataset: bool
    ) -> float:
        """
        Compute confidence score for a suggestion.
        
        Returns:
            Float between 0.0 and 1.0
        """
        scores = {}
        
        # Pattern match score
        if op_type in self.HIGH_CONFIDENCE_PATTERNS:
            scores['pattern_match'] = self.HIGH_CONFIDENCE_PATTERNS[op_type]
        elif issue_type in self.HIGH_CONFIDENCE_PATTERNS:
            scores['pattern_match'] = self.HIGH_CONFIDENCE_PATTERNS[issue_type]
        else:
            scores['pattern_match'] = 0.5  # Unknown pattern
        
        # Complexity score
        scores['complexity'] = self.COMPLEXITY_SCORES.get(complexity, 0.5)
        
        # Documentation score
        scores['documentation'] = 0.9 if has_documentation else 0.3
        
        # Evidence score
        scores['evidence'] = 0.9 if seen_in_dataset else 0.4
        
        # Weighted average
        confidence = sum(
            scores[factor] * weight
            for factor, weight in self.CONFIDENCE_FACTORS.items()
        )
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]


class SuggestionGenerator:
    """
    Main suggestion generator for ONNX models.
    
    Analyzes models and generates prioritized suggestions with
    confidence scores for engineers to implement.
    """
    
    # Suggestion templates for common issues
    SUGGESTION_TEMPLATES = {
        'Einsum': {
            'suggestion': "Replace Einsum with MatMul + Reshape/Transpose sequence",
            'steps': [
                "1. Analyze the Einsum equation to understand the tensor contraction",
                "2. Identify if it's a batch matmul pattern (e.g., 'bhid,bhjd->bhij')",
                "3. If batch matmul: use MatMul directly (handles batched inputs)",
                "4. If complex: decompose into Transpose + MatMul + Reshape",
                "5. Verify output shape matches original Einsum output"
            ],
            'reference': "ONNX Graph Surgery Guide - Section: Einsum Replacement",
            'complexity': 'moderate',
            'category': 'unsupported_operation'
        },
        'NonZero': {
            'suggestion': "Replace NonZero with static indexing or Where operation",
            'steps': [
                "1. Analyze what NonZero is selecting (threshold, mask, etc.)",
                "2. If selecting top-K: use TopK with static K",
                "3. If masking: consider Where with static output size",
                "4. May require model architecture changes if truly dynamic"
            ],
            'reference': "ONNX Graph Surgery Guide - Dynamic Operations",
            'complexity': 'complex',
            'category': 'dynamic_output'
        },
        'Where': {
            'suggestion': "Replace conditional Where with Mul+Add pattern or static selection",
            'steps': [
                "1. Analyze the condition being tested",
                "2. If binary mask: use Mul(x, mask) + Mul(y, 1-mask)",
                "3. If threshold: consider Clip or custom logic",
                "4. Ensure output shape is statically determinable"
            ],
            'reference': "ONNX Graph Surgery Guide - Conditional Operations",
            'complexity': 'moderate',
            'category': 'conditional_operation'
        },
        'Loop': {
            'suggestion': "Unroll loop or replace with static equivalent",
            'steps': [
                "1. Determine if loop count is static or dynamic",
                "2. If static: unroll the loop into sequential operations",
                "3. If dynamic: consider redesigning this part of the model",
                "4. May require architectural changes to the model"
            ],
            'reference': "ONNX Graph Surgery Guide - Control Flow",
            'complexity': 'complex',
            'category': 'control_flow'
        },
        'If': {
            'suggestion': "Replace conditional branch with both-path computation",
            'steps': [
                "1. Compute both branches unconditionally",
                "2. Use Mul+Add pattern to select correct output",
                "3. Condition becomes a selection mask",
                "4. Both branches must have compatible output shapes"
            ],
            'reference': "ONNX Graph Surgery Guide - Control Flow",
            'complexity': 'complex',
            'category': 'control_flow'
        },
        'Identity': {
            'suggestion': "Remove Identity node and rewire connections",
            'steps': [
                "1. Note the Identity node's input tensor name",
                "2. Note all nodes that consume the Identity's output",
                "3. Update those nodes to consume the input directly",
                "4. Remove the Identity node"
            ],
            'reference': "ONNX Graph Surgery Guide - Graph Simplification",
            'complexity': 'trivial',
            'category': 'optimization'
        },
        'Dropout': {
            'suggestion': "Remove Dropout node (training artifact)",
            'steps': [
                "1. Dropout is only needed during training",
                "2. For inference, remove the node entirely",
                "3. Rewire: connect Dropout's input to its output consumers",
                "4. This is a no-op removal with no mathematical impact"
            ],
            'reference': "ONNX Graph Surgery Guide - Inference Optimization",
            'complexity': 'trivial',
            'category': 'training_artifact'
        },
        'Squeeze': {
            'suggestion': "Remove Squeeze and adjust downstream nodes for dimension handling",
            'steps': [
                "1. Identify which dimensions are being squeezed",
                "2. Check if downstream nodes can handle the unsqueezed shape",
                "3. If yes: remove Squeeze and update downstream",
                "4. If no: add Reshape at model output instead"
            ],
            'reference': "ONNX Graph Surgery Guide - Shape Operations",
            'complexity': 'simple',
            'category': 'shape_operation'
        },
        'Unsqueeze': {
            'suggestion': "Remove Unsqueeze and handle dimensions differently",
            'steps': [
                "1. Identify which dimensions are being added",
                "2. Check if upstream node can produce correct shape",
                "3. Consider using Reshape instead for 4D conversion",
                "4. Verify all downstream consumers handle new shape"
            ],
            'reference': "ONNX Graph Surgery Guide - Shape Operations",
            'complexity': 'simple',
            'category': 'shape_operation'
        },
        'dynamic_shape': {
            'suggestion': "Replace dynamic shape with static concrete values",
            'steps': [
                "1. Identify the source of dynamic dimension (batch, sequence, etc.)",
                "2. Determine appropriate static value for deployment",
                "3. Update graph input shape with concrete dimension",
                "4. Run shape inference to propagate changes",
                "5. Fix any downstream shape mismatches"
            ],
            'reference': "ONNX Graph Surgery Guide - Static Shapes",
            'complexity': 'simple',
            'category': 'dynamic_shape'
        },
        'non_4d_tensor': {
            'suggestion': "Add Reshape to convert tensor to 4D format",
            'steps': [
                "1. Determine current tensor shape (e.g., [B, N, C])",
                "2. Choose 4D mapping (e.g., [B, C, N, 1] or [B, C, H, W])",
                "3. Insert Reshape node after the producer",
                "4. Update all consumers to use reshaped output",
                "5. Add inverse Reshape before model output if needed"
            ],
            'reference': "ONNX Graph Surgery Guide - MLA 4D Requirement",
            'complexity': 'simple',
            'category': 'tensor_format'
        },
        'LayerNormalization': {
            'suggestion': "Decompose LayerNorm into primitive operations",
            'steps': [
                "1. LayerNorm = (x - mean) / sqrt(var + eps) * scale + bias",
                "2. Replace with: ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add",
                "3. Create constants for epsilon and exponent (2)",
                "4. Preserve scale and bias initializers",
                "5. Verify numerical equivalence"
            ],
            'reference': "ONNX Graph Surgery Guide - Normalization Layers",
            'complexity': 'moderate',
            'category': 'decomposition'
        },
        'Gelu': {
            'suggestion': "Replace GELU with sigmoid approximation",
            'steps': [
                "1. GELU(x) ≈ x * sigmoid(1.702 * x)",
                "2. Create constant tensor with value 1.702",
                "3. Add Mul node: scaled = x * 1.702",
                "4. Add Sigmoid node: sig = sigmoid(scaled)",
                "5. Add Mul node: output = x * sig",
                "6. Remove original GELU node"
            ],
            'reference': "ONNX Graph Surgery Guide - Activation Functions",
            'complexity': 'simple',
            'category': 'activation'
        },
    }
    
    def _extract_node_context(
        self,
        node: Optional[NodeAnalysis],
        analysis: ModelAnalysis,
        node_id: int
    ) -> Tuple[List[str], List[str], Optional[float]]:
        """
        Extract multi-node context: predecessors, successors, and graph position.
        
        Returns:
            Tuple of (predecessors, successors, graph_position)
        """
        if not node:
            return [], [], None
        
        # Load model to get full graph structure
        try:
            model = onnx.load(analysis.model_path)
        except:
            return [], [], None
        
        # Build tensor producer/consumer maps
        tensor_to_producer = {}
        tensor_to_consumers = defaultdict(list)
        node_name_to_node = {}
        
        for n in model.graph.node:
            node_name_to_node[n.name] = n
            for output in n.output:
                tensor_to_producer[output] = n.name
            for input_tensor in n.input:
                tensor_to_consumers[input_tensor].append(n.name)
        
        # Find predecessors: nodes that produce this node's inputs
        predecessors = []
        for input_tensor in node.inputs:
            producer = tensor_to_producer.get(input_tensor)
            if producer and producer not in predecessors:
                predecessors.append(producer)
        
        # Find successors: nodes that consume this node's outputs
        successors = []
        for output_tensor in node.outputs:
            consumers = tensor_to_consumers.get(output_tensor, [])
            for consumer in consumers:
                if consumer not in successors:
                    successors.append(consumer)
        
        # Calculate graph position: 0.0 = near inputs, 1.0 = near outputs
        if node.name in node_name_to_node:
            nodes = list(model.graph.node)
            try:
                node_index = next(i for i, n in enumerate(nodes) if n.name == node.name)
                graph_position = node_index / max(len(nodes), 1)  # Normalize to [0, 1]
            except (StopIteration, ValueError):
                graph_position = None
        else:
            graph_position = None
        
        return predecessors, successors, graph_position
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize suggestion generator."""
        self.analyzer = ONNXAnalyzer()
        self.scorer = ConfidenceScorer()
        self.api_key = api_key
        
        # Load dataset patterns if available
        self.known_patterns = self._load_known_patterns()
    
    def _load_known_patterns(self) -> Dict:
        """Load patterns from dataset analysis."""
        patterns_file = Path("analysis_report.json")
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def analyze_and_suggest(self, model_path: str) -> SuggestionReport:
        """
        Analyze model and generate suggestions.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            SuggestionReport with all suggestions
        """
        import time
        
        # Analyze model
        analysis = self.analyzer.analyze(model_path)
        
        # Generate suggestions
        suggestions = []
        suggestion_id = 1
        
        # 1. Process compilation blockers
        for blocker in analysis.compilation_blockers:
            suggestion = self._create_suggestion_for_blocker(
                suggestion_id, blocker, analysis
            )
            if suggestion:
                suggestions.append(suggestion)
                suggestion_id += 1
        
        # 2. Check for non-4D tensors
        non_4d_suggestions = self._check_non_4d_tensors(suggestion_id, analysis)
        suggestions.extend(non_4d_suggestions)
        suggestion_id += len(non_4d_suggestions)
        
        # 3. Check for optimization opportunities
        optimization_suggestions = self._check_optimizations(suggestion_id, analysis)
        suggestions.extend(optimization_suggestions)
        suggestion_id += len(optimization_suggestions)
        
        # 4. Check for dynamic shapes
        dynamic_suggestions = self._check_dynamic_shapes(suggestion_id, analysis)
        suggestions.extend(dynamic_suggestions)
        
        # Sort by priority and confidence
        suggestions = self._prioritize_suggestions(suggestions)
        
        # Count by priority
        critical = sum(1 for s in suggestions if s.priority == Priority.CRITICAL)
        high = sum(1 for s in suggestions if s.priority == Priority.HIGH)
        medium = sum(1 for s in suggestions if s.priority == Priority.MEDIUM)
        low = sum(1 for s in suggestions if s.priority == Priority.LOW)
        
        # Determine compilation status
        if critical > 0:
            status = "blocked"
        elif high > 0:
            status = "partially_blocked"
        else:
            status = "likely_compilable"
        
        # Generate summary
        summary = self._generate_summary(suggestions, analysis)
        
        return SuggestionReport(
            model_name=analysis.model_name,
            model_path=model_path,
            compilation_status=status,
            total_issues=len(suggestions),
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            suggestions=suggestions,
            summary=summary,
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _create_suggestion_for_blocker(
        self,
        suggestion_id: int,
        blocker: Dict,
        analysis: ModelAnalysis
    ) -> Optional[Suggestion]:
        """Create suggestion for a compilation blocker."""
        op_type = blocker.get('op_type', 'Unknown')
        node_id = blocker.get('node_id', -1)
        reason = blocker.get('reason', '')
        
        # Get node info
        node = None
        if 0 <= node_id < len(analysis.nodes):
            node = analysis.nodes[node_id]
        
        # Get template
        template = self.SUGGESTION_TEMPLATES.get(op_type, {})
        
        # Extract multi-node context (predecessors, successors, graph position)
        predecessors, successors, graph_position = self._extract_node_context(
            node, analysis, node_id
        )
        
        # Create location with multi-node context
        location = SuggestionLocation(
            node_id=node_id,
            node_name=node.name if node else f"node_{node_id}",
            op_type=op_type,
            inputs=node.inputs if node else [],
            outputs=node.outputs if node else [],
            predecessors=predecessors,
            successors=successors,
            graph_position=graph_position
        )
        
        # Determine priority based on blocker type
        if op_type in ['Loop', 'If', 'Scan']:
            priority = Priority.CRITICAL
        elif op_type in ['Einsum', 'NonZero', 'Where']:
            priority = Priority.CRITICAL
        elif op_type in ['Identity', 'Dropout']:
            priority = Priority.LOW
        else:
            priority = Priority.HIGH
        
        # Get suggestion details
        suggestion_text = template.get('suggestion', f"Address {op_type} operation: {reason}")
        steps = template.get('steps', [f"1. Analyze the {op_type} node", "2. Determine appropriate replacement"])
        reference = template.get('reference', "ONNX Graph Surgery Guide")
        complexity = template.get('complexity', 'moderate')
        category = template.get('category', 'blocker')
        
        # Compute confidence
        confidence = self.scorer.compute_confidence(
            issue_type=op_type,
            op_type=op_type,
            complexity=complexity,
            has_documentation=bool(template),
            seen_in_dataset=op_type in self.known_patterns.get('op_type_statistics', {})
        )
        
        # Calculate impact
        dependent_count = len(node.dependents) if node else 0
        impact = f"Blocking node affects {dependent_count} downstream nodes"
        
        return Suggestion(
            id=suggestion_id,
            priority=priority,
            confidence=confidence,
            issue=f"{op_type}: {reason}",
            location=location,
            suggestion=suggestion_text,
            implementation_steps=steps,
            impact=impact,
            reference=reference,
            category=category,
            estimated_effort=complexity
        )
    
    def _check_non_4d_tensors(
        self,
        start_id: int,
        analysis: ModelAnalysis
    ) -> List[Suggestion]:
        """Check for non-4D tensors that need reshaping."""
        suggestions = []
        seen_tensors = set()
        
        for node in analysis.nodes:
            for i, shape in enumerate(node.output_shapes):
                if shape and len(shape) != 4:
                    output_name = node.outputs[i] if i < len(node.outputs) else "unknown"
                    
                    if output_name in seen_tensors:
                        continue
                    seen_tensors.add(output_name)
                    
                    template = self.SUGGESTION_TEMPLATES.get('non_4d_tensor', {})
                    
                    # Extract multi-node context
                    predecessors, successors, graph_position = self._extract_node_context(
                        node, analysis, node.node_id
                    )
                    
                    location = SuggestionLocation(
                        node_id=node.node_id,
                        node_name=node.name,
                        op_type=node.op_type,
                        inputs=node.inputs,
                        outputs=node.outputs,
                        predecessors=predecessors,
                        successors=successors,
                        graph_position=graph_position
                    )
                    
                    confidence = self.scorer.compute_confidence(
                        issue_type='non_4d_tensor',
                        op_type=node.op_type,
                        complexity='simple',
                        has_documentation=True,
                        seen_in_dataset=True
                    )
                    
                    suggestions.append(Suggestion(
                        id=start_id + len(suggestions),
                        priority=Priority.HIGH,
                        confidence=confidence,
                        issue=f"Non-4D tensor output: {output_name} has shape {list(shape)} ({len(shape)}D)",
                        location=location,
                        suggestion=template.get('suggestion', "Convert tensor to 4D format"),
                        implementation_steps=template.get('steps', [
                            f"1. Current shape: {list(shape)}",
                            f"2. Reshape to 4D: e.g., {self._suggest_4d_shape(shape)}",
                            "3. Insert Reshape node after this operation",
                            "4. Update downstream consumers"
                        ]),
                        impact=f"Required for MLA compilation - affects {len(node.dependents)} nodes",
                        reference=template.get('reference', "ONNX Graph Surgery Guide"),
                        category='tensor_format',
                        estimated_effort='simple'
                    ))
        
        return suggestions
    
    def _suggest_4d_shape(self, shape: Tuple) -> str:
        """Suggest a 4D shape for a non-4D tensor."""
        if len(shape) == 1:
            return f"[1, {shape[0]}, 1, 1]"
        elif len(shape) == 2:
            return f"[{shape[0]}, {shape[1]}, 1, 1]"
        elif len(shape) == 3:
            return f"[{shape[0]}, {shape[1]}, {shape[2]}, 1]"
        elif len(shape) == 5:
            return f"[{shape[0]}, {shape[1]}*{shape[2]}, {shape[3]}, {shape[4]}]"
        else:
            return f"[1, ?, ?, ?] - manual analysis needed"
    
    def _check_optimizations(
        self,
        start_id: int,
        analysis: ModelAnalysis
    ) -> List[Suggestion]:
        """Check for optimization opportunities."""
        suggestions = []
        
        # Check for Identity nodes
        identity_nodes = [n for n in analysis.nodes if n.op_type == 'Identity']
        if identity_nodes:
            template = self.SUGGESTION_TEMPLATES.get('Identity', {})
            
            suggestions.append(Suggestion(
                id=start_id,
                priority=Priority.LOW,
                confidence=0.99,
                issue=f"Found {len(identity_nodes)} Identity nodes that can be removed",
                location=SuggestionLocation(
                    node_id=identity_nodes[0].node_id,
                    node_name="Multiple Identity nodes",
                    op_type="Identity",
                    inputs=[],
                    outputs=[]
                ),
                suggestion=template.get('suggestion', "Remove Identity nodes"),
                implementation_steps=template.get('steps', [
                    "1. For each Identity node, rewire input to output",
                    "2. Remove the Identity node"
                ]),
                impact=f"Simplifies graph by removing {len(identity_nodes)} unnecessary nodes",
                reference=template.get('reference', "ONNX Graph Surgery Guide"),
                category='optimization',
                estimated_effort='trivial'
            ))
        
        # Check for Dropout nodes
        dropout_nodes = [n for n in analysis.nodes if n.op_type == 'Dropout']
        if dropout_nodes:
            template = self.SUGGESTION_TEMPLATES.get('Dropout', {})
            
            suggestions.append(Suggestion(
                id=start_id + len(suggestions),
                priority=Priority.LOW,
                confidence=0.99,
                issue=f"Found {len(dropout_nodes)} Dropout nodes (training artifacts)",
                location=SuggestionLocation(
                    node_id=dropout_nodes[0].node_id,
                    node_name="Multiple Dropout nodes",
                    op_type="Dropout",
                    inputs=[],
                    outputs=[]
                ),
                suggestion=template.get('suggestion', "Remove Dropout nodes for inference"),
                implementation_steps=template.get('steps', [
                    "1. For each Dropout, connect input directly to output consumers",
                    "2. Remove the Dropout node"
                ]),
                impact=f"Removes {len(dropout_nodes)} training artifacts",
                reference=template.get('reference', "ONNX Graph Surgery Guide"),
                category='training_artifact',
                estimated_effort='trivial'
            ))
        
        # Check for consecutive Reshape nodes
        consecutive_reshapes = []
        for i, node in enumerate(analysis.nodes[:-1]):
            if node.op_type == 'Reshape':
                next_node = analysis.nodes[i + 1]
                if next_node.op_type == 'Reshape':
                    consecutive_reshapes.append((node, next_node))
        
        if consecutive_reshapes:
            suggestions.append(Suggestion(
                id=start_id + len(suggestions),
                priority=Priority.LOW,
                confidence=0.90,
                issue=f"Found {len(consecutive_reshapes)} consecutive Reshape pairs that can be fused",
                location=SuggestionLocation(
                    node_id=consecutive_reshapes[0][0].node_id,
                    node_name=f"{consecutive_reshapes[0][0].name} -> {consecutive_reshapes[0][1].name}",
                    op_type="Reshape",
                    inputs=[],
                    outputs=[]
                ),
                suggestion="Fuse consecutive Reshape operations",
                implementation_steps=[
                    "1. For each Reshape pair, keep only the second Reshape's target shape",
                    "2. Update the second Reshape to take the first's input",
                    "3. Remove the first Reshape node"
                ],
                impact=f"Reduces {len(consecutive_reshapes)} unnecessary reshape operations",
                reference="ONNX Graph Surgery Guide - Graph Simplification",
                category='optimization',
                estimated_effort='simple'
            ))
        
        return suggestions
    
    def _check_dynamic_shapes(
        self,
        start_id: int,
        analysis: ModelAnalysis
    ) -> List[Suggestion]:
        """Check for dynamic shapes that need fixing."""
        suggestions = []
        
        if analysis.dynamic_dimensions:
            # Group by tensor
            by_tensor = defaultdict(list)
            for dim in analysis.dynamic_dimensions:
                by_tensor[dim['tensor']].append(dim)
            
            template = self.SUGGESTION_TEMPLATES.get('dynamic_shape', {})
            
            for tensor_name, dims in by_tensor.items():
                dim_info = ", ".join([f"dim[{d['dimension']}]={d['value']}" for d in dims])
                
                confidence = self.scorer.compute_confidence(
                    issue_type='dynamic_shape',
                    op_type='Shape',
                    complexity='simple',
                    has_documentation=True,
                    seen_in_dataset=True
                )
                
                suggestions.append(Suggestion(
                    id=start_id + len(suggestions),
                    priority=Priority.MEDIUM,
                    confidence=confidence,
                    issue=f"Dynamic dimensions in tensor '{tensor_name}': {dim_info}",
                    location=SuggestionLocation(
                        node_id=-1,
                        node_name=tensor_name,
                        op_type="tensor",
                        inputs=[],
                        outputs=[]
                    ),
                    suggestion=template.get('suggestion', "Replace dynamic dimensions with static values"),
                    implementation_steps=[
                        f"1. Tensor '{tensor_name}' has dynamic dims: {dim_info}",
                        "2. Determine appropriate static values for deployment",
                        "3. Update the tensor shape with concrete dimensions",
                        "4. Run shape inference to propagate changes"
                    ],
                    impact="Required for hardware compilation - shapes must be static",
                    reference=template.get('reference', "ONNX Graph Surgery Guide"),
                    category='dynamic_shape',
                    estimated_effort='simple'
                ))
        
        return suggestions
    
    def _prioritize_suggestions(
        self,
        suggestions: List[Suggestion]
    ) -> List[Suggestion]:
        """Sort suggestions by priority and confidence."""
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
            Priority.INFO: 4
        }
        
        return sorted(
            suggestions,
            key=lambda s: (priority_order[s.priority], -s.confidence)
        )
    
    def _generate_summary(
        self,
        suggestions: List[Suggestion],
        analysis: ModelAnalysis
    ) -> str:
        """Generate human-readable summary."""
        if not suggestions:
            return "No compilation issues detected. Model appears to be MLA-compatible."
        
        critical = sum(1 for s in suggestions if s.priority == Priority.CRITICAL)
        high = sum(1 for s in suggestions if s.priority == Priority.HIGH)
        medium = sum(1 for s in suggestions if s.priority == Priority.MEDIUM)
        low = sum(1 for s in suggestions if s.priority == Priority.LOW)
        
        parts = []
        if critical:
            parts.append(f"{critical} critical")
        if high:
            parts.append(f"{high} high")
        if medium:
            parts.append(f"{medium} medium")
        if low:
            parts.append(f"{low} low priority")
        
        issue_summary = ", ".join(parts)
        
        # Find most impactful suggestions
        high_confidence = [s for s in suggestions if s.confidence >= 0.9]
        
        summary = f"Found {len(suggestions)} issues ({issue_summary}). "
        
        if critical > 0:
            summary += f"Model CANNOT compile until {critical} critical issue(s) are resolved. "
        
        if high_confidence:
            summary += f"{len(high_confidence)} suggestions have high confidence (≥90%)."
        
        return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python suggestion_generator.py <model.onnx>")
        sys.exit(1)
    
    generator = SuggestionGenerator()
    report = generator.analyze_and_suggest(sys.argv[1])
    
    print(json.dumps(report.to_dict(), indent=2))

