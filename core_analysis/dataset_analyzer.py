#!/usr/bin/env python3
"""
Dataset Analyzer for Model Surgery Pipeline.

Analyzes original/modified ONNX model pairs to discover:
- Common transformation patterns
- Frequently removed/added operations
- Shape change patterns
- Candidates for deterministic templates

This helps identify which patterns should be added as templates
to reduce reliance on LLM-based modifications.

No API calls required - purely offline analysis.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import onnx
from onnx import numpy_helper


@dataclass
class NodeDiff:
    """Difference for a single node."""
    node_name: str
    op_type: str
    change_type: str  # 'added', 'removed', 'modified'
    details: Dict = field(default_factory=dict)


@dataclass
class ONNXModelDiff:
    """Differences between original and modified ONNX model (direct comparison)."""
    model_name: str
    original_node_count: int
    modified_node_count: int
    added_nodes: List[NodeDiff] = field(default_factory=list)
    removed_nodes: List[NodeDiff] = field(default_factory=list)
    modified_nodes: List[NodeDiff] = field(default_factory=list)
    op_type_changes: Dict[str, int] = field(default_factory=dict)  # op_type -> count
    shape_changes: List[Dict] = field(default_factory=list)
    original_model_path: Optional[str] = None  # NEW: Path to original model for context extraction
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'original_node_count': self.original_node_count,
            'modified_node_count': self.modified_node_count,
            'added_nodes': [asdict(n) for n in self.added_nodes],
            'removed_nodes': [asdict(n) for n in self.removed_nodes],
            'modified_nodes': [asdict(n) for n in self.modified_nodes],
            'op_type_changes': self.op_type_changes,
            'shape_changes': self.shape_changes
        }


@dataclass
class Pattern:
    """A detected transformation pattern."""
    name: str
    description: str
    frequency: int  # How many models exhibit this pattern
    op_types_involved: List[str]
    example_models: List[str]
    template_candidate: bool = False
    implementation_hint: str = ""
    model_category: str = "Other"  # NEW: Model category (YOLO, Transformer, etc.)
    context: Dict = field(default_factory=dict)  # NEW: Context about when pattern applies
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    total_models: int
    model_diffs: List[ONNXModelDiff]
    patterns: List[Pattern]
    op_type_statistics: Dict[str, Dict]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'total_models': self.total_models,
            'model_diffs': [d.to_dict() for d in self.model_diffs],
            'patterns': [p.to_dict() for p in self.patterns],
            'op_type_statistics': self.op_type_statistics,
            'recommendations': self.recommendations
        }


class DatasetAnalyzer:
    """
    Analyze ONNX model pairs to discover transformation patterns.
    
    Usage:
        analyzer = DatasetAnalyzer()
        report = analyzer.analyze_dataset("dataset")
        print(analyzer.generate_report(report))
    """
    
    def __init__(self):
        self.diffs: List[ONNXModelDiff] = []
    
    def _detect_model_category(self, model_name: str) -> str:
        """Detect model category from model name."""
        name_lower = model_name.lower()
        
        if 'yolo' in name_lower:
            return "YOLO"
        elif 'vit' in name_lower or 'vision_transformer' in name_lower:
            return "ViT"
        elif 't5' in name_lower or 'transformer' in name_lower or 'bert' in name_lower or 'mt5' in name_lower:
            return "Transformer"
        elif 'cnn' in name_lower or 'resnet' in name_lower or 'mobilenet' in name_lower:
            return "CNN"
        else:
            return "Other"
    
    def _extract_context(
        self,
        node: NodeDiff,
        diff: ONNXModelDiff,
        original_model: Optional[onnx.ModelProto] = None
    ) -> Dict:
        """Extract context about when operation is removed/added."""
        context = {
            'position': 'unknown',  # 'near_output', 'near_input', 'middle'
            'surrounding_ops': [],  # Operations before/after
            'pattern_type': 'removal' if node.change_type == 'removed' else 'addition'
        }
        
        # Try to get position from original model if available
        if original_model:
            try:
                # Find node in original model
                node_idx = None
                for i, n in enumerate(original_model.graph.node):
                    if n.name == node.node_name:
                        node_idx = i
                        break
                
                if node_idx is not None:
                    total_nodes = len(original_model.graph.node)
                    position_ratio = node_idx / total_nodes if total_nodes > 0 else 0.5
                    
                    if position_ratio > 0.8:
                        context['position'] = 'near_output'
                    elif position_ratio < 0.2:
                        context['position'] = 'near_input'
                    else:
                        context['position'] = 'middle'
                    
                    # Get surrounding operations
                    if node_idx > 0:
                        context['surrounding_ops'].append(
                            original_model.graph.node[node_idx-1].op_type
                        )
                    if node_idx < total_nodes - 1:
                        context['surrounding_ops'].append(
                            original_model.graph.node[node_idx+1].op_type
                        )
            except Exception:
                pass  # If we can't extract context, use defaults
        
        return context
        
    def analyze_dataset(self, dataset_dir: str) -> AnalysisReport:
        """
        Analyze all model pairs in the dataset.
        
        Args:
            dataset_dir: Directory containing model folders with original/modified subdirs
            
        Returns:
            Complete analysis report
        """
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        model_diffs = []
        
        # Find all model pairs
        for model_dir in sorted(dataset_path.iterdir()):
            if not model_dir.is_dir():
                continue
            
            original_dir = model_dir / "original"
            modified_dir = model_dir / "modified"
            
            if not original_dir.exists() or not modified_dir.exists():
                continue
            
            # Find ONNX files
            original_files = list(original_dir.glob("*.onnx"))
            modified_files = list(modified_dir.glob("*.onnx"))
            
            if not original_files or not modified_files:
                continue
            
            # Use first ONNX file found
            original_path = original_files[0]
            modified_path = modified_files[0]
            
            print(f"Analyzing: {model_dir.name}")
            
            try:
                diff = self._compute_diff(
                    str(original_path),
                    str(modified_path),
                    model_dir.name
                )
                diff.original_model_path = str(original_path)  # Store path for context extraction
                model_diffs.append(diff)
            except Exception as e:
                print(f"  Error analyzing {model_dir.name}: {e}")
        
        self.diffs = model_diffs
        
        # Find patterns
        patterns = self._find_patterns(model_diffs)
        
        # Compute statistics
        op_stats = self._compute_op_statistics(model_diffs)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, op_stats)
        
        return AnalysisReport(
            total_models=len(model_diffs),
            model_diffs=model_diffs,
            patterns=patterns,
            op_type_statistics=op_stats,
            recommendations=recommendations
        )
    
    def _compute_diff(
        self,
        original_path: str,
        modified_path: str,
        model_name: str
    ) -> ONNXModelDiff:
        """Compute differences between two ONNX models."""
        original = onnx.load(original_path)
        modified = onnx.load(modified_path)
        
        diff = ONNXModelDiff(
            model_name=model_name,
            original_node_count=len(original.graph.node),
            modified_node_count=len(modified.graph.node)
        )
        
        # Build node maps
        orig_nodes = {self._node_signature(n): n for n in original.graph.node}
        mod_nodes = {self._node_signature(n): n for n in modified.graph.node}
        
        # Also map by output names for better matching
        orig_by_output = {}
        for node in original.graph.node:
            for out in node.output:
                orig_by_output[out] = node
        
        mod_by_output = {}
        for node in modified.graph.node:
            for out in node.output:
                mod_by_output[out] = node
        
        # Find removed nodes
        for sig, node in orig_nodes.items():
            if sig not in mod_nodes:
                # Check if a similar node exists with same outputs
                matched = False
                for out in node.output:
                    if out in mod_by_output:
                        mod_node = mod_by_output[out]
                        if mod_node.op_type != node.op_type:
                            diff.modified_nodes.append(NodeDiff(
                                node_name=node.name,
                                op_type=node.op_type,
                                change_type='modified',
                                details={
                                    'original_op': node.op_type,
                                    'new_op': mod_node.op_type
                                }
                            ))
                            diff.op_type_changes[f"{node.op_type}->{mod_node.op_type}"] = \
                                diff.op_type_changes.get(f"{node.op_type}->{mod_node.op_type}", 0) + 1
                            matched = True
                            break
                
                if not matched:
                    diff.removed_nodes.append(NodeDiff(
                        node_name=node.name,
                        op_type=node.op_type,
                        change_type='removed'
                    ))
        
        # Find added nodes
        for sig, node in mod_nodes.items():
            if sig not in orig_nodes:
                # Check if it's a modification we already counted
                already_counted = False
                for out in node.output:
                    if out in orig_by_output:
                        already_counted = True
                        break
                
                if not already_counted:
                    diff.added_nodes.append(NodeDiff(
                        node_name=node.name,
                        op_type=node.op_type,
                        change_type='added'
                    ))
        
        # Analyze shape changes
        diff.shape_changes = self._analyze_shape_changes(original, modified)
        
        return diff
    
    def _node_signature(self, node: onnx.NodeProto) -> str:
        """Create a signature for a node."""
        attrs = []
        for attr in sorted(node.attribute, key=lambda a: a.name):
            if attr.type == onnx.AttributeProto.INT:
                attrs.append(f"{attr.name}={attr.i}")
            elif attr.type == onnx.AttributeProto.INTS:
                attrs.append(f"{attr.name}={list(attr.ints)}")
        
        return f"{node.op_type}:{','.join(node.input)}:{','.join(attrs)}"
    
    def _analyze_shape_changes(
        self,
        original: onnx.ModelProto,
        modified: onnx.ModelProto
    ) -> List[Dict]:
        """Analyze changes in tensor shapes."""
        changes = []
        
        # Compare initializers
        orig_inits = {i.name: i for i in original.graph.initializer}
        mod_inits = {i.name: i for i in modified.graph.initializer}
        
        for name, orig_init in orig_inits.items():
            if name in mod_inits:
                mod_init = mod_inits[name]
                orig_shape = list(orig_init.dims)
                mod_shape = list(mod_init.dims)
                
                if orig_shape != mod_shape:
                    changes.append({
                        'type': 'weight_reshape',
                        'name': name,
                        'original_shape': orig_shape,
                        'modified_shape': mod_shape
                    })
        
        return changes
    
    def _find_patterns(self, diffs: List[ONNXModelDiff]) -> List[Pattern]:
        """Find common patterns across all model diffs."""
        patterns = []
        
        # Pattern 1: Common removed operations (grouped by model category)
        removed_ops_by_category = defaultdict(lambda: Counter())
        for diff in diffs:
            model_category = self._detect_model_category(diff.model_name)
            for node in diff.removed_nodes:
                removed_ops_by_category[model_category][node.op_type] += 1
        
        # Create patterns per category
        for model_category, removed_ops in removed_ops_by_category.items():
            for op, count in removed_ops.most_common(10):
                if count >= 2:  # At least 2 models in this category
                    example_models = [
                        d.model_name for d in diffs
                        if self._detect_model_category(d.model_name) == model_category
                        and any(n.op_type == op for n in d.removed_nodes)
                    ][:5]
                    
                    # Extract context from first example
                    context = {}
                    for diff in diffs:
                        if self._detect_model_category(diff.model_name) == model_category:
                            matching_nodes = [n for n in diff.removed_nodes if n.op_type == op]
                            if matching_nodes:
                                original_model = None
                                if diff.original_model_path:
                                    try:
                                        original_model = onnx.load(diff.original_model_path)
                                    except:
                                        pass
                                context = self._extract_context(matching_nodes[0], diff, original_model)
                                break
                    
                    patterns.append(Pattern(
                        name=f"Remove {op}",
                        description=f"{op} nodes are frequently removed during surgery in {model_category} models",
                        frequency=count,
                        op_types_involved=[op],
                        example_models=example_models,
                        template_candidate=op in ['Identity', 'Squeeze', 'Unsqueeze', 'Dropout'],
                        implementation_hint=f"Add template: remove_{op.lower()}",
                        model_category=model_category,
                        context=context
                    ))
        
        # Pattern 2: Common added operations (grouped by model category)
        added_ops_by_category = defaultdict(lambda: Counter())
        for diff in diffs:
            model_category = self._detect_model_category(diff.model_name)
            for node in diff.added_nodes:
                added_ops_by_category[model_category][node.op_type] += 1
        
        # Create patterns per category
        for model_category, added_ops in added_ops_by_category.items():
            for op, count in added_ops.most_common(10):
                if count >= 2:
                    example_models = [
                        d.model_name for d in diffs
                        if self._detect_model_category(d.model_name) == model_category
                        and any(n.op_type == op for n in d.added_nodes)
                    ][:5]
                    
                    # Extract context from first example
                    context = {}
                    for diff in diffs:
                        if self._detect_model_category(diff.model_name) == model_category:
                            matching_nodes = [n for n in diff.added_nodes if n.op_type == op]
                            if matching_nodes:
                                original_model = None
                                if diff.original_model_path:
                                    try:
                                        original_model = onnx.load(diff.original_model_path)
                                    except:
                                        pass
                                context = self._extract_context(matching_nodes[0], diff, original_model)
                                break
                    
                    patterns.append(Pattern(
                        name=f"Add {op}",
                        description=f"{op} nodes are frequently added during surgery in {model_category} models",
                        frequency=count,
                        op_types_involved=[op],
                        example_models=example_models,
                        template_candidate=op in ['Reshape', 'Transpose', 'Slice', 'Concat'],
                        implementation_hint=f"Consider template for adding {op} nodes",
                        model_category=model_category,
                        context=context
                    ))
        
        # Pattern 3: Operation replacements
        replacements = Counter()
        for diff in diffs:
            for change, count in diff.op_type_changes.items():
                replacements[change] += count
        
        for replacement, count in replacements.most_common(10):
            if count >= 2:
                example_models = [
                    d.model_name for d in diffs
                    if replacement in d.op_type_changes
                ][:5]
                
                patterns.append(Pattern(
                    name=f"Replace {replacement}",
                    description=f"Operation replacement pattern",
                    frequency=count,
                    op_types_involved=replacement.split('->'),
                    example_models=example_models,
                    template_candidate=True,
                    implementation_hint=f"Add template: replace_{replacement.replace('->', '_to_').lower()}"
                ))
        
        # Pattern 4: Reshape patterns
        reshape_additions = sum(
            1 for d in diffs
            for n in d.added_nodes
            if n.op_type == 'Reshape'
        )
        if reshape_additions > 0:
            patterns.append(Pattern(
                name="4D Tensor Reshaping",
                description="Reshape operations added to convert tensors to 4D for MLA",
                frequency=reshape_additions,
                op_types_involved=['Reshape'],
                example_models=[
                    d.model_name for d in diffs
                    if any(n.op_type == 'Reshape' for n in d.added_nodes)
                ][:5],
                template_candidate=True,
                implementation_hint="Use add_reshape_4d template"
            ))
        
        # Pattern 5: Concat patterns for output merging
        concat_additions = sum(
            1 for d in diffs
            for n in d.added_nodes
            if n.op_type == 'Concat'
        )
        if concat_additions > 0:
            patterns.append(Pattern(
                name="Output Concatenation",
                description="Concat nodes added to merge split outputs",
                frequency=concat_additions,
                op_types_involved=['Concat'],
                example_models=[
                    d.model_name for d in diffs
                    if any(n.op_type == 'Concat' for n in d.added_nodes)
                ][:5],
                template_candidate=False,
                implementation_hint="Part of split-process-merge pattern"
            ))
        
        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        
        return patterns
    
    def _compute_op_statistics(self, diffs: List[ONNXModelDiff]) -> Dict[str, Dict]:
        """Compute statistics for each operation type."""
        stats = defaultdict(lambda: {
            'added_count': 0,
            'removed_count': 0,
            'modified_count': 0,
            'models_affected': set()
        })
        
        for diff in diffs:
            for node in diff.added_nodes:
                stats[node.op_type]['added_count'] += 1
                stats[node.op_type]['models_affected'].add(diff.model_name)
            
            for node in diff.removed_nodes:
                stats[node.op_type]['removed_count'] += 1
                stats[node.op_type]['models_affected'].add(diff.model_name)
            
            for node in diff.modified_nodes:
                stats[node.op_type]['modified_count'] += 1
                stats[node.op_type]['models_affected'].add(diff.model_name)
        
        # Convert sets to counts
        result = {}
        for op, data in stats.items():
            result[op] = {
                'added_count': data['added_count'],
                'removed_count': data['removed_count'],
                'modified_count': data['modified_count'],
                'models_affected': len(data['models_affected'])
            }
        
        return result
    
    def _generate_recommendations(
        self,
        patterns: List[Pattern],
        op_stats: Dict[str, Dict]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Recommend templates for frequent patterns
        template_candidates = [p for p in patterns if p.template_candidate]
        if template_candidates:
            recommendations.append(
                "HIGH PRIORITY - Add deterministic templates for these patterns:\n" +
                "\n".join(f"  - {p.name}: {p.implementation_hint}" for p in template_candidates[:5])
            )
        
        # Recommend based on op statistics
        high_removal_ops = [
            op for op, stats in op_stats.items()
            if stats['removed_count'] >= 3 and stats['added_count'] == 0
        ]
        if high_removal_ops:
            recommendations.append(
                f"Operations frequently removed (consider removal templates):\n" +
                f"  {', '.join(high_removal_ops)}"
            )
        
        high_addition_ops = [
            op for op, stats in op_stats.items()
            if stats['added_count'] >= 3 and stats['removed_count'] == 0
        ]
        if high_addition_ops:
            recommendations.append(
                f"Operations frequently added (consider addition templates):\n" +
                f"  {', '.join(high_addition_ops)}"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "No strong patterns detected. Consider:\n"
                "  - Collecting more model pairs\n"
                "  - Manual inspection of edge cases"
            )
        
        return recommendations
    
    def generate_report(self, analysis: AnalysisReport) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "DATASET ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Total models analyzed: {analysis.total_models}",
            "",
            "-" * 70,
            "PATTERNS DISCOVERED",
            "-" * 70,
        ]
        
        for i, pattern in enumerate(analysis.patterns[:15], 1):
            lines.append(f"\n{i}. {pattern.name}")
            lines.append(f"   Description: {pattern.description}")
            lines.append(f"   Frequency: {pattern.frequency} models")
            lines.append(f"   Operations: {', '.join(pattern.op_types_involved)}")
            lines.append(f"   Template Candidate: {'Yes' if pattern.template_candidate else 'No'}")
            if pattern.implementation_hint:
                lines.append(f"   Hint: {pattern.implementation_hint}")
            lines.append(f"   Examples: {', '.join(pattern.example_models)}")
        
        lines.extend([
            "",
            "-" * 70,
            "OPERATION STATISTICS (top 15)",
            "-" * 70,
            ""
        ])
        
        # Sort by total changes
        sorted_ops = sorted(
            analysis.op_type_statistics.items(),
            key=lambda x: x[1]['added_count'] + x[1]['removed_count'] + x[1]['modified_count'],
            reverse=True
        )[:15]
        
        for op, stats in sorted_ops:
            lines.append(
                f"  {op}: +{stats['added_count']} / -{stats['removed_count']} / "
                f"~{stats['modified_count']} ({stats['models_affected']} models)"
            )
        
        lines.extend([
            "",
            "-" * 70,
            "RECOMMENDATIONS",
            "-" * 70,
            ""
        ])
        
        for rec in analysis.recommendations:
            lines.append(rec)
            lines.append("")
        
        lines.extend([
            "",
            "-" * 70,
            "MODEL SUMMARIES",
            "-" * 70,
        ])
        
        for diff in analysis.model_diffs:
            delta = diff.modified_node_count - diff.original_node_count
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"\n  {diff.model_name}:"
                f" {diff.original_node_count} -> {diff.modified_node_count} nodes ({sign}{delta})"
            )
            lines.append(
                f"    Added: {len(diff.added_nodes)}, "
                f"Removed: {len(diff.removed_nodes)}, "
                f"Modified: {len(diff.modified_nodes)}"
            )
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def save_report(self, analysis: AnalysisReport, output_path: str) -> None:
        """Save analysis report as JSON."""
        with open(output_path, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2)
        print(f"Report saved to: {output_path}")
    
    # =========================================================================
    # Enhanced Pattern Extraction for PatternDatabase
    # =========================================================================
    
    def extract_learned_patterns(
        self,
        analysis: AnalysisReport,
    ) -> List[Dict]:
        """
        Extract node-level patterns from analysis for PatternDatabase.
        
        Args:
            analysis: Analysis report from analyze_dataset()
            
        Returns:
            List of pattern dictionaries for PatternDatabase
        """
        patterns = []
        pattern_counter = 0
        
        for diff in analysis.model_diffs:
            model_category = self._detect_model_category(diff.model_name)
            original_model = None
            
            # Load original model for context extraction
            if diff.original_model_path:
                try:
                    original_model = onnx.load(diff.original_model_path)
                except Exception:
                    pass
            
            # Extract removal patterns
            for node in diff.removed_nodes:
                context = self._extract_node_context(node, original_model)
                pattern = {
                    'id': f'pattern_{pattern_counter}',
                    'action': 'remove',
                    'op_type': node.op_type,
                    'context': context,
                    'model_category': model_category,
                    'frequency': 1,
                    'confidence': 1.0,
                    'example_models': [diff.model_name],
                    'replacement_ops': [],
                }
                patterns.append(pattern)
                pattern_counter += 1
            
            # Extract addition patterns
            for node in diff.added_nodes:
                context = self._extract_node_context_added(node, diff)
                pattern = {
                    'id': f'pattern_{pattern_counter}',
                    'action': 'add',
                    'op_type': node.op_type,
                    'context': context,
                    'model_category': model_category,
                    'frequency': 1,
                    'confidence': 1.0,
                    'example_models': [diff.model_name],
                    'replacement_ops': [],
                }
                patterns.append(pattern)
                pattern_counter += 1
            
            # Extract replacement patterns (from modified nodes)
            for node in diff.modified_nodes:
                if 'original_op' in node.details and 'new_op' in node.details:
                    context = self._extract_node_context(node, original_model)
                    pattern = {
                        'id': f'pattern_{pattern_counter}',
                        'action': 'replace',
                        'op_type': node.details['original_op'],
                        'context': context,
                        'model_category': model_category,
                        'frequency': 1,
                        'confidence': 1.0,
                        'example_models': [diff.model_name],
                        'replacement_ops': [node.details['new_op']],
                    }
                    patterns.append(pattern)
                    pattern_counter += 1
        
        return patterns
    
    def _extract_node_context(
        self,
        node: NodeDiff,
        original_model: Optional[onnx.ModelProto],
    ) -> Dict:
        """Extract detailed context for a node."""
        context = {
            'position': 'middle',
            'input_ops': [],
            'output_ops': [],
            'input_shapes': [],
            'output_shapes': [],
        }
        
        if original_model is None:
            return context
        
        try:
            # Find the node in the original model
            target_node = None
            node_idx = None
            for i, n in enumerate(original_model.graph.node):
                if n.name == node.node_name or n.op_type == node.op_type:
                    # Check if outputs match
                    target_node = n
                    node_idx = i
                    break
            
            if target_node is None:
                return context
            
            total_nodes = len(original_model.graph.node)
            
            # Determine position
            if node_idx is not None:
                position_ratio = node_idx / total_nodes if total_nodes > 0 else 0.5
                if position_ratio > 0.8:
                    context['position'] = 'near_output'
                elif position_ratio < 0.2:
                    context['position'] = 'near_input'
                else:
                    context['position'] = 'middle'
            
            # Build tensor producer map
            tensor_producers = {}
            for n in original_model.graph.node:
                for out in n.output:
                    tensor_producers[out] = n
            
            # Build tensor consumer map
            tensor_consumers = defaultdict(list)
            for n in original_model.graph.node:
                for inp in n.input:
                    tensor_consumers[inp].append(n)
            
            # Get input producers
            for inp in target_node.input:
                if inp in tensor_producers:
                    context['input_ops'].append(tensor_producers[inp].op_type)
            
            # Get output consumers
            for out in target_node.output:
                for consumer in tensor_consumers.get(out, []):
                    context['output_ops'].append(consumer.op_type)
            
        except Exception:
            pass
        
        return context
    
    def _extract_node_context_added(
        self,
        node: NodeDiff,
        diff: ONNXModelDiff,
    ) -> Dict:
        """Extract context for an added node."""
        # For added nodes, context is less certain
        # We use the node's details if available
        context = {
            'position': 'middle',
            'input_ops': [],
            'output_ops': [],
            'input_shapes': [],
            'output_shapes': [],
        }
        
        # Check if details contain context information
        if 'details' in dir(node) and node.details:
            if 'position' in node.details:
                context['position'] = node.details['position']
        
        return context
    
    def build_pattern_database(
        self,
        analysis: AnalysisReport,
        output_path: Optional[str] = None,
    ):
        """
        Build a PatternDatabase from analysis results.
        
        Args:
            analysis: Analysis report from analyze_dataset()
            output_path: Optional path to save the database
            
        Returns:
            PatternDatabase instance
        """
        # Import here to avoid circular imports
        try:
            from knowledge_base.knowledge_base import PatternDatabase, LearnedPattern, NodeContext
        except ImportError:
            print("Warning: PatternDatabase not available. Using dict-based patterns.")
            return self.extract_learned_patterns(analysis)
        
        # Extract patterns as dicts
        pattern_dicts = self.extract_learned_patterns(analysis)
        
        # Build database
        db = PatternDatabase()
        
        for p_dict in pattern_dicts:
            pattern = LearnedPattern(
                id=p_dict['id'],
                action=p_dict['action'],
                op_type=p_dict['op_type'],
                context=NodeContext(
                    position=p_dict['context'].get('position', 'middle'),
                    input_ops=p_dict['context'].get('input_ops', []),
                    output_ops=p_dict['context'].get('output_ops', []),
                ),
                model_category=p_dict['model_category'],
                frequency=p_dict['frequency'],
                confidence=p_dict['confidence'],
                example_models=p_dict['example_models'],
                replacement_ops=p_dict['replacement_ops'],
            )
            db.add_pattern(pattern)
        
        # Compute statistics
        db.compute_pattern_frequencies()
        db.compute_success_rates()
        
        # Save if path provided
        if output_path:
            db.save(output_path)
            print(f"Pattern database saved to: {output_path}")
        
        return db


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ONNX model surgery dataset')
    parser.add_argument('--dataset', '-d', default='dataset', help='Dataset directory')
    parser.add_argument('--output', '-o', default='analysis_report.json', help='Output JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer()
    
    print(f"\nAnalyzing dataset: {args.dataset}\n")
    
    analysis = analyzer.analyze_dataset(args.dataset)
    
    # Print report
    print(analyzer.generate_report(analysis))
    
    # Save JSON
    analyzer.save_report(analysis, args.output)


if __name__ == "__main__":
    main()

