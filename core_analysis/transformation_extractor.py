#!/usr/bin/env python3
"""
Transformation Extractor for ONNX Model Surgery.

Extracts precise transformations from original/modified ONNX model pairs.
This is the core data extraction module that populates the SurgeryDatabase
with detailed node-level transformation information.

Key Features:
- Precise node matching by output tensor names
- Full attribute extraction
- Shape information via shape inference
- Dependency and ordering analysis
- Compilation blocker detection and linking

Author: Automated Model Surgery Pipeline
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import onnx
from onnx import shape_inference, numpy_helper

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.surgery_database import (
    NodeTransformation,
    TransformationRecord,
    TensorInfo,
    NodeContext,
    SurgeryDatabase
)
from core_analysis.onnx_analyzer import ONNXAnalyzer, ModelAnalysis, NodeAnalysis


# =============================================================================
# Helper Data Structures
# =============================================================================

@dataclass
class NodeSignature:
    """Signature for matching nodes between models."""
    node_id: int
    name: str
    op_type: str
    input_names: List[str]
    output_names: List[str]
    attributes: Dict[str, Any]
    input_shapes: List[Optional[List[int]]]
    output_shapes: List[Optional[List[int]]]
    
    def get_output_signature(self) -> str:
        """Get signature based on outputs (most reliable for matching)."""
        return "|".join(sorted(self.output_names))
    
    def get_input_signature(self) -> str:
        """Get signature based on inputs."""
        return "|".join(sorted(self.input_names))
    
    def matches_by_output(self, other: 'NodeSignature') -> bool:
        """Check if this node produces same outputs as other."""
        return set(self.output_names) == set(other.output_names)


@dataclass 
class NodeMatch:
    """Result of matching nodes between original and modified models."""
    original_id: int
    modified_id: Optional[int]  # None if removed
    match_type: str  # "exact", "renamed", "modified", "removed", "added"
    original_sig: Optional[NodeSignature]
    modified_sig: Optional[NodeSignature]
    changes: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Main Transformation Extractor
# =============================================================================

class TransformationExtractor:
    """
    Extract precise transformations from original/modified ONNX model pairs.
    
    This class compares two ONNX models (original and modified) and extracts
    detailed information about every transformation applied, including:
    - Removed nodes (with full context)
    - Added nodes (with full context)
    - Modified nodes (with before/after details)
    - Shape changes
    - Attribute changes
    
    All extracted data is structured for the SurgeryDatabase format.
    """
    
    # Model categories based on path/name patterns
    CATEGORY_PATTERNS = {
        'YOLO': ['yolo', 'yolov', 'ultralytics'],
        'Transformer': ['t5', 'mt5', 'bert', 'gpt', 'transformer', 'marian', 'troc', 'efficient_at'],
        'ViT': ['vit', 'vision_transformer', 'deit'],
        'CNN': ['resnet', 'mobilenet', 'efficientnet', 'cnn', 'midas', 'depth', 'ggcnn'],
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the transformation extractor.
        
        Args:
            verbose: Whether to print detailed extraction progress
        """
        self.verbose = verbose
        self.analyzer = ONNXAnalyzer()
        
        # Cache for loaded models
        self._model_cache: Dict[str, onnx.ModelProto] = {}
        self._analysis_cache: Dict[str, ModelAnalysis] = {}
    
    def extract_transformations(
        self,
        original_path: str,
        modified_path: str,
        model_name: Optional[str] = None
    ) -> TransformationRecord:
        """
        Extract all transformations between original and modified models.
        
        Args:
            original_path: Path to original ONNX model
            modified_path: Path to modified ONNX model
            model_name: Optional model name (derived from path if not provided)
            
        Returns:
            TransformationRecord with all extracted transformations
        """
        if self.verbose:
            print(f"Extracting transformations: {original_path} -> {modified_path}")
        
        # Load and analyze both models
        original_model = self._load_model(original_path)
        modified_model = self._load_model(modified_path)
        
        original_analysis = self._analyze_model(original_path, original_model)
        modified_analysis = self._analyze_model(modified_path, modified_model)
        
        # Detect model category
        model_name = model_name or self._extract_model_name(original_path)
        model_category = self._detect_category(original_path, model_name)
        
        if self.verbose:
            print(f"  Model: {model_name} ({model_category})")
            print(f"  Original: {len(original_analysis.nodes)} nodes")
            print(f"  Modified: {len(modified_analysis.nodes)} nodes")
        
        # Build node signatures for matching
        original_sigs = self._build_signatures(original_model, original_analysis)
        modified_sigs = self._build_signatures(modified_model, modified_analysis)
        
        # Match nodes between models
        matches = self._match_nodes(original_sigs, modified_sigs, original_model, modified_model)
        
        # Extract transformations from matches
        transformations = self._extract_from_matches(
            matches, 
            original_model, modified_model,
            original_analysis, modified_analysis,
            model_name, model_category
        )
        
        # Compute statistics
        nodes_removed = sum(1 for m in matches if m.match_type == 'removed')
        nodes_added = sum(1 for m in matches if m.match_type == 'added')
        nodes_modified = sum(1 for m in matches if m.match_type == 'modified')
        
        # Count shape changes
        shape_changes = self._count_shape_changes(matches)
        
        # Get original blockers
        original_blockers = self._get_blocker_summary(original_analysis)
        
        # Determine transformation order (topological based on dependencies)
        transformation_order = self._compute_transformation_order(
            transformations, original_analysis
        )
        
        # Create record
        record = TransformationRecord(
            model_name=model_name,
            model_category=model_category,
            original_node_count=len(original_analysis.nodes),
            modified_node_count=len(modified_analysis.nodes),
            original_blockers=original_blockers,
            compilation_success=True,  # Assume modified compiles
            transformations=transformations,
            transformation_order=transformation_order,
            nodes_removed=nodes_removed,
            nodes_added=nodes_added,
            nodes_modified=nodes_modified,
            shape_changes=shape_changes,
            source_original_path=original_path,
            source_modified_path=modified_path,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_version="1.0.0"
        )
        
        if self.verbose:
            print(f"  Extracted {len(transformations)} transformations")
            print(f"    Removed: {nodes_removed}, Added: {nodes_added}, Modified: {nodes_modified}")
        
        return record
    
    def _load_model(self, path: str) -> onnx.ModelProto:
        """Load ONNX model with caching."""
        if path not in self._model_cache:
            model = onnx.load(path)
            # Try shape inference
            try:
                model = shape_inference.infer_shapes(model)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Shape inference failed for {path}: {e}")
            self._model_cache[path] = model
        return self._model_cache[path]
    
    def _analyze_model(self, path: str, model: onnx.ModelProto) -> ModelAnalysis:
        """Analyze model with caching."""
        if path not in self._analysis_cache:
            self._analysis_cache[path] = self.analyzer.analyze(path)
        return self._analysis_cache[path]
    
    def _extract_model_name(self, path: str) -> str:
        """Extract model name from path."""
        # Try to get from directory structure (dataset/ModelName/Original/model.onnx)
        parts = Path(path).parts
        for i, part in enumerate(parts):
            if part.lower() in ['original', 'modified']:
                if i > 0:
                    return parts[i - 1]
        # Fallback to filename
        return Path(path).stem
    
    def _detect_category(self, path: str, model_name: str) -> str:
        """Detect model category from path and name."""
        combined = (path + model_name).lower()
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            if any(p in combined for p in patterns):
                return category
        
        return 'Other'
    
    def _build_signatures(
        self, 
        model: onnx.ModelProto, 
        analysis: ModelAnalysis
    ) -> Dict[int, NodeSignature]:
        """Build node signatures for matching."""
        signatures = {}
        
        for i, node in enumerate(model.graph.node):
            # Get corresponding analysis node
            node_analysis = analysis.nodes[i] if i < len(analysis.nodes) else None
            
            # Extract attributes
            attributes = self._extract_attributes(node)
            
            # Get shapes from analysis
            input_shapes = []
            output_shapes = []
            if node_analysis:
                input_shapes = [
                    list(s) if s else None 
                    for s in node_analysis.input_shapes
                ]
                output_shapes = [
                    list(s) if s else None 
                    for s in node_analysis.output_shapes
                ]
            
            sig = NodeSignature(
                node_id=i,
                name=node.name or f"node_{i}",
                op_type=node.op_type,
                input_names=list(node.input),
                output_names=list(node.output),
                attributes=attributes,
                input_shapes=input_shapes,
                output_shapes=output_shapes
            )
            signatures[i] = sig
        
        return signatures
    
    def _extract_attributes(self, node: onnx.NodeProto) -> Dict[str, Any]:
        """Extract all attributes from a node."""
        attributes = {}
        
        for attr in node.attribute:
            try:
                if attr.type == onnx.AttributeProto.INT:
                    attributes[attr.name] = int(attr.i)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attributes[attr.name] = float(attr.f)
                elif attr.type == onnx.AttributeProto.STRING:
                    attributes[attr.name] = attr.s.decode() if isinstance(attr.s, bytes) else str(attr.s)
                elif attr.type == onnx.AttributeProto.INTS:
                    attributes[attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    attributes[attr.name] = list(attr.floats)
                elif attr.type == onnx.AttributeProto.STRINGS:
                    attributes[attr.name] = [s.decode() if isinstance(s, bytes) else str(s) for s in attr.strings]
                elif attr.type == onnx.AttributeProto.TENSOR:
                    # Store tensor shape and type, not values
                    attributes[attr.name] = {
                        'type': 'tensor',
                        'dims': list(attr.t.dims),
                        'data_type': attr.t.data_type
                    }
                elif attr.type == onnx.AttributeProto.GRAPH:
                    attributes[attr.name] = {'type': 'subgraph', 'node_count': len(attr.g.node)}
                elif attr.type == onnx.AttributeProto.TENSORS:
                    attributes[attr.name] = [
                        {'type': 'tensor', 'dims': list(t.dims), 'data_type': t.data_type}
                        for t in attr.tensors
                    ]
                else:
                    attributes[attr.name] = f"<{onnx.AttributeProto.AttributeType.Name(attr.type)}>"
            except Exception as e:
                attributes[attr.name] = f"<error: {str(e)[:50]}>"
        
        return attributes
    
    def _match_nodes(
        self,
        original_sigs: Dict[int, NodeSignature],
        modified_sigs: Dict[int, NodeSignature],
        original_model: onnx.ModelProto,
        modified_model: onnx.ModelProto
    ) -> List[NodeMatch]:
        """
        Match nodes between original and modified models.
        
        Uses multiple strategies:
        1. Exact output name match
        2. Op type + relative position
        3. Attribute similarity
        """
        matches = []
        matched_original = set()
        matched_modified = set()
        
        # Build output -> node maps
        original_output_map: Dict[str, int] = {}
        for node_id, sig in original_sigs.items():
            for output in sig.output_names:
                if output:
                    original_output_map[output] = node_id
        
        modified_output_map: Dict[str, int] = {}
        for node_id, sig in modified_sigs.items():
            for output in sig.output_names:
                if output:
                    modified_output_map[output] = node_id
        
        # Strategy 1: Exact output name match
        for output_name, orig_id in original_output_map.items():
            if output_name in modified_output_map and orig_id not in matched_original:
                mod_id = modified_output_map[output_name]
                if mod_id not in matched_modified:
                    orig_sig = original_sigs[orig_id]
                    mod_sig = modified_sigs[mod_id]
                    
                    # Determine match type
                    if orig_sig.op_type == mod_sig.op_type and orig_sig.attributes == mod_sig.attributes:
                        match_type = "exact"
                        changes = {}
                    else:
                        match_type = "modified"
                        changes = self._compute_changes(orig_sig, mod_sig)
                    
                    matches.append(NodeMatch(
                        original_id=orig_id,
                        modified_id=mod_id,
                        match_type=match_type,
                        original_sig=orig_sig,
                        modified_sig=mod_sig,
                        changes=changes
                    ))
                    matched_original.add(orig_id)
                    matched_modified.add(mod_id)
        
        # Strategy 2: Op type + name similarity for unmatched nodes
        for orig_id, orig_sig in original_sigs.items():
            if orig_id in matched_original:
                continue
            
            best_match = None
            best_score = 0
            
            for mod_id, mod_sig in modified_sigs.items():
                if mod_id in matched_modified:
                    continue
                
                # Calculate similarity score
                score = self._compute_similarity(orig_sig, mod_sig)
                if score > best_score and score > 0.5:  # Threshold
                    best_score = score
                    best_match = mod_id
            
            if best_match is not None:
                mod_sig = modified_sigs[best_match]
                
                if orig_sig.op_type == mod_sig.op_type:
                    match_type = "renamed" if orig_sig.name != mod_sig.name else "modified"
                else:
                    match_type = "modified"
                
                matches.append(NodeMatch(
                    original_id=orig_id,
                    modified_id=best_match,
                    match_type=match_type,
                    original_sig=orig_sig,
                    modified_sig=mod_sig,
                    changes=self._compute_changes(orig_sig, mod_sig)
                ))
                matched_original.add(orig_id)
                matched_modified.add(best_match)
        
        # Removed nodes (in original but not matched)
        for orig_id, orig_sig in original_sigs.items():
            if orig_id not in matched_original:
                matches.append(NodeMatch(
                    original_id=orig_id,
                    modified_id=None,
                    match_type="removed",
                    original_sig=orig_sig,
                    modified_sig=None,
                    changes={}
                ))
        
        # Added nodes (in modified but not matched)
        for mod_id, mod_sig in modified_sigs.items():
            if mod_id not in matched_modified:
                matches.append(NodeMatch(
                    original_id=-1,
                    modified_id=mod_id,
                    match_type="added",
                    original_sig=None,
                    modified_sig=mod_sig,
                    changes={}
                ))
        
        return matches
    
    def _compute_similarity(self, sig1: NodeSignature, sig2: NodeSignature) -> float:
        """Compute similarity score between two node signatures."""
        score = 0.0
        
        # Op type match (most important)
        if sig1.op_type == sig2.op_type:
            score += 0.5
        
        # Name similarity
        if sig1.name and sig2.name:
            # Simple overlap check
            name1_parts = set(sig1.name.lower().split('_'))
            name2_parts = set(sig2.name.lower().split('_'))
            if name1_parts & name2_parts:
                score += 0.2
        
        # Input/output count match
        if len(sig1.input_names) == len(sig2.input_names):
            score += 0.1
        if len(sig1.output_names) == len(sig2.output_names):
            score += 0.1
        
        # Attribute similarity
        if sig1.attributes and sig2.attributes:
            common_keys = set(sig1.attributes.keys()) & set(sig2.attributes.keys())
            if common_keys:
                matching_values = sum(
                    1 for k in common_keys 
                    if sig1.attributes[k] == sig2.attributes[k]
                )
                score += 0.1 * (matching_values / len(common_keys))
        
        return score
    
    def _compute_changes(self, orig_sig: NodeSignature, mod_sig: NodeSignature) -> Dict[str, Any]:
        """Compute detailed changes between two matched nodes."""
        changes = {}
        
        # Op type change
        if orig_sig.op_type != mod_sig.op_type:
            changes['op_type'] = {'from': orig_sig.op_type, 'to': mod_sig.op_type}
        
        # Name change
        if orig_sig.name != mod_sig.name:
            changes['name'] = {'from': orig_sig.name, 'to': mod_sig.name}
        
        # Attribute changes
        attr_changes = {}
        all_keys = set(orig_sig.attributes.keys()) | set(mod_sig.attributes.keys())
        for key in all_keys:
            orig_val = orig_sig.attributes.get(key)
            mod_val = mod_sig.attributes.get(key)
            if orig_val != mod_val:
                attr_changes[key] = {'from': orig_val, 'to': mod_val}
        if attr_changes:
            changes['attributes'] = attr_changes
        
        # Input changes
        if orig_sig.input_names != mod_sig.input_names:
            changes['inputs'] = {
                'from': orig_sig.input_names,
                'to': mod_sig.input_names
            }
        
        # Output changes
        if orig_sig.output_names != mod_sig.output_names:
            changes['outputs'] = {
                'from': orig_sig.output_names,
                'to': mod_sig.output_names
            }
        
        # Shape changes
        if orig_sig.output_shapes != mod_sig.output_shapes:
            changes['output_shapes'] = {
                'from': orig_sig.output_shapes,
                'to': mod_sig.output_shapes
            }
        
        return changes
    
    def _extract_from_matches(
        self,
        matches: List[NodeMatch],
        original_model: onnx.ModelProto,
        modified_model: onnx.ModelProto,
        original_analysis: ModelAnalysis,
        modified_analysis: ModelAnalysis,
        model_name: str,
        model_category: str
    ) -> List[NodeTransformation]:
        """Extract NodeTransformation objects from matches."""
        transformations = []
        
        # Get initializer names
        original_initializers = {init.name for init in original_model.graph.initializer}
        modified_initializers = {init.name for init in modified_model.graph.initializer}
        
        # Build output -> node maps for context extraction
        original_output_to_node = {}
        for i, node in enumerate(original_model.graph.node):
            for output in node.output:
                if output:
                    original_output_to_node[output] = i
        
        modified_output_to_node = {}
        for i, node in enumerate(modified_model.graph.node):
            for output in node.output:
                if output:
                    modified_output_to_node[output] = i
        
        total_original_nodes = len(original_analysis.nodes)
        total_modified_nodes = len(modified_analysis.nodes)
        
        for match in matches:
            if match.match_type == "exact":
                # Skip exact matches - no transformation
                continue
            
            if match.match_type == "removed":
                # Node was removed
                transformation = self._create_removal_transformation(
                    match.original_sig,
                    original_model, original_analysis,
                    original_initializers, original_output_to_node,
                    model_name, total_original_nodes
                )
                transformations.append(transformation)
            
            elif match.match_type == "added":
                # Node was added
                transformation = self._create_addition_transformation(
                    match.modified_sig,
                    modified_model, modified_analysis,
                    modified_initializers, modified_output_to_node,
                    model_name, total_modified_nodes
                )
                transformations.append(transformation)
            
            elif match.match_type in ["modified", "renamed"]:
                # Node was modified
                transformation = self._create_modification_transformation(
                    match.original_sig, match.modified_sig, match.changes,
                    original_model, original_analysis,
                    original_initializers, original_output_to_node,
                    model_name, total_original_nodes
                )
                transformations.append(transformation)
        
        return transformations
    
    def _create_removal_transformation(
        self,
        sig: NodeSignature,
        model: onnx.ModelProto,
        analysis: ModelAnalysis,
        initializers: Set[str],
        output_to_node: Dict[str, int],
        model_name: str,
        total_nodes: int
    ) -> NodeTransformation:
        """Create transformation record for a removed node."""
        node_analysis = analysis.nodes[sig.node_id] if sig.node_id < len(analysis.nodes) else None
        
        # Get predecessors and successors
        predecessors = self._get_predecessor_context(
            sig, model, analysis, initializers, output_to_node
        )
        successors = self._get_successor_context(sig, model, analysis)
        
        # Get input/output tensor info
        input_tensors = self._get_tensor_info(
            sig.input_names, sig.input_shapes, model, initializers
        )
        output_tensors = self._get_tensor_info(
            sig.output_names, sig.output_shapes, model, initializers
        )
        
        # Determine if this was a compilation blocker
        is_blocker = False
        blocker_reason = None
        if node_analysis:
            is_blocker = node_analysis.is_compilation_blocker
            blocker_reason = node_analysis.blocker_reason
        
        # Generate surgery steps
        surgery_steps = self._generate_removal_steps(sig, predecessors, successors)
        
        # Generate code snippet
        code_snippet = self._generate_removal_code(sig)
        
        return NodeTransformation(
            original_node_id=sig.node_id,
            original_node_name=sig.name,
            original_op_type=sig.op_type,
            graph_position=sig.node_id / total_nodes if total_nodes > 0 else 0.5,
            total_nodes_in_graph=total_nodes,
            predecessor_nodes=predecessors,
            successor_nodes=successors,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attributes=sig.attributes,
            action="remove",
            result_node=None,
            replacement_ops=[],
            blocker_reason=blocker_reason,
            compilation_error=None,
            is_compilation_blocker=is_blocker,
            surgery_steps=surgery_steps,
            code_snippet=code_snippet,
            confidence=0.9 if is_blocker else 0.7,  # Higher confidence for blocker removals
            source_model=model_name
        )
    
    def _create_addition_transformation(
        self,
        sig: NodeSignature,
        model: onnx.ModelProto,
        analysis: ModelAnalysis,
        initializers: Set[str],
        output_to_node: Dict[str, int],
        model_name: str,
        total_nodes: int
    ) -> NodeTransformation:
        """Create transformation record for an added node."""
        # Get predecessors and successors
        predecessors = self._get_predecessor_context(
            sig, model, analysis, initializers, output_to_node
        )
        successors = self._get_successor_context(sig, model, analysis)
        
        # Get input/output tensor info
        input_tensors = self._get_tensor_info(
            sig.input_names, sig.input_shapes, model, initializers
        )
        output_tensors = self._get_tensor_info(
            sig.output_names, sig.output_shapes, model, initializers
        )
        
        # Generate surgery steps
        surgery_steps = self._generate_addition_steps(sig, predecessors, successors)
        
        # Generate code snippet
        code_snippet = self._generate_addition_code(sig)
        
        # Store result node details
        result_node = {
            'name': sig.name,
            'op_type': sig.op_type,
            'attributes': sig.attributes,
            'input_names': sig.input_names,
            'output_names': sig.output_names,
            'input_shapes': sig.input_shapes,
            'output_shapes': sig.output_shapes
        }
        
        return NodeTransformation(
            original_node_id=-1,  # No original node
            original_node_name="",
            original_op_type="",
            graph_position=sig.node_id / total_nodes if total_nodes > 0 else 0.5,
            total_nodes_in_graph=total_nodes,
            predecessor_nodes=predecessors,
            successor_nodes=successors,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attributes={},  # No original attributes
            action="add",
            result_node=result_node,
            replacement_ops=[sig.op_type],
            blocker_reason=None,
            compilation_error=None,
            is_compilation_blocker=False,
            surgery_steps=surgery_steps,
            code_snippet=code_snippet,
            confidence=0.8,
            source_model=model_name
        )
    
    def _create_modification_transformation(
        self,
        orig_sig: NodeSignature,
        mod_sig: NodeSignature,
        changes: Dict[str, Any],
        model: onnx.ModelProto,
        analysis: ModelAnalysis,
        initializers: Set[str],
        output_to_node: Dict[str, int],
        model_name: str,
        total_nodes: int
    ) -> NodeTransformation:
        """Create transformation record for a modified node."""
        node_analysis = analysis.nodes[orig_sig.node_id] if orig_sig.node_id < len(analysis.nodes) else None
        
        # Get predecessors and successors
        predecessors = self._get_predecessor_context(
            orig_sig, model, analysis, initializers, output_to_node
        )
        successors = self._get_successor_context(orig_sig, model, analysis)
        
        # Get input/output tensor info
        input_tensors = self._get_tensor_info(
            orig_sig.input_names, orig_sig.input_shapes, model, initializers
        )
        output_tensors = self._get_tensor_info(
            orig_sig.output_names, orig_sig.output_shapes, model, initializers
        )
        
        # Determine action type
        if 'op_type' in changes:
            action = "replace"
            replacement_ops = [mod_sig.op_type]
        elif 'output_shapes' in changes:
            action = "reshape"
            replacement_ops = []
        else:
            action = "modify"
            replacement_ops = []
        
        # Determine if this was a compilation blocker
        is_blocker = False
        blocker_reason = None
        if node_analysis:
            is_blocker = node_analysis.is_compilation_blocker
            blocker_reason = node_analysis.blocker_reason
        
        # Generate surgery steps
        surgery_steps = self._generate_modification_steps(orig_sig, mod_sig, changes)
        
        # Generate code snippet
        code_snippet = self._generate_modification_code(orig_sig, mod_sig, changes)
        
        # Store result node details
        result_node = {
            'name': mod_sig.name,
            'op_type': mod_sig.op_type,
            'attributes': mod_sig.attributes,
            'input_names': mod_sig.input_names,
            'output_names': mod_sig.output_names,
            'input_shapes': mod_sig.input_shapes,
            'output_shapes': mod_sig.output_shapes,
            'changes': changes
        }
        
        return NodeTransformation(
            original_node_id=orig_sig.node_id,
            original_node_name=orig_sig.name,
            original_op_type=orig_sig.op_type,
            graph_position=orig_sig.node_id / total_nodes if total_nodes > 0 else 0.5,
            total_nodes_in_graph=total_nodes,
            predecessor_nodes=predecessors,
            successor_nodes=successors,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attributes=orig_sig.attributes,
            action=action,
            result_node=result_node,
            replacement_ops=replacement_ops,
            blocker_reason=blocker_reason,
            compilation_error=None,
            is_compilation_blocker=is_blocker,
            surgery_steps=surgery_steps,
            code_snippet=code_snippet,
            confidence=0.85,
            source_model=model_name
        )
    
    def _get_predecessor_context(
        self,
        sig: NodeSignature,
        model: onnx.ModelProto,
        analysis: ModelAnalysis,
        initializers: Set[str],
        output_to_node: Dict[str, int]
    ) -> List[Dict]:
        """Get context about predecessor nodes."""
        predecessors = []
        
        for input_name in sig.input_names:
            if not input_name or input_name in initializers:
                continue
            
            if input_name in output_to_node:
                pred_id = output_to_node[input_name]
                if pred_id < len(analysis.nodes):
                    pred_node = analysis.nodes[pred_id]
                    predecessors.append({
                        'name': pred_node.name,
                        'op_type': pred_node.op_type,
                        'output_shapes': [
                            list(s) if s else None 
                            for s in pred_node.output_shapes
                        ]
                    })
        
        return predecessors
    
    def _get_successor_context(
        self,
        sig: NodeSignature,
        model: onnx.ModelProto,
        analysis: ModelAnalysis
    ) -> List[Dict]:
        """Get context about successor nodes."""
        successors = []
        
        output_set = set(sig.output_names)
        
        for node in analysis.nodes:
            # Check if any of this node's inputs are outputs of our target
            node_inputs = set(node.inputs)
            if node_inputs & output_set:
                successors.append({
                    'name': node.name,
                    'op_type': node.op_type,
                    'input_shapes': [
                        list(s) if s else None 
                        for s in node.input_shapes
                    ]
                })
        
        return successors
    
    def _get_tensor_info(
        self,
        names: List[str],
        shapes: List[Optional[List[int]]],
        model: onnx.ModelProto,
        initializers: Set[str]
    ) -> List[Dict]:
        """Get detailed tensor information."""
        tensors = []
        
        for i, name in enumerate(names):
            if not name:
                continue
            
            shape = shapes[i] if i < len(shapes) else None
            is_initializer = name in initializers
            
            # Determine if dynamic
            is_dynamic = False
            dynamic_dims = []
            if shape:
                for j, dim in enumerate(shape):
                    if dim is None or (isinstance(dim, str)):
                        is_dynamic = True
                        dynamic_dims.append(j)
            
            tensors.append({
                'name': name,
                'shape': shape,
                'dtype': 'float32',  # Default, would need value_info lookup for actual type
                'is_initializer': is_initializer,
                'is_dynamic': is_dynamic,
                'dynamic_dims': dynamic_dims
            })
        
        return tensors
    
    def _generate_removal_steps(
        self,
        sig: NodeSignature,
        predecessors: List[Dict],
        successors: List[Dict]
    ) -> List[str]:
        """Generate step-by-step instructions for removing a node."""
        steps = []
        
        steps.append(f"1. Identify {sig.op_type} node '{sig.name}' (node_id: {sig.node_id})")
        
        if predecessors:
            pred_names = [p['name'] for p in predecessors[:3]]
            steps.append(f"2. Note predecessor nodes: {', '.join(pred_names)}")
        
        if successors:
            succ_names = [s['name'] for s in successors[:3]]
            steps.append(f"3. Note successor nodes: {', '.join(succ_names)}")
        
        if len(sig.input_names) == 1 and len(sig.output_names) == 1:
            steps.append(f"4. Rewire: Connect '{sig.input_names[0]}' directly to consumers of '{sig.output_names[0]}'")
        else:
            steps.append(f"4. Rewire graph to bypass this node (inputs: {len(sig.input_names)}, outputs: {len(sig.output_names)})")
        
        steps.append(f"5. Remove node '{sig.name}' from graph")
        steps.append("6. Validate graph structure with onnx.checker.check_model()")
        
        return steps
    
    def _generate_addition_steps(
        self,
        sig: NodeSignature,
        predecessors: List[Dict],
        successors: List[Dict]
    ) -> List[str]:
        """Generate step-by-step instructions for adding a node."""
        steps = []
        
        steps.append(f"1. Create new {sig.op_type} node named '{sig.name}'")
        
        if sig.attributes:
            attr_str = ', '.join(f"{k}={v}" for k, v in list(sig.attributes.items())[:3])
            steps.append(f"2. Set attributes: {attr_str}")
        
        if sig.input_names:
            steps.append(f"3. Connect inputs: {', '.join(sig.input_names[:3])}")
        
        if sig.output_names:
            steps.append(f"4. Define outputs: {', '.join(sig.output_names[:3])}")
        
        steps.append(f"5. Insert node at appropriate position in graph")
        steps.append("6. Validate graph structure with onnx.checker.check_model()")
        
        return steps
    
    def _generate_modification_steps(
        self,
        orig_sig: NodeSignature,
        mod_sig: NodeSignature,
        changes: Dict[str, Any]
    ) -> List[str]:
        """Generate step-by-step instructions for modifying a node."""
        steps = []
        
        steps.append(f"1. Locate {orig_sig.op_type} node '{orig_sig.name}' (node_id: {orig_sig.node_id})")
        
        if 'op_type' in changes:
            steps.append(f"2. Change op_type: {changes['op_type']['from']} -> {changes['op_type']['to']}")
        
        if 'attributes' in changes:
            for attr_name, change in list(changes['attributes'].items())[:3]:
                steps.append(f"3. Update attribute '{attr_name}': {change['from']} -> {change['to']}")
        
        if 'inputs' in changes:
            steps.append(f"4. Update inputs: {changes['inputs']['from']} -> {changes['inputs']['to']}")
        
        if 'outputs' in changes:
            steps.append(f"5. Update outputs: {changes['outputs']['from']} -> {changes['outputs']['to']}")
        
        steps.append("6. Validate graph structure with onnx.checker.check_model()")
        
        return steps
    
    def _generate_removal_code(self, sig: NodeSignature) -> str:
        """Generate GraphSurgeon code for node removal."""
        code = f"""# Remove {sig.op_type} node '{sig.name}'
import onnx_graphsurgeon as gs

# Find the node
for node in graph.nodes:
    if node.name == "{sig.name}":
        # Bypass the node by connecting its inputs to its outputs' consumers
        if len(node.inputs) == 1 and len(node.outputs) == 1:
            input_tensor = node.inputs[0]
            output_tensor = node.outputs[0]
            
            # Replace output tensor with input tensor in all consumers
            for consumer in graph.nodes:
                for i, inp in enumerate(consumer.inputs):
                    if inp == output_tensor:
                        consumer.inputs[i] = input_tensor
        
        # Remove the node
        node.outputs.clear()
        break

# Clean up
graph.cleanup()
"""
        return code
    
    def _generate_addition_code(self, sig: NodeSignature) -> str:
        """Generate GraphSurgeon code for node addition."""
        attrs_str = ', '.join(f'"{k}": {repr(v)}' for k, v in list(sig.attributes.items())[:5])
        
        code = f"""# Add {sig.op_type} node '{sig.name}'
import onnx_graphsurgeon as gs
import numpy as np

# Create output variable
output = gs.Variable("{sig.output_names[0] if sig.output_names else 'output'}", dtype=np.float32)

# Create the node
node = gs.Node(
    op="{sig.op_type}",
    name="{sig.name}",
    inputs=[{', '.join(f'"{n}"' for n in sig.input_names[:3])}],  # Connect to appropriate inputs
    outputs=[output],
    attrs={{{attrs_str}}}
)

# Add to graph
graph.nodes.append(node)
graph.cleanup()
"""
        return code
    
    def _generate_modification_code(
        self,
        orig_sig: NodeSignature,
        mod_sig: NodeSignature,
        changes: Dict[str, Any]
    ) -> str:
        """Generate GraphSurgeon code for node modification."""
        code = f"""# Modify {orig_sig.op_type} node '{orig_sig.name}'
import onnx_graphsurgeon as gs

# Find the node
for node in graph.nodes:
    if node.name == "{orig_sig.name}":
"""
        
        if 'op_type' in changes:
            code += f'        node.op = "{mod_sig.op_type}"\n'
        
        if 'attributes' in changes:
            for attr_name, change in list(changes['attributes'].items())[:5]:
                code += f'        node.attrs["{attr_name}"] = {repr(change["to"])}\n'
        
        code += """        break

graph.cleanup()
"""
        return code
    
    def _count_shape_changes(self, matches: List[NodeMatch]) -> int:
        """Count total shape changes across all matches."""
        count = 0
        for match in matches:
            if match.changes and 'output_shapes' in match.changes:
                count += 1
        return count
    
    def _get_blocker_summary(self, analysis: ModelAnalysis) -> List[str]:
        """Get summary of compilation blockers."""
        blocker_counts = defaultdict(int)
        
        for node in analysis.nodes:
            if node.is_compilation_blocker:
                blocker_counts[node.op_type] += 1
        
        return [f"{op_type} ({count} nodes)" for op_type, count in blocker_counts.items()]
    
    def _compute_transformation_order(
        self,
        transformations: List[NodeTransformation],
        analysis: ModelAnalysis
    ) -> List[int]:
        """Compute recommended transformation order based on dependencies."""
        # Simple approach: order by graph position (earlier nodes first)
        indexed_transforms = [(i, t.graph_position) for i, t in enumerate(transformations)]
        indexed_transforms.sort(key=lambda x: x[1])
        
        return [i for i, _ in indexed_transforms]
    
    def clear_cache(self) -> None:
        """Clear model and analysis caches."""
        self._model_cache.clear()
        self._analysis_cache.clear()


# =============================================================================
# Batch Processing
# =============================================================================

def extract_all_from_dataset(
    dataset_dir: str,
    output_path: str = "rag_data/surgery_database.json",
    verbose: bool = True
) -> SurgeryDatabase:
    """
    Extract transformations from all model pairs in a dataset directory.
    
    Expects directory structure:
    dataset/
        ModelName1/
            Original/
                model.onnx
            Modified/
                model.onnx
        ModelName2/
            ...
    
    Args:
        dataset_dir: Path to dataset directory
        output_path: Path to save the resulting database
        verbose: Whether to print progress
        
    Returns:
        Populated SurgeryDatabase
    """
    from knowledge_base.surgery_database import create_database_with_defaults
    
    extractor = TransformationExtractor(verbose=verbose)
    db = create_database_with_defaults()
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return db
    
    # Find all model directories
    model_dirs = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            # Check for Original/Modified subdirectories
            original_dir = item / "Original"
            modified_dir = item / "Modified"
            
            # Handle case variations
            if not original_dir.exists():
                original_dir = item / "original"
            if not modified_dir.exists():
                modified_dir = item / "modified"
            
            if original_dir.exists() and modified_dir.exists():
                model_dirs.append((item.name, original_dir, modified_dir))
    
    if verbose:
        print(f"Found {len(model_dirs)} model pairs in {dataset_dir}")
    
    # Process each model pair
    successful = 0
    failed = 0
    
    for model_name, original_dir, modified_dir in model_dirs:
        # Find ONNX files
        original_files = list(original_dir.glob("*.onnx"))
        modified_files = list(modified_dir.glob("*.onnx"))
        
        if not original_files or not modified_files:
            if verbose:
                print(f"  Skipping {model_name}: Missing ONNX files")
            failed += 1
            continue
        
        original_path = str(original_files[0])
        modified_path = str(modified_files[0])
        
        try:
            if verbose:
                print(f"\nProcessing: {model_name}")
            
            record = extractor.extract_transformations(
                original_path, modified_path, model_name
            )
            db.add_transformation_record(record)
            successful += 1
            
        except Exception as e:
            if verbose:
                print(f"  Error processing {model_name}: {e}")
            failed += 1
            continue
        
        # Clear cache periodically to manage memory
        if (successful + failed) % 5 == 0:
            extractor.clear_cache()
    
    # Save database
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        db.save(output_path)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Extraction complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total transformations: {db.total_transformations}")
        if output_path:
            print(f"  Saved to: {output_path}")
    
    return db


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract transformations from ONNX model pairs')
    parser.add_argument('--original', help='Path to original ONNX model')
    parser.add_argument('--modified', help='Path to modified ONNX model')
    parser.add_argument('--dataset', help='Path to dataset directory for batch processing')
    parser.add_argument('--output', default='rag_data/surgery_database.json', help='Output path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.dataset:
        # Batch processing
        db = extract_all_from_dataset(args.dataset, args.output, args.verbose)
    elif args.original and args.modified:
        # Single pair processing
        extractor = TransformationExtractor(verbose=args.verbose)
        record = extractor.extract_transformations(args.original, args.modified)
        
        print(f"\nExtracted {len(record.transformations)} transformations from {record.model_name}")
        
        # Save to JSON
        import json
        with open(args.output, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
        print(f"Saved to {args.output}")
    else:
        parser.print_help()
