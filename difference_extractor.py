#!/usr/bin/env python3
"""
Extract differences between original and modified ONNX model maps.
Identifies key architectural changes that enabled compilation.
"""

import re
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class NodeInfo:
    """Represents a single node in the ONNX graph."""
    node_id: str
    op_type: str
    input_node_ids: List[str]
    input_shapes: List[str]
    output_shapes: List[str]
    
    def to_dict(self):
        return {
            'node_id': self.node_id,
            'op_type': self.op_type,
            'input_node_ids': self.input_node_ids,
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes
        }


@dataclass
class ModelDiff:
    """Represents differences between original and modified models."""
    model_name: str
    node_count_diff: int
    added_nodes: List[NodeInfo] = field(default_factory=list)
    removed_nodes: List[NodeInfo] = field(default_factory=list)
    modified_nodes: List[Tuple[NodeInfo, NodeInfo]] = field(default_factory=list)
    op_type_changes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    shape_changes: List[Dict] = field(default_factory=list)
    structural_changes: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'model_name': self.model_name,
            'node_count_diff': self.node_count_diff,
            'added_nodes': [n.to_dict() for n in self.added_nodes],
            'removed_nodes': [n.to_dict() for n in self.removed_nodes],
            'modified_nodes': [
                (orig.to_dict(), mod.to_dict()) 
                for orig, mod in self.modified_nodes
            ],
            'op_type_changes': dict(self.op_type_changes),
            'shape_changes': self.shape_changes,
            'structural_changes': self.structural_changes
        }


def parse_node_line(line: str) -> Optional[NodeInfo]:
    """Parse a single node line from the map format."""
    # Format: <node id>, <op>, <input node ids>  <input shape>, <output shape>
    line = line.strip()
    if not line or line.startswith('#') or 'graph outputs' in line:
        return None
    
    # Match pattern: id, op, [inputs]  [shapes], [shapes]
    match = re.match(r'^(\d+),\s+(\w+),\s+(\[[^\]]+\])\s+(\[[^\]]+\]),\s+(\[[^\]]+\])', line)
    if not match:
        return None
    
    node_id, op_type, inputs_str, input_shapes_str, output_shapes_str = match.groups()
    
    # Parse lists
    input_node_ids = eval(inputs_str) if inputs_str else []
    input_shapes = eval(input_shapes_str) if input_shapes_str else []
    output_shapes = eval(output_shapes_str) if output_shapes_str else []
    
    return NodeInfo(
        node_id=node_id,
        op_type=op_type,
        input_node_ids=input_node_ids,
        input_shapes=input_shapes,
        output_shapes=output_shapes
    )


def parse_model_map(content: str) -> Dict[str, NodeInfo]:
    """Parse a model map and return a dictionary of node_id -> NodeInfo."""
    nodes = {}
    for line in content.split('\n'):
        node = parse_node_line(line)
        if node:
            nodes[node.node_id] = node
    return nodes


def extract_model_sections(content: str) -> Tuple[str, str]:
    """Extract original and modified model sections from a map file."""
    sections = content.split('MODIFIED MODEL MAP')
    if len(sections) < 2:
        return "", ""
    
    original_section = sections[0].split('ORIGINAL MODEL MAP')[1] if 'ORIGINAL MODEL MAP' in sections[0] else ""
    modified_section = sections[1]
    
    return original_section.strip(), modified_section.strip()


def compare_nodes(orig: NodeInfo, mod: NodeInfo) -> List[str]:
    """Compare two nodes and return list of differences."""
    differences = []
    
    if orig.op_type != mod.op_type:
        differences.append(f"OpType changed: {orig.op_type} -> {mod.op_type}")
    
    if orig.input_node_ids != mod.input_node_ids:
        differences.append(f"Input dependencies changed")
    
    if orig.input_shapes != mod.input_shapes:
        differences.append(f"Input shapes changed: {orig.input_shapes} -> {mod.input_shapes}")
    
    if orig.output_shapes != mod.output_shapes:
        differences.append(f"Output shapes changed: {orig.output_shapes} -> {mod.output_shapes}")
    
    return differences


def extract_differences(map_file_path: str) -> ModelDiff:
    """Extract differences from a model map file."""
    with open(map_file_path, 'r') as f:
        content = f.read()
    
    # Extract model name
    model_name_match = re.search(r'# Model: (.+)', content)
    model_name = model_name_match.group(1) if model_name_match else "unknown"
    
    # Extract sections
    original_content, modified_content = extract_model_sections(content)
    
    # Parse nodes
    original_nodes = parse_model_map(original_content)
    modified_nodes = parse_model_map(modified_content)
    
    # Create diff
    diff = ModelDiff(
        model_name=model_name,
        node_count_diff=len(modified_nodes) - len(original_nodes)
    )
    
    # Find added, removed, and modified nodes
    orig_ids = set(original_nodes.keys())
    mod_ids = set(modified_nodes.keys())
    
    added_ids = mod_ids - orig_ids
    removed_ids = orig_ids - mod_ids
    common_ids = orig_ids & mod_ids
    
    for node_id in added_ids:
        diff.added_nodes.append(modified_nodes[node_id])
    
    for node_id in removed_ids:
        diff.removed_nodes.append(original_nodes[node_id])
    
    for node_id in common_ids:
        orig_node = original_nodes[node_id]
        mod_node = modified_nodes[node_id]
        
        if orig_node.op_type != mod_node.op_type:
            diff.op_type_changes[f"{orig_node.op_type}->{mod_node.op_type}"] += 1
        
        node_diffs = compare_nodes(orig_node, mod_node)
        if node_diffs:
            diff.modified_nodes.append((orig_node, mod_node))
            
            # Track shape changes
            if orig_node.input_shapes != mod_node.input_shapes or \
               orig_node.output_shapes != mod_node.output_shapes:
                diff.shape_changes.append({
                    'node_id': node_id,
                    'op_type': orig_node.op_type,
                    'original_input_shapes': orig_node.input_shapes,
                    'modified_input_shapes': mod_node.input_shapes,
                    'original_output_shapes': orig_node.output_shapes,
                    'modified_output_shapes': mod_node.output_shapes
                })
    
    # Identify structural patterns
    if diff.node_count_diff != 0:
        diff.structural_changes.append(
            f"Node count changed: {len(original_nodes)} -> {len(modified_nodes)}"
        )
    
    # Count op type frequency changes
    orig_ops = defaultdict(int)
    mod_ops = defaultdict(int)
    
    for node in original_nodes.values():
        orig_ops[node.op_type] += 1
    for node in modified_nodes.values():
        mod_ops[node.op_type] += 1
    
    for op_type in set(list(orig_ops.keys()) + list(mod_ops.keys())):
        if orig_ops[op_type] != mod_ops[op_type]:
            diff.structural_changes.append(
                f"{op_type} count: {orig_ops[op_type]} -> {mod_ops[op_type]}"
            )
    
    return diff


def extract_all_differences(map_dataset_dir: str) -> List[ModelDiff]:
    """Extract differences for all models in the map dataset."""
    import os
    diffs = []
    
    for filename in os.listdir(map_dataset_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(map_dataset_dir, filename)
            try:
                diff = extract_differences(filepath)
                diffs.append(diff)
                print(f"Extracted differences for {diff.model_name}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return diffs


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diff = extract_differences(sys.argv[1])
        print(f"\nModel: {diff.model_name}")
        print(f"Node count diff: {diff.node_count_diff}")
        print(f"Added nodes: {len(diff.added_nodes)}")
        print(f"Removed nodes: {len(diff.removed_nodes)}")
        print(f"Modified nodes: {len(diff.modified_nodes)}")
        print(f"Op type changes: {dict(diff.op_type_changes)}")
        print(f"Shape changes: {len(diff.shape_changes)}")
        print(f"Structural changes: {diff.structural_changes}")
    else:
        print("Usage: difference_extractor.py <map_file.txt>")

