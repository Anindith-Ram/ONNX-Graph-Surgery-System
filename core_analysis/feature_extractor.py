#!/usr/bin/env python3
"""
Extract high-level features from model differences for RAG indexing.
Converts technical differences into semantic features that can be matched.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from core_analysis.difference_extractor import ModelDiff, NodeInfo


class FeatureExtractor:
    """Extracts semantic features from model differences."""
    
    def __init__(self):
        # Based on ONNX Graph Surgery for Model SDK.pdf
        # Common compilation blockers that need graph surgery
        self.compilation_blockers = [
            'Einsum', 'Complex', 'DynamicSlice', 'NonMaxSuppression',
            'TopK', 'Scan', 'Loop', 'If', 'Sequence'
        ]
        
        # Shape issues that prevent MLA compilation (need 4D tensors)
        self.shape_issues = [
            'dynamic', 'unknown', 'unk__', '?', 'batch_size', 'time_frames'
        ]
        
        # Common graph surgery patterns from PDF
        # 1. Non-4D tensors need reshaping to 4D
        # 2. Unsupported operators replaced by supported ones
        # 3. Reshape/transpose operations that prevent MLA mapping
        self.surgery_patterns = {
            'non_4d_tensors': ['Reshape', 'Transpose', 'Slice', 'Concat'],
            'unsupported_ops': ['Einsum', 'Complex', 'DynamicSlice'],
            'data_reshuffling': ['Reshape', 'Slice', 'Concat', 'Transpose']
        }
    
    def extract_features(self, diff: ModelDiff) -> Dict[str, Any]:
        """Extract semantic features from a model difference."""
        features = {
            'model_name': diff.model_name,
            'summary': self._generate_summary(diff),
            'key_changes': self._identify_key_changes(diff),
            'compilation_issues_fixed': self._identify_compilation_issues(diff),
            'transformation_patterns': self._identify_patterns(diff),
            'node_statistics': self._compute_statistics(diff),
            'recommended_fixes': self._suggest_fixes(diff)
        }
        return features
    
    def _generate_summary(self, diff: ModelDiff) -> str:
        """Generate a human-readable summary of changes."""
        parts = []
        
        if diff.node_count_diff > 0:
            parts.append(f"Added {diff.node_count_diff} nodes")
        elif diff.node_count_diff < 0:
            parts.append(f"Removed {abs(diff.node_count_diff)} nodes")
        
        if diff.added_nodes:
            op_types = [n.op_type for n in diff.added_nodes]
            unique_ops = list(set(op_types))
            parts.append(f"Added operations: {', '.join(unique_ops[:5])}")
        
        if diff.removed_nodes:
            op_types = [n.op_type for n in diff.removed_nodes]
            unique_ops = list(set(op_types))
            parts.append(f"Removed operations: {', '.join(unique_ops[:5])}")
        
        if diff.op_type_changes:
            changes = list(diff.op_type_changes.items())[:3]
            parts.append(f"Operation transformations: {', '.join([f'{k}({v})' for k, v in changes])}")
        
        if diff.shape_changes:
            parts.append(f"Shape modifications: {len(diff.shape_changes)} nodes")
        
        return "; ".join(parts) if parts else "Minor structural changes"
    
    def _identify_key_changes(self, diff: ModelDiff) -> List[str]:
        """Identify the most significant changes."""
        changes = []
        
        # Check for removed problematic ops
        removed_ops = {n.op_type for n in diff.removed_nodes}
        problematic_removed = removed_ops & set(self.compilation_blockers)
        if problematic_removed:
            changes.append(f"Removed compilation-blocking operations: {', '.join(problematic_removed)}")
        
        # Check for added replacement ops
        added_ops = {n.op_type for n in diff.added_nodes}
        if 'Reshape' in added_ops or 'Transpose' in added_ops:
            changes.append("Added shape manipulation operations")
        if 'Split' in added_ops:
            changes.append("Added tensor splitting operations")
        if 'Concat' in added_ops:
            changes.append("Added tensor concatenation operations")
        
        # Check for op type transformations
        for transform, count in diff.op_type_changes.items():
            if '->' in transform:
                orig, mod = transform.split('->')
                if orig in self.compilation_blockers:
                    changes.append(f"Replaced {orig} with {mod} ({count} instances)")
        
        # Check for shape simplifications
        shape_simplified = False
        for sc in diff.shape_changes:
            orig_shapes = str(sc['original_output_shapes'])
            mod_shapes = str(sc['modified_output_shapes'])
            if any(issue in orig_shapes for issue in self.shape_issues):
                if not any(issue in mod_shapes for issue in self.shape_issues):
                    shape_simplified = True
                    break
        
        if shape_simplified:
            changes.append("Simplified dynamic/unknown shapes to concrete shapes")
        
        return changes
    
    def _identify_compilation_issues(self, diff: ModelDiff) -> List[str]:
        """Identify what compilation issues were likely fixed."""
        issues = []
        
        # Check for Einsum removal/replacement
        if any(n.op_type == 'Einsum' for n in diff.removed_nodes):
            issues.append("Einsum operations (often unsupported on edge devices)")
        
        # Check for dynamic shape resolution
        orig_has_dynamic = False
        mod_has_dynamic = False
        
        for node in diff.removed_nodes:
            if any(issue in str(node.output_shapes) for issue in self.shape_issues):
                orig_has_dynamic = True
        
        for node in diff.added_nodes:
            if any(issue in str(node.output_shapes) for issue in self.shape_issues):
                mod_has_dynamic = True
        
        if orig_has_dynamic and not mod_has_dynamic:
            issues.append("Dynamic/unknown tensor shapes")
        
        # Check for complex control flow
        complex_ops = ['Loop', 'Scan', 'If', 'Sequence']
        if any(n.op_type in complex_ops for n in diff.removed_nodes):
            issues.append("Complex control flow operations")
        
        # Check for large tensor operations
        if diff.shape_changes:
            large_tensor_ops = 0
            for sc in diff.shape_changes:
                # Check if shapes were reduced
                orig_size = self._estimate_tensor_size(sc['original_output_shapes'])
                mod_size = self._estimate_tensor_size(sc['modified_output_shapes'])
                if mod_size < orig_size * 0.8:  # 20% reduction
                    large_tensor_ops += 1
            
            if large_tensor_ops > 0:
                issues.append(f"Large tensor operations ({large_tensor_ops} instances)")
        
        return issues
    
    def _identify_patterns(self, diff: ModelDiff) -> List[str]:
        """Identify common transformation patterns based on ONNX Graph Surgery PDF."""
        patterns = []
        
        # Pattern 1: Einsum -> MatMul + Reshape (from PDF case studies)
        einsum_removed = sum(1 for n in diff.removed_nodes if n.op_type == 'Einsum')
        matmul_added = sum(1 for n in diff.added_nodes if n.op_type == 'MatMul')
        reshape_added = sum(1 for n in diff.added_nodes if n.op_type == 'Reshape')
        
        if einsum_removed > 0 and matmul_added > 0 and reshape_added > 0:
            patterns.append("Einsum decomposition: Einsum -> MatMul + Reshape operations (enables MLA mapping)")
        
        # Pattern 2: Non-4D to 4D tensor conversion (PDF requirement: MLA needs 4D tensors)
        # Check if operations were added to maintain 4D tensors throughout
        data_reshuffling_added = sum(1 for n in diff.added_nodes 
                                     if n.op_type in self.surgery_patterns['data_reshuffling'])
        if data_reshuffling_added > 3:
            patterns.append("4D tensor maintenance: Added Reshape/Transpose/Slice/Concat to keep 4D tensors throughout")
        
        # Pattern 3: Dynamic shapes -> Static shapes (PDF: shape inference needed)
        dynamic_to_static = False
        for sc in diff.shape_changes:
            orig = str(sc['original_output_shapes'])
            mod = str(sc['modified_output_shapes'])
            if any(d in orig for d in self.shape_issues) and \
               not any(d in mod for d in self.shape_issues):
                dynamic_to_static = True
                break
        
        if dynamic_to_static:
            patterns.append("Dynamic shape resolution: Unknown dimensions -> Concrete dimensions (required for compilation)")
        
        # Pattern 4: Operation decomposition (PDF: divide-and-conquer approach)
        if diff.node_count_diff > 5:
            patterns.append("Operation decomposition: Large operations split into smaller ones (divide-and-conquer)")
        
        # Pattern 5: Remove unsupported operators (PDF: replace with supported ones)
        unsupported_removed = sum(1 for n in diff.removed_nodes 
                                  if n.op_type in self.surgery_patterns['unsupported_ops'])
        if unsupported_removed > 0:
            patterns.append(f"Unsupported operator removal: {unsupported_removed} unsupported operators replaced")
        
        # Pattern 6: Remove redundant operations (PDF: optimization)
        if diff.node_count_diff < 0:
            patterns.append("Operation elimination: Redundant operations removed")
        
        # Pattern 7: Slicing to avoid reshape/transpose (PDF: YoloV8 case study)
        slice_added = sum(1 for n in diff.added_nodes if n.op_type == 'Slice')
        reshape_removed = sum(1 for n in diff.removed_nodes if n.op_type == 'Reshape')
        transpose_removed = sum(1 for n in diff.removed_nodes if n.op_type == 'Transpose')
        
        if slice_added > 2 and (reshape_removed > 0 or transpose_removed > 0):
            patterns.append("Slicing pattern: Using Slice operations to avoid Reshape/Transpose (enables MLA mapping)")
        
        return patterns
    
    def _compute_statistics(self, diff: ModelDiff) -> Dict[str, Any]:
        """Compute statistical features."""
        stats = {
            'total_changes': len(diff.added_nodes) + len(diff.removed_nodes) + len(diff.modified_nodes),
            'node_count_change': diff.node_count_diff,
            'op_type_transformations': len(diff.op_type_changes),
            'shape_modifications': len(diff.shape_changes),
            'most_common_added_op': self._most_common_op(diff.added_nodes),
            'most_common_removed_op': self._most_common_op(diff.removed_nodes)
        }
        return stats
    
    def _most_common_op(self, nodes: List[NodeInfo]) -> str:
        """Find most common operation type."""
        if not nodes:
            return "None"
        from collections import Counter
        op_counts = Counter(n.op_type for n in nodes)
        return op_counts.most_common(1)[0][0]
    
    def _suggest_fixes(self, diff: ModelDiff) -> List[str]:
        """Suggest fixes based on patterns."""
        fixes = []
        
        if any(n.op_type == 'Einsum' for n in diff.removed_nodes):
            fixes.append("Replace Einsum operations with MatMul + Reshape combinations")
        
        if diff.shape_changes:
            fixes.append("Resolve dynamic shapes by adding shape inference operations")
        
        if any('Slice' in n.op_type for n in diff.added_nodes):
            fixes.append("Use explicit Slice operations instead of dynamic indexing")
        
        return fixes
    
    def _estimate_tensor_size(self, shapes: List[str]) -> int:
        """Estimate tensor size from shape strings."""
        if not shapes:
            return 0
        
        total = 0
        for shape_str in shapes:
            # Extract numbers from shape string like "(1, 3, 224, 224)"
            import re
            numbers = re.findall(r'\d+', shape_str)
            if numbers:
                size = 1
                for num in numbers:
                    size *= int(num)
                total += size
        
        return total


def extract_all_features(map_dataset_dir: str, output_file: str = None) -> List[Dict[str, Any]]:
    """Extract features for all models."""
    from core_analysis.difference_extractor import extract_all_differences
    
    diffs = extract_all_differences(map_dataset_dir)
    extractor = FeatureExtractor()
    
    features = []
    for diff in diffs:
        feat = extractor.extract_features(diff)
        features.append(feat)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)
    
    return features


if __name__ == "__main__":
    import sys
    import os
    
    map_dataset_dir = os.path.join(os.path.dirname(__file__), "map_dataset")
    output_file = os.path.join(os.path.dirname(__file__), "rag_data", "features.json")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    features = extract_all_features(map_dataset_dir, output_file)
    print(f"Extracted features for {len(features)} models")
    print(f"Saved to {output_file}")

