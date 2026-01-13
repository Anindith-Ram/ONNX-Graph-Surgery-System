#!/usr/bin/env python3
"""
Compare pipeline-modified model vs ground truth modified model.
"""

import onnx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from difference_extractor import extract_differences, ModelDiff
from collections import defaultdict, Counter

# Optional dependency for numerical output comparison
try:
    import onnxruntime as ort  # type: ignore
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ort = None
    ONNXRUNTIME_AVAILABLE = False


class ModelComparator:
    """Compare two ONNX models and compute similarity metrics."""
    
    def compare_models(
        self,
        pipeline_model: onnx.ModelProto,
        ground_truth_model: onnx.ModelProto,
        original_model: Optional[onnx.ModelProto] = None
    ) -> Dict[str, Any]:
        """
        Compare pipeline-modified vs ground truth modified models.
        
        Args:
            pipeline_model: Model modified by pipeline rules
            ground_truth_model: Ground truth modified model
            original_model: Optional original model for reference
        
        Returns:
            Dictionary with comparison metrics
        """
        # Extract structural differences
        pipeline_nodes = {node.name: node for node in pipeline_model.graph.node}
        gt_nodes = {node.name: node for node in ground_truth_model.graph.node}
        
        # Node-level comparison
        node_metrics = self._compare_nodes(pipeline_nodes, gt_nodes)
        
        # Operation type comparison
        op_metrics = self._compare_operations(pipeline_model, ground_truth_model)
        
        # Shape comparison
        shape_metrics = self._compare_shapes(pipeline_model, ground_truth_model)
        
        # Structural similarity
        structural_similarity = self._compute_structural_similarity(
            pipeline_model, ground_truth_model
        )
        
        # Transformation accuracy (if original model provided)
        transformation_accuracy = None
        if original_model is not None:
            transformation_accuracy = self._calculate_transformation_accuracy(
                original_model, pipeline_model, ground_truth_model
            )
        
        # Operation comparison with jaccard similarity (for compatibility)
        operation_comparison = {
            'jaccard_similarity': op_metrics['operation_similarity']
        }
        
        result = {
            'node_metrics': node_metrics,
            'operation_metrics': op_metrics,
            'operation_comparison': operation_comparison,  # Added for compatibility
            'shape_metrics': shape_metrics,
            'structural_similarity': structural_similarity,
            'overall_similarity': self._compute_overall_similarity(
                node_metrics, op_metrics, shape_metrics, structural_similarity
            )
        }
        
        if transformation_accuracy is not None:
            result['transformation_accuracy'] = transformation_accuracy
        
        return result
    
    def _compare_nodes(
        self, 
        pipeline_nodes: Dict, 
        gt_nodes: Dict
    ) -> Dict[str, Any]:
        """Compare node structures."""
        pipeline_node_names = set(pipeline_nodes.keys())
        gt_node_names = set(gt_nodes.keys())
        
        common_nodes = pipeline_node_names & gt_node_names
        pipeline_only = pipeline_node_names - gt_node_names
        gt_only = gt_node_names - pipeline_node_names
        
        # Compare operation types for common nodes
        op_type_matches = 0
        for node_name in common_nodes:
            if pipeline_nodes[node_name].op_type == gt_nodes[node_name].op_type:
                op_type_matches += 1
        
        return {
            'pipeline_node_count': len(pipeline_node_names),
            'ground_truth_node_count': len(gt_node_names),
            'common_nodes': len(common_nodes),
            'pipeline_only_nodes': len(pipeline_only),
            'ground_truth_only_nodes': len(gt_only),
            'operation_type_match_rate': op_type_matches / len(common_nodes) if common_nodes else 0,
            'node_overlap': len(common_nodes) / max(len(pipeline_node_names), len(gt_node_names), 1)
        }
    
    def _compare_operations(
        self,
        pipeline_model: onnx.ModelProto,
        gt_model: onnx.ModelProto
    ) -> Dict[str, Any]:
        """Compare operation types."""
        pipeline_ops = defaultdict(int)
        gt_ops = defaultdict(int)
        
        for node in pipeline_model.graph.node:
            pipeline_ops[node.op_type] += 1
        
        for node in gt_model.graph.node:
            gt_ops[node.op_type] += 1
        
        all_ops = set(list(pipeline_ops.keys()) + list(gt_ops.keys()))
        matching_ops = {op for op in all_ops if pipeline_ops[op] == gt_ops[op]}
        
        return {
            'pipeline_operations': dict(pipeline_ops),
            'ground_truth_operations': dict(gt_ops),
            'matching_operations': len(matching_ops),
            'operation_similarity': len(matching_ops) / len(all_ops) if all_ops else 0
        }
    
    def _compare_shapes(
        self,
        pipeline_model: onnx.ModelProto,
        gt_model: onnx.ModelProto
    ) -> Dict[str, Any]:
        """Compare tensor shapes."""
        pipeline_shapes = {}
        gt_shapes = {}
        
        # Extract output shapes
        for output in pipeline_model.graph.output:
            shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
            pipeline_shapes[output.name] = tuple(shape)
        
        for output in gt_model.graph.output:
            shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
            gt_shapes[output.name] = tuple(shape)
        
        # Compare
        matching_shapes = sum(
            1 for name in pipeline_shapes 
            if name in gt_shapes and pipeline_shapes[name] == gt_shapes[name]
        )
        
        return {
            'pipeline_output_shapes': pipeline_shapes,
            'ground_truth_output_shapes': gt_shapes,
            'matching_shapes': matching_shapes,
            'shape_similarity': matching_shapes / max(len(pipeline_shapes), len(gt_shapes), 1)
        }
    
    def _compute_structural_similarity(
        self,
        pipeline_model: onnx.ModelProto,
        gt_model: onnx.ModelProto
    ) -> float:
        """Compute overall structural similarity score."""
        # Simple similarity: node count, operation types, graph structure
        pipeline_node_count = len(pipeline_model.graph.node)
        gt_node_count = len(gt_model.graph.node)
        
        node_count_similarity = 1.0 - abs(pipeline_node_count - gt_node_count) / max(pipeline_node_count, gt_node_count, 1)
        
        # Operation type similarity
        pipeline_ops = {node.op_type for node in pipeline_model.graph.node}
        gt_ops = {node.op_type for node in gt_model.graph.node}
        op_similarity = len(pipeline_ops & gt_ops) / len(pipeline_ops | gt_ops) if (pipeline_ops | gt_ops) else 0
        
        return (node_count_similarity + op_similarity) / 2
    
    def _compute_overall_similarity(
        self,
        node_metrics: Dict,
        op_metrics: Dict,
        shape_metrics: Dict,
        structural_similarity: float
    ) -> float:
        """Compute weighted overall similarity."""
        weights = {
            'node_overlap': 0.3,
            'operation_similarity': 0.3,
            'shape_similarity': 0.2,
            'structural': 0.2
        }
        
        overall = (
            node_metrics['node_overlap'] * weights['node_overlap'] +
            op_metrics['operation_similarity'] * weights['operation_similarity'] +
            shape_metrics['shape_similarity'] * weights['shape_similarity'] +
            structural_similarity * weights['structural']
        )
        
        return overall
    
    def compare_outputs(
        self,
        pipeline_model: onnx.ModelProto,
        gt_model: onnx.ModelProto,
        test_inputs: Dict[str, np.ndarray],
        rtol: float = 1e-5,
        atol: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Compare model outputs numerically.
        
        Args:
            pipeline_model: Pipeline-modified model
            gt_model: Ground truth model
            test_inputs: Dictionary of input_name -> input_array
            rtol: Relative tolerance
            atol: Absolute tolerance
        
        Returns:
            Dictionary with output comparison metrics
        """
        if not ONNXRUNTIME_AVAILABLE:
            return {'error': 'onnxruntime not available. Install with: pip install onnxruntime'}
        
        try:
            # Create inference sessions
            pipeline_session = ort.InferenceSession(pipeline_model.SerializeToString())
            gt_session = ort.InferenceSession(gt_model.SerializeToString())
            
            # Run inference
            pipeline_outputs = pipeline_session.run(None, test_inputs)
            gt_outputs = gt_session.run(None, test_inputs)
            
            # Compare outputs
            output_diffs = []
            max_diffs = []
            mean_diffs = []
            
            for i, (p_out, gt_out) in enumerate(zip(pipeline_outputs, gt_outputs)):
                if p_out.shape != gt_out.shape:
                    output_diffs.append({
                        'output_index': i,
                        'shape_match': False,
                        'pipeline_shape': p_out.shape,
                        'ground_truth_shape': gt_out.shape
                    })
                    continue
                
                diff = np.abs(p_out - gt_out)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                # Check if outputs are close
                is_close = np.allclose(p_out, gt_out, rtol=rtol, atol=atol)
                
                output_diffs.append({
                    'output_index': i,
                    'shape_match': True,
                    'max_difference': float(max_diff),
                    'mean_difference': float(mean_diff),
                    'is_close': bool(is_close),
                    'within_tolerance': is_close
                })
                
                max_diffs.append(max_diff)
                mean_diffs.append(mean_diff)
            
            return {
                'output_comparisons': output_diffs,
                'max_difference_across_outputs': float(np.max(max_diffs)) if max_diffs else None,
                'mean_difference_across_outputs': float(np.mean(mean_diffs)) if mean_diffs else None,
                'all_outputs_match': all(d['within_tolerance'] for d in output_diffs if d.get('shape_match', False))
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'output_comparisons': []
            }
    
    def _get_node_by_name(self, model: onnx.ModelProto, node_name: str) -> Optional[onnx.NodeProto]:
        """Get a node by name from a model."""
        for node in model.graph.node:
            if node.name == node_name:
                return node
        return None
    
    def _calculate_graph_position(
        self,
        node: onnx.NodeProto,
        model: onnx.ModelProto
    ) -> Tuple[int, int]:
        """
        Calculate node's position in graph: distance from inputs and outputs.
        
        Uses topological ordering as a proxy for graph position.
        
        Returns:
            Tuple of (input_distance, output_distance)
            - input_distance: Position in forward topological order (0 = near inputs)
            - output_distance: Position in reverse topological order (0 = near outputs)
        """
        # Build node list and find node index
        nodes = list(model.graph.node)
        try:
            node_index = nodes.index(node)
        except ValueError:
            return (999, 999)  # Node not found
        
        # Calculate input distance: position in forward order (normalized by total nodes)
        # Nodes near the beginning are closer to inputs
        total_nodes = len(nodes)
        input_distance = node_index  # Raw index (0 = first node, likely near inputs)
        
        # Calculate output distance: position in reverse order
        # Nodes near the end are closer to outputs
        output_distance = total_nodes - node_index - 1  # Reverse index (0 = last node, near outputs)
        
        # Alternative: Use a simpler approach - just use index as proxy
        # This is faster and works well for most graphs
        return (input_distance, output_distance)
    
    def _get_output_consumers(
        self,
        node: onnx.NodeProto,
        model: onnx.ModelProto
    ) -> List[str]:
        """Get list of node names that consume this node's outputs."""
        consumers = []
        node_outputs = set(node.output)
        
        for n in model.graph.node:
            for input_tensor in n.input:
                if input_tensor in node_outputs:
                    consumers.append(n.name)
                    break  # Only count once per consumer node
        
        return consumers
    
    def _identify_removed_nodes_by_name(
        self,
        original_model: onnx.ModelProto,
        modified_model: onnx.ModelProto
    ) -> Dict[str, List[str]]:
        """
        Identify which specific nodes were removed, grouped by operation type.
        
        Returns:
            Dict mapping op_type -> list of removed node names
        """
        original_nodes = {node.name: node for node in original_model.graph.node}
        modified_nodes = {node.name: node for node in modified_model.graph.node}
        
        removed_nodes = defaultdict(list)
        for node_name, node in original_nodes.items():
            if node_name not in modified_nodes:
                removed_nodes[node.op_type].append(node_name)
        
        return dict(removed_nodes)
    
    def _identify_added_nodes_by_name(
        self,
        original_model: onnx.ModelProto,
        modified_model: onnx.ModelProto
    ) -> Dict[str, List[str]]:
        """
        Identify which specific nodes were added, grouped by operation type.
        
        Returns:
            Dict mapping op_type -> list of added node names
        """
        original_nodes = {node.name: node for node in original_model.graph.node}
        modified_nodes = {node.name: node for node in modified_model.graph.node}
        
        added_nodes = defaultdict(list)
        for node_name, node in modified_nodes.items():
            if node_name not in original_nodes:
                added_nodes[node.op_type].append(node_name)
        
        return dict(added_nodes)
    
    def _find_node_by_context(
        self,
        model: onnx.ModelProto,
        target_node: onnx.NodeProto,
        original_model: onnx.ModelProto
    ) -> Optional[str]:
        """
        Find a node in model that matches target_node by context using multi-factor matching.
        
        Uses multiple signals:
        1. Input producers (40% weight)
        2. Output consumers (30% weight) - NEW
        3. Input tensors (20% weight)
        4. Graph position (10% weight) - NEW
        
        This is used when exact name matching fails, to find nodes that were renamed
        but are in the same graph location (same inputs/outputs/position).
        
        Returns:
            Node name if found, None otherwise
        """
        # Build tensor producer map for original model
        original_tensor_producers = {}
        for node in original_model.graph.node:
            for output in node.output:
                original_tensor_producers[output] = node.name
        
        # Build tensor producer map for target model (GT or suggested)
        model_tensor_producers = {}
        for node in model.graph.node:
            for output in node.output:
                model_tensor_producers[output] = node.name
        
        # Build tensor consumer map for original model
        original_tensor_consumers = defaultdict(list)
        for node in original_model.graph.node:
            for input_tensor in node.input:
                original_tensor_consumers[input_tensor].append(node.name)
        
        # Build tensor consumer map for target model
        model_tensor_consumers = defaultdict(list)
        for node in model.graph.node:
            for input_tensor in node.input:
                model_tensor_consumers[input_tensor].append(node.name)
        
        # Find which original node produced target_node's inputs
        target_input_producers = []
        target_input_tensors = []
        for input_tensor in target_node.input:
            # Check if this input comes from original model
            producer = original_tensor_producers.get(input_tensor)
            if producer:
                target_input_producers.append(producer)
                target_input_tensors.append(input_tensor)
        
        # Get target node's output consumers (from original model)
        target_output_consumers = []
        for output_tensor in target_node.output:
            consumers = original_tensor_consumers.get(output_tensor, [])
            target_output_consumers.extend(consumers)
        target_output_consumers = list(set(target_output_consumers))  # Remove duplicates
        
        # Calculate target node's graph position
        target_position = self._calculate_graph_position(target_node, original_model)
        
        # If no original inputs found, can't match by context
        if not target_input_producers:
            return None
        
        # Look for nodes in model with same operation type and similar context
        best_match = None
        best_match_score = 0.0
        
        for node in model.graph.node:
            if node.op_type != target_node.op_type:
                continue
            
            # Factor 1: Input producers match (40% weight)
            node_input_producers = []
            node_input_tensors = []
            for input_tensor in node.input:
                # Check original model first
                producer = original_tensor_producers.get(input_tensor)
                if producer:
                    node_input_producers.append(producer)
                    node_input_tensors.append(input_tensor)
            
            producer_match_count = len(set(target_input_producers) & set(node_input_producers))
            producer_match_ratio = producer_match_count / max(len(target_input_producers), 1)
            
            # Factor 2: Output consumers match (30% weight) - NEW
            node_output_consumers = []
            for output_tensor in node.output:
                consumers = model_tensor_consumers.get(output_tensor, [])
                node_output_consumers.extend(consumers)
            node_output_consumers = list(set(node_output_consumers))
            
            # Map to original model consumers for comparison
            node_output_consumers_original = []
            for output_tensor in node.output:
                consumers = original_tensor_consumers.get(output_tensor, [])
                node_output_consumers_original.extend(consumers)
            node_output_consumers_original = list(set(node_output_consumers_original))
            
            consumer_match_count = len(set(target_output_consumers) & set(node_output_consumers_original))
            consumer_match_ratio = consumer_match_count / max(len(target_output_consumers), 1) if target_output_consumers else 0.0
            
            # Factor 3: Input tensors match (20% weight)
            tensor_match_count = len(set(target_input_tensors) & set(node_input_tensors))
            tensor_match_ratio = tensor_match_count / max(len(target_input_tensors), 1)
            
            # Factor 4: Graph position match (10% weight) - NEW
            node_position = self._calculate_graph_position(node, model)
            position_diff_input = abs(target_position[0] - node_position[0])
            position_diff_output = abs(target_position[1] - node_position[1])
            # Normalize: 0 = perfect match, 1 = very different (max diff = 100)
            position_match_ratio = 1.0 - min(1.0, (position_diff_input + position_diff_output) / 100.0)
            
            # Calculate weighted multi-factor match score
            match_score = (
                0.4 * producer_match_ratio +      # Input producers (most important)
                0.3 * consumer_match_ratio +      # Output consumers (very strong signal)
                0.2 * tensor_match_ratio +        # Input tensors
                0.1 * position_match_ratio        # Graph position
            )
            
            # If multi-factor score is better, update best match
            if match_score > best_match_score:
                best_match = node.name
                best_match_score = match_score
        
        # Only return if we have a reasonable match (50% threshold on combined score)
        if best_match_score >= 0.5:
            return best_match
        
        return None
    
    def _match_nodes_by_location(
        self,
        gt_nodes: List[str],
        suggested_nodes: List[str],
        original_model: onnx.ModelProto,
        suggested_model: onnx.ModelProto,
        gt_model: onnx.ModelProto
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Match nodes by location (name, then context).
        
        Returns:
            Tuple of (exact_matches, context_matches, unmatched)
        """
        # Step 1: Exact name matching
        exact_matches = list(set(gt_nodes) & set(suggested_nodes))
        
        # Step 2: Context-based matching for remaining nodes
        gt_remaining = set(gt_nodes) - set(exact_matches)
        suggested_remaining = set(suggested_nodes) - set(exact_matches)
        
        context_matches = []
        matched_suggested = set()
        
        for gt_node_name in gt_remaining:
            # CRITICAL FIX: Get node from original model, not GT model
            # GT model doesn't have removed nodes - they were removed!
            # We need the original node to match by context
            gt_node = self._get_node_by_name(original_model, gt_node_name)
            if not gt_node:
                continue
            
            # Find matching node in suggested model by context
            matched_name = self._find_node_by_context(
                suggested_model, gt_node, original_model
            )
            if matched_name and matched_name in suggested_remaining:
                context_matches.append(gt_node_name)
                matched_suggested.add(matched_name)
        
        unmatched = list(gt_remaining - set(context_matches))
        
        return exact_matches, context_matches, unmatched
    
    def _calculate_transformation_accuracy(
        self,
        original_model: onnx.ModelProto,
        suggested_model: onnx.ModelProto,
        ground_truth_model: onnx.ModelProto
    ) -> Dict[str, Any]:
        """
        Calculate transformation accuracy by comparing:
        - Original → Suggested transformations
        - Original → Ground Truth transformations
        
        Returns transformation_score, critical_areas_match, and location-based metrics.
        """
        # Count operations in each model (for backward compatibility)
        original_ops = Counter(node.op_type for node in original_model.graph.node)
        suggested_ops = Counter(node.op_type for node in suggested_model.graph.node)
        gt_ops = Counter(node.op_type for node in ground_truth_model.graph.node)
        
        # Calculate removals: operations that decreased in count
        original_ops_set = set(original_ops.keys())
        suggested_ops_set = set(suggested_ops.keys())
        gt_ops_set = set(gt_ops.keys())
        
        # Removals: operations that exist in original but not in modified (or count decreased)
        suggested_removals = {}
        gt_removals = {}
        
        for op_type in original_ops_set:
            orig_count = original_ops[op_type]
            suggested_count = suggested_ops.get(op_type, 0)
            gt_count = gt_ops.get(op_type, 0)
            
            if suggested_count < orig_count:
                suggested_removals[op_type] = orig_count - suggested_count
            if gt_count < orig_count:
                gt_removals[op_type] = orig_count - gt_count
        
        # Additions: operations that exist in modified but not in original (or count increased)
        suggested_additions = {}
        gt_additions = {}
        
        for op_type in suggested_ops_set | gt_ops_set:
            orig_count = original_ops.get(op_type, 0)
            suggested_count = suggested_ops.get(op_type, 0)
            gt_count = gt_ops.get(op_type, 0)
            
            if suggested_count > orig_count:
                suggested_additions[op_type] = suggested_count - orig_count
            if gt_count > orig_count:
                gt_additions[op_type] = gt_count - orig_count
        
        # Calculate matches and mismatches
        removal_matches = []
        removal_mismatches_gt = []  # GT removed but we didn't
        removal_mismatches_suggested = []  # We removed but GT didn't
        
        for op_type in set(list(suggested_removals.keys()) + list(gt_removals.keys())):
            suggested_removed = suggested_removals.get(op_type, 0)
            gt_removed = gt_removals.get(op_type, 0)
            
            if suggested_removed > 0 and gt_removed > 0:
                # Both removed - match!
                removal_matches.append({
                    'op_type': op_type,
                    'suggested_count': suggested_removed,
                    'gt_count': gt_removed
                })
            elif gt_removed > 0 and suggested_removed == 0:
                # GT removed but we didn't - mismatch
                removal_mismatches_gt.append({
                    'op_type': op_type,
                    'gt_count': gt_removed
                })
            elif suggested_removed > 0 and gt_removed == 0:
                # We removed but GT didn't - mismatch
                removal_mismatches_suggested.append({
                    'op_type': op_type,
                    'suggested_count': suggested_removed
                })
        
        addition_matches = []
        addition_mismatches_gt = []  # GT added but we didn't
        addition_mismatches_suggested = []  # We added but GT didn't
        
        for op_type in set(list(suggested_additions.keys()) + list(gt_additions.keys())):
            suggested_added = suggested_additions.get(op_type, 0)
            gt_added = gt_additions.get(op_type, 0)
            
            if suggested_added > 0 and gt_added > 0:
                # Both added - match!
                addition_matches.append({
                    'op_type': op_type,
                    'suggested_count': suggested_added,
                    'gt_count': gt_added
                })
            elif gt_added > 0 and suggested_added == 0:
                # GT added but we didn't - mismatch
                addition_mismatches_gt.append({
                    'op_type': op_type,
                    'gt_count': gt_added
                })
            elif suggested_added > 0 and gt_added == 0:
                # We added but GT didn't - mismatch
                addition_mismatches_suggested.append({
                    'op_type': op_type,
                    'suggested_count': suggested_added
                })
        
        # Calculate count accuracy for matches
        removal_count_accuracy = 1.0
        if removal_matches:
            total_suggested_removed = sum(m['suggested_count'] for m in removal_matches)
            total_gt_removed = sum(m['gt_count'] for m in removal_matches)
            if total_gt_removed > 0:
                removal_count_accuracy = min(1.0, total_suggested_removed / total_gt_removed)
        
        addition_count_accuracy = 1.0
        if addition_matches:
            total_suggested_added = sum(m['suggested_count'] for m in addition_matches)
            total_gt_added = sum(m['gt_count'] for m in addition_matches)
            if total_gt_added > 0:
                addition_count_accuracy = min(1.0, total_suggested_added / total_gt_added)
        
        # Calculate transformation score
        # Weighted combination of:
        # - Removal match rate (how many GT removals we matched)
        # - Addition match rate (how many GT additions we matched)
        # - Count accuracy (how close are the counts for matched operations)
        # - Penalties for mismatches
        
        total_gt_removals = len(gt_removals)
        total_gt_additions = len(gt_additions)
        total_critical_changes = total_gt_removals + total_gt_additions
        
        removal_match_rate = len(removal_matches) / total_gt_removals if total_gt_removals > 0 else 1.0
        addition_match_rate = len(addition_matches) / total_gt_additions if total_gt_additions > 0 else 1.0
        
        # Penalty for mismatches (we did something GT didn't)
        mismatch_penalty = (
            len(removal_mismatches_suggested) + len(addition_mismatches_suggested)
        ) / max(total_critical_changes, 1)
        
        # Transformation score: PRIORITIZE critical areas match over count accuracy
        # For engineer verification, identifying the right operations is more important than exact counts
        # Reduced weight on count accuracy and mismatch penalties
        transformation_score = (
            0.5 * removal_match_rate +           # Increased from 0.4: prioritize matching removal types
            0.5 * addition_match_rate +          # Increased from 0.4: prioritize matching addition types
            0.05 * removal_count_accuracy +      # Reduced from 0.1: counts less critical for engineers
            0.05 * addition_count_accuracy -    # Reduced from 0.1: counts less critical for engineers
            0.05 * mismatch_penalty              # Reduced from 0.1: extra changes less penalized (engineers can verify)
        )
        transformation_score = max(0.0, min(1.0, transformation_score))  # Clamp to [0, 1]
        
        # Critical areas match: what % of GT's critical changes did we match?
        matched_critical_changes = len(removal_matches) + len(addition_matches)
        critical_areas_match = (
            matched_critical_changes / total_critical_changes
            if total_critical_changes > 0 else 1.0
        )
        
        # ===== LOCATION-BASED MATCHING =====
        # Identify removed nodes by name (not just type)
        gt_removed_nodes_by_type = self._identify_removed_nodes_by_name(
            original_model, ground_truth_model
        )
        suggested_removed_nodes_by_type = self._identify_removed_nodes_by_name(
            original_model, suggested_model
        )
        
        # Identify added nodes by name (not just type)
        gt_added_nodes_by_type = self._identify_added_nodes_by_name(
            original_model, ground_truth_model
        )
        suggested_added_nodes_by_type = self._identify_added_nodes_by_name(
            original_model, suggested_model
        )
        
        # Match removals by location
        removal_location_matches = {
            'exact_matches': [],
            'context_matches': [],
            'unmatched': []
        }
        total_gt_removed_node_count = 0
        
        for op_type in set(list(gt_removed_nodes_by_type.keys()) + list(suggested_removed_nodes_by_type.keys())):
            gt_nodes = gt_removed_nodes_by_type.get(op_type, [])
            suggested_nodes = suggested_removed_nodes_by_type.get(op_type, [])
            
            if not gt_nodes:
                continue
            
            total_gt_removed_node_count += len(gt_nodes)
            
            # Match nodes by location
            exact_matches, context_matches, unmatched = self._match_nodes_by_location(
                gt_nodes, suggested_nodes,
                original_model, suggested_model, ground_truth_model
            )
            
            removal_location_matches['exact_matches'].extend(exact_matches)
            removal_location_matches['context_matches'].extend(context_matches)
            removal_location_matches['unmatched'].extend(unmatched)
        
        # Match additions by location
        # For additions, nodes are new so we match by:
        # 1. Exact name (if names happen to match)
        # 2. Context (where they were added - input/output tensors)
        addition_location_matches = {
            'exact_matches': [],
            'context_matches': [],
            'unmatched': []
        }
        total_gt_added_node_count = 0
        
        for op_type in set(list(gt_added_nodes_by_type.keys()) + list(suggested_added_nodes_by_type.keys())):
            gt_nodes = gt_added_nodes_by_type.get(op_type, [])
            suggested_nodes = suggested_added_nodes_by_type.get(op_type, [])
            
            if not gt_nodes:
                continue
            
            total_gt_added_node_count += len(gt_nodes)
            
            # For additions, try exact name matching first
            exact_matches = list(set(gt_nodes) & set(suggested_nodes))
            gt_remaining = set(gt_nodes) - set(exact_matches)
            suggested_remaining = set(suggested_nodes) - set(exact_matches)
            
            # Then try context matching (match by where nodes were inserted)
            context_matches = []
            matched_suggested = set()
            
            for gt_node_name in gt_remaining:
                gt_node = self._get_node_by_name(ground_truth_model, gt_node_name)
                if not gt_node:
                    continue
                
                # For additions, match by multi-factor context (where the node was inserted)
                # Find nodes in suggested model with same op_type and similar context
                best_match = None
                best_match_score = 0.0
                
                # Get GT node's context for matching
                gt_inputs = set(gt_node.input)
                gt_output_consumers = self._get_output_consumers(gt_node, ground_truth_model)
                gt_position = self._calculate_graph_position(gt_node, ground_truth_model)
                
                for suggested_node_name in suggested_remaining:
                    if suggested_node_name in matched_suggested:
                        continue
                    
                    suggested_node = self._get_node_by_name(suggested_model, suggested_node_name)
                    if not suggested_node or suggested_node.op_type != gt_node.op_type:
                        continue
                    
                    # Factor 1: Input tensors match (40% weight)
                    suggested_inputs = set(suggested_node.input)
                    input_overlap = len(gt_inputs & suggested_inputs)
                    input_match_ratio = input_overlap / max(len(gt_inputs), 1)
                    
                    # Factor 2: Output consumers match (30% weight) - NEW
                    suggested_output_consumers = self._get_output_consumers(suggested_node, suggested_model)
                    consumer_overlap = len(set(gt_output_consumers) & set(suggested_output_consumers))
                    consumer_match_ratio = consumer_overlap / max(len(gt_output_consumers), 1) if gt_output_consumers else 0.0
                    
                    # Factor 3: Graph position match (30% weight) - NEW
                    suggested_position = self._calculate_graph_position(suggested_node, suggested_model)
                    position_diff_input = abs(gt_position[0] - suggested_position[0])
                    position_diff_output = abs(gt_position[1] - suggested_position[1])
                    # Normalize: 0 = perfect match, 1 = very different (max diff = 100)
                    position_match_ratio = 1.0 - min(1.0, (position_diff_input + position_diff_output) / 100.0)
                    
                    # Calculate weighted multi-factor match score
                    match_score = (
                        0.4 * input_match_ratio +      # Input tensors (most important for additions)
                        0.3 * consumer_match_ratio +   # Output consumers (very strong signal)
                        0.3 * position_match_ratio     # Graph position
                    )
                    
                    if match_score > best_match_score:
                        best_match = suggested_node_name
                        best_match_score = match_score
                
                # Require at least 50% combined score for a match
                if best_match and best_match_score >= 0.5:
                    context_matches.append(gt_node_name)
                    matched_suggested.add(best_match)
            
            unmatched = list(gt_remaining - set(context_matches))
            
            addition_location_matches['exact_matches'].extend(exact_matches)
            addition_location_matches['context_matches'].extend(context_matches)
            addition_location_matches['unmatched'].extend(unmatched)
        
        # Calculate location-based metrics
        total_gt_changed_nodes = total_gt_removed_node_count + total_gt_added_node_count
        total_location_matches = (
            len(removal_location_matches['exact_matches']) +
            len(removal_location_matches['context_matches']) +
            len(addition_location_matches['exact_matches']) +
            len(addition_location_matches['context_matches'])
        )
        
        location_based_critical_areas_match = (
            total_location_matches / total_gt_changed_nodes
            if total_gt_changed_nodes > 0 else 1.0
        )
        
        # Combined metric: weighted combination of type and location matching
        combined_critical_areas_match = (
            0.6 * critical_areas_match +  # 60% weight on type matching
            0.4 * location_based_critical_areas_match  # 40% weight on location matching
        )
        
        return {
            'transformation_score': transformation_score,
            'critical_areas_match': critical_areas_match,  # Type-based (existing)
            'location_based_critical_areas_match': location_based_critical_areas_match,  # Location-based (new)
            'combined_critical_areas_match': combined_critical_areas_match,  # Combined (new)
            'removal_matches': [m['op_type'] for m in removal_matches],
            'removal_mismatches_gt': [m['op_type'] for m in removal_mismatches_gt],
            'removal_mismatches_suggested': [m['op_type'] for m in removal_mismatches_suggested],
            'addition_matches': [m['op_type'] for m in addition_matches],
            'addition_mismatches_gt': [m['op_type'] for m in addition_mismatches_gt],
            'addition_mismatches_suggested': [m['op_type'] for m in addition_mismatches_suggested],
            'removal_count_accuracy': removal_count_accuracy,
            'addition_count_accuracy': addition_count_accuracy,
            'expected_removals': list(gt_removals.keys()),
            'actual_removals': list(suggested_removals.keys()),
            'expected_additions': list(gt_additions.keys()),
            'actual_additions': list(suggested_additions.keys()),
            'total_gt_removals': total_gt_removals,
            'total_gt_additions': total_gt_additions,
            'total_critical_changes': total_critical_changes,
            # Location-based matching details
            'removal_location_matches': removal_location_matches,
            'addition_location_matches': addition_location_matches,
            'total_gt_removed_node_count': total_gt_removed_node_count,
            'total_gt_added_node_count': total_gt_added_node_count,
            'total_gt_changed_node_count': total_gt_changed_nodes
        }

