#!/usr/bin/env python3
"""
Deep ONNX Model Analyzer for Production Pipeline.

Extracts comprehensive model structure including:
- Full node information with attributes
- Shape inference results
- Tensor types and dimensions
- Graph topology and dependencies
- Potential compilation blockers
- Weight statistics (without actual values)
"""

import onnx
from onnx import shape_inference, numpy_helper
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NodeAnalysis:
    """Complete analysis of a single ONNX node."""
    node_id: int
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    input_shapes: List[Optional[Tuple]]
    output_shapes: List[Optional[Tuple]]
    attributes: Dict[str, Any]
    input_types: List[str]
    output_types: List[str]
    dependencies: List[int]  # Node IDs this depends on
    dependents: List[int]    # Node IDs that depend on this
    is_compilation_blocker: bool
    blocker_reason: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'name': self.name,
            'op_type': self.op_type,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'input_shapes': [list(s) if s else None for s in self.input_shapes],
            'output_shapes': [list(s) if s else None for s in self.output_shapes],
            'attributes': self.attributes,
            'input_types': self.input_types,
            'output_types': self.output_types,
            'dependencies': self.dependencies,
            'dependents': self.dependents,
            'is_compilation_blocker': self.is_compilation_blocker,
            'blocker_reason': self.blocker_reason
        }


@dataclass
class ModelAnalysis:
    """Complete analysis of an ONNX model."""
    model_name: str
    opset_version: int
    ir_version: int
    producer_name: str
    nodes: List[NodeAnalysis]
    graph_inputs: List[Dict]
    graph_outputs: List[Dict]
    initializers: List[Dict]  # Metadata only, not values
    total_parameters: int
    compilation_blockers: List[Dict]
    dynamic_dimensions: List[Dict]
    topology: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'opset_version': self.opset_version,
            'ir_version': self.ir_version,
            'producer_name': self.producer_name,
            'node_count': len(self.nodes),
            'nodes': [n.to_dict() for n in self.nodes],
            'graph_inputs': self.graph_inputs,
            'graph_outputs': self.graph_outputs,
            'initializers': self.initializers,
            'total_parameters': self.total_parameters,
            'compilation_blockers': self.compilation_blockers,
            'dynamic_dimensions': self.dynamic_dimensions,
            'topology': self.topology
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Model: {self.model_name}",
            f"Opset: {self.opset_version}, IR: {self.ir_version}",
            f"Nodes: {len(self.nodes)}",
            f"Parameters: {self.total_parameters:,}",
            f"Compilation Blockers: {len(self.compilation_blockers)}",
            f"Dynamic Dimensions: {len(self.dynamic_dimensions)}",
        ]
        
        if self.compilation_blockers:
            lines.append("Blockers: " + ", ".join(
                b['op_type'] for b in self.compilation_blockers[:5]
            ))
        
        return "\n".join(lines)


class ONNXAnalyzer:
    """Deep ONNX model analyzer."""
    
    # Operations known to block hardware compilation
    # Based on SiMa MLA documentation and ONNX Graph Surgery best practices
    COMPILATION_BLOCKERS = {
        # Core unsupported operations
        'Einsum': 'Complex tensor contraction not supported - replace with MatMul+Reshape',
        'NonZero': 'Dynamic output shape based on data values',
        'Where': 'Conditional execution not supported on MLA',
        'Loop': 'Dynamic control flow not supported',
        'If': 'Conditional branching not supported',
        'Scan': 'Sequential processing not supported',
        
        # Sequence operations
        'SequenceAt': 'Sequence operations not supported',
        'SequenceConstruct': 'Sequence operations not supported',
        'SequenceInsert': 'Sequence operations not supported',
        'SequenceErase': 'Sequence operations not supported',
        'SequenceLength': 'Sequence operations not supported',
        'SplitToSequence': 'Sequence operations not supported',
        'ConcatFromSequence': 'Sequence operations not supported',
        
        # Dynamic output size operations
        'NonMaxSuppression': 'Dynamic output size based on data',
        'TopK': 'Dynamic k value not supported',
        'Unique': 'Dynamic output size based on data',
        'Compress': 'Dynamic output size based on condition',
        
        # Complex indexing operations
        'GatherND': 'Complex N-dimensional indexing',
        'ScatterND': 'Complex N-dimensional scatter',
        'GatherElements': 'Element-wise gather may have issues',
        'ScatterElements': 'Element-wise scatter may have issues',
        
        # Region operations
        'RoiAlign': 'Region of interest operations limited support',
        'RoiPool': 'Region pooling not supported',
        
        # Quantization issues
        'DynamicQuantizeLinear': 'Dynamic quantization not supported',
        'QuantizeLinear': 'May need static scale/zero_point',
        'DequantizeLinear': 'May need static scale/zero_point',
        
        # Optional/new operations
        'Optional': 'Optional type not supported',
        'OptionalGetElement': 'Optional type not supported',
        'OptionalHasElement': 'Optional type not supported',
        
        # String operations
        'StringNormalizer': 'String operations not supported',
        'TfIdfVectorizer': 'Text processing not supported',
        
        # Random operations (non-deterministic)
        'RandomNormal': 'Random operations affect reproducibility',
        'RandomNormalLike': 'Random operations affect reproducibility',
        'RandomUniform': 'Random operations affect reproducibility',
        'RandomUniformLike': 'Random operations affect reproducibility',
        'Multinomial': 'Random sampling not supported',
        
        # Deprecated/problematic
        'Upsample': 'Deprecated - use Resize instead',
        'Scatter': 'Deprecated - use ScatterElements',
        'Gather': 'May need axis validation',
    }
    
    # Operations that are blockers only under certain conditions
    CONDITIONAL_BLOCKERS = {
        'Reshape': 'Blocker if shape contains -1 or dynamic values',
        'Expand': 'Blocker if shape is dynamic',
        'Tile': 'Blocker if repeats are dynamic',
        'Slice': 'Blocker if starts/ends/axes are dynamic',
        'Pad': 'Blocker if pads are dynamic',
        'Resize': 'Blocker if scales/sizes are dynamic',
        'ConstantOfShape': 'Blocker if shape is dynamic',
    }
    
    # Operations that may need attention but aren't blockers
    WARNING_OPERATIONS = {
        'Dropout': 'Should be removed for inference (training artifact)',
        'Identity': 'Can be removed to simplify graph',
        'Cast': 'May indicate type mismatches',
        'Shape': 'May introduce dynamic dimensions',
        'Size': 'May introduce dynamic dimensions',
        'Range': 'May create dynamic sequences',
    }
    
    # Tensor types that may cause issues
    PROBLEMATIC_TYPES = {
        'complex64': 'Complex numbers not supported on MLA',
        'complex128': 'Complex numbers not supported on MLA',
        'string': 'String tensors not supported',
        'bfloat16': 'BFloat16 may have limited support',
    }
    
    # Hardware limits (SiMa MLA typical limits)
    HARDWARE_LIMITS = {
        'max_tensor_dims': 4,  # MLA requires 4D tensors
        'max_channels': 2048,  # Maximum channel dimension
        'max_spatial': 8192,   # Maximum spatial dimension
        'max_batch': 32,       # Maximum batch size
        'min_supported_opset': 9,
        'max_supported_opset': 17,
    }
    
    def __init__(self):
        self.dtype_map = {
            1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16',
            5: 'int16', 6: 'int32', 7: 'int64', 8: 'string',
            9: 'bool', 10: 'float16', 11: 'float64', 12: 'uint32',
            13: 'uint64', 14: 'complex64', 15: 'complex128', 16: 'bfloat16'
        }
    
    def analyze(self, model_path: str) -> ModelAnalysis:
        """
        Perform deep analysis of ONNX model.
        
        Args:
            model_path: Path to ONNX model file
            
        Returns:
            Complete ModelAnalysis
        """
        # Load model
        model = onnx.load(model_path)
        
        # Try shape inference
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"Warning: Shape inference failed: {e}")
        
        graph = model.graph
        
        # Build shape and type tables
        shape_table, type_table = self._build_tensor_tables(graph)
        
        # Get initializer names (weights)
        initializer_names = {init.name for init in graph.initializer}
        
        # Build producer map (tensor -> node that produces it)
        producer_map = self._build_producer_map(graph, initializer_names)
        
        # Analyze all nodes
        nodes = []
        dependency_graph = defaultdict(list)  # node_id -> [dependent_node_ids]
        
        for i, node in enumerate(graph.node):
            node_analysis = self._analyze_node(
                i, node, shape_table, type_table, 
                initializer_names, producer_map
            )
            nodes.append(node_analysis)
        
        # Build dependency relationships
        self._build_dependencies(nodes, producer_map, graph)
        
        # Extract compilation blockers
        blockers = [
            {'node_id': n.node_id, 'op_type': n.op_type, 'reason': n.blocker_reason}
            for n in nodes if n.is_compilation_blocker
        ]
        
        # Find dynamic dimensions
        dynamic_dims = self._find_dynamic_dimensions(graph, shape_table)
        
        # Extract graph inputs/outputs
        graph_inputs = self._extract_io_info(graph.input, initializer_names, shape_table, type_table)
        graph_outputs = self._extract_io_info(graph.output, set(), shape_table, type_table)
        
        # Extract initializer metadata
        initializers = self._extract_initializer_metadata(graph.initializer)
        total_params = sum(init['size'] for init in initializers)
        
        # Build topology info
        topology = self._analyze_topology(nodes)
        
        # Get model name
        model_name = model_path.split('/')[-1].replace('.onnx', '')
        
        return ModelAnalysis(
            model_name=model_name,
            opset_version=model.opset_import[0].version if model.opset_import else 0,
            ir_version=model.ir_version,
            producer_name=model.producer_name or 'unknown',
            nodes=nodes,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            initializers=initializers,
            total_parameters=total_params,
            compilation_blockers=blockers,
            dynamic_dimensions=dynamic_dims,
            topology=topology
        )
    
    def _build_tensor_tables(
        self, graph: onnx.GraphProto
    ) -> Tuple[Dict[str, Tuple], Dict[str, str]]:
        """Build shape and type lookup tables."""
        shape_table = {}
        type_table = {}
        
        def process_value_info(vi):
            if not vi.name:
                return
            
            tt = vi.type.tensor_type
            if tt.HasField('shape'):
                dims = []
                for d in tt.shape.dim:
                    if d.HasField('dim_value'):
                        dims.append(int(d.dim_value))
                    elif d.HasField('dim_param'):
                        dims.append(str(d.dim_param))
                    else:
                        dims.append(None)
                shape_table[vi.name] = tuple(dims)
            
            if tt.elem_type:
                type_table[vi.name] = self.dtype_map.get(tt.elem_type, 'unknown')
        
        for vi in graph.input:
            process_value_info(vi)
        for vi in graph.output:
            process_value_info(vi)
        for vi in graph.value_info:
            process_value_info(vi)
        
        return shape_table, type_table
    
    def _build_producer_map(
        self, graph: onnx.GraphProto, initializer_names: Set[str]
    ) -> Dict[str, Tuple[int, str]]:
        """Build map of tensor name -> (producing node id, 'node'/'input'/'init')."""
        producer_map = {}
        
        # Graph inputs
        for vi in graph.input:
            if vi.name and vi.name not in initializer_names:
                producer_map[vi.name] = (-1, 'input')
        
        # Initializers
        for init in graph.initializer:
            producer_map[init.name] = (-1, 'init')
        
        # Node outputs
        for i, node in enumerate(graph.node):
            for out_name in node.output:
                if out_name:
                    producer_map[out_name] = (i, 'node')
        
        return producer_map
    
    def _analyze_node(
        self, 
        node_id: int,
        node: onnx.NodeProto,
        shape_table: Dict,
        type_table: Dict,
        initializer_names: Set[str],
        producer_map: Dict
    ) -> NodeAnalysis:
        """Analyze a single node with comprehensive blocker detection."""
        # Get input/output shapes and types
        input_shapes = [shape_table.get(inp) for inp in node.input if inp]
        output_shapes = [shape_table.get(out) for out in node.output if out]
        input_types = [type_table.get(inp, 'unknown') for inp in node.input if inp]
        output_types = [type_table.get(out, 'unknown') for out in node.output if out]
        
        # Extract attributes
        attributes = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attributes[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attributes[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                attributes[attr.name] = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
            elif attr.type == onnx.AttributeProto.INTS:
                attributes[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attributes[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.TENSOR:
                attributes[attr.name] = f"tensor({attr.t.dims})"
        
        # Check for compilation blockers
        is_blocker = False
        blocker_reason = None
        blocker_reasons = []
        
        # 1. Check direct blocker operations
        if node.op_type in self.COMPILATION_BLOCKERS:
            is_blocker = True
            blocker_reasons.append(self.COMPILATION_BLOCKERS[node.op_type])
        
        # 2. Check for dynamic shape issues
        for shape in output_shapes:
            if shape and any(isinstance(d, str) or d is None for d in shape):
                is_blocker = True
                blocker_reasons.append("Dynamic/unknown output shape")
                break
        
        # 3. Check for non-4D tensors (MLA requires 4D)
        for i, shape in enumerate(output_shapes):
            if shape:
                num_dims = len([d for d in shape if d is not None])
                if num_dims > 0 and num_dims != 4:
                    is_blocker = True
                    blocker_reasons.append(f"Non-4D tensor output ({num_dims}D) - MLA requires 4D")
                    break
        
        # 4. Check for problematic data types
        for dtype in output_types + input_types:
            if dtype in self.PROBLEMATIC_TYPES:
                is_blocker = True
                blocker_reasons.append(self.PROBLEMATIC_TYPES[dtype])
        
        # 5. Check for large tensor dimensions exceeding hardware limits
        for shape in output_shapes:
            if shape:
                for i, dim in enumerate(shape):
                    if isinstance(dim, int):
                        if i == 0 and dim > self.HARDWARE_LIMITS['max_batch']:
                            blocker_reasons.append(f"Batch size {dim} exceeds limit {self.HARDWARE_LIMITS['max_batch']}")
                        elif i == 1 and dim > self.HARDWARE_LIMITS['max_channels']:
                            blocker_reasons.append(f"Channel count {dim} exceeds limit {self.HARDWARE_LIMITS['max_channels']}")
                        elif i >= 2 and dim > self.HARDWARE_LIMITS['max_spatial']:
                            blocker_reasons.append(f"Spatial dimension {dim} exceeds limit {self.HARDWARE_LIMITS['max_spatial']}")
        
        # 6. Check conditional blockers
        if node.op_type in self.CONDITIONAL_BLOCKERS:
            # Check if the node has dynamic inputs that make it a blocker
            if node.op_type == 'Reshape':
                # Reshape is a blocker if shape input is not an initializer
                if len(node.input) > 1 and node.input[1] not in initializer_names:
                    is_blocker = True
                    blocker_reasons.append("Reshape with dynamic shape input")
            
            elif node.op_type == 'Slice':
                # Slice is a blocker if starts/ends/steps are dynamic
                slice_inputs = node.input[1:]  # Everything after the data input
                dynamic_slice_inputs = [inp for inp in slice_inputs if inp and inp not in initializer_names]
                if dynamic_slice_inputs:
                    is_blocker = True
                    blocker_reasons.append("Slice with dynamic starts/ends/steps")
            
            elif node.op_type == 'Resize':
                # Resize is a blocker if scales/sizes are dynamic
                if len(node.input) > 2:
                    for inp in node.input[2:]:
                        if inp and inp not in initializer_names:
                            is_blocker = True
                            blocker_reasons.append("Resize with dynamic scales/sizes")
                            break
            
            elif node.op_type in ['Expand', 'Tile', 'Pad', 'ConstantOfShape']:
                # These are blockers if shape/pads/repeats input is not constant
                if len(node.input) > 1 and node.input[1] not in initializer_names:
                    is_blocker = True
                    blocker_reasons.append(f"{node.op_type} with dynamic parameter")
        
        # 7. Check for custom/unknown operators
        known_ops = set(self.COMPILATION_BLOCKERS.keys()) | set(self.CONDITIONAL_BLOCKERS.keys()) | set(self.WARNING_OPERATIONS.keys())
        known_ops.update([
            # Standard ONNX ops that are typically supported
            'Conv', 'ConvTranspose', 'MatMul', 'Gemm', 'Add', 'Sub', 'Mul', 'Div',
            'Relu', 'LeakyRelu', 'PRelu', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
            'MaxPool', 'AveragePool', 'GlobalMaxPool', 'GlobalAveragePool',
            'BatchNormalization', 'InstanceNormalization', 'LayerNormalization',
            'Concat', 'Split', 'Slice', 'Gather', 'Reshape', 'Transpose', 'Squeeze', 'Unsqueeze',
            'Flatten', 'Reduce*', 'Clip', 'Abs', 'Neg', 'Sqrt', 'Pow', 'Exp', 'Log',
            'Sin', 'Cos', 'Pad', 'Resize', 'Upsample', 'DepthToSpace', 'SpaceToDepth',
            'Constant', 'ConstantOfShape', 'Shape', 'Size', 'Cast', 'Floor', 'Ceil', 'Round',
            'Min', 'Max', 'Mean', 'Sum', 'Erf', 'Gelu', 'Elu', 'Selu', 'Celu', 'HardSigmoid',
            'HardSwish', 'Mish', 'Swish', 'Softplus', 'Softsign', 'ThresholdedRelu',
            'Expand', 'Tile', 'ReduceMax', 'ReduceMin', 'ReduceMean', 'ReduceSum', 'ReduceProd',
            'ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp', 'ReduceSumSquare',
            'ArgMax', 'ArgMin', 'Equal', 'Greater', 'Less', 'GreaterOrEqual', 'LessOrEqual',
            'And', 'Or', 'Not', 'Xor', 'Mod', 'Sign', 'BitShift', 'LpNormalization',
        ])
        
        if node.op_type not in known_ops and not any(node.op_type.startswith(p) for p in ['Reduce']):
            is_blocker = True
            blocker_reasons.append(f"Unknown/custom operator: {node.op_type}")
        
        # Combine blocker reasons
        if blocker_reasons:
            blocker_reason = "; ".join(blocker_reasons[:3])  # Limit to first 3 reasons
        
        # Get data inputs (exclude initializers)
        data_inputs = [inp for inp in node.input if inp and inp not in initializer_names]
        
        return NodeAnalysis(
            node_id=node_id,
            name=node.name or f"node_{node_id}",
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            attributes=attributes,
            input_types=input_types,
            output_types=output_types,
            dependencies=[],  # Filled later
            dependents=[],    # Filled later
            is_compilation_blocker=is_blocker,
            blocker_reason=blocker_reason
        )
    
    def _build_dependencies(
        self, 
        nodes: List[NodeAnalysis],
        producer_map: Dict,
        graph: onnx.GraphProto
    ):
        """Build node dependency relationships."""
        initializer_names = {init.name for init in graph.initializer}
        
        # Map output tensor -> node id
        output_to_node = {}
        for node in nodes:
            for out in node.outputs:
                output_to_node[out] = node.node_id
        
        # Build dependencies
        for node in nodes:
            for inp in node.inputs:
                if inp and inp not in initializer_names:
                    if inp in output_to_node:
                        dep_id = output_to_node[inp]
                        if dep_id not in node.dependencies:
                            node.dependencies.append(dep_id)
                        if node.node_id not in nodes[dep_id].dependents:
                            nodes[dep_id].dependents.append(node.node_id)
    
    def _find_dynamic_dimensions(
        self, graph: onnx.GraphProto, shape_table: Dict
    ) -> List[Dict]:
        """Find all tensors with dynamic dimensions."""
        dynamic = []
        
        for name, shape in shape_table.items():
            if shape:
                for i, dim in enumerate(shape):
                    if isinstance(dim, str) or dim is None:
                        dynamic.append({
                            'tensor': name,
                            'dimension': i,
                            'value': str(dim) if dim else 'unknown'
                        })
        
        return dynamic
    
    def _extract_io_info(
        self,
        value_infos,
        exclude_names: Set[str],
        shape_table: Dict,
        type_table: Dict
    ) -> List[Dict]:
        """Extract input/output information."""
        result = []
        for vi in value_infos:
            if vi.name and vi.name not in exclude_names:
                result.append({
                    'name': vi.name,
                    'shape': list(shape_table.get(vi.name, [])) if shape_table.get(vi.name) else None,
                    'dtype': type_table.get(vi.name, 'unknown')
                })
        return result
    
    def _extract_initializer_metadata(
        self, initializers
    ) -> List[Dict]:
        """Extract initializer metadata (not values)."""
        result = []
        for init in initializers:
            size = 1
            for dim in init.dims:
                size *= dim
            
            result.append({
                'name': init.name,
                'shape': list(init.dims),
                'dtype': self.dtype_map.get(init.data_type, 'unknown'),
                'size': size
            })
        return result
    
    def _analyze_topology(self, nodes: List[NodeAnalysis]) -> Dict[str, Any]:
        """Analyze graph topology."""
        op_counts = defaultdict(int)
        max_depth = 0
        
        for node in nodes:
            op_counts[node.op_type] += 1
        
        # Find max depth (longest path)
        depths = {n.node_id: 0 for n in nodes}
        for node in nodes:
            for dep_id in node.dependencies:
                depths[node.node_id] = max(depths[node.node_id], depths[dep_id] + 1)
        
        max_depth = max(depths.values()) if depths else 0
        
        return {
            'operation_counts': dict(op_counts),
            'max_depth': max_depth,
            'unique_operations': len(op_counts),
            'has_branches': any(len(n.dependents) > 1 for n in nodes),
            'has_residuals': any(len(n.dependencies) > 1 for n in nodes)
        }
    
    def get_nodes_by_op(self, analysis: ModelAnalysis, op_types: List[str]) -> List[NodeAnalysis]:
        """Get all nodes matching given operation types."""
        return [n for n in analysis.nodes if n.op_type in op_types]
    
    def get_subgraph(
        self, 
        analysis: ModelAnalysis, 
        center_node_ids: List[int],
        depth: int = 2
    ) -> List[NodeAnalysis]:
        """Extract subgraph around specified nodes."""
        included = set(center_node_ids)
        
        # Expand outward
        for _ in range(depth):
            new_included = set()
            for node_id in included:
                node = analysis.nodes[node_id]
                new_included.update(node.dependencies)
                new_included.update(node.dependents)
            included.update(new_included)
        
        # Filter to valid node IDs
        included = {nid for nid in included if 0 <= nid < len(analysis.nodes)}
        
        return [analysis.nodes[nid] for nid in sorted(included)]
    
    def get_blocker_summary(self, analysis: ModelAnalysis) -> Dict[str, Any]:
        """
        Get comprehensive summary of all compilation blockers.
        
        Returns dict with:
        - total_blockers: Total blocker count
        - by_type: Blockers grouped by type
        - by_reason: Blockers grouped by reason
        - recommendations: Suggested fixes
        """
        blockers = [n for n in analysis.nodes if n.is_compilation_blocker]
        
        by_type = defaultdict(list)
        by_reason = defaultdict(list)
        
        for node in blockers:
            by_type[node.op_type].append(node.node_id)
            if node.blocker_reason:
                # Extract primary reason
                primary_reason = node.blocker_reason.split(';')[0].strip()
                by_reason[primary_reason].append(node.node_id)
        
        # Generate recommendations
        recommendations = []
        
        if 'Einsum' in by_type:
            recommendations.append(
                f"Replace {len(by_type['Einsum'])} Einsum nodes with MatMul+Reshape equivalents"
            )
        
        if any('Non-4D' in r for r in by_reason):
            recommendations.append(
                "Add Reshape nodes to convert tensors to 4D format for MLA"
            )
        
        if any('dynamic' in r.lower() for r in by_reason):
            recommendations.append(
                "Replace dynamic shapes with static values where possible"
            )
        
        if any('Identity' in by_type or 'Dropout' in by_type):
            recommendations.append(
                "Remove Identity and Dropout nodes (inference artifacts)"
            )
        
        # Check for conditional blockers that might be fixable
        fixable_ops = {'Reshape', 'Slice', 'Resize', 'Pad'}
        fixable_count = sum(len(by_type.get(op, [])) for op in fixable_ops)
        if fixable_count > 0:
            recommendations.append(
                f"{fixable_count} nodes may be fixable by making inputs constant"
            )
        
        return {
            'total_blockers': len(blockers),
            'by_type': dict(by_type),
            'by_reason': dict(by_reason),
            'recommendations': recommendations,
            'severity': 'high' if len(blockers) > 10 else ('medium' if len(blockers) > 3 else 'low')
        }
    
    def check_opset_compatibility(self, model_path: str) -> Dict[str, Any]:
        """
        Check if model opset version is compatible with target hardware.
        
        Returns compatibility info and any version-specific issues.
        """
        model = onnx.load(model_path)
        
        opset_version = model.opset_import[0].version if model.opset_import else 0
        
        compatible = (
            self.HARDWARE_LIMITS['min_supported_opset'] <= opset_version <= 
            self.HARDWARE_LIMITS['max_supported_opset']
        )
        
        issues = []
        if opset_version < self.HARDWARE_LIMITS['min_supported_opset']:
            issues.append(f"Opset {opset_version} is below minimum supported ({self.HARDWARE_LIMITS['min_supported_opset']})")
        if opset_version > self.HARDWARE_LIMITS['max_supported_opset']:
            issues.append(f"Opset {opset_version} is above maximum supported ({self.HARDWARE_LIMITS['max_supported_opset']})")
        
        # Check for domain-specific opsets
        for opset in model.opset_import:
            if opset.domain and opset.domain != '':
                issues.append(f"Non-standard domain: {opset.domain} (version {opset.version})")
        
        return {
            'opset_version': opset_version,
            'compatible': compatible,
            'min_supported': self.HARDWARE_LIMITS['min_supported_opset'],
            'max_supported': self.HARDWARE_LIMITS['max_supported_opset'],
            'issues': issues
        }
    
    def get_warnings(self, analysis: ModelAnalysis) -> List[Dict]:
        """
        Get non-blocking warnings for the model.
        
        These are issues that won't prevent compilation but may affect
        performance or indicate suboptimal model structure.
        """
        warnings = []
        
        for node in analysis.nodes:
            if node.op_type in self.WARNING_OPERATIONS:
                warnings.append({
                    'node_id': node.node_id,
                    'op_type': node.op_type,
                    'warning': self.WARNING_OPERATIONS[node.op_type],
                    'name': node.name
                })
        
        # Check for suboptimal patterns
        # Pattern: consecutive Reshape nodes
        for i, node in enumerate(analysis.nodes[:-1]):
            if node.op_type == 'Reshape':
                next_node = analysis.nodes[i + 1]
                if next_node.op_type == 'Reshape' and i + 1 in node.dependents:
                    warnings.append({
                        'node_id': node.node_id,
                        'op_type': 'Pattern',
                        'warning': f"Consecutive Reshape nodes ({node.node_id}, {next_node.node_id}) can be fused",
                        'name': f"{node.name} -> {next_node.name}"
                    })
        
        # Pattern: Identity nodes
        identity_count = sum(1 for n in analysis.nodes if n.op_type == 'Identity')
        if identity_count > 0:
            warnings.append({
                'node_id': -1,
                'op_type': 'Optimization',
                'warning': f"{identity_count} Identity nodes can be removed",
                'name': 'Graph simplification'
            })
        
        # Pattern: Dropout nodes (should be removed for inference)
        dropout_count = sum(1 for n in analysis.nodes if n.op_type == 'Dropout')
        if dropout_count > 0:
            warnings.append({
                'node_id': -1,
                'op_type': 'Inference',
                'warning': f"{dropout_count} Dropout nodes should be removed for inference",
                'name': 'Training artifact'
            })
        
        return warnings


    # =========================================================================
    # Enhanced Methods for Surgery Database
    # =========================================================================
    
    def get_dependency_chain(
        self, 
        analysis: ModelAnalysis, 
        node_id: int, 
        direction: str = 'both',
        max_depth: int = 10
    ) -> Dict[str, List[int]]:
        """
        Get full dependency chain for a node (multi-hop).
        
        Args:
            analysis: Model analysis
            node_id: Starting node ID
            direction: 'predecessors', 'successors', or 'both'
            max_depth: Maximum chain depth to traverse
            
        Returns:
            Dict with 'predecessors' and/or 'successors' lists of node IDs
        """
        result = {}
        
        if direction in ('predecessors', 'both'):
            predecessors = []
            current_level = {node_id}
            visited = {node_id}
            
            for depth in range(max_depth):
                next_level = set()
                for nid in current_level:
                    if 0 <= nid < len(analysis.nodes):
                        node = analysis.nodes[nid]
                        for dep_id in node.dependencies:
                            if dep_id not in visited:
                                predecessors.append(dep_id)
                                next_level.add(dep_id)
                                visited.add(dep_id)
                
                if not next_level:
                    break
                current_level = next_level
            
            result['predecessors'] = predecessors
        
        if direction in ('successors', 'both'):
            successors = []
            current_level = {node_id}
            visited = {node_id}
            
            for depth in range(max_depth):
                next_level = set()
                for nid in current_level:
                    if 0 <= nid < len(analysis.nodes):
                        node = analysis.nodes[nid]
                        for dep_id in node.dependents:
                            if dep_id not in visited:
                                successors.append(dep_id)
                                next_level.add(dep_id)
                                visited.add(dep_id)
                
                if not next_level:
                    break
                current_level = next_level
            
            result['successors'] = successors
        
        return result
    
    def detect_subgraph_patterns(
        self, 
        analysis: ModelAnalysis
    ) -> List[Dict[str, Any]]:
        """
        Detect common subgraph patterns in the model.
        
        Patterns detected:
        - Conv-BN-Relu (convolution block)
        - Conv-BN (convolution with normalization)
        - Attention block (MatMul-Softmax-MatMul)
        - SE block (GlobalPool-FC-Relu-FC-Sigmoid-Mul)
        - Residual connection (Add with skip)
        - LayerNorm pattern
        
        Returns:
            List of detected patterns with node IDs
        """
        patterns = []
        nodes = analysis.nodes
        node_count = len(nodes)
        
        # Build quick op_type lookup
        op_type_map = {n.node_id: n.op_type for n in nodes}
        
        # Pattern 1: Conv-BN-Relu / Conv-BN-ReLU
        for i, node in enumerate(nodes):
            if node.op_type == 'Conv':
                # Check for BN after Conv
                for dep_id in node.dependents:
                    if dep_id < node_count and nodes[dep_id].op_type == 'BatchNormalization':
                        bn_node = nodes[dep_id]
                        # Check for Relu after BN
                        for relu_id in bn_node.dependents:
                            if relu_id < node_count and nodes[relu_id].op_type in ('Relu', 'LeakyRelu'):
                                patterns.append({
                                    'pattern_type': 'Conv-BN-Relu',
                                    'node_ids': [i, dep_id, relu_id],
                                    'op_sequence': ['Conv', 'BatchNormalization', nodes[relu_id].op_type]
                                })
                        # Also record Conv-BN without Relu
                        if not any(p['pattern_type'] == 'Conv-BN-Relu' and i in p['node_ids'] for p in patterns):
                            patterns.append({
                                'pattern_type': 'Conv-BN',
                                'node_ids': [i, dep_id],
                                'op_sequence': ['Conv', 'BatchNormalization']
                            })
        
        # Pattern 2: Attention pattern (MatMul-Softmax-MatMul)
        for i, node in enumerate(nodes):
            if node.op_type == 'MatMul':
                for softmax_id in node.dependents:
                    if softmax_id < node_count and nodes[softmax_id].op_type == 'Softmax':
                        softmax_node = nodes[softmax_id]
                        for matmul2_id in softmax_node.dependents:
                            if matmul2_id < node_count and nodes[matmul2_id].op_type == 'MatMul':
                                patterns.append({
                                    'pattern_type': 'Attention',
                                    'node_ids': [i, softmax_id, matmul2_id],
                                    'op_sequence': ['MatMul', 'Softmax', 'MatMul'],
                                    'description': 'Self-attention QK^T * V pattern'
                                })
        
        # Pattern 3: Einsum attention (single Einsum for attention)
        for i, node in enumerate(nodes):
            if node.op_type == 'Einsum':
                # Check if this looks like attention
                equation = node.attributes.get('equation', '')
                if 'bhid' in equation or 'bhjd' in equation or 'bhij' in equation:
                    patterns.append({
                        'pattern_type': 'Einsum-Attention',
                        'node_ids': [i],
                        'op_sequence': ['Einsum'],
                        'description': f'Attention via Einsum: {equation}',
                        'is_blocker': True
                    })
        
        # Pattern 4: Residual/Skip connection (Add with two different depth inputs)
        for i, node in enumerate(nodes):
            if node.op_type == 'Add' and len(node.dependencies) >= 2:
                deps = node.dependencies
                if len(deps) >= 2:
                    # Check if inputs come from different depths (skip connection)
                    dep_depths = []
                    for dep_id in deps:
                        if dep_id >= 0:
                            chain = self.get_dependency_chain(analysis, dep_id, 'predecessors', max_depth=5)
                            dep_depths.append(len(chain.get('predecessors', [])))
                    
                    # If depth difference > 2, likely a skip connection
                    if dep_depths and max(dep_depths) - min(dep_depths) > 2:
                        patterns.append({
                            'pattern_type': 'Residual',
                            'node_ids': [i] + deps,
                            'op_sequence': ['Add (residual)'],
                            'description': 'Skip/residual connection'
                        })
        
        # Pattern 5: LayerNorm decomposition (ReduceMean-Sub-Mul-ReduceMean-Add-Sqrt-Div-Mul-Add)
        for i, node in enumerate(nodes):
            if node.op_type == 'ReduceMean':
                # Look for Sub following ReduceMean
                for sub_id in node.dependents:
                    if sub_id < node_count and nodes[sub_id].op_type == 'Sub':
                        # This might be start of LayerNorm decomposition
                        patterns.append({
                            'pattern_type': 'LayerNorm-Decomposed',
                            'node_ids': [i, sub_id],
                            'op_sequence': ['ReduceMean', 'Sub', '...'],
                            'description': 'Possible LayerNorm decomposition'
                        })
                        break
        
        # Pattern 6: SE Block (Squeeze-and-Excitation)
        for i, node in enumerate(nodes):
            if node.op_type in ('GlobalAveragePool', 'ReduceMean'):
                # Look for FC/Conv following
                for fc1_id in node.dependents:
                    if fc1_id < node_count and nodes[fc1_id].op_type in ('Gemm', 'Conv', 'MatMul'):
                        fc1_node = nodes[fc1_id]
                        # Look for activation
                        for act_id in fc1_node.dependents:
                            if act_id < node_count and nodes[act_id].op_type in ('Relu', 'Sigmoid', 'HardSigmoid'):
                                patterns.append({
                                    'pattern_type': 'SE-Block',
                                    'node_ids': [i, fc1_id, act_id],
                                    'op_sequence': [node.op_type, nodes[fc1_id].op_type, nodes[act_id].op_type],
                                    'description': 'Squeeze-and-Excitation block'
                                })
                                break
        
        return patterns
    
    def get_node_context_for_surgery(
        self, 
        analysis: ModelAnalysis, 
        node_id: int
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for a node for surgery recommendations.
        
        Returns all information needed by LLM to understand and fix the node.
        """
        if node_id < 0 or node_id >= len(analysis.nodes):
            return {}
        
        node = analysis.nodes[node_id]
        total_nodes = len(analysis.nodes)
        
        # Get dependency chain
        chains = self.get_dependency_chain(analysis, node_id, 'both', max_depth=3)
        
        # Get immediate predecessor/successor details
        predecessor_details = []
        for dep_id in node.dependencies:
            if 0 <= dep_id < total_nodes:
                dep_node = analysis.nodes[dep_id]
                predecessor_details.append({
                    'node_id': dep_id,
                    'name': dep_node.name,
                    'op_type': dep_node.op_type,
                    'output_shapes': [list(s) if s else None for s in dep_node.output_shapes]
                })
        
        successor_details = []
        for dep_id in node.dependents:
            if 0 <= dep_id < total_nodes:
                dep_node = analysis.nodes[dep_id]
                successor_details.append({
                    'node_id': dep_id,
                    'name': dep_node.name,
                    'op_type': dep_node.op_type,
                    'input_shapes': [list(s) if s else None for s in dep_node.input_shapes]
                })
        
        # Detect patterns this node participates in
        all_patterns = self.detect_subgraph_patterns(analysis)
        node_patterns = [p for p in all_patterns if node_id in p.get('node_ids', [])]
        
        return {
            'node_id': node_id,
            'name': node.name,
            'op_type': node.op_type,
            'graph_position': node_id / total_nodes if total_nodes > 0 else 0.5,
            'total_nodes': total_nodes,
            
            # Shapes and types
            'input_shapes': [list(s) if s else None for s in node.input_shapes],
            'output_shapes': [list(s) if s else None for s in node.output_shapes],
            'input_types': node.input_types,
            'output_types': node.output_types,
            
            # Attributes
            'attributes': node.attributes,
            
            # Connectivity
            'inputs': node.inputs,
            'outputs': node.outputs,
            'predecessors': predecessor_details,
            'successors': successor_details,
            
            # Extended chains
            'predecessor_chain': chains.get('predecessors', [])[:10],
            'successor_chain': chains.get('successors', [])[:10],
            
            # Patterns
            'patterns': node_patterns,
            
            # Blocker info
            'is_compilation_blocker': node.is_compilation_blocker,
            'blocker_reason': node.blocker_reason
        }
    
    def get_all_tensor_info(self, model_path: str) -> Dict[str, Dict]:
        """
        Get complete tensor information for all tensors in the model.
        
        Returns dict mapping tensor name to {shape, dtype, is_input, is_output, is_initializer}
        """
        model = onnx.load(model_path)
        
        try:
            model = shape_inference.infer_shapes(model)
        except:
            pass
        
        graph = model.graph
        tensor_info = {}
        
        # Build shape and type tables
        shape_table, type_table = self._build_tensor_tables(graph)
        
        # Get initializer names
        initializer_names = {init.name for init in graph.initializer}
        
        # Get input/output names
        input_names = {vi.name for vi in graph.input if vi.name not in initializer_names}
        output_names = {vi.name for vi in graph.output}
        
        # All tensors from value_info, inputs, outputs
        all_tensor_names = set(shape_table.keys()) | input_names | output_names
        
        # Add tensors from node inputs/outputs
        for node in graph.node:
            all_tensor_names.update(node.input)
            all_tensor_names.update(node.output)
        
        # Remove empty names
        all_tensor_names.discard('')
        
        for name in all_tensor_names:
            shape = shape_table.get(name)
            dtype = type_table.get(name, 'unknown')
            
            tensor_info[name] = {
                'name': name,
                'shape': list(shape) if shape else None,
                'dtype': dtype,
                'is_input': name in input_names,
                'is_output': name in output_names,
                'is_initializer': name in initializer_names,
                'is_dynamic': shape and any(isinstance(d, str) or d is None for d in shape)
            }
        
        return tensor_info


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python onnx_analyzer.py <model.onnx>")
        sys.exit(1)
    
    analyzer = ONNXAnalyzer()
    analysis = analyzer.analyze(sys.argv[1])
    
    print(analysis.get_summary())
    print("\n" + "="*60)
    print(json.dumps(analysis.to_dict(), indent=2, default=str)[:5000])

