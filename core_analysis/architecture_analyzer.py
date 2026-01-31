#!/usr/bin/env python3
"""
Architecture Analyzer for ONNX Models.

Detects high-level architectural patterns in ONNX models including:
- Transformer attention blocks
- Feed-forward networks
- Layer normalization patterns
- Detection heads (YOLO-style)
- Residual connections
- Multi-scale feature pyramids

This enables strategic surgery planning at the architectural level,
not just individual node operations.

Author: Automated Model Surgery Pipeline
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import onnx
from onnx import shape_inference

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ArchitectureType(Enum):
    """Model architecture types."""
    TRANSFORMER = "Transformer"
    VIT = "ViT"
    YOLO = "YOLO"
    CNN = "CNN"
    HYBRID = "Hybrid"
    UNKNOWN = "Unknown"


class BlockType(Enum):
    """Architectural block types."""
    ATTENTION = "attention"
    EINSUM_ATTENTION = "einsum_attention"
    FEED_FORWARD = "feed_forward"
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    RESIDUAL = "residual"
    CONV_BLOCK = "conv_block"
    DETECTION_HEAD = "detection_head"
    POOLING = "pooling"
    EMBEDDING = "embedding"
    OUTPUT_HEAD = "output_head"
    UNKNOWN = "unknown"


@dataclass
class ArchitecturalBlock:
    """Represents a detected architectural block."""
    block_id: str
    block_type: BlockType
    start_node_id: int
    end_node_id: int
    node_ids: List[int]
    op_sequence: List[str]  # e.g., ["MatMul", "Softmax", "MatMul"]
    description: str
    
    # Block-specific metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"equation": "bhid,bhjd->bhij"} for Einsum attention
    
    # Compilation issues in this block
    has_blockers: bool = False
    blocker_nodes: List[int] = field(default_factory=list)
    blocker_reasons: List[str] = field(default_factory=list)
    
    # Dependencies
    input_blocks: List[str] = field(default_factory=list)  # Block IDs
    output_blocks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'block_id': self.block_id,
            'block_type': self.block_type.value,
            'start_node_id': self.start_node_id,
            'end_node_id': self.end_node_id,
            'node_ids': self.node_ids,
            'op_sequence': self.op_sequence,
            'description': self.description,
            'attributes': self.attributes,
            'has_blockers': self.has_blockers,
            'blocker_nodes': self.blocker_nodes,
            'blocker_reasons': self.blocker_reasons,
            'input_blocks': self.input_blocks,
            'output_blocks': self.output_blocks
        }


@dataclass
class DataFlowPath:
    """Represents a data flow path through the model."""
    path_id: str
    name: str  # e.g., "main", "skip", "scale_1"
    block_sequence: List[str]  # Block IDs in order
    is_main_path: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'path_id': self.path_id,
            'name': self.name,
            'block_sequence': self.block_sequence,
            'is_main_path': self.is_main_path
        }


@dataclass
class ModelArchitecture:
    """Complete architectural analysis of a model."""
    model_name: str
    architecture_type: ArchitectureType
    architecture_confidence: float  # 0.0 to 1.0
    
    # Detected blocks
    blocks: List[ArchitecturalBlock] = field(default_factory=list)
    
    # Data flow
    data_flows: List[DataFlowPath] = field(default_factory=list)
    
    # Summary statistics
    total_nodes: int = 0
    nodes_in_blocks: int = 0
    block_coverage: float = 0.0  # % of nodes in identified blocks
    
    # Architecture-specific metadata
    num_transformer_layers: int = 0
    num_attention_heads: int = 0
    has_multi_scale: bool = False
    scale_levels: int = 0
    
    # Compilation blockers by block
    blocks_with_blockers: int = 0
    total_blocker_nodes: int = 0
    
    # Recommended strategy hints
    recommended_strategy: str = ""
    divide_and_conquer_recommended: bool = False
    suggested_split_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'architecture_type': self.architecture_type.value,
            'architecture_confidence': self.architecture_confidence,
            'blocks': [b.to_dict() for b in self.blocks],
            'data_flows': [d.to_dict() for d in self.data_flows],
            'total_nodes': self.total_nodes,
            'nodes_in_blocks': self.nodes_in_blocks,
            'block_coverage': self.block_coverage,
            'num_transformer_layers': self.num_transformer_layers,
            'num_attention_heads': self.num_attention_heads,
            'has_multi_scale': self.has_multi_scale,
            'scale_levels': self.scale_levels,
            'blocks_with_blockers': self.blocks_with_blockers,
            'total_blocker_nodes': self.total_blocker_nodes,
            'recommended_strategy': self.recommended_strategy,
            'divide_and_conquer_recommended': self.divide_and_conquer_recommended,
            'suggested_split_points': self.suggested_split_points
        }
    
    def get_blocks_by_type(self, block_type: BlockType) -> List[ArchitecturalBlock]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks if b.block_type == block_type]
    
    def get_blocker_blocks(self) -> List[ArchitecturalBlock]:
        """Get all blocks that contain compilation blockers."""
        return [b for b in self.blocks if b.has_blockers]
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Model: {self.model_name}",
            f"Architecture: {self.architecture_type.value} (confidence: {self.architecture_confidence:.2f})",
            f"Total Nodes: {self.total_nodes}",
            f"Blocks Detected: {len(self.blocks)} ({self.block_coverage:.1%} coverage)",
        ]
        
        # Block type summary
        block_counts = defaultdict(int)
        for block in self.blocks:
            block_counts[block.block_type.value] += 1
        
        if block_counts:
            lines.append("Block Types: " + ", ".join(
                f"{t}: {c}" for t, c in sorted(block_counts.items())
            ))
        
        # Blocker summary
        if self.blocks_with_blockers > 0:
            lines.append(f"Blocks with Blockers: {self.blocks_with_blockers} ({self.total_blocker_nodes} nodes)")
        
        # Strategy hint
        if self.recommended_strategy:
            lines.append(f"Recommended Strategy: {self.recommended_strategy}")
        
        if self.divide_and_conquer_recommended:
            lines.append(f"Divide & Conquer: Recommended (split at: {', '.join(self.suggested_split_points[:3])})")
        
        return "\n".join(lines)


class ArchitectureAnalyzer:
    """
    Analyze ONNX models to detect architectural patterns.
    
    This analyzer identifies high-level structures like attention blocks,
    feed-forward networks, and detection heads, enabling strategic
    transformation planning.
    """
    
    # ==========================================================================
    # Pattern Definitions
    # ==========================================================================
    
    # Attention patterns (op sequences that indicate attention)
    ATTENTION_PATTERNS = [
        # Standard attention: Q @ K^T @ V
        ["MatMul", "Softmax", "MatMul"],
        ["MatMul", "Div", "Softmax", "MatMul"],  # With scaling
        ["MatMul", "Add", "Softmax", "MatMul"],  # With mask
        ["MatMul", "Mul", "Softmax", "MatMul"],  # With scaling via Mul
    ]
    
    # Einsum attention (single op doing QK^T or attention)
    EINSUM_ATTENTION_EQUATIONS = [
        "bhid,bhjd->bhij",  # QK^T
        "bhij,bhjd->bhid",  # Attention @ V
        "bnqd,bnkd->bnqk",  # Alternative naming
        "bnqk,bnkd->bnqd",  # Alternative naming
    ]
    
    # Feed-forward patterns
    FFN_PATTERNS = [
        ["MatMul", "Relu", "MatMul"],
        ["MatMul", "Gelu", "MatMul"],
        ["Gemm", "Relu", "Gemm"],
        ["Gemm", "Gelu", "Gemm"],
        ["MatMul", "Add", "Relu", "MatMul"],
        ["MatMul", "Add", "Gelu", "MatMul"],
    ]
    
    # Layer normalization patterns
    LAYERNORM_PATTERNS = [
        ["ReduceMean", "Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div"],
        ["ReduceMean", "Sub", "Mul", "ReduceMean", "Add", "Sqrt", "Div"],
    ]
    
    # Batch normalization pattern
    BATCHNORM_PATTERN = ["BatchNormalization"]
    
    # Conv block patterns
    CONV_BLOCK_PATTERNS = [
        ["Conv", "BatchNormalization", "Relu"],
        ["Conv", "BatchNormalization", "LeakyRelu"],
        ["Conv", "Relu"],
        ["Conv", "LeakyRelu"],
        ["Conv", "BatchNormalization", "Sigmoid", "Mul"],  # SiLU/Swish
    ]
    
    # Detection head patterns (YOLO-style)
    DETECTION_HEAD_INDICATORS = [
        "Sigmoid",  # Class probabilities
        "Concat",   # Multi-scale fusion
        "Reshape",  # Output reshaping
        "Transpose",  # Layout change
    ]
    
    # MLA Compilation blockers
    COMPILATION_BLOCKERS = {
        'Einsum', 'NonZero', 'Where', 'Loop', 'If', 'Scan',
        'SequenceAt', 'SequenceConstruct', 'NonMaxSuppression',
        'TopK', 'Unique', 'GatherND', 'ScatterND'
    }
    
    def __init__(self):
        """Initialize the architecture analyzer."""
        self.dtype_map = {
            1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16',
            5: 'int16', 6: 'int32', 7: 'int64', 8: 'string',
            9: 'bool', 10: 'float16', 11: 'float64', 12: 'uint32',
            13: 'uint64', 14: 'complex64', 15: 'complex128', 16: 'bfloat16'
        }
    
    def analyze(self, model_path: str) -> ModelArchitecture:
        """
        Analyze an ONNX model's architecture.
        
        Args:
            model_path: Path to ONNX model file
            
        Returns:
            ModelArchitecture with detected patterns
        """
        # Load model
        model = onnx.load(model_path)
        
        # Try shape inference
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass
        
        graph = model.graph
        model_name = Path(model_path).stem
        
        # Build node lookup structures
        nodes = list(graph.node)
        node_by_output = {}
        node_by_name = {}
        
        for i, node in enumerate(nodes):
            node_by_name[node.name or f"node_{i}"] = i
            for output in node.output:
                if output:
                    node_by_output[output] = i
        
        # Detect architecture type
        arch_type, confidence = self._detect_architecture_type(nodes, model_name)
        
        # Detect architectural blocks
        blocks = []
        used_nodes = set()
        
        # 1. Detect Einsum attention blocks (highest priority - these are blockers)
        einsum_blocks = self._detect_einsum_attention(nodes, used_nodes)
        blocks.extend(einsum_blocks)
        
        # 2. Detect standard attention blocks
        attention_blocks = self._detect_attention_blocks(nodes, used_nodes, node_by_output)
        blocks.extend(attention_blocks)
        
        # 3. Detect layer normalization
        layernorm_blocks = self._detect_layernorm_blocks(nodes, used_nodes)
        blocks.extend(layernorm_blocks)
        
        # 4. Detect feed-forward networks
        ffn_blocks = self._detect_ffn_blocks(nodes, used_nodes, node_by_output)
        blocks.extend(ffn_blocks)
        
        # 5. Detect conv blocks
        conv_blocks = self._detect_conv_blocks(nodes, used_nodes)
        blocks.extend(conv_blocks)
        
        # 6. Detect detection heads
        detection_blocks = self._detect_detection_heads(nodes, used_nodes, node_by_output)
        blocks.extend(detection_blocks)
        
        # 7. Detect residual connections
        residual_blocks = self._detect_residuals(nodes, used_nodes, node_by_output)
        blocks.extend(residual_blocks)
        
        # Mark blockers in each block
        self._mark_blockers_in_blocks(blocks, nodes)
        
        # Compute statistics
        nodes_in_blocks = len(set().union(*[set(b.node_ids) for b in blocks])) if blocks else 0
        block_coverage = nodes_in_blocks / len(nodes) if nodes else 0.0
        
        blocks_with_blockers = sum(1 for b in blocks if b.has_blockers)
        total_blocker_nodes = sum(len(b.blocker_nodes) for b in blocks)
        
        # Compute transformer-specific stats
        num_transformer_layers = len([b for b in blocks if b.block_type in 
                                       [BlockType.ATTENTION, BlockType.EINSUM_ATTENTION]])
        
        # Detect multi-scale (YOLO-style)
        has_multi_scale = self._detect_multi_scale(nodes)
        scale_levels = self._count_scale_levels(nodes) if has_multi_scale else 0
        
        # Generate strategy recommendations
        recommended_strategy, divide_conquer, split_points = self._generate_strategy_hints(
            arch_type, blocks, total_blocker_nodes, len(nodes)
        )
        
        # Build data flow paths (simplified)
        data_flows = self._build_data_flows(blocks)
        
        return ModelArchitecture(
            model_name=model_name,
            architecture_type=arch_type,
            architecture_confidence=confidence,
            blocks=blocks,
            data_flows=data_flows,
            total_nodes=len(nodes),
            nodes_in_blocks=nodes_in_blocks,
            block_coverage=block_coverage,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=0,  # Would need deeper analysis
            has_multi_scale=has_multi_scale,
            scale_levels=scale_levels,
            blocks_with_blockers=blocks_with_blockers,
            total_blocker_nodes=total_blocker_nodes,
            recommended_strategy=recommended_strategy,
            divide_and_conquer_recommended=divide_conquer,
            suggested_split_points=split_points
        )
    
    def _detect_architecture_type(
        self, 
        nodes: List, 
        model_name: str
    ) -> Tuple[ArchitectureType, float]:
        """Detect the overall architecture type."""
        name_lower = model_name.lower()
        
        # Count key operations
        op_counts = defaultdict(int)
        for node in nodes:
            op_counts[node.op_type] += 1
        
        # Name-based detection (high confidence)
        if any(x in name_lower for x in ['yolo', 'yolov']):
            return ArchitectureType.YOLO, 0.95
        
        if any(x in name_lower for x in ['vit', 'deit', 'vision_transformer']):
            return ArchitectureType.VIT, 0.95
        
        if any(x in name_lower for x in ['t5', 'bert', 'gpt', 'transformer', 'marian', 'trocr']):
            return ArchitectureType.TRANSFORMER, 0.95
        
        # Operation-based detection
        has_attention_ops = (
            op_counts.get('Einsum', 0) > 0 or
            (op_counts.get('MatMul', 0) > 5 and op_counts.get('Softmax', 0) > 0)
        )
        
        has_conv_heavy = op_counts.get('Conv', 0) > 10
        has_detection_head = (
            op_counts.get('Sigmoid', 0) > 0 and
            op_counts.get('Concat', 0) > 0 and
            op_counts.get('Reshape', 0) > 0
        )
        
        # Determine type
        if has_attention_ops and has_conv_heavy:
            return ArchitectureType.VIT, 0.7
        elif has_attention_ops:
            return ArchitectureType.TRANSFORMER, 0.7
        elif has_conv_heavy and has_detection_head:
            return ArchitectureType.YOLO, 0.7
        elif has_conv_heavy:
            return ArchitectureType.CNN, 0.7
        
        return ArchitectureType.UNKNOWN, 0.3
    
    def _detect_einsum_attention(
        self, 
        nodes: List, 
        used_nodes: Set[int]
    ) -> List[ArchitecturalBlock]:
        """Detect Einsum-based attention patterns."""
        blocks = []
        
        for i, node in enumerate(nodes):
            if node.op_type == 'Einsum' and i not in used_nodes:
                # Get equation attribute
                equation = ""
                for attr in node.attribute:
                    if attr.name == "equation":
                        equation = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
                        break
                
                # Check if this is an attention pattern
                is_attention = any(
                    eq in equation for eq in self.EINSUM_ATTENTION_EQUATIONS
                )
                
                block_type = BlockType.EINSUM_ATTENTION if is_attention else BlockType.UNKNOWN
                description = f"Einsum operation with equation: {equation}"
                if is_attention:
                    description = f"Einsum attention ({equation})"
                
                block = ArchitecturalBlock(
                    block_id=f"einsum_{i}",
                    block_type=block_type,
                    start_node_id=i,
                    end_node_id=i,
                    node_ids=[i],
                    op_sequence=["Einsum"],
                    description=description,
                    attributes={"equation": equation},
                    has_blockers=True,  # Einsum is always a blocker
                    blocker_nodes=[i],
                    blocker_reasons=["Einsum tensor contraction not supported on MLA"]
                )
                blocks.append(block)
                used_nodes.add(i)
        
        return blocks
    
    def _detect_attention_blocks(
        self, 
        nodes: List, 
        used_nodes: Set[int],
        node_by_output: Dict[str, int]
    ) -> List[ArchitecturalBlock]:
        """Detect standard MatMul-Softmax-MatMul attention patterns."""
        blocks = []
        
        for i, node in enumerate(nodes):
            if node.op_type == 'Softmax' and i not in used_nodes:
                # Look for MatMul before Softmax
                pre_matmul = None
                for inp in node.input:
                    if inp in node_by_output:
                        pre_idx = node_by_output[inp]
                        if nodes[pre_idx].op_type in ['MatMul', 'Div', 'Add', 'Mul']:
                            # Trace back to MatMul
                            if nodes[pre_idx].op_type == 'MatMul':
                                pre_matmul = pre_idx
                            else:
                                # Check one more level
                                for inp2 in nodes[pre_idx].input:
                                    if inp2 in node_by_output:
                                        pre_idx2 = node_by_output[inp2]
                                        if nodes[pre_idx2].op_type == 'MatMul':
                                            pre_matmul = pre_idx2
                                            break
                
                # Look for MatMul after Softmax
                post_matmul = None
                softmax_output = node.output[0] if node.output else None
                if softmax_output:
                    for j, other_node in enumerate(nodes):
                        if j != i and softmax_output in other_node.input:
                            if other_node.op_type == 'MatMul':
                                post_matmul = j
                                break
                
                if pre_matmul is not None and post_matmul is not None:
                    # Found attention pattern
                    node_ids = sorted(set([pre_matmul, i, post_matmul]))
                    
                    # Skip if any are already used
                    if any(nid in used_nodes for nid in node_ids):
                        continue
                    
                    block = ArchitecturalBlock(
                        block_id=f"attention_{i}",
                        block_type=BlockType.ATTENTION,
                        start_node_id=min(node_ids),
                        end_node_id=max(node_ids),
                        node_ids=node_ids,
                        op_sequence=["MatMul", "Softmax", "MatMul"],
                        description="Self-attention block (Q @ K^T @ V)"
                    )
                    blocks.append(block)
                    used_nodes.update(node_ids)
        
        return blocks
    
    def _detect_layernorm_blocks(
        self, 
        nodes: List, 
        used_nodes: Set[int]
    ) -> List[ArchitecturalBlock]:
        """Detect layer normalization patterns."""
        blocks = []
        
        for i, node in enumerate(nodes):
            if node.op_type == 'ReduceMean' and i not in used_nodes:
                # Try to match LayerNorm pattern
                # Look ahead for Sub, Pow, ReduceMean, Add, Sqrt, Div
                matched_ids = [i]
                expected_ops = ['Sub', 'Pow', 'ReduceMean', 'Add', 'Sqrt', 'Div']
                
                current_idx = i + 1
                for expected_op in expected_ops:
                    if current_idx < len(nodes):
                        if nodes[current_idx].op_type == expected_op:
                            matched_ids.append(current_idx)
                            current_idx += 1
                        elif nodes[current_idx].op_type == 'Mul' and expected_op == 'Pow':
                            # Alternative pattern with Mul instead of Pow
                            matched_ids.append(current_idx)
                            current_idx += 1
                        else:
                            break
                    else:
                        break
                
                # Check if we matched enough of the pattern
                if len(matched_ids) >= 5:  # At least ReduceMean, Sub, *, ReduceMean, Add
                    # Skip if any are already used
                    if any(nid in used_nodes for nid in matched_ids):
                        continue
                    
                    block = ArchitecturalBlock(
                        block_id=f"layernorm_{i}",
                        block_type=BlockType.LAYER_NORM,
                        start_node_id=min(matched_ids),
                        end_node_id=max(matched_ids),
                        node_ids=matched_ids,
                        op_sequence=[nodes[idx].op_type for idx in matched_ids],
                        description="Layer normalization"
                    )
                    blocks.append(block)
                    used_nodes.update(matched_ids)
        
        return blocks
    
    def _detect_ffn_blocks(
        self, 
        nodes: List, 
        used_nodes: Set[int],
        node_by_output: Dict[str, int]
    ) -> List[ArchitecturalBlock]:
        """Detect feed-forward network patterns."""
        blocks = []
        
        for i, node in enumerate(nodes):
            if node.op_type in ['Gelu', 'Relu', 'LeakyRelu'] and i not in used_nodes:
                # Look for MatMul/Gemm before activation
                pre_linear = None
                for inp in node.input:
                    if inp in node_by_output:
                        pre_idx = node_by_output[inp]
                        if nodes[pre_idx].op_type in ['MatMul', 'Gemm', 'Add']:
                            if nodes[pre_idx].op_type == 'Add':
                                # Check one more level for MatMul
                                for inp2 in nodes[pre_idx].input:
                                    if inp2 in node_by_output:
                                        pre_idx2 = node_by_output[inp2]
                                        if nodes[pre_idx2].op_type in ['MatMul', 'Gemm']:
                                            pre_linear = pre_idx2
                                            break
                            else:
                                pre_linear = pre_idx
                
                # Look for MatMul/Gemm after activation
                post_linear = None
                act_output = node.output[0] if node.output else None
                if act_output:
                    for j in range(i + 1, min(i + 5, len(nodes))):
                        if act_output in nodes[j].input:
                            if nodes[j].op_type in ['MatMul', 'Gemm']:
                                post_linear = j
                                break
                
                if pre_linear is not None and post_linear is not None:
                    node_ids = sorted(set([pre_linear, i, post_linear]))
                    
                    if any(nid in used_nodes for nid in node_ids):
                        continue
                    
                    block = ArchitecturalBlock(
                        block_id=f"ffn_{i}",
                        block_type=BlockType.FEED_FORWARD,
                        start_node_id=min(node_ids),
                        end_node_id=max(node_ids),
                        node_ids=node_ids,
                        op_sequence=[nodes[nid].op_type for nid in node_ids],
                        description="Feed-forward network"
                    )
                    blocks.append(block)
                    used_nodes.update(node_ids)
        
        return blocks
    
    def _detect_conv_blocks(
        self, 
        nodes: List, 
        used_nodes: Set[int]
    ) -> List[ArchitecturalBlock]:
        """Detect convolutional blocks (Conv-BN-Relu patterns)."""
        blocks = []
        
        for i, node in enumerate(nodes):
            if node.op_type == 'Conv' and i not in used_nodes:
                matched_ids = [i]
                current_idx = i + 1
                
                # Look for BatchNorm
                if current_idx < len(nodes) and nodes[current_idx].op_type == 'BatchNormalization':
                    matched_ids.append(current_idx)
                    current_idx += 1
                
                # Look for activation
                if current_idx < len(nodes) and nodes[current_idx].op_type in [
                    'Relu', 'LeakyRelu', 'Sigmoid', 'Mul'
                ]:
                    matched_ids.append(current_idx)
                    current_idx += 1
                    
                    # SiLU pattern: Sigmoid followed by Mul
                    if nodes[matched_ids[-1]].op_type == 'Sigmoid':
                        if current_idx < len(nodes) and nodes[current_idx].op_type == 'Mul':
                            matched_ids.append(current_idx)
                
                if len(matched_ids) >= 2:  # At least Conv + something
                    if any(nid in used_nodes for nid in matched_ids):
                        continue
                    
                    block = ArchitecturalBlock(
                        block_id=f"conv_{i}",
                        block_type=BlockType.CONV_BLOCK,
                        start_node_id=min(matched_ids),
                        end_node_id=max(matched_ids),
                        node_ids=matched_ids,
                        op_sequence=[nodes[idx].op_type for idx in matched_ids],
                        description="Convolutional block"
                    )
                    blocks.append(block)
                    used_nodes.update(matched_ids)
        
        return blocks
    
    def _detect_detection_heads(
        self, 
        nodes: List, 
        used_nodes: Set[int],
        node_by_output: Dict[str, int]
    ) -> List[ArchitecturalBlock]:
        """Detect detection head patterns (YOLO-style)."""
        blocks = []
        
        # Look for Concat operations near the end (common in detection heads)
        concat_indices = [i for i, n in enumerate(nodes) if n.op_type == 'Concat']
        
        for concat_idx in concat_indices:
            if concat_idx in used_nodes:
                continue
            
            # Check if this concat is near the end (detection head region)
            if concat_idx < len(nodes) * 0.6:  # Skip if in first 60% of graph
                continue
            
            # Look for Sigmoid and Reshape nearby
            nearby_ops = set()
            for j in range(max(0, concat_idx - 10), min(len(nodes), concat_idx + 10)):
                nearby_ops.add(nodes[j].op_type)
            
            if 'Sigmoid' in nearby_ops or 'Reshape' in nearby_ops:
                # This might be a detection head
                node_ids = [concat_idx]
                
                # Include nearby Sigmoid, Reshape, Transpose
                for j in range(max(0, concat_idx - 5), min(len(nodes), concat_idx + 5)):
                    if nodes[j].op_type in ['Sigmoid', 'Reshape', 'Transpose', 'Split']:
                        if j not in used_nodes:
                            node_ids.append(j)
                
                node_ids = sorted(set(node_ids))
                
                if any(nid in used_nodes for nid in node_ids):
                    continue
                
                block = ArchitecturalBlock(
                    block_id=f"detection_head_{concat_idx}",
                    block_type=BlockType.DETECTION_HEAD,
                    start_node_id=min(node_ids),
                    end_node_id=max(node_ids),
                    node_ids=node_ids,
                    op_sequence=[nodes[idx].op_type for idx in node_ids],
                    description="Detection head (multi-scale fusion)"
                )
                blocks.append(block)
                used_nodes.update(node_ids)
        
        return blocks
    
    def _detect_residuals(
        self, 
        nodes: List, 
        used_nodes: Set[int],
        node_by_output: Dict[str, int]
    ) -> List[ArchitecturalBlock]:
        """Detect residual connections."""
        blocks = []
        
        for i, node in enumerate(nodes):
            if node.op_type == 'Add' and i not in used_nodes:
                # Check if this Add has inputs from different "depths"
                # (indicating a skip connection)
                input_depths = []
                for inp in node.input:
                    if inp in node_by_output:
                        input_depths.append(node_by_output[inp])
                
                if len(input_depths) >= 2:
                    depth_diff = max(input_depths) - min(input_depths)
                    if depth_diff > 3:  # Significant depth difference = skip connection
                        block = ArchitecturalBlock(
                            block_id=f"residual_{i}",
                            block_type=BlockType.RESIDUAL,
                            start_node_id=i,
                            end_node_id=i,
                            node_ids=[i],
                            op_sequence=["Add"],
                            description=f"Residual connection (skip depth: {depth_diff})"
                        )
                        blocks.append(block)
                        used_nodes.add(i)
        
        return blocks
    
    def _mark_blockers_in_blocks(
        self, 
        blocks: List[ArchitecturalBlock], 
        nodes: List
    ) -> None:
        """Mark compilation blockers in each block."""
        for block in blocks:
            blocker_nodes = []
            blocker_reasons = []
            
            for node_id in block.node_ids:
                if node_id < len(nodes):
                    node = nodes[node_id]
                    if node.op_type in self.COMPILATION_BLOCKERS:
                        blocker_nodes.append(node_id)
                        blocker_reasons.append(
                            f"{node.op_type}: Not supported on MLA"
                        )
                    
                    # Check for non-4D tensors (simplified check)
                    # In a full implementation, we'd use shape inference
            
            if blocker_nodes:
                block.has_blockers = True
                block.blocker_nodes = blocker_nodes
                block.blocker_reasons = blocker_reasons
    
    def _detect_multi_scale(self, nodes: List) -> bool:
        """Detect if model uses multi-scale features (YOLO-style)."""
        # Count Concat operations
        concat_count = sum(1 for n in nodes if n.op_type == 'Concat')
        
        # Count Resize/Upsample operations
        resize_count = sum(1 for n in nodes if n.op_type in ['Resize', 'Upsample'])
        
        # Multi-scale typically has multiple concats and resizes
        return concat_count >= 3 and resize_count >= 2
    
    def _count_scale_levels(self, nodes: List) -> int:
        """Count the number of scale levels in a multi-scale model."""
        # Simplified: count unique Concat operations
        return sum(1 for n in nodes if n.op_type == 'Concat')
    
    def _generate_strategy_hints(
        self, 
        arch_type: ArchitectureType,
        blocks: List[ArchitecturalBlock],
        total_blockers: int,
        total_nodes: int
    ) -> Tuple[str, bool, List[str]]:
        """Generate strategy recommendations based on architecture analysis."""
        
        # Count blocker types
        einsum_blocks = [b for b in blocks if b.block_type == BlockType.EINSUM_ATTENTION]
        detection_blocks = [b for b in blocks if b.block_type == BlockType.DETECTION_HEAD]
        
        # Determine strategy
        strategy = ""
        divide_conquer = False
        split_points = []
        
        if arch_type == ArchitectureType.TRANSFORMER:
            if len(einsum_blocks) > 0:
                strategy = "transformer_attention_decomposition"
                # If many Einsum blocks throughout, might need divide and conquer
                if len(einsum_blocks) > 6 and total_blockers > 20:
                    divide_conquer = True
                    split_points = ["encoder_input", "encoder_output"]
            else:
                strategy = "transformer_optimization"
        
        elif arch_type == ArchitectureType.VIT:
            strategy = "vit_full_rewrite"
            divide_conquer = True  # ViT typically needs divide and conquer
            split_points = ["patch_embed", "transformer_blocks", "head"]
        
        elif arch_type == ArchitectureType.YOLO:
            if len(detection_blocks) > 0:
                strategy = "yolo_detection_head_separation"
            else:
                strategy = "yolo_backbone_optimization"
        
        elif arch_type == ArchitectureType.CNN:
            strategy = "cnn_standard_optimization"
        
        else:
            strategy = "generic_blocker_removal"
        
        # Force divide and conquer if blockers are spread throughout
        if total_blockers > 0:
            blocker_blocks = [b for b in blocks if b.has_blockers]
            if len(blocker_blocks) > 5:
                # Blockers spread across many blocks
                blocker_positions = [b.start_node_id / total_nodes for b in blocker_blocks]
                if max(blocker_positions) - min(blocker_positions) > 0.5:
                    divide_conquer = True
        
        return strategy, divide_conquer, split_points
    
    def _build_data_flows(
        self, 
        blocks: List[ArchitecturalBlock]
    ) -> List[DataFlowPath]:
        """Build simplified data flow paths."""
        if not blocks:
            return []
        
        # Sort blocks by start position
        sorted_blocks = sorted(blocks, key=lambda b: b.start_node_id)
        
        # Create main path
        main_path = DataFlowPath(
            path_id="main",
            name="main",
            block_sequence=[b.block_id for b in sorted_blocks],
            is_main_path=True
        )
        
        return [main_path]


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_architecture(model_path: str) -> ModelArchitecture:
    """Convenience function to analyze model architecture."""
    analyzer = ArchitectureAnalyzer()
    return analyzer.analyze(model_path)


def get_architecture_summary(model_path: str) -> str:
    """Get a human-readable architecture summary."""
    arch = analyze_architecture(model_path)
    return arch.get_summary()


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python architecture_analyzer.py <model.onnx>")
        sys.exit(1)
    
    analyzer = ArchitectureAnalyzer()
    architecture = analyzer.analyze(sys.argv[1])
    
    print(architecture.get_summary())
    print("\n" + "=" * 60)
    print("\nDetailed Blocks:")
    for block in architecture.blocks[:10]:  # Show first 10
        print(f"  {block.block_id}: {block.block_type.value} ({len(block.node_ids)} nodes)")
        if block.has_blockers:
            print(f"    BLOCKERS: {block.blocker_reasons}")
    
    if len(architecture.blocks) > 10:
        print(f"  ... and {len(architecture.blocks) - 10} more blocks")
