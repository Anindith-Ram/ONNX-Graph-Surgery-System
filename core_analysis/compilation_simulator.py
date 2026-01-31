#!/usr/bin/env python3
"""
MLA Compilation Simulator for ONNX Models.

Simulates the SiMa MLA compilation process to identify potential blockers
before running the actual Model SDK. This allows for faster iteration
during surgery planning.

Key features:
- Check each node against MLA constraints
- Identify nodes that would be mapped to CVU/APU
- Predict number of LM files that would be generated
- Generate error messages matching SDK format

Based on the ONNX Graph Surgery for Model SDK documentation.

Author: Automated Model Surgery Pipeline
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import onnx
from onnx import shape_inference, numpy_helper

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ProcessorTarget(Enum):
    """Processor targets for node mapping."""
    MLA = "MLA"  # Machine Learning Accelerator
    CVU = "CVU"  # Computer Vision Unit
    APU = "APU"  # Application Processing Unit (CPU fallback)
    UNKNOWN = "Unknown"


class BlockerSeverity(Enum):
    """Severity of compilation blockers."""
    CRITICAL = "critical"  # Will definitely fail compilation
    WARNING = "warning"    # May cause issues
    INFO = "info"          # Informational only


@dataclass
class TensorConstraintViolation:
    """Describes a tensor constraint violation."""
    tensor_name: str
    constraint: str
    actual_value: Any
    expected_value: Any
    severity: BlockerSeverity = BlockerSeverity.CRITICAL
    
    def to_dict(self) -> Dict:
        return {
            'tensor_name': self.tensor_name,
            'constraint': self.constraint,
            'actual_value': str(self.actual_value),
            'expected_value': str(self.expected_value),
            'severity': self.severity.value
        }
    
    def to_error_message(self) -> str:
        return f"Tensor '{self.tensor_name}': {self.constraint} - got {self.actual_value}, expected {self.expected_value}"


@dataclass
class NodeCompilationResult:
    """Result of compiling a single node."""
    node_index: int
    node_name: str
    op_type: str
    target: ProcessorTarget
    
    # Status
    is_blocker: bool = False
    blocker_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Constraint violations
    tensor_violations: List[TensorConstraintViolation] = field(default_factory=list)
    
    # Suggested fixes
    suggested_fixes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'node_index': self.node_index,
            'node_name': self.node_name,
            'op_type': self.op_type,
            'target': self.target.value,
            'is_blocker': self.is_blocker,
            'blocker_reasons': self.blocker_reasons,
            'warnings': self.warnings,
            'tensor_violations': [v.to_dict() for v in self.tensor_violations],
            'suggested_fixes': self.suggested_fixes
        }


@dataclass
class CompilationReport:
    """Complete compilation simulation report."""
    model_name: str
    total_nodes: int
    
    # Node mapping
    nodes_on_mla: int = 0
    nodes_on_cvu: int = 0
    nodes_on_apu: int = 0
    
    # Blockers
    blocker_count: int = 0
    blocker_nodes: List[NodeCompilationResult] = field(default_factory=list)
    
    # Warnings
    warning_count: int = 0
    warning_nodes: List[NodeCompilationResult] = field(default_factory=list)
    
    # LM file prediction
    predicted_lm_files: int = 1
    lm_file_reasons: List[str] = field(default_factory=list)
    
    # Summary by op type
    blocker_ops: Dict[str, int] = field(default_factory=dict)
    
    # All node results
    node_results: List[NodeCompilationResult] = field(default_factory=list)
    
    # Overall status
    will_compile: bool = True
    compilation_score: float = 1.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'total_nodes': self.total_nodes,
            'nodes_on_mla': self.nodes_on_mla,
            'nodes_on_cvu': self.nodes_on_cvu,
            'nodes_on_apu': self.nodes_on_apu,
            'blocker_count': self.blocker_count,
            'blocker_nodes': [n.to_dict() for n in self.blocker_nodes],
            'warning_count': self.warning_count,
            'predicted_lm_files': self.predicted_lm_files,
            'lm_file_reasons': self.lm_file_reasons,
            'blocker_ops': self.blocker_ops,
            'will_compile': self.will_compile,
            'compilation_score': self.compilation_score
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        status = "PASS" if self.will_compile else "FAIL"
        lines = [
            f"Compilation Simulation Report: {self.model_name}",
            f"=" * 50,
            f"Status: {status} (score: {self.compilation_score:.2f})",
            f"",
            f"Node Distribution:",
            f"  Total Nodes: {self.total_nodes}",
            f"  MLA: {self.nodes_on_mla} ({self.nodes_on_mla / self.total_nodes * 100:.1f}%)" if self.total_nodes > 0 else "",
            f"  CVU: {self.nodes_on_cvu}",
            f"  APU: {self.nodes_on_apu}",
            f"",
            f"Predicted LM Files: {self.predicted_lm_files}",
        ]
        
        if self.blocker_count > 0:
            lines.extend([
                f"",
                f"BLOCKERS: {self.blocker_count}",
            ])
            for op, count in sorted(self.blocker_ops.items(), key=lambda x: -x[1]):
                lines.append(f"  {op}: {count} nodes")
        
        if self.warning_count > 0:
            lines.append(f"")
            lines.append(f"Warnings: {self.warning_count}")
        
        return "\n".join(filter(None, lines))
    
    def get_blocker_summary(self) -> Dict[str, List[str]]:
        """Get blockers grouped by operation type."""
        summary = defaultdict(list)
        for node in self.blocker_nodes:
            for reason in node.blocker_reasons:
                summary[node.op_type].append(f"{node.node_name}: {reason}")
        return dict(summary)


class CompilationSimulator:
    """
    Simulate MLA compilation for ONNX models.
    
    Checks each node against MLA constraints to identify potential
    compilation issues before running the actual Model SDK.
    """
    
    # ==========================================================================
    # MLA Constraints (based on ONNX Graph Surgery documentation)
    # ==========================================================================
    
    # Operations that are never supported on MLA
    UNSUPPORTED_OPS = {
        'Einsum',           # Tensor contraction - must decompose
        'NonZero',          # Dynamic output shape
        'Where',            # Conditional - dynamic
        'Loop',             # Dynamic control flow
        'If',               # Conditional control flow
        'Scan',             # Sequence processing
        'SequenceAt',       # Sequence operations
        'SequenceConstruct',
        'NonMaxSuppression', # Post-processing
        'TopK',             # Dynamic output
        'Unique',           # Dynamic output
        'GatherND',         # Dynamic indexing (depends on use)
        'ScatterND',        # Dynamic scatter
        'Compress',         # Dynamic masking
        'DynamicQuantizeLinear',  # Dynamic quantization
        'Range',            # Dynamic range generation
    }
    
    # Operations with limited support (may work with constraints)
    LIMITED_SUPPORT_OPS = {
        'Reshape': "Must have static shape",
        'Squeeze': "Must have static axes",
        'Unsqueeze': "Must have static axes",
        'Slice': "Must have static starts/ends/steps",
        'Gather': "Must have static indices for some cases",
        'Split': "Must have static split sizes",
        'Concat': "Supported but may affect LM file count",
        'Pad': "Pad values must be static",
    }
    
    # Operations that go to CVU
    CVU_OPS = {
        'Resize',           # Image resizing
        'Upsample',         # Deprecated but still used
        'MaxPool',          # Some pooling configurations
        'AveragePool',      # Some pooling configurations
    }
    
    # Data type constraints
    SUPPORTED_DTYPES = {
        'float32', 'float16', 'int8', 'uint8', 'int32'
    }
    
    DTYPE_MAP = {
        1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16',
        5: 'int16', 6: 'int32', 7: 'int64', 8: 'string',
        9: 'bool', 10: 'float16', 11: 'float64', 12: 'uint32',
        13: 'uint64', 14: 'complex64', 15: 'complex128', 16: 'bfloat16'
    }
    
    def __init__(self, verbose: bool = False):
        """Initialize the compilation simulator."""
        self.verbose = verbose
    
    def simulate(self, model_path: str) -> CompilationReport:
        """
        Simulate MLA compilation for a model.
        
        Args:
            model_path: Path to ONNX model file
            
        Returns:
            CompilationReport with detailed results
        """
        # Load model
        model = onnx.load(model_path)
        model_name = Path(model_path).stem
        
        # Try shape inference
        try:
            model = shape_inference.infer_shapes(model)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Shape inference failed: {e}")
        
        graph = model.graph
        nodes = list(graph.node)
        
        # Build tensor info
        tensor_shapes = self._get_tensor_shapes(graph)
        tensor_dtypes = self._get_tensor_dtypes(graph)
        initializer_names = {init.name for init in graph.initializer}
        
        # Analyze each node
        node_results = []
        blocker_nodes = []
        warning_nodes = []
        blocker_ops = defaultdict(int)
        
        nodes_on_mla = 0
        nodes_on_cvu = 0
        nodes_on_apu = 0
        
        for i, node in enumerate(nodes):
            result = self._analyze_node(
                node, i, tensor_shapes, tensor_dtypes, initializer_names
            )
            node_results.append(result)
            
            if result.is_blocker:
                blocker_nodes.append(result)
                blocker_ops[result.op_type] += 1
                nodes_on_apu += 1
            elif result.warnings:
                warning_nodes.append(result)
                if result.target == ProcessorTarget.CVU:
                    nodes_on_cvu += 1
                elif result.target == ProcessorTarget.MLA:
                    nodes_on_mla += 1
                else:
                    nodes_on_apu += 1
            else:
                if result.target == ProcessorTarget.CVU:
                    nodes_on_cvu += 1
                elif result.target == ProcessorTarget.MLA:
                    nodes_on_mla += 1
                else:
                    nodes_on_apu += 1
        
        # Predict LM files
        predicted_lm_files, lm_reasons = self._predict_lm_files(
            node_results, nodes_on_cvu, nodes_on_apu
        )
        
        # Calculate compilation score
        if len(nodes) > 0:
            mla_ratio = nodes_on_mla / len(nodes)
            blocker_ratio = len(blocker_nodes) / len(nodes)
            score = max(0.0, mla_ratio - blocker_ratio * 2)
        else:
            score = 0.0
        
        # Determine if it will compile
        will_compile = len(blocker_nodes) == 0
        
        return CompilationReport(
            model_name=model_name,
            total_nodes=len(nodes),
            nodes_on_mla=nodes_on_mla,
            nodes_on_cvu=nodes_on_cvu,
            nodes_on_apu=nodes_on_apu,
            blocker_count=len(blocker_nodes),
            blocker_nodes=blocker_nodes,
            warning_count=len(warning_nodes),
            warning_nodes=warning_nodes,
            predicted_lm_files=predicted_lm_files,
            lm_file_reasons=lm_reasons,
            blocker_ops=dict(blocker_ops),
            node_results=node_results,
            will_compile=will_compile,
            compilation_score=score
        )
    
    def _get_tensor_shapes(self, graph) -> Dict[str, Optional[List[int]]]:
        """Extract tensor shapes from graph."""
        shapes = {}
        
        # From value_info
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.type.HasField('tensor_type'):
                shape = []
                has_dynamic = False
                for dim in vi.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                    else:
                        shape.append(-1)  # Dynamic
                        has_dynamic = True
                shapes[vi.name] = shape if not has_dynamic else None
        
        # From initializers
        for init in graph.initializer:
            shapes[init.name] = list(init.dims)
        
        return shapes
    
    def _get_tensor_dtypes(self, graph) -> Dict[str, str]:
        """Extract tensor data types from graph."""
        dtypes = {}
        
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.type.HasField('tensor_type'):
                elem_type = vi.type.tensor_type.elem_type
                dtypes[vi.name] = self.DTYPE_MAP.get(elem_type, 'unknown')
        
        for init in graph.initializer:
            dtypes[init.name] = self.DTYPE_MAP.get(init.data_type, 'unknown')
        
        return dtypes
    
    def _analyze_node(
        self,
        node,
        index: int,
        tensor_shapes: Dict[str, Optional[List[int]]],
        tensor_dtypes: Dict[str, str],
        initializer_names: Set[str]
    ) -> NodeCompilationResult:
        """Analyze a single node for MLA compatibility."""
        result = NodeCompilationResult(
            node_index=index,
            node_name=node.name or f"node_{index}",
            op_type=node.op_type,
            target=ProcessorTarget.MLA
        )
        
        # Check if operation is unsupported
        if node.op_type in self.UNSUPPORTED_OPS:
            result.is_blocker = True
            result.blocker_reasons.append(
                f"Operation '{node.op_type}' is not supported on MLA"
            )
            result.target = ProcessorTarget.APU
            result.suggested_fixes.append(
                self._get_fix_suggestion(node.op_type)
            )
            return result
        
        # Check if operation has limited support
        if node.op_type in self.LIMITED_SUPPORT_OPS:
            limitation = self.LIMITED_SUPPORT_OPS[node.op_type]
            # Check if the limitation applies
            is_dynamic = self._check_dynamic_inputs(node, tensor_shapes, initializer_names)
            if is_dynamic:
                result.is_blocker = True
                result.blocker_reasons.append(
                    f"Operation '{node.op_type}': {limitation}"
                )
                result.target = ProcessorTarget.APU
        
        # Check if operation goes to CVU
        if node.op_type in self.CVU_OPS:
            result.target = ProcessorTarget.CVU
            result.warnings.append(
                f"Operation '{node.op_type}' will be mapped to CVU"
            )
        
        # Check tensor constraints
        for input_name in node.input:
            if not input_name:
                continue
            
            # Check shape (must be 4D for most MLA ops)
            shape = tensor_shapes.get(input_name)
            if shape is not None:
                if len(shape) != 4 and node.op_type in ['Conv', 'BatchNormalization']:
                    violation = TensorConstraintViolation(
                        tensor_name=input_name,
                        constraint="Must be 4D tensor",
                        actual_value=f"{len(shape)}D",
                        expected_value="4D"
                    )
                    result.tensor_violations.append(violation)
                    result.warnings.append(violation.to_error_message())
            elif shape is None and input_name not in initializer_names:
                # Dynamic shape
                result.warnings.append(
                    f"Input '{input_name}' has dynamic shape"
                )
            
            # Check dtype
            dtype = tensor_dtypes.get(input_name)
            if dtype and dtype not in self.SUPPORTED_DTYPES:
                violation = TensorConstraintViolation(
                    tensor_name=input_name,
                    constraint="Unsupported data type",
                    actual_value=dtype,
                    expected_value=", ".join(self.SUPPORTED_DTYPES)
                )
                result.tensor_violations.append(violation)
                result.blocker_reasons.append(violation.to_error_message())
                result.is_blocker = True
        
        # Check for Einsum-specific issues
        if node.op_type == 'Einsum':
            equation = ""
            for attr in node.attribute:
                if attr.name == "equation":
                    equation = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
            result.blocker_reasons.append(
                f"Einsum operation with equation: {equation}"
            )
            result.suggested_fixes.append(
                "Decompose Einsum into MatMul + Transpose operations"
            )
        
        return result
    
    def _check_dynamic_inputs(
        self,
        node,
        tensor_shapes: Dict[str, Optional[List[int]]],
        initializer_names: Set[str]
    ) -> bool:
        """Check if node has dynamic inputs (shapes not known at compile time)."""
        # Specific checks for shape-dependent operations
        if node.op_type == 'Reshape':
            # Check if shape input is dynamic
            if len(node.input) > 1:
                shape_input = node.input[1]
                if shape_input not in initializer_names:
                    return True
        
        elif node.op_type in ['Slice']:
            # Check if starts/ends/axes are dynamic
            for i in range(1, len(node.input)):
                if node.input[i] and node.input[i] not in initializer_names:
                    return True
        
        elif node.op_type == 'Gather':
            # Check if indices are dynamic
            if len(node.input) > 1:
                indices_input = node.input[1]
                if indices_input not in initializer_names:
                    return True
        
        return False
    
    def _get_fix_suggestion(self, op_type: str) -> str:
        """Get fix suggestion for a blocking operation."""
        suggestions = {
            'Einsum': "Decompose into MatMul + Transpose operations",
            'NonZero': "Replace with Where + indices if possible, or move to post-processing",
            'Where': "Use Mul + Add pattern for simple cases, or split model",
            'Loop': "Unroll loop if iteration count is known",
            'If': "Split model at conditional, run branches separately",
            'NonMaxSuppression': "Move to post-processing on CPU",
            'TopK': "Move to post-processing on CPU",
            'GatherND': "Replace with Gather if indices are simple",
            'ScatterND': "Restructure to avoid dynamic scatter",
        }
        return suggestions.get(op_type, f"Remove or replace {op_type} operation")
    
    def _predict_lm_files(
        self,
        node_results: List[NodeCompilationResult],
        nodes_on_cvu: int,
        nodes_on_apu: int
    ) -> Tuple[int, List[str]]:
        """
        Predict number of LM files the model would generate.
        
        Multiple LM files are generated when:
        - Model has CVU operations (requires graph split)
        - Model has APU operations interspersed
        - Model is too large for single MLA execution
        """
        lm_files = 1
        reasons = []
        
        if nodes_on_cvu > 0:
            lm_files += 1
            reasons.append(f"CVU operations present ({nodes_on_cvu} nodes)")
        
        if nodes_on_apu > 0:
            # Each APU section may cause a split
            apu_sections = self._count_apu_sections(node_results)
            if apu_sections > 1:
                lm_files += apu_sections - 1
                reasons.append(f"Multiple APU sections ({apu_sections})")
        
        # Check for operations that typically cause splits
        concat_count = sum(1 for r in node_results if r.op_type == 'Concat')
        if concat_count > 10:
            reasons.append(f"Many Concat operations ({concat_count}) may cause graph splits")
        
        return lm_files, reasons
    
    def _count_apu_sections(self, node_results: List[NodeCompilationResult]) -> int:
        """Count distinct APU sections (consecutive APU nodes)."""
        sections = 0
        in_apu_section = False
        
        for result in node_results:
            if result.target == ProcessorTarget.APU or result.is_blocker:
                if not in_apu_section:
                    sections += 1
                    in_apu_section = True
            else:
                in_apu_section = False
        
        return sections
    
    def get_verbose_output(self, report: CompilationReport) -> str:
        """
        Generate verbose output similar to Model SDK.
        
        This mimics the verbose output format described in the
        ONNX Graph Surgery documentation.
        """
        lines = [
            "=" * 60,
            "MLA COMPILATION SIMULATION (VERBOSE)",
            "=" * 60,
            "",
            f"Model: {report.model_name}",
            f"Total Nodes: {report.total_nodes}",
            "",
            "-" * 40,
            "NODE ANALYSIS",
            "-" * 40,
        ]
        
        for result in report.blocker_nodes[:20]:  # Show first 20 blockers
            lines.append("")
            lines.append(f"[BLOCKER] Node {result.node_index}: {result.node_name}")
            lines.append(f"  Op Type: {result.op_type}")
            for reason in result.blocker_reasons:
                lines.append(f"  Reason: {reason}")
            for fix in result.suggested_fixes:
                lines.append(f"  Suggestion: {fix}")
        
        if len(report.blocker_nodes) > 20:
            lines.append(f"  ... and {len(report.blocker_nodes) - 20} more blockers")
        
        lines.extend([
            "",
            "-" * 40,
            "SUMMARY",
            "-" * 40,
            f"Nodes on MLA: {report.nodes_on_mla}",
            f"Nodes on CVU: {report.nodes_on_cvu}",
            f"Nodes on APU: {report.nodes_on_apu}",
            f"",
            f"Blockers: {report.blocker_count}",
            f"Predicted LM Files: {report.predicted_lm_files}",
            f"",
            f"Will Compile: {'YES' if report.will_compile else 'NO'}",
            f"Compilation Score: {report.compilation_score:.2f}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def simulate_compilation(model_path: str, verbose: bool = False) -> CompilationReport:
    """Convenience function to simulate compilation."""
    simulator = CompilationSimulator(verbose=verbose)
    return simulator.simulate(model_path)


def check_mla_compatibility(model_path: str) -> bool:
    """Quick check if model is MLA compatible."""
    report = simulate_compilation(model_path)
    return report.will_compile


def get_blockers(model_path: str) -> Dict[str, int]:
    """Get blocking operations and their counts."""
    report = simulate_compilation(model_path)
    return report.blocker_ops


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compilation_simulator.py <model.onnx> [--verbose]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    simulator = CompilationSimulator(verbose=verbose)
    report = simulator.simulate(model_path)
    
    print(report.get_summary())
    
    if verbose:
        print()
        print(simulator.get_verbose_output(report))
