#!/usr/bin/env python3
"""
Numerical Verification Module for ONNX Model Surgery.

Verifies numerical equivalence between original and modified models
by running inference and comparing outputs with configurable tolerances.

Tolerance Categories:
- Identical (0.0): Data reshuffling only
- Close (~1e-6): Math order changes
- Acceptable (~1e-4): Complex rewrites (DETR-level)
- Divergent (> 1e-4): Potential bug

Author: Automated Model Surgery Pipeline
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import onnx
import onnxruntime as ort

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ToleranceCategory(Enum):
    """Tolerance categories for numerical comparison."""
    IDENTICAL = "identical"        # Max diff < 1e-10
    CLOSE = "close"                # Max diff < 1e-6
    ACCEPTABLE = "acceptable"      # Max diff < 1e-4
    MARGINAL = "marginal"          # Max diff < 1e-2
    DIVERGENT = "divergent"        # Max diff >= 1e-2


@dataclass
class OutputComparison:
    """Comparison result for a single output tensor."""
    output_name: str
    shape_original: Tuple[int, ...]
    shape_modified: Tuple[int, ...]
    
    # Differences
    max_absolute_diff: float = 0.0
    mean_absolute_diff: float = 0.0
    max_relative_diff: float = 0.0
    mean_relative_diff: float = 0.0
    
    # Statistics
    num_elements: int = 0
    num_different: int = 0  # Elements above threshold
    
    # Category
    tolerance_category: ToleranceCategory = ToleranceCategory.IDENTICAL
    
    # Shape match
    shapes_match: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'output_name': self.output_name,
            'shape_original': list(self.shape_original),
            'shape_modified': list(self.shape_modified),
            'max_absolute_diff': float(self.max_absolute_diff),
            'mean_absolute_diff': float(self.mean_absolute_diff),
            'max_relative_diff': float(self.max_relative_diff),
            'mean_relative_diff': float(self.mean_relative_diff),
            'num_elements': self.num_elements,
            'num_different': self.num_different,
            'tolerance_category': self.tolerance_category.value,
            'shapes_match': self.shapes_match
        }


@dataclass
class NumericalVerificationResult:
    """Complete numerical verification result."""
    model_name: str
    
    # Overall metrics
    max_absolute_difference: float = 0.0
    mean_absolute_difference: float = 0.0
    max_relative_difference: float = 0.0
    
    # Status
    outputs_match: bool = True
    all_shapes_match: bool = True
    tolerance_category: ToleranceCategory = ToleranceCategory.IDENTICAL
    
    # Per-output comparisons
    output_comparisons: List[OutputComparison] = field(default_factory=list)
    
    # Test configuration
    num_test_samples: int = 1
    tolerance_threshold: float = 1e-4
    
    # Errors
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'max_absolute_difference': float(self.max_absolute_difference),
            'mean_absolute_difference': float(self.mean_absolute_difference),
            'max_relative_difference': float(self.max_relative_difference),
            'outputs_match': self.outputs_match,
            'all_shapes_match': self.all_shapes_match,
            'tolerance_category': self.tolerance_category.value,
            'output_comparisons': [c.to_dict() for c in self.output_comparisons],
            'num_test_samples': self.num_test_samples,
            'tolerance_threshold': self.tolerance_threshold,
            'error_message': self.error_message
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        status = "PASS" if self.outputs_match else "FAIL"
        lines = [
            f"Numerical Verification: {self.model_name}",
            f"=" * 50,
            f"Status: {status}",
            f"Tolerance Category: {self.tolerance_category.value}",
            f"",
            f"Maximum Absolute Difference: {self.max_absolute_difference:.2e}",
            f"Mean Absolute Difference: {self.mean_absolute_difference:.2e}",
            f"Maximum Relative Difference: {self.max_relative_difference:.2e}",
            f"",
            f"Outputs Compared: {len(self.output_comparisons)}",
            f"All Shapes Match: {self.all_shapes_match}",
            f"Test Samples: {self.num_test_samples}",
            f"Tolerance Threshold: {self.tolerance_threshold}"
        ]
        
        if self.error_message:
            lines.append(f"\nError: {self.error_message}")
        
        return "\n".join(lines)


class NumericalVerifier:
    """
    Verify numerical equivalence between ONNX models.
    
    Runs inference on both original and modified models with the same
    inputs and compares outputs with configurable tolerances.
    """
    
    # Tolerance thresholds
    IDENTICAL_THRESHOLD = 1e-10
    CLOSE_THRESHOLD = 1e-6
    ACCEPTABLE_THRESHOLD = 1e-4
    MARGINAL_THRESHOLD = 1e-2
    
    def __init__(
        self,
        tolerance: float = 1e-4,
        num_samples: int = 3,
        seed: int = 42,
        verbose: bool = False
    ):
        """
        Initialize the numerical verifier.
        
        Args:
            tolerance: Default tolerance threshold
            num_samples: Number of test samples to run
            seed: Random seed for reproducibility
            verbose: Enable verbose output
        """
        self.tolerance = tolerance
        self.num_samples = num_samples
        self.seed = seed
        self.verbose = verbose
        
        np.random.seed(seed)
    
    def verify(
        self,
        original_model: Union[str, onnx.ModelProto],
        modified_model: Union[str, onnx.ModelProto],
        test_inputs: Optional[List[Dict[str, np.ndarray]]] = None,
        tolerance: Optional[float] = None
    ) -> NumericalVerificationResult:
        """
        Verify numerical equivalence between two models.
        
        Args:
            original_model: Original model (path or proto)
            modified_model: Modified model (path or proto)
            test_inputs: Optional list of test input dictionaries
            tolerance: Override default tolerance
            
        Returns:
            NumericalVerificationResult
        """
        tolerance = tolerance or self.tolerance
        
        # Load models
        if isinstance(original_model, str):
            original = onnx.load(original_model)
            model_name = Path(original_model).stem
        else:
            original = original_model
            model_name = original.graph.name or "model"
        
        if isinstance(modified_model, str):
            modified = onnx.load(modified_model)
        else:
            modified = modified_model
        
        result = NumericalVerificationResult(
            model_name=model_name,
            tolerance_threshold=tolerance,
            num_test_samples=self.num_samples
        )
        
        try:
            # Create inference sessions
            original_session = ort.InferenceSession(
                original.SerializeToString(),
                providers=['CPUExecutionProvider']
            )
            modified_session = ort.InferenceSession(
                modified.SerializeToString(),
                providers=['CPUExecutionProvider']
            )
            
            # Get input/output info
            original_inputs = {inp.name: inp for inp in original_session.get_inputs()}
            modified_inputs = {inp.name: inp for inp in modified_session.get_inputs()}
            
            # Generate test inputs if not provided
            if test_inputs is None:
                test_inputs = self._generate_test_inputs(
                    original_inputs, self.num_samples
                )
            
            # Run inference and compare
            all_comparisons = []
            
            for sample_idx, inputs in enumerate(test_inputs):
                if self.verbose:
                    print(f"  Running sample {sample_idx + 1}/{len(test_inputs)}...")
                
                # Adjust inputs for modified model if needed
                modified_inputs_data = self._adapt_inputs(
                    inputs, original_inputs, modified_inputs
                )
                
                # Run inference
                original_outputs = original_session.run(None, inputs)
                modified_outputs = modified_session.run(None, modified_inputs_data)
                
                # Get output names
                original_output_names = [o.name for o in original_session.get_outputs()]
                modified_output_names = [o.name for o in modified_session.get_outputs()]
                
                # Compare outputs
                comparisons = self._compare_outputs(
                    original_outputs, modified_outputs,
                    original_output_names, modified_output_names,
                    tolerance
                )
                all_comparisons.extend(comparisons)
            
            # Aggregate results
            self._aggregate_results(result, all_comparisons, tolerance)
            
        except Exception as e:
            result.error_message = str(e)
            result.outputs_match = False
        
        return result
    
    def verify_from_paths(
        self,
        original_path: str,
        modified_path: str,
        tolerance: Optional[float] = None
    ) -> NumericalVerificationResult:
        """Convenience method to verify from file paths."""
        return self.verify(original_path, modified_path, tolerance=tolerance)
    
    def _generate_test_inputs(
        self,
        input_info: Dict[str, Any],
        num_samples: int
    ) -> List[Dict[str, np.ndarray]]:
        """Generate random test inputs based on model input specifications."""
        samples = []
        
        for _ in range(num_samples):
            sample = {}
            for name, info in input_info.items():
                # Get shape (handle dynamic dimensions)
                shape = []
                for dim in info.shape:
                    if isinstance(dim, int) and dim > 0:
                        shape.append(dim)
                    else:
                        # Dynamic dimension - use reasonable default
                        shape.append(1)
                
                # Get dtype
                dtype = self._onnx_type_to_numpy(info.type)
                
                # Generate random data
                if dtype in [np.float32, np.float16, np.float64]:
                    data = np.random.randn(*shape).astype(dtype)
                elif dtype in [np.int32, np.int64]:
                    data = np.random.randint(0, 10, size=shape).astype(dtype)
                else:
                    data = np.random.rand(*shape).astype(dtype)
                
                sample[name] = data
            
            samples.append(sample)
        
        return samples
    
    def _onnx_type_to_numpy(self, onnx_type: str) -> type:
        """Convert ONNX type string to numpy dtype."""
        type_map = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
            'tensor(double)': np.float64,
            'tensor(int32)': np.int32,
            'tensor(int64)': np.int64,
            'tensor(int8)': np.int8,
            'tensor(uint8)': np.uint8,
            'tensor(bool)': np.bool_,
        }
        return type_map.get(onnx_type, np.float32)
    
    def _adapt_inputs(
        self,
        inputs: Dict[str, np.ndarray],
        original_info: Dict[str, Any],
        modified_info: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Adapt inputs if modified model has different input specs."""
        adapted = {}
        
        for name, data in inputs.items():
            if name in modified_info:
                adapted[name] = data
            else:
                # Input name might have changed - try to match by position
                # This is a simplified approach
                adapted[name] = data
        
        return adapted
    
    def _compare_outputs(
        self,
        original_outputs: List[np.ndarray],
        modified_outputs: List[np.ndarray],
        original_names: List[str],
        modified_names: List[str],
        tolerance: float
    ) -> List[OutputComparison]:
        """Compare output tensors."""
        comparisons = []
        
        # Match outputs by position (simpler approach)
        num_outputs = min(len(original_outputs), len(modified_outputs))
        
        for i in range(num_outputs):
            orig = original_outputs[i]
            mod = modified_outputs[i]
            name = original_names[i] if i < len(original_names) else f"output_{i}"
            
            comparison = OutputComparison(
                output_name=name,
                shape_original=orig.shape,
                shape_modified=mod.shape
            )
            
            # Check shape match
            if orig.shape != mod.shape:
                comparison.shapes_match = False
                # Try to compare if total elements match
                if orig.size != mod.size:
                    comparison.tolerance_category = ToleranceCategory.DIVERGENT
                    comparisons.append(comparison)
                    continue
                # Reshape for comparison
                mod = mod.reshape(orig.shape)
            
            # Compute differences
            abs_diff = np.abs(orig.astype(np.float64) - mod.astype(np.float64))
            
            comparison.max_absolute_diff = float(np.max(abs_diff))
            comparison.mean_absolute_diff = float(np.mean(abs_diff))
            comparison.num_elements = orig.size
            comparison.num_different = int(np.sum(abs_diff > tolerance))
            
            # Compute relative difference (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = abs_diff / (np.abs(orig.astype(np.float64)) + 1e-10)
                rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
            
            comparison.max_relative_diff = float(np.max(rel_diff))
            comparison.mean_relative_diff = float(np.mean(rel_diff))
            
            # Categorize
            comparison.tolerance_category = self._categorize_difference(
                comparison.max_absolute_diff
            )
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _categorize_difference(self, max_diff: float) -> ToleranceCategory:
        """Categorize the maximum difference."""
        if max_diff < self.IDENTICAL_THRESHOLD:
            return ToleranceCategory.IDENTICAL
        elif max_diff < self.CLOSE_THRESHOLD:
            return ToleranceCategory.CLOSE
        elif max_diff < self.ACCEPTABLE_THRESHOLD:
            return ToleranceCategory.ACCEPTABLE
        elif max_diff < self.MARGINAL_THRESHOLD:
            return ToleranceCategory.MARGINAL
        else:
            return ToleranceCategory.DIVERGENT
    
    def _aggregate_results(
        self,
        result: NumericalVerificationResult,
        comparisons: List[OutputComparison],
        tolerance: float
    ) -> None:
        """Aggregate comparison results into final result."""
        result.output_comparisons = comparisons
        
        if not comparisons:
            result.outputs_match = False
            result.error_message = "No outputs to compare"
            return
        
        # Compute overall metrics
        result.max_absolute_difference = max(c.max_absolute_diff for c in comparisons)
        result.mean_absolute_difference = np.mean([c.mean_absolute_diff for c in comparisons])
        result.max_relative_difference = max(c.max_relative_diff for c in comparisons)
        
        # Check shapes
        result.all_shapes_match = all(c.shapes_match for c in comparisons)
        
        # Determine overall category
        result.tolerance_category = self._categorize_difference(result.max_absolute_difference)
        
        # Determine pass/fail
        result.outputs_match = (
            result.max_absolute_difference <= tolerance and
            result.all_shapes_match
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def verify_numerical_equivalence(
    original_path: str,
    modified_path: str,
    tolerance: float = 1e-4,
    verbose: bool = False
) -> NumericalVerificationResult:
    """
    Convenience function to verify numerical equivalence.
    
    Args:
        original_path: Path to original model
        modified_path: Path to modified model
        tolerance: Tolerance threshold
        verbose: Enable verbose output
        
    Returns:
        NumericalVerificationResult
    """
    verifier = NumericalVerifier(tolerance=tolerance, verbose=verbose)
    return verifier.verify_from_paths(original_path, modified_path)


def quick_verify(
    original_path: str,
    modified_path: str
) -> bool:
    """Quick verification - returns True if models match within acceptable tolerance."""
    result = verify_numerical_equivalence(original_path, modified_path)
    return result.outputs_match


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python numerical_verifier.py <original.onnx> <modified.onnx> [--tolerance=1e-4]")
        sys.exit(1)
    
    original_path = sys.argv[1]
    modified_path = sys.argv[2]
    
    # Parse tolerance
    tolerance = 1e-4
    for arg in sys.argv[3:]:
        if arg.startswith("--tolerance="):
            tolerance = float(arg.split("=")[1])
    
    verifier = NumericalVerifier(tolerance=tolerance, verbose=True)
    result = verifier.verify_from_paths(original_path, modified_path)
    
    print(result.get_summary())
    
    # Exit with appropriate code
    sys.exit(0 if result.outputs_match else 1)
