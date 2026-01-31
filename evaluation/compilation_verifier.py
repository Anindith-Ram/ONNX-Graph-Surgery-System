#!/usr/bin/env python3
"""
Compilation Verification Module for ONNX Model Surgery.

Verifies that modified models will compile successfully to a single LM file
on the SiMa MLA. Checks node mappings, blocker resolution, and predicts
compilation outcomes.

Key Features:
- Simulate MLA compilation check
- Count predicted LM file segments
- Verify all nodes mapped to MLA
- Generate compilation report

Author: Automated Model Surgery Pipeline
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import onnx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_analysis.compilation_simulator import (
    CompilationSimulator, CompilationReport, ProcessorTarget
)


class CompilationStatus(Enum):
    """Overall compilation status."""
    WILL_COMPILE = "will_compile"          # Single LM file, all MLA
    PARTIAL_COMPILE = "partial_compile"    # Multiple LM files or some CVU/APU
    WILL_NOT_COMPILE = "will_not_compile"  # Blockers remain
    UNKNOWN = "unknown"                     # Could not determine


@dataclass
class NodeMappingStats:
    """Statistics about node processor mappings."""
    total_nodes: int = 0
    nodes_on_mla: int = 0
    nodes_on_cvu: int = 0
    nodes_on_apu: int = 0
    
    @property
    def mla_percentage(self) -> float:
        return self.nodes_on_mla / self.total_nodes * 100 if self.total_nodes > 0 else 0.0
    
    @property
    def all_on_mla(self) -> bool:
        return self.nodes_on_mla == self.total_nodes and self.total_nodes > 0
    
    def to_dict(self) -> Dict:
        return {
            'total_nodes': self.total_nodes,
            'nodes_on_mla': self.nodes_on_mla,
            'nodes_on_cvu': self.nodes_on_cvu,
            'nodes_on_apu': self.nodes_on_apu,
            'mla_percentage': self.mla_percentage,
            'all_on_mla': self.all_on_mla
        }


@dataclass
class BlockerResolutionStats:
    """Statistics about blocker resolution."""
    original_blockers: int = 0
    remaining_blockers: int = 0
    resolved_blockers: int = 0
    
    # By operation type
    blockers_by_op: Dict[str, int] = field(default_factory=dict)
    
    @property
    def resolution_rate(self) -> float:
        return self.resolved_blockers / self.original_blockers if self.original_blockers > 0 else 1.0
    
    @property
    def all_resolved(self) -> bool:
        return self.remaining_blockers == 0
    
    def to_dict(self) -> Dict:
        return {
            'original_blockers': self.original_blockers,
            'remaining_blockers': self.remaining_blockers,
            'resolved_blockers': self.resolved_blockers,
            'resolution_rate': self.resolution_rate,
            'all_resolved': self.all_resolved,
            'blockers_by_op': self.blockers_by_op
        }


@dataclass
class CompilationVerificationResult:
    """Complete compilation verification result."""
    model_name: str
    
    # Overall status
    status: CompilationStatus = CompilationStatus.UNKNOWN
    all_nodes_mla_compatible: bool = False
    single_lm_file: bool = False
    
    # Predicted outputs
    predicted_lm_files: int = 1
    lm_file_reasons: List[str] = field(default_factory=list)
    
    # Node mapping
    node_mapping: NodeMappingStats = field(default_factory=NodeMappingStats)
    
    # Blocker resolution
    blocker_stats: BlockerResolutionStats = field(default_factory=BlockerResolutionStats)
    
    # Unmapped nodes
    unmapped_nodes: List[str] = field(default_factory=list)
    blockers_remaining: List[str] = field(default_factory=list)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    # Raw simulation data
    simulation_report: Optional[CompilationReport] = None
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'status': self.status.value,
            'all_nodes_mla_compatible': self.all_nodes_mla_compatible,
            'single_lm_file': self.single_lm_file,
            'predicted_lm_files': self.predicted_lm_files,
            'lm_file_reasons': self.lm_file_reasons,
            'node_mapping': self.node_mapping.to_dict(),
            'blocker_stats': self.blocker_stats.to_dict(),
            'unmapped_nodes': self.unmapped_nodes,
            'blockers_remaining': self.blockers_remaining,
            'warnings': self.warnings
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        status_emoji = {
            CompilationStatus.WILL_COMPILE: "PASS",
            CompilationStatus.PARTIAL_COMPILE: "PARTIAL",
            CompilationStatus.WILL_NOT_COMPILE: "FAIL",
            CompilationStatus.UNKNOWN: "UNKNOWN"
        }
        
        lines = [
            f"Compilation Verification: {self.model_name}",
            f"=" * 50,
            f"Status: {status_emoji.get(self.status, 'UNKNOWN')}",
            f"",
            f"Node Mapping:",
            f"  Total Nodes: {self.node_mapping.total_nodes}",
            f"  MLA: {self.node_mapping.nodes_on_mla} ({self.node_mapping.mla_percentage:.1f}%)",
            f"  CVU: {self.node_mapping.nodes_on_cvu}",
            f"  APU: {self.node_mapping.nodes_on_apu}",
            f"",
            f"Predicted LM Files: {self.predicted_lm_files}",
            f"Single LM File: {self.single_lm_file}",
        ]
        
        if self.blocker_stats.original_blockers > 0:
            lines.extend([
                f"",
                f"Blocker Resolution:",
                f"  Original: {self.blocker_stats.original_blockers}",
                f"  Remaining: {self.blocker_stats.remaining_blockers}",
                f"  Resolution Rate: {self.blocker_stats.resolution_rate:.1%}"
            ])
        
        if self.blockers_remaining:
            lines.append(f"\nRemaining Blockers: {', '.join(self.blockers_remaining[:5])}")
            if len(self.blockers_remaining) > 5:
                lines.append(f"  ... and {len(self.blockers_remaining) - 5} more")
        
        if self.warnings:
            lines.append(f"\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        return "\n".join(lines)


class CompilationVerifier:
    """
    Verify compilation compatibility for ONNX models.
    
    Uses the compilation simulator to check if models will compile
    to MLA and predicts the number of LM files.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the compilation verifier.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.simulator = CompilationSimulator(verbose=verbose)
    
    def verify(
        self,
        model_path: str,
        original_model_path: Optional[str] = None
    ) -> CompilationVerificationResult:
        """
        Verify compilation compatibility of a model.
        
        Args:
            model_path: Path to model to verify
            original_model_path: Optional path to original model for comparison
            
        Returns:
            CompilationVerificationResult
        """
        model_name = Path(model_path).stem
        result = CompilationVerificationResult(model_name=model_name)
        
        try:
            # Run simulation on modified model
            sim_result = self.simulator.simulate(model_path)
            result.simulation_report = sim_result
            
            # Extract node mapping stats
            result.node_mapping = NodeMappingStats(
                total_nodes=sim_result.total_nodes,
                nodes_on_mla=sim_result.nodes_on_mla,
                nodes_on_cvu=sim_result.nodes_on_cvu,
                nodes_on_apu=sim_result.nodes_on_apu
            )
            
            # Extract blocker stats
            result.blocker_stats = BlockerResolutionStats(
                remaining_blockers=sim_result.blocker_count,
                blockers_by_op=sim_result.blocker_ops
            )
            
            # Get original blockers if original model provided
            if original_model_path:
                orig_sim = self.simulator.simulate(original_model_path)
                result.blocker_stats.original_blockers = orig_sim.blocker_count
                result.blocker_stats.resolved_blockers = (
                    orig_sim.blocker_count - sim_result.blocker_count
                )
            else:
                result.blocker_stats.original_blockers = sim_result.blocker_count
            
            # Predicted LM files
            result.predicted_lm_files = sim_result.predicted_lm_files
            result.lm_file_reasons = sim_result.lm_file_reasons
            result.single_lm_file = sim_result.predicted_lm_files == 1
            
            # Check MLA compatibility
            result.all_nodes_mla_compatible = (
                sim_result.blocker_count == 0 and
                sim_result.nodes_on_apu == 0
            )
            
            # Get remaining blockers
            result.blockers_remaining = [
                f"{n.op_type}:{n.node_name}" for n in sim_result.blocker_nodes
            ]
            
            # Determine overall status
            if result.all_nodes_mla_compatible and result.single_lm_file:
                result.status = CompilationStatus.WILL_COMPILE
            elif sim_result.blocker_count > 0:
                result.status = CompilationStatus.WILL_NOT_COMPILE
            else:
                result.status = CompilationStatus.PARTIAL_COMPILE
            
            # Generate warnings
            if sim_result.nodes_on_cvu > 0:
                result.warnings.append(
                    f"{sim_result.nodes_on_cvu} nodes mapped to CVU"
                )
            if not result.single_lm_file:
                result.warnings.append(
                    f"Model will generate {result.predicted_lm_files} LM files"
                )
            
        except Exception as e:
            result.status = CompilationStatus.UNKNOWN
            result.warnings.append(f"Verification error: {e}")
        
        return result
    
    def verify_model(
        self,
        model: onnx.ModelProto,
        model_name: str = "model"
    ) -> CompilationVerificationResult:
        """
        Verify compilation compatibility from model proto.
        
        Args:
            model: ONNX model proto
            model_name: Name for the model
            
        Returns:
            CompilationVerificationResult
        """
        import tempfile
        import os
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx.save(model, f.name)
            temp_path = f.name
        
        try:
            result = self.verify(temp_path)
            result.model_name = model_name
            return result
        finally:
            os.unlink(temp_path)
    
    def compare_compilation(
        self,
        original_path: str,
        modified_path: str
    ) -> Tuple[CompilationVerificationResult, CompilationVerificationResult, Dict]:
        """
        Compare compilation status between original and modified models.
        
        Args:
            original_path: Path to original model
            modified_path: Path to modified model
            
        Returns:
            Tuple of (original_result, modified_result, comparison_dict)
        """
        original_result = self.verify(original_path)
        modified_result = self.verify(modified_path, original_path)
        
        # Build comparison
        comparison = {
            'blockers_resolved': (
                original_result.blocker_stats.remaining_blockers -
                modified_result.blocker_stats.remaining_blockers
            ),
            'mla_nodes_gained': (
                modified_result.node_mapping.nodes_on_mla -
                original_result.node_mapping.nodes_on_mla
            ),
            'lm_files_change': (
                modified_result.predicted_lm_files -
                original_result.predicted_lm_files
            ),
            'compilation_improved': (
                modified_result.status.value != CompilationStatus.WILL_NOT_COMPILE.value and
                (original_result.status == CompilationStatus.WILL_NOT_COMPILE or
                 modified_result.blocker_stats.remaining_blockers < 
                 original_result.blocker_stats.remaining_blockers)
            ),
            'will_compile': modified_result.status == CompilationStatus.WILL_COMPILE
        }
        
        return original_result, modified_result, comparison


# =============================================================================
# Convenience Functions
# =============================================================================

def verify_compilation(
    model_path: str,
    verbose: bool = False
) -> CompilationVerificationResult:
    """Convenience function to verify compilation."""
    verifier = CompilationVerifier(verbose=verbose)
    return verifier.verify(model_path)


def will_compile(model_path: str) -> bool:
    """Quick check if model will compile to single LM file."""
    result = verify_compilation(model_path)
    return result.status == CompilationStatus.WILL_COMPILE


def get_blocker_count(model_path: str) -> int:
    """Get count of remaining blockers."""
    result = verify_compilation(model_path)
    return result.blocker_stats.remaining_blockers


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compilation_verifier.py <model.onnx> [original.onnx]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    original_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    verifier = CompilationVerifier(verbose=True)
    result = verifier.verify(model_path, original_path)
    
    print(result.get_summary())
    
    # Exit with appropriate code
    sys.exit(0 if result.status == CompilationStatus.WILL_COMPILE else 1)
