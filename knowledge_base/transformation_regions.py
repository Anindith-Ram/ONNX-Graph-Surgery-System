#!/usr/bin/env python3
"""
Transformation Regions Module for ONNX Model Surgery.

Provides data structures and utilities for tracking coherent transformation
regions in ONNX models. Regions represent groups of nodes that should be
transformed together as a unit.

Key Components:
- SubgraphSignature: Identifies subgraph patterns
- TransformationRegion: Tracks a region being transformed
- TransformationPlan: Complete transformation plan
- RegionTracker: Manages regions across transformation process

Author: Automated Model Surgery Pipeline
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import hashlib


class RegionStatus(Enum):
    """Status of a transformation region."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class TransformationResult(Enum):
    """Result of a transformation attempt."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubgraphSignature:
    """
    Signature for identifying subgraph patterns.
    
    Used to match similar structures across different models
    and identify opportunities for reusing transformation strategies.
    """
    signature_id: str
    op_sequence: List[str]  # Ordered list of op types
    op_counts: Dict[str, int]  # Count of each op type
    edge_count: int  # Number of internal edges
    input_count: int  # Number of external inputs
    output_count: int  # Number of outputs
    
    # Pattern characteristics
    has_loop: bool = False
    has_conditional: bool = False
    has_attention: bool = False
    has_normalization: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'signature_id': self.signature_id,
            'op_sequence': self.op_sequence,
            'op_counts': self.op_counts,
            'edge_count': self.edge_count,
            'input_count': self.input_count,
            'output_count': self.output_count,
            'has_loop': self.has_loop,
            'has_conditional': self.has_conditional,
            'has_attention': self.has_attention,
            'has_normalization': self.has_normalization
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SubgraphSignature':
        return cls(**data)
    
    @classmethod
    def compute_from_ops(
        cls,
        op_sequence: List[str],
        input_count: int = 0,
        output_count: int = 1
    ) -> 'SubgraphSignature':
        """Compute signature from operation sequence."""
        op_counts = defaultdict(int)
        for op in op_sequence:
            op_counts[op] += 1
        
        # Generate signature ID from hash
        content = f"{sorted(op_sequence)}_{input_count}_{output_count}"
        sig_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Detect characteristics
        has_attention = any(op in ['Softmax', 'Einsum'] for op in op_sequence)
        has_normalization = any(op in ['LayerNormalization', 'BatchNormalization', 'ReduceMean'] for op in op_sequence)
        has_loop = 'Loop' in op_sequence
        has_conditional = 'If' in op_sequence
        
        return cls(
            signature_id=sig_id,
            op_sequence=op_sequence,
            op_counts=dict(op_counts),
            edge_count=len(op_sequence) - 1,  # Simplified
            input_count=input_count,
            output_count=output_count,
            has_loop=has_loop,
            has_conditional=has_conditional,
            has_attention=has_attention,
            has_normalization=has_normalization
        )
    
    def similarity_score(self, other: 'SubgraphSignature') -> float:
        """Compute similarity score with another signature (0.0 to 1.0)."""
        # Op type overlap
        ops1 = set(self.op_counts.keys())
        ops2 = set(other.op_counts.keys())
        
        if not ops1 or not ops2:
            return 0.0
        
        overlap = len(ops1 & ops2)
        union = len(ops1 | ops2)
        jaccard = overlap / union if union > 0 else 0.0
        
        # Op count similarity
        count_sim = 0.0
        for op in ops1 & ops2:
            c1 = self.op_counts[op]
            c2 = other.op_counts[op]
            count_sim += min(c1, c2) / max(c1, c2)
        count_sim = count_sim / len(ops1 | ops2) if ops1 | ops2 else 0.0
        
        # Characteristic match
        char_match = sum([
            self.has_attention == other.has_attention,
            self.has_normalization == other.has_normalization,
            self.has_loop == other.has_loop,
            self.has_conditional == other.has_conditional
        ]) / 4.0
        
        # Weighted combination
        return 0.4 * jaccard + 0.4 * count_sim + 0.2 * char_match


@dataclass
class TransformationAttempt:
    """Record of a transformation attempt."""
    attempt_id: int
    strategy_used: str
    phases_applied: List[str]
    result: TransformationResult
    error_message: str = ""
    duration_ms: float = 0.0
    nodes_changed: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'attempt_id': self.attempt_id,
            'strategy_used': self.strategy_used,
            'phases_applied': self.phases_applied,
            'result': self.result.value,
            'error_message': self.error_message,
            'duration_ms': self.duration_ms,
            'nodes_changed': self.nodes_changed
        }


@dataclass
class TransformationRegion:
    """
    A coherent region of the model to transform together.
    
    This is the runtime representation of a region, including
    status tracking and transformation history.
    """
    region_id: str
    region_type: str  # "attention_block", "detection_head", etc.
    
    # Structure
    signature: SubgraphSignature
    node_indices: List[int]
    op_types: List[str]
    
    # Context
    original_purpose: str
    architectural_issue: str
    
    # Transformation info
    recommended_strategy: str
    fallback_strategies: List[str] = field(default_factory=list)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    required_by: List[str] = field(default_factory=list)
    
    # Status tracking
    status: RegionStatus = RegionStatus.PENDING
    attempts: List[TransformationAttempt] = field(default_factory=list)
    
    # Results
    nodes_before: int = 0
    nodes_after: int = 0
    blockers_resolved: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'region_id': self.region_id,
            'region_type': self.region_type,
            'signature': self.signature.to_dict(),
            'node_indices': self.node_indices,
            'op_types': self.op_types,
            'original_purpose': self.original_purpose,
            'architectural_issue': self.architectural_issue,
            'recommended_strategy': self.recommended_strategy,
            'fallback_strategies': self.fallback_strategies,
            'depends_on': self.depends_on,
            'required_by': self.required_by,
            'status': self.status.value,
            'attempts': [a.to_dict() for a in self.attempts],
            'nodes_before': self.nodes_before,
            'nodes_after': self.nodes_after,
            'blockers_resolved': self.blockers_resolved
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransformationRegion':
        signature = SubgraphSignature.from_dict(data['signature'])
        attempts = []
        for a in data.get('attempts', []):
            attempt = TransformationAttempt(
                attempt_id=a['attempt_id'],
                strategy_used=a['strategy_used'],
                phases_applied=a['phases_applied'],
                result=TransformationResult(a['result']),
                error_message=a.get('error_message', ''),
                duration_ms=a.get('duration_ms', 0.0),
                nodes_changed=a.get('nodes_changed', 0)
            )
            attempts.append(attempt)
        
        return cls(
            region_id=data['region_id'],
            region_type=data['region_type'],
            signature=signature,
            node_indices=data['node_indices'],
            op_types=data['op_types'],
            original_purpose=data['original_purpose'],
            architectural_issue=data['architectural_issue'],
            recommended_strategy=data['recommended_strategy'],
            fallback_strategies=data.get('fallback_strategies', []),
            depends_on=data.get('depends_on', []),
            required_by=data.get('required_by', []),
            status=RegionStatus(data.get('status', 'pending')),
            attempts=attempts,
            nodes_before=data.get('nodes_before', 0),
            nodes_after=data.get('nodes_after', 0),
            blockers_resolved=data.get('blockers_resolved', 0)
        )
    
    def record_attempt(
        self,
        strategy: str,
        phases: List[str],
        result: TransformationResult,
        error: str = "",
        duration: float = 0.0,
        nodes_changed: int = 0
    ) -> None:
        """Record a transformation attempt."""
        attempt = TransformationAttempt(
            attempt_id=len(self.attempts) + 1,
            strategy_used=strategy,
            phases_applied=phases,
            result=result,
            error_message=error,
            duration_ms=duration,
            nodes_changed=nodes_changed
        )
        self.attempts.append(attempt)
        
        # Update status
        if result == TransformationResult.SUCCESS:
            self.status = RegionStatus.COMPLETED
        elif result == TransformationResult.PARTIAL_SUCCESS:
            self.status = RegionStatus.IN_PROGRESS
        elif result == TransformationResult.FAILED:
            self.status = RegionStatus.FAILED
    
    @property
    def last_attempt(self) -> Optional[TransformationAttempt]:
        """Get the last transformation attempt."""
        return self.attempts[-1] if self.attempts else None
    
    @property
    def success_rate(self) -> float:
        """Compute success rate across attempts."""
        if not self.attempts:
            return 0.0
        successful = sum(
            1 for a in self.attempts 
            if a.result in [TransformationResult.SUCCESS, TransformationResult.PARTIAL_SUCCESS]
        )
        return successful / len(self.attempts)


@dataclass
class ValidationCheckpoint:
    """Checkpoint for validating transformation progress."""
    checkpoint_id: str
    name: str
    after_region: str  # Region ID
    validation_type: str  # "shape", "numerical", "compilation"
    expected_result: str
    tolerance: float = 1e-6
    
    # Result
    passed: Optional[bool] = None
    actual_result: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'checkpoint_id': self.checkpoint_id,
            'name': self.name,
            'after_region': self.after_region,
            'validation_type': self.validation_type,
            'expected_result': self.expected_result,
            'tolerance': self.tolerance,
            'passed': self.passed,
            'actual_result': self.actual_result
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ValidationCheckpoint':
        return cls(**data)


@dataclass
class TransformationPlan:
    """
    Complete transformation plan for a model.
    
    This is the runtime plan that tracks the state of all transformations,
    including which regions have been transformed and their results.
    """
    plan_id: str
    model_name: str
    model_path: str
    architecture_type: str
    
    # Regions
    regions: List[TransformationRegion] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    
    # Strategy
    primary_strategy: str = ""
    fallback_strategy: str = ""
    
    # Checkpoints
    checkpoints: List[ValidationCheckpoint] = field(default_factory=list)
    
    # Status
    current_region_idx: int = 0
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    
    # Results
    total_blockers_before: int = 0
    total_blockers_after: int = 0
    transformation_success_rate: float = 0.0
    
    # Timestamps
    started_at: str = ""
    completed_at: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'plan_id': self.plan_id,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'architecture_type': self.architecture_type,
            'regions': [r.to_dict() for r in self.regions],
            'execution_order': self.execution_order,
            'primary_strategy': self.primary_strategy,
            'fallback_strategy': self.fallback_strategy,
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'current_region_idx': self.current_region_idx,
            'status': self.status,
            'total_blockers_before': self.total_blockers_before,
            'total_blockers_after': self.total_blockers_after,
            'transformation_success_rate': self.transformation_success_rate,
            'started_at': self.started_at,
            'completed_at': self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransformationPlan':
        regions = [TransformationRegion.from_dict(r) for r in data.get('regions', [])]
        checkpoints = [ValidationCheckpoint.from_dict(c) for c in data.get('checkpoints', [])]
        
        return cls(
            plan_id=data['plan_id'],
            model_name=data['model_name'],
            model_path=data.get('model_path', ''),
            architecture_type=data['architecture_type'],
            regions=regions,
            execution_order=data.get('execution_order', []),
            primary_strategy=data.get('primary_strategy', ''),
            fallback_strategy=data.get('fallback_strategy', ''),
            checkpoints=checkpoints,
            current_region_idx=data.get('current_region_idx', 0),
            status=data.get('status', 'pending'),
            total_blockers_before=data.get('total_blockers_before', 0),
            total_blockers_after=data.get('total_blockers_after', 0),
            transformation_success_rate=data.get('transformation_success_rate', 0.0),
            started_at=data.get('started_at', ''),
            completed_at=data.get('completed_at', '')
        )
    
    def get_region(self, region_id: str) -> Optional[TransformationRegion]:
        """Get region by ID."""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None
    
    def get_current_region(self) -> Optional[TransformationRegion]:
        """Get the current region to transform."""
        if self.current_region_idx < len(self.execution_order):
            region_id = self.execution_order[self.current_region_idx]
            return self.get_region(region_id)
        return None
    
    def advance_to_next_region(self) -> bool:
        """Advance to the next region. Returns False if no more regions."""
        self.current_region_idx += 1
        return self.current_region_idx < len(self.execution_order)
    
    def get_pending_regions(self) -> List[TransformationRegion]:
        """Get all regions that haven't been transformed yet."""
        return [r for r in self.regions if r.status == RegionStatus.PENDING]
    
    def get_completed_regions(self) -> List[TransformationRegion]:
        """Get all successfully transformed regions."""
        return [r for r in self.regions if r.status == RegionStatus.COMPLETED]
    
    def get_failed_regions(self) -> List[TransformationRegion]:
        """Get all regions that failed to transform."""
        return [r for r in self.regions if r.status == RegionStatus.FAILED]
    
    def compute_success_rate(self) -> float:
        """Compute overall transformation success rate."""
        if not self.regions:
            return 0.0
        completed = len(self.get_completed_regions())
        return completed / len(self.regions)
    
    def save(self, path: str) -> None:
        """Save plan to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TransformationPlan':
        """Load plan from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_summary(self) -> str:
        """Get human-readable summary of plan status."""
        completed = len(self.get_completed_regions())
        failed = len(self.get_failed_regions())
        pending = len(self.get_pending_regions())
        
        lines = [
            f"Transformation Plan: {self.plan_id}",
            f"=" * 50,
            f"Model: {self.model_name}",
            f"Architecture: {self.architecture_type}",
            f"Strategy: {self.primary_strategy}",
            f"Status: {self.status}",
            f"",
            f"Regions:",
            f"  Completed: {completed}",
            f"  Failed: {failed}",
            f"  Pending: {pending}",
            f"  Total: {len(self.regions)}",
            f"",
            f"Progress: {self.current_region_idx}/{len(self.execution_order)}",
            f"Success Rate: {self.compute_success_rate():.1%}",
        ]
        
        if self.total_blockers_before > 0:
            lines.append(f"Blockers: {self.total_blockers_after}/{self.total_blockers_before} remaining")
        
        return "\n".join(lines)


class RegionTracker:
    """
    Tracks transformation regions across the transformation process.
    
    Provides utilities for:
    - Creating regions from model analysis
    - Tracking region transformations
    - Finding similar regions across models
    - Learning from transformation outcomes
    """
    
    def __init__(self):
        """Initialize the region tracker."""
        self.region_history: List[TransformationRegion] = []
        self.signature_index: Dict[str, List[str]] = defaultdict(list)  # sig_id -> region_ids
    
    def add_region(self, region: TransformationRegion) -> None:
        """Add a region to the tracker."""
        self.region_history.append(region)
        self.signature_index[region.signature.signature_id].append(region.region_id)
    
    def find_similar_regions(
        self,
        signature: SubgraphSignature,
        min_similarity: float = 0.7
    ) -> List[TransformationRegion]:
        """Find regions with similar signatures."""
        similar = []
        
        for region in self.region_history:
            sim = signature.similarity_score(region.signature)
            if sim >= min_similarity:
                similar.append((region, sim))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in similar]
    
    def get_successful_strategies_for_signature(
        self,
        signature: SubgraphSignature
    ) -> List[Tuple[str, float]]:
        """
        Get strategies that worked for similar signatures.
        
        Returns: List of (strategy_id, success_rate) tuples
        """
        similar = self.find_similar_regions(signature)
        
        strategy_stats = defaultdict(lambda: {'success': 0, 'total': 0})
        
        for region in similar:
            for attempt in region.attempts:
                strategy = attempt.strategy_used
                strategy_stats[strategy]['total'] += 1
                if attempt.result in [TransformationResult.SUCCESS, TransformationResult.PARTIAL_SUCCESS]:
                    strategy_stats[strategy]['success'] += 1
        
        results = []
        for strategy, stats in strategy_stats.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                results.append((strategy, success_rate))
        
        # Sort by success rate
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def save(self, path: str) -> None:
        """Save tracker state to file."""
        data = {
            'regions': [r.to_dict() for r in self.region_history],
            'signature_index': dict(self.signature_index)
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RegionTracker':
        """Load tracker state from file."""
        tracker = cls()
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            for region_data in data.get('regions', []):
                region = TransformationRegion.from_dict(region_data)
                tracker.region_history.append(region)
            
            tracker.signature_index = defaultdict(list, data.get('signature_index', {}))
        
        return tracker
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked regions."""
        total = len(self.region_history)
        if total == 0:
            return {'total_regions': 0}
        
        completed = sum(1 for r in self.region_history if r.status == RegionStatus.COMPLETED)
        failed = sum(1 for r in self.region_history if r.status == RegionStatus.FAILED)
        
        # Region types
        type_counts = defaultdict(int)
        for r in self.region_history:
            type_counts[r.region_type] += 1
        
        # Strategy success rates
        strategy_stats = defaultdict(lambda: {'success': 0, 'total': 0})
        for region in self.region_history:
            for attempt in region.attempts:
                strategy = attempt.strategy_used
                strategy_stats[strategy]['total'] += 1
                if attempt.result == TransformationResult.SUCCESS:
                    strategy_stats[strategy]['success'] += 1
        
        return {
            'total_regions': total,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total if total > 0 else 0.0,
            'region_types': dict(type_counts),
            'unique_signatures': len(self.signature_index),
            'strategy_success_rates': {
                s: stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
                for s, stats in strategy_stats.items()
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_region_from_nodes(
    region_id: str,
    region_type: str,
    node_indices: List[int],
    op_types: List[str],
    purpose: str,
    issue: str,
    strategy: str
) -> TransformationRegion:
    """Create a transformation region from node information."""
    signature = SubgraphSignature.compute_from_ops(op_types)
    
    return TransformationRegion(
        region_id=region_id,
        region_type=region_type,
        signature=signature,
        node_indices=node_indices,
        op_types=op_types,
        original_purpose=purpose,
        architectural_issue=issue,
        recommended_strategy=strategy
    )


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test signature computation
    sig1 = SubgraphSignature.compute_from_ops(
        ["MatMul", "Softmax", "MatMul"],
        input_count=3,
        output_count=1
    )
    print(f"Signature 1: {sig1.signature_id}")
    print(f"  Has attention: {sig1.has_attention}")
    
    sig2 = SubgraphSignature.compute_from_ops(
        ["Einsum", "Softmax"],
        input_count=2,
        output_count=1
    )
    print(f"\nSignature 2: {sig2.signature_id}")
    print(f"  Similarity with sig1: {sig1.similarity_score(sig2):.2f}")
    
    # Test region creation
    region = create_region_from_nodes(
        region_id="test_region_1",
        region_type="attention_block",
        node_indices=[10, 11, 12, 13],
        op_types=["MatMul", "Softmax", "MatMul"],
        purpose="Self-attention computation",
        issue="Einsum not supported on MLA",
        strategy="transformer_einsum_decomposition"
    )
    print(f"\nRegion: {region.region_id}")
    print(f"  Type: {region.region_type}")
    print(f"  Status: {region.status.value}")
    
    # Test tracker
    tracker = RegionTracker()
    tracker.add_region(region)
    
    stats = tracker.get_statistics()
    print(f"\nTracker Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
