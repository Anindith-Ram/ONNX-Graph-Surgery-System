#!/usr/bin/env python3
"""
Unit tests for agents/diagnostics.py

Tests the diagnostic data structures:
- ErrorCategory
- GraphSnapshot
- TransformationDelta
- TransformationResult
- FeedbackCollector
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.diagnostics import (
    ErrorCategory,
    GraphSnapshot,
    TransformationDelta,
    TransformationResult,
    FeedbackCollector,
)


class TestErrorCategory(unittest.TestCase):
    """Tests for ErrorCategory enum."""
    
    def test_error_category_values(self):
        """Test error category enum values."""
        self.assertEqual(ErrorCategory.NODE_NOT_FOUND.value, "node_not_found")
        self.assertEqual(ErrorCategory.SHAPE_MISMATCH.value, "shape_mismatch")
        self.assertEqual(ErrorCategory.HANDLER_NOT_IMPLEMENTED.value, "no_handler")
    
    def test_from_exception_node_not_found(self):
        """Test categorizing 'not found' exceptions."""
        e = Exception("Node xyz not found in graph")
        category = ErrorCategory.from_exception(e)
        self.assertEqual(category, ErrorCategory.NODE_NOT_FOUND)
    
    def test_from_exception_shape_mismatch(self):
        """Test categorizing shape mismatch exceptions."""
        e = Exception("Shape mismatch: expected [1,2,3] got [1,2,4]")
        category = ErrorCategory.from_exception(e)
        self.assertEqual(category, ErrorCategory.SHAPE_MISMATCH)
    
    def test_from_exception_unknown(self):
        """Test categorizing unknown exceptions."""
        e = Exception("Some random error")
        category = ErrorCategory.from_exception(e)
        self.assertEqual(category, ErrorCategory.UNKNOWN)


class TestGraphSnapshot(unittest.TestCase):
    """Tests for GraphSnapshot class."""
    
    def test_snapshot_creation(self):
        """Test manual snapshot creation."""
        snapshot = GraphSnapshot(
            node_count=10,
            op_type_counts={"Conv": 3, "Relu": 2},
            node_names={"conv1", "conv2", "relu1"},
            input_names={"input"},
            output_names={"output"},
            edge_count=15,
            initializer_count=5,
        )
        
        self.assertEqual(snapshot.node_count, 10)
        self.assertEqual(snapshot.edge_count, 15)
        self.assertEqual(snapshot.op_type_counts["Conv"], 3)
    
    def test_to_dict(self):
        """Test snapshot serialization."""
        snapshot = GraphSnapshot(
            node_count=5,
            op_type_counts={"MatMul": 2},
            node_names={"mm1", "mm2"},
            input_names={"x"},
            output_names={"y"},
            edge_count=8,
            initializer_count=2,
        )
        
        d = snapshot.to_dict()
        
        self.assertIn("node_count", d)
        self.assertIn("op_type_counts", d)
        self.assertIn("timestamp", d)
        self.assertEqual(d["node_count"], 5)


class TestTransformationDelta(unittest.TestCase):
    """Tests for TransformationDelta class."""
    
    def test_compute_no_change(self):
        """Test delta when no transformation occurred."""
        before = GraphSnapshot(
            node_count=10,
            op_type_counts={"Conv": 3},
            node_names={"a", "b", "c"},
            input_names=set(),
            output_names=set(),
            edge_count=15,
            initializer_count=5,
        )
        
        delta = TransformationDelta.compute(before, None)
        
        self.assertFalse(delta.has_changes)
        self.assertEqual(len(delta.nodes_added), 0)
        self.assertEqual(len(delta.nodes_removed), 0)
    
    def test_compute_with_changes(self):
        """Test delta when changes occurred."""
        before = GraphSnapshot(
            node_count=10,
            op_type_counts={"Conv": 3, "Relu": 2},
            node_names={"conv1", "conv2", "conv3", "relu1", "relu2"},
            input_names=set(),
            output_names=set(),
            edge_count=15,
            initializer_count=5,
        )
        
        after = GraphSnapshot(
            node_count=9,
            op_type_counts={"Conv": 3, "Relu": 1},  # One Relu removed
            node_names={"conv1", "conv2", "conv3", "relu1"},  # relu2 removed
            input_names=set(),
            output_names=set(),
            edge_count=13,
            initializer_count=5,
        )
        
        delta = TransformationDelta.compute(before, after)
        
        self.assertTrue(delta.has_changes)
        self.assertIn("relu2", delta.nodes_removed)
        self.assertEqual(delta.ops_removed.get("Relu", 0), 1)
        self.assertEqual(delta.node_count_delta, -1)
    
    def test_str_representation(self):
        """Test string representation of delta."""
        delta = TransformationDelta(
            nodes_added=set(),
            nodes_removed={"node1"},
            ops_added={},
            ops_removed={"Identity": 1},
            node_count_delta=-1,
            edge_count_delta=-2,
            initializer_delta=0,
        )
        
        s = str(delta)
        self.assertIn("removed", s)
        self.assertIn("Identity", s)


class TestTransformationResult(unittest.TestCase):
    """Tests for TransformationResult class."""
    
    def test_successful_result(self):
        """Test creating a successful result."""
        result = TransformationResult(
            suggestion_id=1,
            node_name="identity_0",
            op_type="Identity",
            action_type="remove",
            success=True,
            was_transformed=True,
            duration_ms=50.5,
        )
        
        self.assertTrue(result.success)
        self.assertTrue(result.was_transformed)
        self.assertTrue(result.effective)
    
    def test_failed_result(self):
        """Test creating a failed result."""
        result = TransformationResult(
            suggestion_id=2,
            node_name="einsum_0",
            op_type="Einsum",
            action_type="replace",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.HANDLER_NOT_IMPLEMENTED,
            error_message="No handler for complex Einsum",
        )
        
        self.assertFalse(result.success)
        self.assertFalse(result.effective)
        self.assertEqual(result.error_category, ErrorCategory.HANDLER_NOT_IMPLEMENTED)
    
    def test_to_observation_string(self):
        """Test observation string format."""
        result = TransformationResult(
            suggestion_id=1,
            node_name="test_node",
            op_type="Conv",
            action_type="remove",
            success=True,
            was_transformed=True,
            duration_ms=25.0,
        )
        
        obs = result.to_observation_string()
        
        self.assertIn("Suggestion 1", obs)
        self.assertIn("SUCCESS", obs)
        self.assertIn("test_node", obs)
    
    def test_retry_count_tracking(self):
        """Test that retry_count is properly tracked."""
        # Result with no retries
        result1 = TransformationResult(
            suggestion_id=1,
            node_name="node1",
            op_type="Identity",
            action_type="remove",
            success=True,
            was_transformed=True,
            retry_count=0,
        )
        self.assertEqual(result1.retry_count, 0)
        
        # Result after retries
        result2 = TransformationResult(
            suggestion_id=2,
            node_name="node2",
            op_type="Einsum",
            action_type="replace",
            success=True,
            was_transformed=True,
            retry_count=2,
        )
        self.assertEqual(result2.retry_count, 2)
        
        # Failed result with retries exhausted
        result3 = TransformationResult(
            suggestion_id=3,
            node_name="node3",
            op_type="Custom",
            action_type="transform",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.HANDLER_FAILED,
            retry_count=2,
        )
        self.assertEqual(result3.retry_count, 2)
        self.assertFalse(result3.success)


class TestFeedbackCollector(unittest.TestCase):
    """Tests for FeedbackCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = FeedbackCollector(model_name="test_model")
    
    def test_empty_collector(self):
        """Test empty collector properties."""
        self.assertEqual(self.collector.attempted_count, 0)
        self.assertEqual(self.collector.success_rate(), 0.0)
    
    def test_add_results(self):
        """Test adding results."""
        result1 = TransformationResult(
            suggestion_id=1,
            node_name="node1",
            op_type="Identity",
            action_type="remove",
            success=True,
            was_transformed=True,
        )
        result2 = TransformationResult(
            suggestion_id=2,
            node_name="node2",
            op_type="Einsum",
            action_type="replace",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.HANDLER_FAILED,
        )
        
        self.collector.add(result1)
        self.collector.add(result2)
        
        self.assertEqual(self.collector.attempted_count, 2)
        self.assertEqual(self.collector.applied_count, 1)
        self.assertEqual(self.collector.failed_count, 1)
        self.assertEqual(self.collector.success_rate(), 0.5)
    
    def test_error_distribution(self):
        """Test error distribution calculation."""
        result1 = TransformationResult(
            suggestion_id=1,
            node_name="node1",
            op_type="A",
            action_type="remove",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.NODE_NOT_FOUND,
        )
        result2 = TransformationResult(
            suggestion_id=2,
            node_name="node2",
            op_type="B",
            action_type="remove",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.NODE_NOT_FOUND,
        )
        result3 = TransformationResult(
            suggestion_id=3,
            node_name="node3",
            op_type="C",
            action_type="replace",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.HANDLER_NOT_IMPLEMENTED,
        )
        
        self.collector.add(result1)
        self.collector.add(result2)
        self.collector.add(result3)
        
        dist = self.collector.error_distribution()
        
        self.assertEqual(dist[ErrorCategory.NODE_NOT_FOUND], 2)
        self.assertEqual(dist[ErrorCategory.HANDLER_NOT_IMPLEMENTED], 1)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with SuggestionApplicator interface."""
        result = TransformationResult(
            suggestion_id=1,
            node_name="node",
            op_type="Op",
            action_type="remove",
            success=True,
            was_transformed=True,
        )
        self.collector.add(result)
        
        # These should match the old SuggestionApplicator counters
        self.assertEqual(self.collector.applied_count, 1)
        self.assertEqual(self.collector.transformed_count, 1)
        self.assertEqual(self.collector.failed_count, 0)
        self.assertEqual(self.collector.skipped_count, 0)
    
    def test_get_summary(self):
        """Test get_summary method."""
        result = TransformationResult(
            suggestion_id=1,
            node_name="node",
            op_type="Op",
            action_type="remove",
            success=True,
            was_transformed=True,
        )
        self.collector.add(result)
        
        summary = self.collector.get_summary()
        
        self.assertIn("applied", summary)
        self.assertIn("failed", summary)
        self.assertIn("success_rate", summary)
    
    def test_to_observation_string(self):
        """Test observation string format for ReAct."""
        result = TransformationResult(
            suggestion_id=1,
            node_name="node",
            op_type="Op",
            action_type="remove",
            success=True,
            was_transformed=True,
        )
        self.collector.add(result)
        
        obs = self.collector.to_observation_string()
        
        self.assertIn("Transformation Session Feedback", obs)
        self.assertIn("Success Rate", obs)
        self.assertIn("test_model", obs)
    
    def test_reset(self):
        """Test collector reset."""
        result = TransformationResult(
            suggestion_id=1,
            node_name="node",
            op_type="Op",
            action_type="remove",
            success=True,
            was_transformed=True,
        )
        self.collector.add(result)
        
        self.assertEqual(self.collector.attempted_count, 1)
        
        self.collector.reset()
        
        self.assertEqual(self.collector.attempted_count, 0)
    
    def test_suggestions_needing_retry(self):
        """Test identification of suggestions that need retry."""
        # Retryable error: HANDLER_FAILED
        result1 = TransformationResult(
            suggestion_id=1,
            node_name="node1",
            op_type="A",
            action_type="remove",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.HANDLER_FAILED,
            retry_count=0,
        )
        # Retryable error: TIMEOUT
        result2 = TransformationResult(
            suggestion_id=2,
            node_name="node2",
            op_type="B",
            action_type="replace",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.TIMEOUT,
            retry_count=1,
        )
        # Non-retryable error: NODE_NOT_FOUND
        result3 = TransformationResult(
            suggestion_id=3,
            node_name="node3",
            op_type="C",
            action_type="remove",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.NODE_NOT_FOUND,
            retry_count=0,
        )
        # Success - should not need retry
        result4 = TransformationResult(
            suggestion_id=4,
            node_name="node4",
            op_type="D",
            action_type="remove",
            success=True,
            was_transformed=True,
            retry_count=2,
        )
        
        self.collector.add(result1)
        self.collector.add(result2)
        self.collector.add(result3)
        self.collector.add(result4)
        
        retry_ids = self.collector.suggestions_needing_retry()
        
        # Should include suggestions 1 and 2 (retryable errors)
        # Should NOT include 3 (non-retryable) or 4 (success)
        self.assertIn(1, retry_ids)
        self.assertIn(2, retry_ids)
        self.assertNotIn(3, retry_ids)
        self.assertNotIn(4, retry_ids)
        self.assertEqual(len(retry_ids), 2)


if __name__ == "__main__":
    unittest.main()
