#!/usr/bin/env python3
"""
Integration tests for agents module.

Tests the ReAct agent, ToT planner, and pipeline integration.
These tests use mocks for LLM calls to avoid API dependencies.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.config import AgentConfig, ToTConfig, PipelineConfig
from agents.state import AgentState, TransformationStrategy, StateManager
from agents.diagnostics import FeedbackCollector, TransformationResult, ErrorCategory


class TestAgentConfig(unittest.TestCase):
    """Tests for AgentConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        
        self.assertEqual(config.max_iterations, 15)
        self.assertEqual(config.temperature, 0.1)
        self.assertTrue(config.use_tot_planning)
    
    def test_should_use_tot(self):
        """Test ToT usage decision logic."""
        config = AgentConfig(
            tot_threshold_critical=3,
            tot_threshold_total=10,
        )
        
        # Should use ToT for complex cases
        self.assertTrue(config.should_use_tot(critical_count=5, total_issues=5, status="ok"))
        self.assertTrue(config.should_use_tot(critical_count=1, total_issues=15, status="ok"))
        self.assertTrue(config.should_use_tot(critical_count=1, total_issues=5, status="blocked"))
        
        # Should not use ToT for simple cases
        self.assertFalse(config.should_use_tot(critical_count=1, total_issues=5, status="ok"))


class TestToTConfig(unittest.TestCase):
    """Tests for ToTConfig."""
    
    def test_default_config(self):
        """Test default ToT configuration."""
        config = ToTConfig()
        
        self.assertEqual(config.num_strategies, 3)
        self.assertEqual(config.selection_method, "weighted_score")
    
    def test_evaluation_weights(self):
        """Test evaluation weights sum to approximately 1."""
        config = ToTConfig()
        
        total_weight = sum(config.evaluation_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)


class TestTransformationStrategy(unittest.TestCase):
    """Tests for TransformationStrategy."""
    
    def test_strategy_creation(self):
        """Test strategy creation."""
        strategy = TransformationStrategy(
            name="Test Strategy",
            priority_order=["critical_blockers", "shape_issues"],
            approach="aggressive_removal",
            estimated_success=0.75,
            rationale="Test rationale",
        )
        
        self.assertEqual(strategy.name, "Test Strategy")
        self.assertEqual(strategy.approach, "aggressive_removal")
        self.assertEqual(len(strategy.priority_order), 2)
    
    def test_to_dict(self):
        """Test strategy serialization."""
        strategy = TransformationStrategy(
            name="Test",
            priority_order=["a", "b"],
            approach="hybrid",
            estimated_success=0.5,
            rationale="test",
        )
        
        d = strategy.to_dict()
        
        self.assertIn("name", d)
        self.assertIn("priority_order", d)
        self.assertIn("estimated_success", d)
    
    def test_from_dict(self):
        """Test strategy deserialization."""
        data = {
            "name": "Restored Strategy",
            "priority_order": ["x", "y", "z"],
            "approach": "conservative",
            "estimated_success": 0.8,
            "rationale": "restored",
        }
        
        strategy = TransformationStrategy.from_dict(data)
        
        self.assertEqual(strategy.name, "Restored Strategy")
        self.assertEqual(strategy.estimated_success, 0.8)


class TestAgentState(unittest.TestCase):
    """Tests for AgentState."""
    
    def test_state_creation_manual(self):
        """Test manual state creation."""
        state = AgentState(
            model_path="/path/to/model.onnx",
            model_name="test_model",
        )
        
        self.assertEqual(state.model_name, "test_model")
        self.assertEqual(state.iteration, 0)
        self.assertEqual(len(state.pending_suggestions), 0)
    
    def test_should_continue_no_suggestions(self):
        """Test should_continue with no pending suggestions."""
        state = AgentState(
            model_path="/path/to/model.onnx",
            pending_suggestions=[],
        )
        
        self.assertFalse(state.should_continue())
    
    def test_should_continue_max_iterations(self):
        """Test should_continue at max iterations."""
        state = AgentState(
            model_path="/path/to/model.onnx",
            pending_suggestions=[{"id": 1}],
            iteration=15,
            max_iterations=15,
        )
        
        self.assertFalse(state.should_continue())
    
    def test_should_continue_normal(self):
        """Test should_continue in normal state."""
        config = AgentConfig(min_iterations_before_stop=5, early_stop_threshold=0.1)
        state = AgentState(
            model_path="/path/to/model.onnx",
            pending_suggestions=[{"id": 1}, {"id": 2}],
            iteration=3,
            max_iterations=15,
            config=config,
        )
        
        self.assertTrue(state.should_continue())
    
    def test_mark_applied_success(self):
        """Test marking suggestion as applied successfully."""
        state = AgentState(
            model_path="/path/to/model.onnx",
            pending_suggestions=[{"id": 1}, {"id": 2}],
        )
        
        suggestion = {"id": 1}
        result = TransformationResult(
            suggestion_id=1,
            node_name="node",
            op_type="Op",
            action_type="remove",
            success=True,
            was_transformed=True,
        )
        
        state.mark_applied(suggestion, result)
        
        self.assertEqual(len(state.pending_suggestions), 1)
        self.assertEqual(len(state.applied_suggestions), 1)
    
    def test_mark_applied_failure(self):
        """Test marking suggestion as failed."""
        state = AgentState(
            model_path="/path/to/model.onnx",
            pending_suggestions=[{"id": 1}],
        )
        
        suggestion = {"id": 1}
        result = TransformationResult(
            suggestion_id=1,
            node_name="node",
            op_type="Op",
            action_type="remove",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.HANDLER_FAILED,
        )
        
        state.mark_applied(suggestion, result)
        
        self.assertEqual(len(state.pending_suggestions), 0)
        self.assertEqual(len(state.failed_suggestions), 1)
    
    def test_get_progress_summary(self):
        """Test progress summary generation."""
        state = AgentState(
            model_path="/path/to/model.onnx",
            model_name="test",
            pending_suggestions=[{"id": 1}],
            applied_suggestions=[{"id": 2}],
            iteration=5,
            max_iterations=10,
        )
        
        summary = state.get_progress_summary()
        
        self.assertEqual(summary["iteration"], 5)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["applied"], 1)
    
    def test_update_strategy(self):
        """Test strategy update."""
        state = AgentState(model_path="/path/to/model.onnx")
        
        strategy = TransformationStrategy(
            name="New Strategy",
            priority_order=["a"],
            approach="hybrid",
            estimated_success=0.7,
            rationale="test",
        )
        
        state.update_strategy(strategy)
        
        self.assertEqual(state.strategy.name, "New Strategy")
        self.assertTrue(state.strategy_changed)


class TestStateManager(unittest.TestCase):
    """Tests for StateManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "/tmp/test_checkpoints"
        self.manager = StateManager(checkpoint_dir=self.test_dir)
    
    def test_has_checkpoint_false(self):
        """Test has_checkpoint when no checkpoint exists."""
        self.assertFalse(self.manager.has_checkpoint("nonexistent_model"))


class TestPipelineConfig(unittest.TestCase):
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()
        
        self.assertTrue(config.use_rag)
        self.assertTrue(config.checkpoint_enabled)
        self.assertIsInstance(config.agent_config, AgentConfig)
        self.assertIsInstance(config.tot_config, ToTConfig)
    
    def test_from_dict(self):
        """Test config creation from dictionary."""
        data = {
            "use_rag": False,
            "output_dir": "/custom/output",
            "agent_config": {
                "max_iterations": 20,
            },
            "tot_config": {
                "num_strategies": 5,
            },
        }
        
        config = PipelineConfig.from_dict(data)
        
        self.assertFalse(config.use_rag)
        self.assertEqual(config.output_dir, "/custom/output")
        self.assertEqual(config.agent_config.max_iterations, 20)
        self.assertEqual(config.tot_config.num_strategies, 5)
    
    def test_to_dict(self):
        """Test config serialization."""
        config = PipelineConfig(
            use_rag=True,
            output_dir="/test/output",
        )
        
        d = config.to_dict()
        
        self.assertIn("use_rag", d)
        self.assertIn("agent_config", d)
        self.assertIn("tot_config", d)


class TestAgentTools(unittest.TestCase):
    """Tests for agent tools (mocked)."""
    
    @patch('agents.tools.COMPONENTS_AVAILABLE', True)
    def test_agent_context_creation(self):
        """Test AgentContext creation."""
        from agents.tools import AgentContext
        
        context = AgentContext(
            model_path="/path/to/model.onnx",
            api_key="test_key",
        )
        
        self.assertEqual(context.model_path, "/path/to/model.onnx")
        self.assertEqual(context.api_key, "test_key")
    
    @patch('agents.tools.COMPONENTS_AVAILABLE', True)
    def test_set_suggestions(self):
        """Test setting suggestions in context."""
        from agents.tools import AgentContext
        
        context = AgentContext(model_path="/path/to/model.onnx")
        
        suggestions = [
            {"id": 1, "priority": "high"},
            {"id": 2, "priority": "medium"},
        ]
        
        context.set_suggestions(suggestions)
        
        self.assertEqual(len(context.suggestions), 2)
        self.assertIn(1, context.suggestions)
        self.assertIn(2, context.suggestions)


class TestReActAgentMocked(unittest.TestCase):
    """Tests for ReAct agent with mocked LLM."""
    
    @patch('agents.react_agent.LANGCHAIN_AVAILABLE', False)
    def test_agent_requires_langchain(self):
        """Test that agent raises error without LangChain."""
        from agents.react_agent import GraphSurgeryReActAgent
        
        with self.assertRaises(ImportError):
            GraphSurgeryReActAgent(api_key="test_key")


class TestSimpleReActAgent(unittest.TestCase):
    """Tests for SimpleReActAgent."""
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_simple_agent_creation(self, mock_model, mock_configure):
        """Test SimpleReActAgent creation."""
        from agents.react_agent import SimpleReActAgent
        
        agent = SimpleReActAgent(api_key="test_key")
        
        self.assertIsNotNone(agent)
        mock_configure.assert_called_once_with(api_key="test_key")


if __name__ == "__main__":
    unittest.main()
