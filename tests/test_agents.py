#!/usr/bin/env python3
"""
Tests for the new agentic pipeline configuration and base agent behavior.
"""

import unittest

from agents.config import (
    PipelineConfig,
    DiagnosisAgentConfig,
    StrategyAgentConfig,
    SurgeryAgentConfig,
)
from agents.base.agent_base import BaseAgent
from agents.base.unified_retriever import RetrievalProfile


class DummyAgent(BaseAgent):
    """Minimal agent for testing BaseAgent behaviors."""

    def __init__(self, pipeline_config: PipelineConfig):
        super().__init__(
            api_key="test-key",
            pipeline_config=pipeline_config,
            agent_model_name="gemini/gemini-3-pro-preview",
            temperature=0.1,
            max_tokens=256,
        )


class TestAgentConfigs(unittest.TestCase):
    def test_default_agent_configs(self):
        diag = DiagnosisAgentConfig()
        strat = StrategyAgentConfig()
        surg = SurgeryAgentConfig()

        self.assertEqual(diag.model_name, "gemini/gemini-3-pro-preview")
        self.assertEqual(strat.model_name, "gemini/gemini-3-pro-preview")
        self.assertEqual(surg.model_name, "gemini/gemini-3-pro-preview")

        self.assertGreaterEqual(diag.max_context_chunks, 1)
        self.assertGreaterEqual(strat.max_strategy_examples, 1)
        self.assertGreaterEqual(surg.max_retries_per_phase, 1)

    def test_pipeline_config_has_agent_configs(self):
        cfg = PipelineConfig()
        self.assertIsNotNone(cfg.diagnosis_agent_config)
        self.assertIsNotNone(cfg.strategy_agent_config)
        self.assertIsNotNone(cfg.surgery_agent_config)


class TestBaseAgent(unittest.TestCase):
    def test_retrieve_context_missing_kb(self):
        cfg = PipelineConfig(kb_path="nonexistent_kb.json")
        agent = DummyAgent(cfg)
        combined = agent.retrieve_combined(
            profile=RetrievalProfile.diagnosis(),
            query="test query"
        )
        self.assertIsInstance(combined.kb_text, str)


if __name__ == "__main__":
    unittest.main()
