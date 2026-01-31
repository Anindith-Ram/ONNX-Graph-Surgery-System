#!/usr/bin/env python3
"""
RAG-Enhanced Suggestion Generator for ONNX Model Compilation.

Uses Retrieval-Augmented Generation (RAG) to provide context-aware,
domain-specific suggestions based on:
1. PDF documentation (ONNX Graph Surgery best practices)
2. Dataset transformation patterns
3. Model-specific knowledge

This extends the base SuggestionGenerator with RAG capabilities.
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_analysis.onnx_analyzer import ONNXAnalyzer, ModelAnalysis, NodeAnalysis
from suggestion_pipeline.suggestion_generator import (
    SuggestionGenerator, SuggestionReport, Suggestion, SuggestionLocation,
    Priority, ConfidenceScorer
)
from knowledge_base.rag_retriever import RAGRetriever, detect_model_category
from knowledge_base.response_cache import cached_gemini_call

# Import surgery database and context generator (optional - for enhanced context)
try:
    from knowledge_base.surgery_database import SurgeryDatabase
    from knowledge_base.llm_context_generator import LLMContextGenerator
    SURGERY_DB_AVAILABLE = True
except ImportError:
    SURGERY_DB_AVAILABLE = False
from suggestion_pipeline.suggestion_scorer import SuggestionScorer
from utilities.api_quota_manager import APIQuotaManager
from utilities.checkpoint_manager import CheckpointManager
from datetime import datetime


class RAGSuggestionGenerator(SuggestionGenerator):
    """
    RAG-enhanced suggestion generator.
    
    Uses knowledge base to provide context-aware suggestions that leverage:
    - Official graph surgery documentation
    - Proven transformation patterns from dataset
    - Model-specific best practices
    """
    
    # Enhanced prompt template with RAG context (for individual suggestions - legacy)
    RAG_SUGGESTION_PROMPT = """You are an expert ONNX graph surgery engineer specializing in making models compilable for edge hardware (MLA).

You have access to domain knowledge from:
1. Official ONNX Graph Surgery documentation
2. Proven transformation patterns from a dataset of successfully modified models
3. Model-specific best practices

CONTEXT FROM KNOWLEDGE BASE:
{context}

MODEL INFORMATION:
- Model Category: {model_category}
- Operation Type: {op_type}
- Issue: {issue_description}
- Node Details: {node_details}

CURRENT SUGGESTION (from rule-based system):
{suggestion_text}

TASK:
Generate an ACTUAL TRANSFORMATION suggestion (remove, replace, add, reshape) - NOT a "retain" suggestion.
The goal is to FIX compilation blockers and make the model compilable on MLA hardware.

CRITICAL: If the base suggestion says to remove/modify something, ENHANCE it with specific transformation steps.
If the base suggestion is unclear, generate a concrete transformation based on the context.

Provide:
1. A specific, actionable transformation (e.g., "Remove Einsum, replace with MatMul+Reshape" NOT "Retain operation")
2. Detailed implementation steps that follow best practices from the documentation
3. Confidence assessment based on how well this matches documented patterns
4. Any warnings or considerations from the knowledge base

AVOID generic "retain" or "preserve" suggestions unless the operation is explicitly required for a specific pattern.

OUTPUT FORMAT (JSON only, no markdown, no code blocks):
{{
    "enhanced_suggestion": "Improved suggestion text (keep concise, escape quotes with backslash)",
    "implementation_steps": ["step1", "step2"],
    "confidence_boost": 0.1,
    "warnings": ["warning1"],
    "reference": "Specific reference",
    "alternative_approaches": ["approach1"]
}}

IMPORTANT: Return ONLY valid JSON. Escape all quotes inside strings with backslash. Keep strings on single lines when possible.
"""
    
    # Batch prompt template - FOCUSED ON ACTUAL TRANSFORMATIONS WITH EXACT LOCATIONS
    RAG_BATCH_SUGGESTION_PROMPT = """You are an ONNX graph modification expert. Generate ACTUAL TRANSFORMATIONS to make models compilable on MLA hardware.

CRITICAL: Generate TRANSFORMATIONS (remove, replace, add, reshape), NOT "retain" suggestions.
The goal is to FIX compilation blockers, not preserve the status quo.

CONTEXT:
{context}

MODEL: {model_category} | OPERATION: {op_type}

SUGGESTIONS ({count} total):
{suggestions}

TRANSFORMATION EXAMPLES (what we want - WITH EXACT LOCATIONS):
- "Remove Einsum node 'einsum_123' (node_id: 45) and replace with MatMul+Reshape for MLA compatibility"
- "Add Sigmoid node after 'concat_456' (node_id: 78) at output heads to convert logits to probabilities"
- "Reshape tensor at 'reshape_789' (node_id: 12) from 5D [1,1,64,320,320] to 4D [1,64,320,320] for MLA"
- "Replace DynamicSlice node 'slice_abc' (node_id: 34) with Slice using concrete indices"
- "Insert Reshape node after 'mul_xyz' (node_id: 56) to convert 3D [1,2,8400] to 4D [1,2,1,8400]"

AVOID (what we DON'T want):
- "Retain Mul node" (unless it's explicitly needed for a specific pattern)
- "Preserve operation" (we need changes, not preservation)
- Generic "keep as-is" suggestions
- Vague locations like "somewhere in the graph"

RESPOND WITH ONLY A JSON ARRAY. Each object must have:
- suggestion_id: (integer matching input - REQUIRED)
- enhanced_suggestion: (string, max 300 chars, MUST include: transformation action + exact node name + node_id for engineer clarity)
- location: (object with node_name, node_id, op_type - REQUIRED for exact location)
- steps: (array of 2-4 short strings describing HOW to transform at this exact location - include node name in steps)
- confidence_boost: (number 0.0-0.2)

EXAMPLE FORMAT (engineer-friendly):
[{{"suggestion_id":1,"enhanced_suggestion":"Remove Einsum node 'einsum_123' (node_id: 45) and replace with MatMul+Reshape for MLA compatibility","location":{{"node_name":"einsum_123","node_id":45,"op_type":"Einsum"}},"steps":["Locate Einsum node 'einsum_123' (node_id: 45) in the graph","Replace 'einsum_123' with MatMul operation","Add Reshape node after MatMul for correct output shape","Validate graph structure with onnx.checker"],"confidence_boost":0.15}}]

CRITICAL: The enhanced_suggestion MUST include the exact node name and node_id so engineers know WHERE to apply the change.

RULES:
- Return ONLY valid JSON array
- Keep strings SHORT (under 250 chars for enhanced_suggestion)
- No markdown, no explanation text
- Exactly {count} objects in array
- MUST specify transformation action (remove/replace/add/reshape), not "retain"
- MUST include exact location (node_name and node_id from input) in enhanced_suggestion
- Location object must match the input suggestion's node_name and node_id
"""
    
    # Enhanced individual prompt template for Phase 2 (individual enhancement)
    RAG_INDIVIDUAL_SUGGESTION_PROMPT = """You are an ONNX graph modification expert. Generate a detailed TRANSFORMATION for a single suggestion to make the model compilable on MLA hardware.

CRITICAL: Generate a TRANSFORMATION (remove, replace, add, reshape), NOT a "retain" suggestion.
The goal is to FIX compilation blockers, not preserve the status quo.

CONTEXT FROM KNOWLEDGE BASE:
{context}

MODEL INFORMATION:
- Model Category: {model_category}
- Operation Type: {op_type}

NODE DETAILS:
- Node Name: {node_name}
- Node ID: {node_id}
- Graph Position: {graph_position} (0.0 = inputs, 1.0 = outputs)
- Predecessors: {predecessors}
- Successors: {successors}
- Inputs: {inputs}
- Outputs: {outputs}

CURRENT ISSUE:
{issue_description}

CURRENT SUGGESTION (from rule-based system):
{suggestion_text}

TASK:
Generate a detailed, actionable transformation with exact location information.

OUTPUT FORMAT (JSON only, no markdown, no code blocks):
{{
    "enhanced_suggestion": "Detailed transformation text (max 300 chars, MUST include node name '{node_name}' and node_id {node_id})",
    "location": {{
        "node_name": "{node_name}",
        "node_id": {node_id},
        "op_type": "{op_type}"
    }},
    "steps": ["step1 with node name", "step2", "step3", "step4"],
    "confidence_boost": 0.1,
    "warnings": ["warning1"],
    "reference": "Specific reference"
}}

CRITICAL REQUIREMENTS:
- enhanced_suggestion MUST include the exact node name '{node_name}' and node_id {node_id}
- steps MUST reference the node name '{node_name}' for clarity
- MUST specify transformation action (remove/replace/add/reshape), not "retain"
- Location object must match the input (node_name: "{node_name}", node_id: {node_id})

IMPORTANT: Return ONLY valid JSON. Escape all quotes inside strings with backslash.
"""
    
    def __init__(
        self,
        kb_path: str = "knowledge_base.json",
        api_key: Optional[str] = None,
        use_rag: bool = True,
        use_hybrid_mode: bool = False,
        daily_api_limit: int = 250,
        checkpoint_dir: str = "checkpoints",
        individual_enhancement_threshold: Optional[int] = None,
        individual_context_chunks: int = 10,
        checkpoint_frequency: int = 1
    ):
        """
        Initialize RAG-enhanced suggestion generator.
        
        Args:
            kb_path: Path to knowledge base JSON file
            api_key: Gemini API key
            use_rag: Whether to use RAG enhancement (can disable for testing)
            use_hybrid_mode: Enable hybrid mode (batch + individual enhancement)
            daily_api_limit: Maximum API calls per day (default: 250)
            checkpoint_dir: Directory for checkpoint files
            individual_enhancement_threshold: Top N suggestions for individual enhancement (None = auto)
            individual_context_chunks: Number of KB chunks for individual enhancement (default: 10)
            checkpoint_frequency: Save checkpoint every N suggestions (default: 1)
        """
        super().__init__(api_key=api_key)
        
        self.use_rag = use_rag and GEMINI_AVAILABLE
        self.api_key = api_key
        self.use_hybrid_mode = use_hybrid_mode
        self.daily_api_limit = daily_api_limit
        self.checkpoint_dir = checkpoint_dir
        self.individual_enhancement_threshold = individual_enhancement_threshold
        self.individual_context_chunks = individual_context_chunks
        self.checkpoint_frequency = checkpoint_frequency
        
        # Initialize quota and checkpoint managers if hybrid mode
        if self.use_hybrid_mode:
            self.quota_manager = APIQuotaManager(daily_limit=daily_api_limit)
            self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            self.scorer = SuggestionScorer()
            print(f"Hybrid mode enabled (daily limit: {daily_api_limit}, checkpoint dir: {checkpoint_dir})")
        else:
            self.quota_manager = None
            self.checkpoint_manager = None
            # Keep the scorer from parent class (ConfidenceScorer) - don't override to None
        
        # Initialize surgery database and context generator (if available)
        self.surgery_db = None
        self.context_generator = None
        surgery_db_path = os.path.join(os.path.dirname(kb_path), "surgery_database.json")
        
        if SURGERY_DB_AVAILABLE and os.path.exists(surgery_db_path):
            try:
                self.surgery_db = SurgeryDatabase.load(surgery_db_path)
                self.context_generator = LLMContextGenerator(self.surgery_db)
                print(f"Surgery database loaded ({self.surgery_db.total_models} models, {self.surgery_db.total_transformations} transformations)")
            except Exception as e:
                print(f"Warning: Could not load surgery database: {e}")
        
        if self.use_rag:
            if not api_key:
                print("Warning: No API key provided, RAG features disabled")
                self.use_rag = False
            else:
                try:
                    genai.configure(api_key=api_key)
                    self.retriever = RAGRetriever(kb_path, api_key=api_key)
                    print(f"RAG-enhanced suggestions enabled (KB: {kb_path})")
                except Exception as e:
                    print(f"Warning: Could not initialize RAG: {e}")
                    self.use_rag = False
        else:
            self.retriever = None
    
    def analyze_and_suggest(self, model_path: str) -> SuggestionReport:
        """
        Analyze model and generate RAG-enhanced suggestions.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            SuggestionReport with enhanced suggestions
        """
        # Detect model category
        model_category = detect_model_category(model_path)
        
        # Get base suggestions
        report = super().analyze_and_suggest(model_path)
        
        # Generate pattern-based suggestions (NEW: Phase 1)
        # Need to get analysis separately since SuggestionReport doesn't store it
        if self.use_rag and self.retriever:
            try:
                # Re-analyze model to get ModelAnalysis for pattern-based suggestions
                analysis = self.analyzer.analyze(model_path)
                pattern_suggestions = self._generate_pattern_based_suggestions(
                    model_path, model_category, analysis, report
                )
                # Merge with base suggestions
                if pattern_suggestions:
                    print(f"  Generated {len(pattern_suggestions)} pattern-based suggestions")
                    report.suggestions = report.suggestions + pattern_suggestions
                    report = self._recompute_report_stats(report)
                else:
                    print(f"  No pattern-based suggestions generated for {model_category} model")
            except Exception as e:
                print(f"  Warning: Could not generate pattern-based suggestions: {e}")
                import traceback
                traceback.print_exc()
        
        # Enhance suggestions with RAG if enabled (using batched processing)
        if self.use_rag and self.retriever:
            print(f"\nEnhancing suggestions with RAG (Model Category: {model_category})...")
            
            # Group suggestions by op_type and split into optimal batches
            grouped_suggestions = self._group_suggestions_by_context(report.suggestions)
            
            enhanced_suggestions = []
            total_batches = sum(len(batches) for batches in grouped_suggestions.values())
            current_batch = 0
            
            for op_type, batches in grouped_suggestions.items():
                for batch_idx, suggestions_group in enumerate(batches):
                    current_batch += 1
                    batch_label = f"{op_type}" if len(batches) == 1 else f"{op_type} (batch {batch_idx + 1}/{len(batches)})"
                    print(f"  Processing batch {current_batch}/{total_batches}: {batch_label} ({len(suggestions_group)} suggestions)")
                    
                    # Batch enhance all suggestions in this group with retry
                    enhanced_batch = self._enhance_suggestions_batch_with_retry(
                        suggestions_group,
                        model_path,
                        model_category,
                        report,
                        max_retries=3
                    )
                    enhanced_suggestions.extend(enhanced_batch)
            
            # Update report with enhanced suggestions
            report.suggestions = enhanced_suggestions
            report = self._recompute_report_stats(report)
        
        return report
    
    def analyze_and_suggest_hybrid(self, model_path: str, resume: bool = True) -> SuggestionReport:
        """
        Hybrid approach with quota management and checkpointing.
        
        Phase 1: Batch enhancement (all suggestions)
        Phase 2: Individual enhancement for top suggestions (with quota management)
        
        Args:
            model_path: Path to ONNX model
            resume: Whether to resume from checkpoint if exists
            
        Returns:
            SuggestionReport with enhanced suggestions
        """
        if not self.use_hybrid_mode:
            # Fall back to regular mode
            return self.analyze_and_suggest(model_path)
        
        model_name = Path(model_path).stem
        model_category = detect_model_category(model_path)
        
        # Check for existing checkpoint
        checkpoint = None
        if resume:
            checkpoint = self.checkpoint_manager.load_checkpoint(model_name, "phase2")
        
        if checkpoint:
            print(f"\nResuming from checkpoint (date: {checkpoint.get('date', 'unknown')})")
            print(f"  Processed: {checkpoint.get('processed_count', 0)}")
            print(f"  Remaining: {checkpoint.get('remaining_count', 0)}")
            
            # Load processed and remaining suggestions
            # Note: We'll need to reconstruct Suggestion objects from dicts
            processed = self._load_suggestions_from_dict(checkpoint.get('processed_suggestions', []))
            remaining = self._load_suggestions_from_dict(checkpoint.get('remaining_suggestions', []))
            
            # Get the base report for context (we'll merge with processed suggestions)
            report = super().analyze_and_suggest(model_path)
            # Don't do pattern-based or batch enhancement again if resuming
        else:
            # Phase 1: Batch enhancement (all suggestions)
            print("\nPhase 1: Batch enhancement...")
            report = super().analyze_and_suggest(model_path)
            
            # Generate pattern-based suggestions if enabled
            if self.use_rag and self.retriever:
                try:
                    analysis = self.analyzer.analyze(model_path)
                    pattern_suggestions = self._generate_pattern_based_suggestions(
                        model_path, model_category, analysis, report
                    )
                    if pattern_suggestions:
                        print(f"  Generated {len(pattern_suggestions)} pattern-based suggestions")
                        report.suggestions = report.suggestions + pattern_suggestions
                        report = self._recompute_report_stats(report)
                except Exception as e:
                    print(f"  Warning: Could not generate pattern-based suggestions: {e}")
            
            # Enhance all suggestions with batch RAG
            if self.use_rag and self.retriever:
                print(f"\nEnhancing suggestions with RAG (Model Category: {model_category})...")
                grouped_suggestions = self._group_suggestions_by_context(report.suggestions)
                
                enhanced_suggestions = []
                total_batches = sum(len(batches) for batches in grouped_suggestions.values())
                current_batch = 0
                
                for op_type, batches in grouped_suggestions.items():
                    for batch_idx, suggestions_group in enumerate(batches):
                        current_batch += 1
                        batch_label = f"{op_type}" if len(batches) == 1 else f"{op_type} (batch {batch_idx + 1}/{len(batches)})"
                        print(f"  Processing batch {current_batch}/{total_batches}: {batch_label} ({len(suggestions_group)} suggestions)")
                        
                        # Check quota before batch
                        if not self.quota_manager.can_make_request():
                            print(f"  Quota exhausted, saving checkpoint...")
                            self.checkpoint_manager.save_checkpoint(
                                model_name=model_name,
                                phase="phase1",
                                processed_suggestions=enhanced_suggestions,
                                remaining_suggestions=report.suggestions[len(enhanced_suggestions):],
                                api_calls_used=self.quota_manager.used_today,
                                additional_data={'phase1_complete': False}
                            )
                            print("  Checkpoint saved. Resume tomorrow.")
                            # Return partial report
                            report.suggestions = enhanced_suggestions + report.suggestions[len(enhanced_suggestions):]
                            return report
                        
                        enhanced_batch = self._enhance_suggestions_batch_with_retry(
                            suggestions_group,
                            model_path,
                            model_category,
                            report,
                            max_retries=3
                        )
                        enhanced_suggestions.extend(enhanced_batch)
                        
                        # Record API usage
                        self.quota_manager.record_request()
                
                report.suggestions = enhanced_suggestions
                report = self._recompute_report_stats(report)
            
            # Score and select critical suggestions for Phase 2
            print("\nScoring suggestions for Phase 2...")
            scored = self.scorer.rank_suggestions(report.suggestions)
            
            # Select top suggestions that fit in remaining quota
            remaining_quota = self.quota_manager.get_remaining_quota()
            if self.individual_enhancement_threshold is None:
                # Auto-select: top 20% or top 50, whichever is smaller, but respect quota
                threshold = min(
                    max(50, len(report.suggestions) // 5),  # Top 20% or at least 50
                    remaining_quota  # But not more than quota allows
                )
            else:
                threshold = min(self.individual_enhancement_threshold, remaining_quota)
            
            critical_suggestions = [s for s, score in scored[:threshold]]
            remaining = [s for s, score in scored[threshold:]]
            processed = []
            
            print(f"  Selected {len(critical_suggestions)} suggestions for individual enhancement")
            print(f"  Remaining quota: {remaining_quota}")
        
        # Phase 2: Individual enhancement (with quota checking)
        if remaining:
            print(f"\nPhase 2: Individual enhancement (quota: {self.quota_manager.get_remaining_quota()})...")
            enhanced_suggestions = []
            
            for idx, suggestion in enumerate(remaining):
                # Check quota before processing
                if not self.quota_manager.can_make_request():
                    print(f"\nQuota exhausted ({self.quota_manager.used_today}/{self.quota_manager.DAILY_LIMIT})")
                    print(f"Saving checkpoint with {len(remaining) - idx} remaining suggestions...")
                    self.checkpoint_manager.save_checkpoint(
                        model_name=model_name,
                        phase="phase2",
                        processed_suggestions=processed + enhanced_suggestions,
                        remaining_suggestions=remaining[idx:],
                        api_calls_used=self.quota_manager.used_today,
                        additional_data={'model_category': model_category}
                    )
                    print("Checkpoint saved. Resume tomorrow with --resume flag.")
                    break
                
                # Enhance suggestion individually
                try:
                    enhanced = self._enhance_suggestion_individually(
                        suggestion, model_path, model_category, report
                    )
                    enhanced_suggestions.append(enhanced)
                    processed.append(enhanced)
                    
                    # Record API usage
                    self.quota_manager.record_request()
                    
                    # Save checkpoint periodically
                    if len(enhanced_suggestions) % self.checkpoint_frequency == 0:
                        self.checkpoint_manager.save_checkpoint(
                            model_name=model_name,
                            phase="phase2",
                            processed_suggestions=processed,
                            remaining_suggestions=remaining[idx + 1:],
                            api_calls_used=self.quota_manager.used_today,
                            additional_data={'model_category': model_category}
                        )
                    
                    if (idx + 1) % 10 == 0:
                        print(f"  Processed {idx + 1}/{len(remaining)} suggestions (quota: {self.quota_manager.get_remaining_quota()})")
                except Exception as e:
                    print(f"  Warning: Failed to enhance suggestion {suggestion.id}: {e}")
                    # Keep original suggestion
                    processed.append(suggestion)
                    enhanced_suggestions.append(suggestion)
            
            # If all processed, clear checkpoint
            if len(enhanced_suggestions) == len(remaining):
                self.checkpoint_manager.clear_checkpoint(model_name, "phase2")
                print("\nAll suggestions processed. Checkpoint cleared.")
            
            # Combine enhanced + batch-enhanced suggestions
            # Get all suggestions that weren't in the remaining list
            processed_ids = {s.id for s in processed}
            remaining_ids = {s.id for s in remaining}
            other_suggestions = [s for s in report.suggestions if s.id not in processed_ids and s.id not in remaining_ids]
            final_suggestions = processed + other_suggestions
            report.suggestions = final_suggestions
        else:
            # No remaining suggestions, clear checkpoint
            self.checkpoint_manager.clear_checkpoint(model_name, "phase2")
        
        report = self._recompute_report_stats(report)
        return report
    
    def _load_suggestions_from_dict(self, suggestion_dicts: List[Dict]) -> List[Suggestion]:
        """
        Load Suggestion objects from dictionary representations.
        
        Args:
            suggestion_dicts: List of suggestion dictionaries
            
        Returns:
            List of Suggestion objects
        """
        suggestions = []
        for s_dict in suggestion_dicts:
            try:
                # Reconstruct Suggestion from dict
                location_dict = s_dict.get('location', {})
                location = SuggestionLocation(
                    node_id=location_dict.get('node_id', -1),
                    node_name=location_dict.get('node_name', ''),
                    op_type=location_dict.get('op_type', ''),
                    inputs=location_dict.get('inputs', []),
                    outputs=location_dict.get('outputs', []),
                    predecessors=location_dict.get('predecessors', []),
                    successors=location_dict.get('successors', []),
                    graph_position=location_dict.get('graph_position')
                )
                
                # Get priority
                priority_str = s_dict.get('priority', 'medium')
                priority = Priority[priority_str.upper()] if priority_str.upper() in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'] else Priority.MEDIUM
                
                suggestion = Suggestion(
                    id=s_dict.get('id', 0),
                    priority=priority,
                    confidence=s_dict.get('confidence', 0.5),
                    issue=s_dict.get('issue', ''),
                    location=location,
                    suggestion=s_dict.get('suggestion', ''),
                    implementation_steps=s_dict.get('implementation_steps', []),
                    impact=s_dict.get('impact', ''),
                    reference=s_dict.get('reference', ''),
                    category=s_dict.get('category', ''),
                    estimated_effort=s_dict.get('estimated_effort', '')
                )
                suggestions.append(suggestion)
            except Exception as e:
                print(f"  Warning: Failed to load suggestion from dict: {e}")
                continue
        
        return suggestions
    
    def _enhance_suggestion_individually(
        self,
        suggestion: Suggestion,
        model_path: str,
        model_category: str,
        analysis: SuggestionReport
    ) -> Suggestion:
        """
        Enhance a single suggestion individually with enhanced context.
        
        Args:
            suggestion: Suggestion to enhance
            model_path: Path to model
            model_category: Detected model category
            analysis: Full analysis report
            
        Returns:
            Enhanced suggestion
        """
        if not self.use_rag or not self.retriever:
            return suggestion
        
        # Build query for retrieval
        query = f"{suggestion.issue} {suggestion.location.op_type} {model_category}"
        
        # Retrieve enhanced context (more chunks for individual)
        context = self.retriever.retrieve(
            query=query,
            model_category=model_category,
            op_type=suggestion.location.op_type,
            top_k=self.individual_context_chunks
        )
        
        if not context.chunks:
            return suggestion
        
        context_text = context.get_text(max_chunks=self.individual_context_chunks)
        
        # Build detailed prompt with full graph context
        graph_position = suggestion.location.graph_position if suggestion.location.graph_position is not None else "unknown"
        predecessors = ", ".join(suggestion.location.predecessors) if hasattr(suggestion.location, 'predecessors') and suggestion.location.predecessors else "none"
        successors = ", ".join(suggestion.location.successors) if hasattr(suggestion.location, 'successors') and suggestion.location.successors else "none"
        
        prompt = self.RAG_INDIVIDUAL_SUGGESTION_PROMPT.format(
            context=context_text,
            model_category=model_category,
            op_type=suggestion.location.op_type,
            node_name=suggestion.location.node_name,
            node_id=suggestion.location.node_id,
            graph_position=graph_position,
            predecessors=predecessors,
            successors=successors,
            inputs=", ".join(suggestion.location.inputs) if suggestion.location.inputs else "none",
            outputs=", ".join(suggestion.location.outputs) if suggestion.location.outputs else "none",
            issue_description=suggestion.issue,
            suggestion_text=suggestion.suggestion
        )
        
        # Call Gemini with caching
        try:
            response = cached_gemini_call(
                prompt=prompt,
                api_key=self.api_key,
                model_name="models/gemini-3-pro-preview",
                temperature=0.1,  # Lower temperature for more consistent JSON
                max_tokens=3000
            )
            
            if not response:
                return suggestion
            
            # Parse response
            enhancement = self._parse_enhancement_response(response)
            
            # Apply enhancement
            if enhancement:
                suggestion = self._apply_enhancement(suggestion, enhancement, context)
        
        except Exception as e:
            if "parsing" not in str(e).lower() and "json" not in str(e).lower():
                print(f"    Warning: Individual enhancement failed for suggestion {suggestion.id}: {e}")
        
        return suggestion
    
    def _select_critical_suggestions(
        self,
        suggestions: List[Suggestion],
        max_count: Optional[int] = None
    ) -> Tuple[List[Suggestion], List[Suggestion]]:
        """
        Select critical suggestions for individual enhancement.
        
        Args:
            suggestions: List of all suggestions
            max_count: Maximum number to select (None = auto)
            
        Returns:
            Tuple of (critical_suggestions, remaining_suggestions)
        """
        if not self.scorer:
            return [], suggestions
        
        # Rank by impact score
        scored = self.scorer.rank_suggestions(suggestions)
        
        if max_count is None:
            # Auto-select: top 20% or top 50, whichever is smaller
            max_count = min(max(50, len(suggestions) // 5), len(suggestions))
        
        critical = [s for s, score in scored[:max_count]]
        remaining = [s for s, score in scored[max_count:]]
        
        return critical, remaining
    
    def _enhance_suggestion_with_rag(
        self,
        suggestion: Suggestion,
        model_path: str,
        model_category: str,
        analysis: SuggestionReport
    ) -> Suggestion:
        """
        Enhance a suggestion using RAG.
        
        Args:
            suggestion: Base suggestion
            model_path: Path to model
            model_category: Detected model category
            analysis: Full analysis report
            
        Returns:
            Enhanced suggestion
        """
        if not self.use_rag or not self.retriever:
            return suggestion
        
        # Build query for retrieval
        query = f"{suggestion.issue} {suggestion.location.op_type} {model_category}"
        
        # Retrieve relevant context
        context = self.retriever.retrieve(
            query=query,
            model_category=model_category,
            op_type=suggestion.location.op_type,
            top_k=5
        )
        
        if not context.chunks:
            # No relevant context found, return original
            return suggestion
        
        # Build prompt with context (text is already sanitized in get_text())
        context_text = context.get_text(max_chunks=5)
        
        node_details = {
            'name': suggestion.location.node_name,
            'op_type': suggestion.location.op_type,
            'inputs': suggestion.location.inputs,
            'outputs': suggestion.location.outputs
        }
        
        prompt = self.RAG_SUGGESTION_PROMPT.format(
            context=context_text,
            model_category=model_category,
            op_type=suggestion.location.op_type,
            issue_description=suggestion.issue,
            node_details=json.dumps(node_details, indent=2),
            suggestion_text=suggestion.suggestion
        )
        
        # Call Gemini with caching
        try:
            response = cached_gemini_call(
                prompt=prompt,
                api_key=self.api_key,
                model_name="models/gemini-3-pro-preview",
                temperature=0.2,  # Lower temperature for more consistent JSON
                max_tokens=3000  # Increased to prevent truncation
            )
            
            if not response:
                # Handle None response gracefully
                return suggestion
            
            # Parse response
            enhancement = self._parse_enhancement_response(response)
            
            # Apply enhancement
            if enhancement:
                suggestion = self._apply_enhancement(suggestion, enhancement, context)
            # If enhancement is None, silently use original suggestion (errors already logged in parser)
        
        except Exception as e:
            # Only log if it's not a parsing error (those are handled in _parse_enhancement_response)
            if "parsing" not in str(e).lower() and "json" not in str(e).lower():
                print(f"  Warning: RAG enhancement failed for suggestion {suggestion.id}: {e}")
            # Return original suggestion on error
        
        return suggestion
    
    def _calculate_optimal_batch_size(self, suggestions: List[Suggestion], context_length: int = 2000) -> int:
        """
        Calculate optimal batch size based on token estimates.
        
        Args:
            suggestions: List of suggestions to batch
            context_length: Estimated length of context text
            
        Returns:
            Optimal batch size (1-5) - reduced from 10 to prevent truncation
        """
        # Estimate tokens per suggestion (~400 tokens for complete JSON response)
        avg_tokens_per_suggestion = 400
        # Reserve tokens for prompt, context, and response overhead
        # Context consumes ~1 token per 4 characters
        context_tokens = context_length // 4
        reserved_tokens = 2500 + context_tokens
        # Gemini max output tokens - increased
        max_output_tokens = 8192
        
        # Calculate how many suggestions we can fit
        available_tokens = max_output_tokens - reserved_tokens
        optimal_size = min(
            available_tokens // avg_tokens_per_suggestion,
            len(suggestions),
            10  # Increased to 10 to reduce API calls (quality is good at this size)
        )
        return max(optimal_size, 1)  # At least 1
    
    def _group_suggestions_by_context(self, suggestions: List[Suggestion]) -> Dict[str, List[List[Suggestion]]]:
        """
        Group suggestions by op_type and split into optimal-sized batches.
        
        Args:
            suggestions: List of suggestions to group
            
        Returns:
            Dictionary mapping op_type to list of batches (each batch is a list of suggestions)
        """
        # First, group by op_type
        grouped_by_op = {}
        for suggestion in suggestions:
            op_type = suggestion.location.op_type
            if op_type not in grouped_by_op:
                grouped_by_op[op_type] = []
            grouped_by_op[op_type].append(suggestion)
        
        # Calculate optimal batch size (estimate context at ~2000 chars average)
        optimal_size = self._calculate_optimal_batch_size(suggestions, context_length=2000)
        
        # Split large groups into optimal-sized batches
        final_groups = {}
        for op_type, group in grouped_by_op.items():
            if len(group) <= optimal_size:
                # Small enough, single batch
                final_groups[op_type] = [group]
            else:
                # Split into multiple batches
                batches = []
                for i in range(0, len(group), optimal_size):
                    batches.append(group[i:i + optimal_size])
                final_groups[op_type] = batches
        
        return final_groups
    
    def _enhance_suggestions_batch_with_retry(
        self,
        suggestions: List[Suggestion],
        model_path: str,
        model_category: str,
        analysis: SuggestionReport,
        max_retries: int = 3
    ) -> List[Suggestion]:
        """
        Enhance suggestions with retry logic.
        
        Args:
            suggestions: List of suggestions to enhance
            model_path: Path to model
            model_category: Detected model category
            analysis: Full analysis report
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of enhanced suggestions
        """
        for attempt in range(max_retries):
            try:
                result, enhanced_count = self._enhance_suggestions_batch(
                    suggestions,
                    model_path,
                    model_category,
                    analysis
                )
                
                # If at least 30% enhanced, consider it successful
                if enhanced_count >= len(suggestions) * 0.3:
                    if attempt > 0:
                        print(f"    Success after retry {attempt + 1}")
                    return result
                elif attempt < max_retries - 1:
                    print(f"    Low enhancement rate ({enhanced_count}/{len(suggestions)}), retrying...")
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    return result
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    Retry {attempt + 1}/{max_retries} after error: {str(e)[:100]}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"  Warning: All retries failed for batch: {e}")
                    return suggestions  # Return originals after all retries fail
        
        return suggestions
    
    def _get_enriched_context(
        self,
        op_type: str,
        model_category: str,
        node_context: Optional[Dict] = None,
        max_chunks: int = 5
    ) -> str:
        """
        Get enriched context from both RAGRetriever and SurgeryDatabase.
        
        Combines:
        1. Traditional RAG context (from knowledge base)
        2. Surgery database context (WHY/HOW explanations)
        
        Args:
            op_type: Operation type (e.g., "Einsum")
            model_category: Model category (e.g., "Transformer")
            node_context: Optional additional node context
            max_chunks: Maximum chunks from RAG retriever
            
        Returns:
            Combined context string for LLM prompts
        """
        context_parts = []
        
        # Part 1: Get context from surgery database (if available)
        if self.context_generator:
            try:
                generated = self.context_generator.generate_blocker_context(
                    op_type=op_type,
                    model_category=model_category,
                    node_context=node_context,
                    max_examples=3
                )
                
                if generated.relevance_score > 0.3:  # Only include if relevant
                    surgery_context = generated.to_string(max_tokens=1500)
                    if surgery_context.strip():
                        context_parts.append("=== SURGERY DATABASE CONTEXT ===")
                        context_parts.append(surgery_context)
                        context_parts.append("")
            except Exception as e:
                # Silently fall back to RAG-only context
                pass
        
        # Part 2: Get context from RAG retriever
        if self.retriever:
            try:
                query = f"{op_type} {model_category} ONNX graph surgery compilation"
                rag_context = self.retriever.retrieve(
                    query=query,
                    model_category=model_category,
                    op_type=op_type,
                    top_k=max_chunks
                )
                
                if rag_context and rag_context.chunks:
                    rag_text = rag_context.get_text(max_chunks=max_chunks)
                    if rag_text.strip():
                        context_parts.append("=== KNOWLEDGE BASE CONTEXT ===")
                        context_parts.append(rag_text)
            except Exception as e:
                # Silently fall back to surgery-db-only context
                pass
        
        # Combine contexts
        if not context_parts:
            return ""
        
        return "\n".join(context_parts)
    
    def _enhance_suggestions_batch(
        self,
        suggestions: List[Suggestion],
        model_path: str,
        model_category: str,
        analysis: SuggestionReport
    ) -> Tuple[List[Suggestion], int]:
        """
        Enhance multiple suggestions in a single API call.
        
        Args:
            suggestions: List of suggestions to enhance (should share same op_type)
            model_path: Path to model
            model_category: Detected model category
            analysis: Full analysis report
            
        Returns:
            Tuple of (List of enhanced suggestions, count of enhancements applied)
        """
        if not self.use_rag or (not self.retriever and not self.context_generator) or not suggestions:
            return suggestions, 0
        
        # Use the first suggestion's op_type (they should all be the same after grouping)
        op_type = suggestions[0].location.op_type
        
        # Get enriched context from both sources (surgery db + RAG retriever)
        context_text = self._get_enriched_context(
            op_type=op_type,
            model_category=model_category,
            max_chunks=5
        )
        
        if not context_text.strip():
            # No relevant context found, return originals
            return suggestions, 0
        
        # Build batch prompt with all suggestions (include exact location info)
        suggestions_data = []
        for suggestion in suggestions:
            node_details = {
                'suggestion_id': suggestion.id,
                'node_name': suggestion.location.node_name,  # Exact node name for location
                'node_id': suggestion.location.node_id,  # Exact node ID for location
                'op_type': suggestion.location.op_type,
                'issue': suggestion.issue,
                'current_suggestion': suggestion.suggestion,
                'priority': suggestion.priority.value if hasattr(suggestion.priority, 'value') else str(suggestion.priority),
                'inputs': suggestion.location.inputs,
                'outputs': suggestion.location.outputs,
                'location_info': f"Node '{suggestion.location.node_name}' (node_id: {suggestion.location.node_id}, op_type: {suggestion.location.op_type})"  # Human-readable location
            }
            suggestions_data.append(node_details)
        
        prompt = self.RAG_BATCH_SUGGESTION_PROMPT.format(
            context=context_text,
            model_category=model_category,
            op_type=op_type,
            count=len(suggestions),
            suggestions=json.dumps(suggestions_data, indent=2)
        )
        
        # Single API call for the batch (cached via cached_gemini_call)
        try:
            response = cached_gemini_call(
                prompt=prompt,
                api_key=self.api_key,
                model_name="models/gemini-3-pro-preview",
                temperature=0.1,  # Lower temperature for more consistent JSON
                max_tokens=8192  # Increased tokens to prevent truncation
            )
            
            if not response:
                # Handle None response gracefully - return originals
                print(f"    Warning: No response from API for {op_type} batch ({len(suggestions)} suggestions)")
                return suggestions, 0
            
            # Get the actual suggestion IDs for matching
            suggestion_ids = [s.id for s in suggestions]
            
            # Parse batch response with actual IDs
            enhancements = self._parse_batch_enhancement_response(response, suggestion_ids)
            
            # Apply enhancements to suggestions (matched by ID)
            enhanced_suggestions = []
            enhanced_count = 0
            for suggestion in suggestions:
                enhancement = enhancements.get(suggestion.id)
                if enhancement:
                    suggestion = self._apply_enhancement(suggestion, enhancement, context)
                    enhanced_count += 1
                enhanced_suggestions.append(suggestion)
            
            if enhanced_count > 0:
                print(f"    Successfully enhanced {enhanced_count}/{len(suggestions)} suggestions")
            else:
                print(f"    Warning: No enhancements applied for {op_type} batch - using original suggestions")
            
            return enhanced_suggestions, enhanced_count
        
        except Exception as e:
            print(f"  Warning: Batch RAG enhancement failed for {op_type} ({len(suggestions)} suggestions): {e}")
            print(f"  Continuing with original suggestions for this batch...")
            return suggestions, 0  # Return originals on error
    
    def _validate_enhancement(self, enhancement: Dict) -> bool:
        """Validate enhancement has required fields and valid data."""
        # Check for suggestion_id (required)
        if 'suggestion_id' not in enhancement:
            return False
        
        # Validate suggestion_id is valid integer
        sid = enhancement['suggestion_id']
        if not isinstance(sid, (int, float)):
            return False
        
        # Check for enhanced_suggestion OR steps (need at least one useful field)
        has_suggestion = enhancement.get('enhanced_suggestion', '').strip()
        has_steps = enhancement.get('steps') or enhancement.get('implementation_steps')
        
        if not has_suggestion and not has_steps:
            return False
        
        return True
    
    def _parse_batch_enhancement_response(self, response: str, suggestion_ids: List[int]) -> Dict[int, Dict]:
        """
        Parse batch enhancement response with multi-stage recovery.
        
        Args:
            response: Gemini API response
            suggestion_ids: List of actual suggestion IDs to match
            
        Returns:
            Dictionary mapping suggestion_id to enhancement dict
        """
        expected_count = len(suggestion_ids)
        
        if not response:
            return {}
        
        original_response = response
        
        # Stage 1: Try standard JSON parsing
        try:
            result = self._parse_standard_json_array(response, suggestion_ids)
            if result and len(result) >= expected_count * 0.5:
                return result
        except:
            pass
        
        # Stage 2: Extract individual complete objects
        try:
            result = self._extract_individual_objects(response, suggestion_ids)
            if result and len(result) >= expected_count * 0.3:
                print(f"    Recovered {len(result)}/{expected_count} enhancements from broken JSON")
                return result
        except:
            pass
        
        # Stage 3: Recover partial objects
        try:
            result = self._recover_partial_objects(response, suggestion_ids)
            if result and len(result) >= expected_count * 0.2:
                print(f"    Partially recovered {len(result)}/{expected_count} enhancements")
                return result
        except:
            pass
        
        # Stage 4: Regex-based field extraction (last resort)
        try:
            result = self._extract_fields_regex(response, suggestion_ids)
            if result:
                print(f"    Extracted {len(result)}/{expected_count} enhancements via regex")
                return result
        except:
            pass
        
        print(f"    Error: Could not parse batch response (tried 4 recovery strategies)")
        print(f"    Response preview: {original_response[:300]}...")
        return {}
    
    def _parse_standard_json_array(self, response: str, suggestion_ids: List[int]) -> Dict[int, Dict]:
        """Stage 1: Standard JSON array parsing."""
        response = response.strip()
        
        # Extract JSON array from response
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end == -1:
                end = len(response)
            response = response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end == -1:
                end = len(response)
            response = response[start:end].strip()
        
        # Find array boundaries
        if not response.startswith('['):
            start_idx = response.find('[')
            if start_idx != -1:
                response = response[start_idx:]
        
        if not response.endswith(']'):
            end_idx = response.rfind(']')
            if end_idx != -1:
                response = response[:end_idx + 1]
        
        # Clean up common JSON issues
        response = self._fix_json_string(response)
        
        # Parse JSON array
        enhancements = json.loads(response)
        
        if not isinstance(enhancements, list):
            raise ValueError("Response is not a JSON array")
        
        # Convert to dict keyed by suggestion_id (using actual IDs from response)
        enhancement_map = {}
        valid_ids = set(suggestion_ids)
        for enh in enhancements:
            if isinstance(enh, dict) and self._validate_enhancement(enh):
                suggestion_id = int(enh['suggestion_id'])
                # Only include if ID matches one we're looking for
                if suggestion_id in valid_ids:
                    enhancement_map[suggestion_id] = enh
        
        return enhancement_map
    
    def _extract_individual_objects(self, response: str, suggestion_ids: List[int]) -> Dict[int, Dict]:
        """Stage 2: Extract individual complete JSON objects."""
        valid_ids = set(suggestion_ids)
        enhancement_map = {}
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(response):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    # Found a complete object
                    obj_str = response[start_pos:i+1]
                    try:
                        obj = json.loads(obj_str)
                        if self._validate_enhancement(obj):
                            sid = int(obj['suggestion_id'])
                            if sid in valid_ids:
                                enhancement_map[sid] = obj
                    except:
                        # Try to fix and parse
                        try:
                            fixed_obj = self._fix_json_string(obj_str)
                            obj = json.loads(fixed_obj)
                            if self._validate_enhancement(obj):
                                sid = int(obj['suggestion_id'])
                                if sid in valid_ids:
                                    enhancement_map[sid] = obj
                        except:
                            pass
                    start_pos = -1
        
        return enhancement_map
    
    def _recover_partial_objects(self, response: str, suggestion_ids: List[int]) -> Dict[int, Dict]:
        """Stage 3: Recover partial objects and complete them."""
        import re
        valid_ids = set(suggestion_ids)
        enhancement_map = {}
        brace_count = 0
        start_pos = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(response):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        start_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        # Found a potentially complete object
                        obj_str = response[start_pos:i+1]
                        # Try to extract suggestion_id even if JSON is broken
                        id_match = re.search(r'"suggestion_id"\s*:\s*(\d+)', obj_str)
                        if id_match:
                            suggestion_id = int(id_match.group(1))
                            if suggestion_id in valid_ids:
                                # Try to extract other fields
                                enhanced_match = re.search(r'"enhanced_suggestion"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', obj_str)
                                if enhanced_match:
                                    enhancement_map[suggestion_id] = {
                                        'suggestion_id': suggestion_id,
                                        'enhanced_suggestion': enhanced_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                                    }
                        start_pos = -1
        
        return enhancement_map
    
    def _extract_fields_regex(self, response: str, suggestion_ids: List[int]) -> Dict[int, Dict]:
        """Stage 4: Regex-based field extraction (last resort)."""
        import re
        enhancement_map = {}
        
        for sid in suggestion_ids:
            # Find suggestion_id in the response
            id_pattern = rf'"suggestion_id"\s*:\s*{sid}\s*[,}}\n]'
            id_match = re.search(id_pattern, response)
            
            if id_match:
                # Found suggestion_id, now try to extract enhanced_suggestion
                start_pos = id_match.start()
                # Search in a window around the suggestion_id (next 1000 chars)
                search_window = response[start_pos:start_pos + 1000]
                
                # Try to find enhanced_suggestion in this window
                enhanced_pattern = r'"enhanced_suggestion"\s*:\s*"((?:[^"\\]|\\.)*)"'
                enhanced_match = re.search(enhanced_pattern, search_window)
                
                if enhanced_match:
                    enhanced_text = enhanced_match.group(1)
                    enhanced_text = enhanced_text.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                    enhancement_map[sid] = {
                        'suggestion_id': sid,
                        'enhanced_suggestion': enhanced_text
                    }
                else:
                    # Try simpler pattern
                    simple_pattern = r'"enhanced_suggestion"\s*:\s*"([^"]+)"'
                    simple_match = re.search(simple_pattern, search_window)
                    if simple_match:
                        enhancement_map[sid] = {
                            'suggestion_id': sid,
                            'enhanced_suggestion': simple_match.group(1)
                        }
        
        return enhancement_map
    
    def _parse_enhancement_response(self, response: str) -> Optional[Dict]:
        """Parse Gemini response to extract enhancement."""
        if not response:
            return None
        
        try:
            # Try to extract JSON from response
            original_response = response
            response = response.strip()
            
            # Find JSON block
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end == -1:
                    end = len(response)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end == -1:
                    end = len(response)
                response = response[start:end].strip()
            
            # Try to find JSON object boundaries if not in code block
            if not response.startswith('{'):
                # Look for first {
                start_idx = response.find('{')
                if start_idx != -1:
                    response = response[start_idx:]
            
            # Clean up common JSON issues before parsing
            response = self._fix_json_string(response)
            
            # Try to find the end of the JSON object
            if not response.endswith('}'):
                # Look for last } that might close the object
                end_idx = response.rfind('}')
                if end_idx != -1:
                    response = response[:end_idx + 1]
            
            # Parse JSON
            enhancement = json.loads(response)
            return enhancement
        
        except json.JSONDecodeError as e:
            # Try more aggressive fixes
            try:
                # Find first { and try to fix the string
                start_idx = original_response.find('{')
                if start_idx != -1:
                    # Extract from first { to end
                    json_candidate = original_response[start_idx:]
                    json_candidate = self._fix_json_string(json_candidate)
                    
                    # Try to find a valid closing brace
                    brace_count = 0
                    last_valid_pos = -1
                    for i, char in enumerate(json_candidate):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                last_valid_pos = i
                                break
                    
                    if last_valid_pos > 0:
                        json_candidate = json_candidate[:last_valid_pos + 1]
                        enhancement = json.loads(json_candidate)
                        return enhancement
            except:
                pass
            
            # Last resort: try to extract just the fields we need
            try:
                enhancement = self._extract_fields_from_broken_json(original_response)
                if enhancement:
                    return enhancement
            except:
                pass
            
            # If all else fails, return None (will use original suggestion)
            return None
        except Exception as e:
            # Handle None response or other errors
            return None
    
    def _fix_json_string(self, json_str: str) -> str:
        """Fix common JSON issues in string."""
        import re
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _extract_fields_from_broken_json(self, text: str) -> Optional[Dict]:
        """Extract fields from broken JSON using regex as last resort."""
        import re
        result = {}
        
        # Try to extract enhanced_suggestion
        match = re.search(r'"enhanced_suggestion"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
        if not match:
            # Try without proper escaping
            match = re.search(r'"enhanced_suggestion"\s*:\s*"([^"]+)"', text)
        if match:
            result['enhanced_suggestion'] = match.group(1).replace('\\"', '"').replace('\\n', '\n')
        
        # Try to extract implementation_steps
        steps_match = re.search(r'"implementation_steps"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if steps_match:
            steps_text = steps_match.group(1)
            steps = re.findall(r'"([^"]+)"', steps_text)
            if steps:
                result['implementation_steps'] = steps
        
        # Try to extract confidence_boost
        conf_match = re.search(r'"confidence_boost"\s*:\s*([0-9.]+)', text)
        if conf_match:
            try:
                result['confidence_boost'] = float(conf_match.group(1))
            except:
                pass
        
        return result if result else None
    
    def _apply_enhancement(
        self,
        suggestion: Suggestion,
        enhancement: Dict,
        context: Any
    ) -> Suggestion:
        """Apply enhancement to suggestion."""
        # Update suggestion text (should include location info from prompt)
        if 'enhanced_suggestion' in enhancement:
            suggestion.suggestion = enhancement['enhanced_suggestion']
        
        # Update location if provided in enhancement (for clarity)
        if 'location' in enhancement and isinstance(enhancement['location'], dict):
            loc = enhancement['location']
            if 'node_name' in loc:
                suggestion.location.node_name = loc['node_name']
            if 'node_id' in loc:
                suggestion.location.node_id = int(loc['node_id'])
            if 'op_type' in loc:
                suggestion.location.op_type = loc['op_type']
        
        # Update implementation steps (use 'steps' or 'implementation_steps')
        if 'steps' in enhancement:
            suggestion.implementation_steps = enhancement['steps']
        elif 'implementation_steps' in enhancement:
            suggestion.implementation_steps = enhancement['implementation_steps']
        
        # Boost confidence
        if 'confidence_boost' in enhancement:
            boost = float(enhancement['confidence_boost'])
            suggestion.confidence = min(1.0, suggestion.confidence + boost)
        
        # Add warnings to impact
        if 'warnings' in enhancement and enhancement['warnings']:
            warnings_text = "\n".join([f"⚠️ {w}" for w in enhancement['warnings']])
            suggestion.impact = f"{suggestion.impact}\n\n{warnings_text}"
        
        # Update reference
        if 'reference' in enhancement:
            suggestion.reference = enhancement['reference']
        
        # Add alternative approaches to impact
        if 'alternative_approaches' in enhancement and enhancement['alternative_approaches']:
            alt_text = "\n".join([f"• {alt}" for alt in enhancement['alternative_approaches']])
            suggestion.impact = f"{suggestion.impact}\n\nAlternative approaches:\n{alt_text}"
        
        return suggestion
    
    def _generate_pattern_based_suggestions(
        self,
        model_path: str,
        model_category: str,
        analysis: ModelAnalysis,
        report: SuggestionReport
    ) -> List[Suggestion]:
        """
        Generate suggestions based on learned patterns from KB.
        
        This is Phase 1 of generalization: Use patterns learned from training data
        to generate suggestions for operations that should be removed/added.
        """
        if not self.use_rag or not self.retriever:
            return []
        
        suggestions = []
        
        try:
            import onnx
            model = onnx.load(model_path)
        except Exception as e:
            print(f"  Warning: Could not load model for pattern-based suggestions: {e}")
            return []
        
        # 1. Query KB for removal patterns for this model category
        # Use keyword matching for pattern queries (simpler, no API calls needed)
        # Pattern queries are straightforward and don't need semantic search
        removal_query = f"remove {model_category} operations"
        removal_context = self.retriever.retrieve(removal_query, top_k=15)
        
        # If semantic search didn't find results, try keyword-only retrieval
        if not removal_context or not removal_context.chunks:
            # Force keyword matching by temporarily disabling embeddings
            original_use_embeddings = self.retriever.use_embeddings
            self.retriever.use_embeddings = False
            removal_context = self.retriever.retrieve(removal_query, top_k=15)
            self.retriever.use_embeddings = original_use_embeddings
        
        if not removal_context or not removal_context.chunks:
            print(f"    No removal patterns found in KB for {model_category} models")
            return []
        
        # 2. Extract operation types with sufficient frequency
        ops_to_remove = self._extract_ops_from_patterns(
            removal_context.chunks,
            min_frequency=2,  # At least 2 models in training
            pattern_type='removal',
            model_category=model_category
        )
        
        if not ops_to_remove:
            print(f"    No removal operations found with sufficient frequency (≥2) for {model_category}")
            return []
        
        print(f"    Found {len(ops_to_remove)} operation types to check for removal: {list(ops_to_remove.keys())}")
        
        # 3. Scan model for these operations
        node_map = {node.name: node for node in model.graph.node}
        # Track existing suggestions to avoid duplicates
        existing_suggestion_nodes = {s.location.node_name for s in report.suggestions}
        
        for node in model.graph.node:
            if node.op_type in ops_to_remove:
                # Skip if we already have a suggestion for this node
                if node.name in existing_suggestion_nodes:
                    continue
                
                # Check if pattern applies (use context from KB)
                pattern_context = self._get_pattern_context(
                    node.op_type,
                    model_category,
                    removal_context.chunks
                )
                
                if self._should_apply_pattern(node, pattern_context, model):
                    suggestion = self._create_removal_suggestion(
                        node,
                        pattern_context,
                        analysis,
                        len(report.suggestions) + len(suggestions) + 1000  # Offset to avoid conflicts
                    )
                    suggestions.append(suggestion)
                    existing_suggestion_nodes.add(node.name)  # Track to avoid duplicates
        
        # 4. Similar for addition patterns
        # Use multiple query strategies to find addition patterns
        addition_queries = [
            f"add {model_category} operations",
            f"add operations {model_category}",
            f"{model_category} add",
            "add sigmoid",  # Explicit query for common patterns
            "add activation",
            "add operations"
        ]
        
        addition_context = None
        for query in addition_queries:
            addition_context = self.retriever.retrieve(query, top_k=15)
            if addition_context and addition_context.chunks:
                break
        
        # If semantic search didn't find results, try keyword-only retrieval
        if not addition_context or not addition_context.chunks:
            # Force keyword matching by temporarily disabling embeddings
            original_use_embeddings = self.retriever.use_embeddings
            self.retriever.use_embeddings = False
            for query in addition_queries:
                addition_context = self.retriever.retrieve(query, top_k=15)
                if addition_context and addition_context.chunks:
                    break
            self.retriever.use_embeddings = original_use_embeddings
        
        ops_to_add = {}
        if addition_context and addition_context.chunks:
            # Lower min_frequency to catch patterns learned from training data
            # The KB has patterns with frequency 45, so min_frequency=2 is safe
            ops_to_add = self._extract_ops_from_patterns(
                addition_context.chunks,
                min_frequency=2,  # Pattern must appear in at least 2 training models
                pattern_type='addition',
                model_category=model_category
            )
            
            if ops_to_add:
                print(f"    Found {len(ops_to_add)} addition patterns from KB: {list(ops_to_add.keys())}")
        
        # Generate addition suggestions
        for op_type in ops_to_add:
            target_nodes = self._find_addition_targets(
                model,
                op_type,
                model_category,
                addition_context.chunks if addition_context else []
            )
            for target_node in target_nodes:
                pattern_context = self._get_pattern_context(
                    op_type,
                    model_category,
                    addition_context.chunks if addition_context else []
                )
                suggestion = self._create_addition_suggestion(
                    target_node,
                    op_type,
                    pattern_context,
                    analysis,
                    len(report.suggestions) + len(suggestions) + 1000  # Offset to avoid conflicts
                )
                suggestions.append(suggestion)
        
        # Don't print here - it's printed in analyze_and_suggest() after merging
        return suggestions
    
    def _extract_ops_from_patterns(
        self,
        chunks: List[Any],
        min_frequency: int,
        pattern_type: str,  # 'removal' or 'addition'
        model_category: str
    ) -> Dict[str, int]:
        """Extract operation types and their frequencies from KB chunks."""
        op_frequencies = {}
        
        for chunk in chunks:
            if not hasattr(chunk, 'metadata'):
                continue
            
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            
            # Check if this is the right pattern type and category
            pattern_name = metadata.get('pattern_name', '').lower()
            chunk_pattern_type = metadata.get('pattern_type', '').lower()
            chunk_category = metadata.get('model_category', 'Other')
            
            # More flexible pattern matching - check both pattern_name and pattern_type
            if pattern_type == 'removal':
                # Match if pattern_type is 'removal' OR pattern_name contains 'remove'/'delete'
                if chunk_pattern_type != 'removal' and 'remove' not in pattern_name and 'delete' not in pattern_name:
                    continue
            elif pattern_type == 'addition':
                # Match if pattern_type is 'addition' OR pattern_name contains add-related words
                if chunk_pattern_type != 'addition' and not any(word in pattern_name for word in ['add', 'insert', 'create', 'introduce', 'append']):
                    continue
            
            # Category matching: allow 'Other' to match any category, or exact match
            if chunk_category != model_category and model_category != 'Other' and chunk_category != 'Other':
                continue  # Only match patterns for same category (unless one is 'Other')
            
            # Extract operation types
            op_types = metadata.get('op_types', [])
            frequency = metadata.get('frequency', 0)
            
            # Also try to extract from 'removed_ops' or 'added_ops' if op_types is empty
            if not op_types:
                if pattern_type == 'removal':
                    op_types = metadata.get('removed_ops', [])
                elif pattern_type == 'addition':
                    op_types = metadata.get('added_ops', [])
            
            if frequency >= min_frequency:
                for op_type in op_types:
                    # Use max frequency to prioritize high-confidence patterns
                    op_frequencies[op_type] = max(
                        op_frequencies.get(op_type, 0),
                        frequency
                    )
        
        return op_frequencies
    
    def _get_pattern_context(
        self,
        op_type: str,
        model_category: str,
        chunks: List[Any]
    ) -> Dict:
        """Extract context about when pattern applies."""
        context = {
            'frequency': 0,
            'model_category': model_category,
            'position': 'any',  # 'near_output', 'near_input', 'any'
            'surrounding_ops': [],
            'confidence': 0.0,
            'example_models': []
        }
        
        # Find matching chunk
        for chunk in chunks:
            if not hasattr(chunk, 'metadata'):
                continue
            
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            op_types = metadata.get('op_types', [])
            
            # Also check removed_ops and added_ops
            if not op_types:
                op_types = metadata.get('removed_ops', []) + metadata.get('added_ops', [])
            
            chunk_category = metadata.get('model_category', 'Other')
            
            # Match if op_type is in the chunk's op_types and category matches (or is 'Other')
            if op_type in op_types and (chunk_category == model_category or model_category == 'Other' or chunk_category == 'Other'):
                context['frequency'] = max(context['frequency'], metadata.get('frequency', 1))
                chunk_context = metadata.get('context', {})
                if isinstance(chunk_context, dict):
                    context['position'] = chunk_context.get('position', 'any')
                    context['surrounding_ops'] = chunk_context.get('surrounding_ops', [])
                
                # Get example models
                example_models = metadata.get('example_models', [])
                if example_models:
                    for model in example_models:
                        if model not in context['example_models']:
                            context['example_models'].append(model)
                
                # Calculate confidence based on frequency
                # Higher frequency = higher confidence
                context['confidence'] = min(context['frequency'] / 5.0, 1.0)
                # Don't break - continue to aggregate from multiple matching chunks
        
        return context
    
    def _should_apply_pattern(
        self,
        node: Any,
        pattern_context: Dict,
        model: Any
    ) -> bool:
        """Determine if pattern should be applied to this node."""
        # 1. Check frequency threshold
        if pattern_context['frequency'] < 2:
            return False
        
        # 2. Check position if specified
        if pattern_context['position'] != 'any':
            try:
                node_idx = list(model.graph.node).index(node)
                total_nodes = len(model.graph.node)
                position_ratio = node_idx / total_nodes if total_nodes > 0 else 0.5
                
                if pattern_context['position'] == 'near_output' and position_ratio < 0.8:
                    return False
                if pattern_context['position'] == 'near_input' and position_ratio > 0.2:
                    return False
            except (ValueError, AttributeError):
                pass  # If we can't determine position, allow it
        
        return True
    
    def _create_removal_suggestion(
        self,
        node: Any,
        pattern_context: Dict,
        analysis: ModelAnalysis,
        suggestion_id: int
    ) -> Suggestion:
        """Create a suggestion for removing a node based on pattern."""
        # Find corresponding NodeAnalysis or create minimal one
        node_analysis = None
        for n in analysis.nodes:
            if n.name == node.name:  # NodeAnalysis uses 'name', not 'node_name'
                node_analysis = n
                break
        
        if not node_analysis:
            # Create minimal NodeAnalysis - need to provide all required fields
            from core_analysis.onnx_analyzer import NodeAnalysis
            node_analysis = NodeAnalysis(
                node_id=len(analysis.nodes),
                name=node.name,  # Use 'name', not 'node_name'
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                input_shapes=[None] * len(node.input),
                output_shapes=[None] * len(node.output),
                attributes={},
                input_types=[''] * len(node.input),
                output_types=[''] * len(node.output),
                dependencies=[],
                dependents=[],
                is_compilation_blocker=False,
                blocker_reason=None
            )
        
        # Determine priority based on frequency
        if pattern_context['frequency'] >= 3:
            priority = Priority.HIGH
        else:
            priority = Priority.MEDIUM
        
        # Create suggestion
        suggestion = Suggestion(
            id=suggestion_id,
            priority=priority,
            confidence=0.7 + (pattern_context['confidence'] * 0.2),  # 0.7-0.9
            issue=(
                f"Pattern indicates {node.op_type} should be removed in {pattern_context['model_category']} models "
                f"(seen in {pattern_context['frequency']} training models)"
            ),
            location=SuggestionLocation(
                node_id=node_analysis.node_id,
                node_name=node_analysis.name,  # Use node_analysis.name
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output)
            ),
            suggestion=(
                f"Remove {node.op_type} node based on learned pattern from "
                f"{pattern_context['frequency']} similar {pattern_context['model_category']} models"
            ),
            implementation_steps=[
                f"1. Identify {node.op_type} node: {node.name}",
                f"2. Rewire inputs to outputs (remove node)",
                f"3. Remove unused constants if any",
                f"4. Validate model structure"
            ],
            impact=(
                f"Follows pattern learned from {pattern_context['frequency']} training models. "
                f"This operation is commonly removed in {pattern_context['model_category']} models."
            ),
            reference=(
                f"Knowledge Base Pattern: Remove {node.op_type} in {pattern_context['model_category']} models"
            ),
            category='pattern_based_removal',
            estimated_effort='simple'
        )
        
        return suggestion
    
    def _find_addition_targets(
        self,
        model: Any,
        op_type: str,
        model_category: str,
        chunks: List[Any]
    ) -> List[Any]:
        """Find target nodes where operation should be added based on learned patterns."""
        targets = []
        nodes = list(model.graph.node)
        graph_outputs = set(out.name for out in model.graph.output)
        
        # Build tensor to producer map (graph outputs are tensor names, not node outputs)
        tensor_to_producer = {}
        for node in nodes:
            for output in node.output:
                tensor_to_producer[output] = node
        
        # Strategy for Sigmoid: Add at output heads (common in YOLO models)
        if op_type == 'Sigmoid':
            # Check if Sigmoid already exists at outputs
            has_sigmoid_at_output = False
            for node in nodes:
                if node.op_type == 'Sigmoid':
                    for output in node.output:
                        if output in graph_outputs:
                            has_sigmoid_at_output = True
                            break
                    if has_sigmoid_at_output:
                        break
            
            if has_sigmoid_at_output:
                return []  # Already has Sigmoid at outputs
            
            # Strategy 1: Find producer nodes of graph outputs
            for graph_output_name in graph_outputs:
                producer = tensor_to_producer.get(graph_output_name)
                if producer:
                    # Check if Sigmoid already exists after this producer
                    has_sigmoid_after = False
                    for n in nodes:
                        if n.op_type == 'Sigmoid' and graph_output_name in n.input:
                            has_sigmoid_after = True
                            break
                    
                    if not has_sigmoid_after and producer not in targets:
                        targets.append(producer)
                        break
            
            # Strategy 2: Look for Split nodes before outputs (common in YOLO)
            if not targets:
                for node in reversed(nodes):
                    if node.op_type == 'Split':
                        for output in node.output:
                            if output in graph_outputs:
                                has_sigmoid_after = False
                                for n in nodes:
                                    if n.op_type == 'Sigmoid' and output in n.input:
                                        has_sigmoid_after = True
                                        break
                                if not has_sigmoid_after and node not in targets:
                                    targets.append(node)
                            break
                        if targets:
                            break
            
            # Strategy 3: Look for Concat nodes before outputs
            if not targets:
                for node in reversed(nodes):
                    if node.op_type == 'Concat':
                        for output in node.output:
                            if output in graph_outputs:
                                has_sigmoid_after = False
                                for n in nodes:
                                    if n.op_type == 'Sigmoid' and output in n.input:
                                        has_sigmoid_after = True
                                        break
                                if not has_sigmoid_after and node not in targets:
                                    targets.append(node)
                                    break
                        if targets:
                            break
        
        # Generic strategy for other operations: add after nodes that feed into outputs
        else:
            for graph_output_name in graph_outputs:
                producer = tensor_to_producer.get(graph_output_name)
                if producer and producer not in targets:
                    targets.append(producer)
                    if len(targets) >= 3:  # Limit targets
                        break
        
        return targets
    
    def _create_addition_suggestion(
        self,
        target_node: Any,
        op_type: str,
        pattern_context: Dict,
        analysis: ModelAnalysis,
        suggestion_id: int
    ) -> Suggestion:
        """Create a suggestion for adding a node based on pattern."""
        # Find corresponding NodeAnalysis
        node_analysis = None
        for n in analysis.nodes:
            if n.name == target_node.name:  # NodeAnalysis uses 'name', not 'node_name'
                node_analysis = n
                break
        
        if not node_analysis:
            from core_analysis.onnx_analyzer import NodeAnalysis
            node_analysis = NodeAnalysis(
                node_id=len(analysis.nodes),
                name=target_node.name,  # Use 'name', not 'node_name'
                op_type=target_node.op_type,
                inputs=list(target_node.input),
                outputs=list(target_node.output),
                input_shapes=[None] * len(target_node.input),
                output_shapes=[None] * len(target_node.output),
                attributes={},
                input_types=[''] * len(target_node.input),
                output_types=[''] * len(target_node.output),
                dependencies=[],
                dependents=[],
                is_compilation_blocker=False,
                blocker_reason=None
            )
        
        priority = Priority.HIGH if pattern_context['frequency'] >= 3 else Priority.MEDIUM
        
        suggestion = Suggestion(
            id=suggestion_id,
            priority=priority,
            confidence=0.7 + (pattern_context['confidence'] * 0.2),
            issue=(
                f"Pattern indicates {op_type} should be added in {pattern_context['model_category']} models "
                f"(seen in {pattern_context['frequency']} training models)"
            ),
            location=SuggestionLocation(
                node_id=node_analysis.node_id,
                node_name=node_analysis.name,  # Use node_analysis.name
                op_type=target_node.op_type,
                inputs=list(target_node.input),
                outputs=list(target_node.output)
            ),
            suggestion=(
                f"Add {op_type} node after {target_node.name} based on learned pattern from "
                f"{pattern_context['frequency']} similar {pattern_context['model_category']} models"
            ),
            implementation_steps=[
                f"1. Identify target node: {target_node.name}",
                f"2. Create {op_type} node",
                f"3. Insert after target node",
                f"4. Update graph outputs if needed",
                f"5. Validate model structure"
            ],
            impact=(
                f"Follows pattern learned from {pattern_context['frequency']} training models. "
                f"This operation is commonly added in {pattern_context['model_category']} models."
            ),
            reference=(
                f"Knowledge Base Pattern: Add {op_type} in {pattern_context['model_category']} models"
            ),
            category='pattern_based_addition',
            estimated_effort='simple'
        )
        
        return suggestion
    
    def _recompute_report_stats(self, report: SuggestionReport) -> SuggestionReport:
        """Recompute report statistics after enhancement."""
        # Re-sort by priority and confidence
        report.suggestions = self._prioritize_suggestions(report.suggestions)
        
        # Recompute counts
        report.critical_count = sum(1 for s in report.suggestions if s.priority == Priority.CRITICAL)
        report.high_count = sum(1 for s in report.suggestions if s.priority == Priority.HIGH)
        report.medium_count = sum(1 for s in report.suggestions if s.priority == Priority.MEDIUM)
        report.low_count = sum(1 for s in report.suggestions if s.priority == Priority.LOW)
        report.total_issues = len(report.suggestions)
        
        return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG-enhanced suggestion generator')
    parser.add_argument('model', help='Path to ONNX model')
    parser.add_argument('--kb', default='knowledge_base.json',
                       help='Path to knowledge base')
    parser.add_argument('--api-key', help='Gemini API key')
    parser.add_argument('--no-rag', action='store_true',
                       help='Disable RAG enhancement')
    parser.add_argument('--hybrid', action='store_true',
                       help='Enable hybrid mode (batch + individual enhancement)')
    parser.add_argument('--daily-limit', type=int, default=250,
                       help='Daily API limit (default: 250)')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory for checkpoint files (default: checkpoints)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if exists')
    parser.add_argument('--individual-threshold', type=int, default=None,
                       help='Number of suggestions for individual enhancement (default: auto)')
    
    args = parser.parse_args()
    
    generator = RAGSuggestionGenerator(
        kb_path=args.kb,
        api_key=args.api_key,
        use_rag=not args.no_rag,
        use_hybrid_mode=args.hybrid,
        daily_api_limit=args.daily_limit,
        checkpoint_dir=args.checkpoint_dir,
        individual_enhancement_threshold=args.individual_threshold
    )
    
    if args.hybrid:
        report = generator.analyze_and_suggest_hybrid(args.model, resume=args.resume)
    else:
        report = generator.analyze_and_suggest(args.model)
    
    print(json.dumps(report.to_dict(), indent=2))

