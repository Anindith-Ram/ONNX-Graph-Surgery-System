#!/usr/bin/env python3
"""
Build Surgery Database from Raw ONNX Model Pairs.

This script builds the unified surgery database from scratch by:
1. Scanning the dataset directory for original/modified model pairs
2. Extracting precise transformations from each pair
3. Generating templates from common patterns
4. Saving everything in JSON format

NO LEGACY DATA MIGRATION - This builds fresh from raw data only.

Usage:
    python scripts/build_surgery_db.py --dataset dataset/ --output rag_data/
    python scripts/build_surgery_db.py --dataset dataset/ --verbose
    
Author: Automated Model Surgery Pipeline
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.surgery_database import (
    SurgeryDatabase,
    SurgeryTemplate,
    SurgeryStep,
    CompilationBlocker,
    NodeTransformation,
    TransformationRecord,
    create_database_with_defaults,
    DEFAULT_COMPILATION_BLOCKERS
)
from core_analysis.transformation_extractor import (
    TransformationExtractor,
    extract_all_from_dataset
)
from core_analysis.onnx_analyzer import ONNXAnalyzer


# =============================================================================
# Template Generation
# =============================================================================

class TemplateGenerator:
    """
    Generate surgery templates from extracted transformations.
    
    Analyzes common patterns across models and creates reusable templates.
    """
    
    def __init__(self, min_occurrences: int = 2, min_confidence: float = 0.7):
        """
        Args:
            min_occurrences: Minimum number of occurrences to create a template
            min_confidence: Minimum confidence threshold for templates
        """
        self.min_occurrences = min_occurrences
        self.min_confidence = min_confidence
    
    def generate_templates(
        self, 
        db: SurgeryDatabase
    ) -> List[SurgeryTemplate]:
        """
        Generate templates from transformation patterns in the database.
        
        Returns:
            List of generated SurgeryTemplate objects
        """
        templates = []
        
        # Group transformations by pattern
        patterns = self._group_by_pattern(db)
        
        # Generate templates for common patterns
        for pattern_key, transformations in patterns.items():
            if len(transformations) >= self.min_occurrences:
                template = self._create_template_from_pattern(
                    pattern_key, transformations
                )
                if template and template.confidence >= self.min_confidence:
                    templates.append(template)
        
        return templates
    
    def _group_by_pattern(
        self, 
        db: SurgeryDatabase
    ) -> Dict[str, List[NodeTransformation]]:
        """Group transformations by pattern key."""
        patterns = defaultdict(list)
        
        for record in db.transformation_records:
            for transformation in record.transformations:
                # Create pattern key based on op_type, action, and context
                key = self._create_pattern_key(transformation, record.model_category)
                patterns[key].append(transformation)
        
        return patterns
    
    def _create_pattern_key(
        self, 
        transformation: NodeTransformation,
        model_category: str
    ) -> str:
        """Create a pattern key for grouping similar transformations."""
        # Include op_type, action, and category
        parts = [
            transformation.original_op_type or transformation.action,
            transformation.action,
            model_category
        ]
        
        # Include predecessor op types (sorted for consistency)
        pred_ops = sorted([
            p.get('op_type', '') 
            for p in transformation.predecessor_nodes
        ])[:3]
        if pred_ops:
            parts.append(f"pred:{','.join(pred_ops)}")
        
        # Include successor op types
        succ_ops = sorted([
            s.get('op_type', '') 
            for s in transformation.successor_nodes
        ])[:3]
        if succ_ops:
            parts.append(f"succ:{','.join(succ_ops)}")
        
        return "|".join(parts)
    
    def _create_template_from_pattern(
        self, 
        pattern_key: str,
        transformations: List[NodeTransformation]
    ) -> Optional[SurgeryTemplate]:
        """Create a template from a group of similar transformations."""
        if not transformations:
            return None
        
        # Get representative transformation
        rep = transformations[0]
        
        # Parse pattern key
        parts = pattern_key.split("|")
        op_type = parts[0] if parts else ""
        action = parts[1] if len(parts) > 1 else ""
        category = parts[2] if len(parts) > 2 else ""
        
        # Calculate confidence from blocker transformations
        blocker_count = sum(1 for t in transformations if t.is_compilation_blocker)
        success_count = len(transformations)  # All are successful (from modified models)
        confidence = min(0.95, 0.5 + (success_count / 10) + (blocker_count / success_count if blocker_count else 0) * 0.2)
        
        # Generate template ID
        template_id = f"{op_type.lower()}_{action}_{category.lower()}"
        template_id = template_id.replace(" ", "_").replace("-", "_")
        
        # Generate name
        name = f"{op_type} {action.title()} for {category}"
        
        # Generate description
        description = self._generate_description(rep, transformations)
        
        # Collect example models
        example_models = list(set(t.source_model for t in transformations))[:5]
        
        # Generate steps from first transformation with steps
        steps = []
        for t in transformations:
            if t.surgery_steps:
                steps = [
                    SurgeryStep(
                        step_number=i + 1,
                        action=self._infer_action_type(step),
                        description=step,
                        target_pattern=self._extract_target_pattern(step),
                        operation=self._extract_operation(step),
                        validation=""
                    )
                    for i, step in enumerate(t.surgery_steps[:6])
                ]
                break
        
        # Get GraphSurgeon code from first transformation with code
        graphsurgeon_code = ""
        for t in transformations:
            if t.code_snippet:
                graphsurgeon_code = t.code_snippet
                break
        
        # Generate warnings
        warnings = self._generate_warnings(transformations)
        
        # Generate contraindications
        contraindications = self._generate_contraindications(rep, category)
        
        # Get applicable categories
        applicable_categories = [category] if category and category != "Other" else []
        
        return SurgeryTemplate(
            template_id=template_id,
            name=name,
            description=description,
            trigger_op_type=op_type,
            trigger_conditions={
                'action': action,
                'min_occurrences': len(transformations)
            },
            applicable_categories=applicable_categories,
            confidence=confidence,
            success_count=success_count,
            failure_count=0,
            steps=steps,
            graphsurgeon_code=graphsurgeon_code,
            warnings=warnings,
            contraindications=contraindications,
            example_models=example_models
        )
    
    def _generate_description(
        self, 
        rep: NodeTransformation,
        all_transforms: List[NodeTransformation]
    ) -> str:
        """Generate a human-readable description for the template."""
        action = rep.action
        op_type = rep.original_op_type
        
        if action == "remove":
            if rep.is_compilation_blocker:
                return f"Remove {op_type} nodes that block compilation. {rep.blocker_reason or ''}"
            else:
                return f"Remove {op_type} nodes for graph simplification."
        
        elif action == "replace":
            replacement = " -> ".join(rep.replacement_ops) if rep.replacement_ops else "equivalent operations"
            return f"Replace {op_type} with {replacement} for hardware compatibility."
        
        elif action == "add":
            result_op = rep.result_node.get('op_type', 'node') if rep.result_node else 'node'
            return f"Add {result_op} node for proper graph structure."
        
        elif action == "reshape":
            return f"Modify {op_type} node shapes for static shape requirements."
        
        else:
            return f"{action.title()} {op_type} operations."
    
    def _infer_action_type(self, step: str) -> str:
        """Infer action type from step description."""
        step_lower = step.lower()
        
        if any(w in step_lower for w in ['identify', 'find', 'locate', 'note']):
            return 'identify'
        elif any(w in step_lower for w in ['create', 'add', 'insert']):
            return 'create_node'
        elif any(w in step_lower for w in ['rewire', 'connect', 'bypass']):
            return 'rewire'
        elif any(w in step_lower for w in ['remove', 'delete']):
            return 'remove'
        elif any(w in step_lower for w in ['validate', 'verify', 'check']):
            return 'validate'
        elif any(w in step_lower for w in ['update', 'change', 'modify']):
            return 'modify'
        else:
            return 'action'
    
    def _extract_target_pattern(self, step: str) -> str:
        """Extract target pattern from step description."""
        # Look for node names or op types in quotes or patterns
        if "'" in step:
            start = step.find("'")
            end = step.find("'", start + 1)
            if end > start:
                return step[start + 1:end]
        return ""
    
    def _extract_operation(self, step: str) -> str:
        """Extract operation details from step description."""
        # Return the step itself as the operation
        # Remove leading number if present
        if step and step[0].isdigit():
            idx = step.find('. ')
            if idx > 0:
                return step[idx + 2:]
        return step
    
    def _generate_warnings(
        self, 
        transformations: List[NodeTransformation]
    ) -> List[str]:
        """Generate warnings based on transformation patterns."""
        warnings = []
        
        # Check for shape-sensitive transformations
        shape_changes = sum(1 for t in transformations if any(
            tensor.get('is_dynamic', False) 
            for tensor in t.input_tensors
        ))
        if shape_changes > 0:
            warnings.append("Some inputs may have dynamic shapes - verify static dimensions")
        
        # Check for attribute variations
        attr_variations = set()
        for t in transformations:
            if t.attributes:
                attr_variations.update(t.attributes.keys())
        if len(attr_variations) > 3:
            warnings.append(f"Multiple attribute variations detected: {', '.join(list(attr_variations)[:3])}")
        
        return warnings[:3]  # Limit warnings
    
    def _generate_contraindications(
        self, 
        rep: NodeTransformation,
        category: str
    ) -> List[str]:
        """Generate contraindications for the template."""
        contraindications = []
        
        # Op-specific contraindications
        op_type = rep.original_op_type
        
        if op_type == "Einsum":
            contraindications.append("Einsum with more than 2 inputs")
            contraindications.append("Non-standard equation patterns")
        elif op_type in ("Reshape", "Transpose"):
            contraindications.append("If shape inference fails")
        elif op_type == "Loop":
            contraindications.append("If iteration count is truly dynamic")
        
        return contraindications


# =============================================================================
# Blocker Analyzer
# =============================================================================

class BlockerAnalyzer:
    """Analyze compilation blockers across all models."""
    
    def __init__(self, db: SurgeryDatabase):
        self.db = db
    
    def update_blocker_statistics(self) -> List[CompilationBlocker]:
        """
        Update compilation blocker statistics from the database.
        
        Returns updated list of CompilationBlocker objects.
        """
        # Count occurrences of each blocker type
        blocker_counts = defaultdict(int)
        blocker_categories = defaultdict(set)
        blocker_templates = defaultdict(set)
        
        for record in self.db.transformation_records:
            for t in record.transformations:
                if t.is_compilation_blocker:
                    blocker_counts[t.original_op_type] += 1
                    blocker_categories[t.original_op_type].add(record.model_category)
        
        # Map blockers to templates
        for template in self.db.surgery_templates:
            if template.trigger_op_type:
                blocker_templates[template.trigger_op_type].add(template.template_id)
        
        # Update existing blockers and add new ones
        updated_blockers = []
        existing_op_types = {b.op_type for b in self.db.compilation_blockers}
        
        for blocker in self.db.compilation_blockers:
            blocker.occurrence_count = blocker_counts.get(blocker.op_type, 0)
            blocker.affected_categories = list(blocker_categories.get(blocker.op_type, []))
            blocker.solution_templates = list(blocker_templates.get(blocker.op_type, blocker.solution_templates))
            updated_blockers.append(blocker)
        
        # Add newly discovered blockers
        for op_type, count in blocker_counts.items():
            if op_type not in existing_op_types:
                updated_blockers.append(CompilationBlocker(
                    op_type=op_type,
                    reason=f"Blocking operation found in {count} model transformations",
                    hardware_limitation="Detected from model surgery patterns",
                    solution_templates=list(blocker_templates.get(op_type, [])),
                    affected_categories=list(blocker_categories.get(op_type, [])),
                    severity="medium" if count < 5 else "high",
                    occurrence_count=count
                ))
        
        return updated_blockers


# =============================================================================
# Strategy Extraction (Phase 5 Enhancement)
# =============================================================================

class StrategyExtractor:
    """
    Extract transformation strategies from successful model transformations.
    
    Analyzes model pairs to learn what strategies (not just node transforms)
    were effective for different architecture types.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize strategy extractor."""
        self.verbose = verbose
        
        # Import architecture analyzer
        try:
            from core_analysis.architecture_analyzer import (
                ArchitectureAnalyzer, ArchitectureType
            )
            from core_analysis.compilation_simulator import CompilationSimulator
            self.arch_analyzer = ArchitectureAnalyzer()
            self.comp_simulator = CompilationSimulator()
            self.components_available = True
        except ImportError:
            self.components_available = False
    
    def extract_strategies(
        self,
        db: SurgeryDatabase,
        model_pairs: List[Tuple[str, str, str]]  # (name, original_path, modified_path)
    ) -> List[Dict]:
        """
        Extract strategies from model transformation pairs.
        
        Args:
            db: Surgery database with transformation records
            model_pairs: List of (model_name, original_path, modified_path) tuples
            
        Returns:
            List of learned strategy patterns
        """
        if not self.components_available:
            if self.verbose:
                print("  Warning: Strategy extraction components not available")
            return []
        
        learned_strategies = []
        
        # Group models by architecture
        arch_groups = defaultdict(list)
        
        for model_name, orig_path, mod_path in model_pairs:
            try:
                # Analyze original architecture
                architecture = self.arch_analyzer.analyze(orig_path)
                arch_type = architecture.architecture_type.value
                
                # Get compilation info
                orig_comp = self.comp_simulator.simulate(orig_path)
                mod_comp = self.comp_simulator.simulate(mod_path)
                
                # Get transformation record from database
                record = None
                for rec in db.transformation_records:
                    if rec.model_name == model_name:
                        record = rec
                        break
                
                if record:
                    arch_groups[arch_type].append({
                        'model_name': model_name,
                        'architecture': architecture.to_dict(),
                        'original_blockers': orig_comp.blocker_count,
                        'final_blockers': mod_comp.blocker_count,
                        'blockers_resolved': orig_comp.blocker_count - mod_comp.blocker_count,
                        'transformations': record.transformations,
                        'patterns_detected': [b.block_type.value for b in architecture.blocks if b.has_blockers],
                        'divide_conquer_recommended': architecture.divide_and_conquer_recommended
                    })
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not extract strategy from {model_name}: {e}")
                continue
        
        # Analyze patterns for each architecture type
        for arch_type, models in arch_groups.items():
            if len(models) < 2:
                continue  # Need at least 2 models to learn pattern
            
            strategy = self._learn_strategy_from_group(arch_type, models)
            if strategy:
                learned_strategies.append(strategy)
        
        return learned_strategies
    
    def _learn_strategy_from_group(
        self,
        arch_type: str,
        models: List[Dict]
    ) -> Optional[Dict]:
        """Learn strategy patterns from a group of similar models."""
        # Count common patterns
        all_patterns = []
        all_actions = []
        all_op_types = []
        
        total_blockers_resolved = 0
        total_blockers_initial = 0
        
        for model in models:
            all_patterns.extend(model.get('patterns_detected', []))
            total_blockers_resolved += model.get('blockers_resolved', 0)
            total_blockers_initial += model.get('original_blockers', 0)
            
            for trans in model.get('transformations', []):
                if hasattr(trans, 'action'):
                    all_actions.append(trans.action)
                if hasattr(trans, 'original_op_type') and trans.original_op_type:
                    all_op_types.append(trans.original_op_type)
        
        # Count frequencies
        from collections import Counter
        pattern_counts = Counter(all_patterns)
        action_counts = Counter(all_actions)
        op_type_counts = Counter(all_op_types)
        
        # Most common patterns become target_patterns
        target_patterns = [p for p, c in pattern_counts.most_common(5) if c >= len(models) // 2]
        
        # Calculate confidence
        confidence = 0.5
        if total_blockers_initial > 0:
            resolution_rate = total_blockers_resolved / total_blockers_initial
            confidence = min(0.95, 0.5 + resolution_rate * 0.4)
        
        # Determine if divide and conquer is common
        dc_count = sum(1 for m in models if m.get('divide_conquer_recommended', False))
        divide_conquer = dc_count > len(models) // 2
        
        # Build learned strategy
        strategy = {
            'strategy_id': f"learned_{arch_type.lower()}_v1",
            'name': f"Learned {arch_type} Strategy",
            'description': f"Strategy learned from {len(models)} {arch_type} models",
            'target_architecture': arch_type,
            'target_patterns': target_patterns,
            'divide_and_conquer': divide_conquer,
            'common_actions': dict(action_counts.most_common(5)),
            'common_op_types': dict(op_type_counts.most_common(10)),
            'confidence': confidence,
            'success_count': len(models),
            'total_blockers_resolved': total_blockers_resolved,
            'models_analyzed': [m['model_name'] for m in models]
        }
        
        return strategy
    
    def save_strategies(self, strategies: List[Dict], output_path: str) -> None:
        """Save learned strategies to JSON file."""
        data = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'learned_strategies': strategies
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"  Saved {len(strategies)} learned strategies to {output_path}")


# =============================================================================
# Main Build Process
# =============================================================================

def build_surgery_database(
    dataset_dir: str,
    output_dir: str = "rag_data",
    verbose: bool = True
) -> SurgeryDatabase:
    """
    Build the complete surgery database from scratch.
    
    Args:
        dataset_dir: Path to dataset directory with model pairs
        output_dir: Directory for output JSON files
        verbose: Whether to print progress
        
    Returns:
        Populated SurgeryDatabase
    """
    start_time = datetime.now()
    
    if verbose:
        print("=" * 70)
        print("BUILDING SURGERY DATABASE FROM RAW DATA")
        print("=" * 70)
        print(f"Dataset: {dataset_dir}")
        print(f"Output: {output_dir}")
        print(f"Started: {start_time.isoformat()}")
        print("=" * 70)
        print()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================================================
    # Step 1: Create database with default blockers
    # ==========================================================================
    if verbose:
        print("[Step 1/5] Creating database with default compilation blockers...")
    
    db = create_database_with_defaults()
    
    if verbose:
        print(f"  Default blockers loaded: {len(db.compilation_blockers)}")
        for blocker in db.compilation_blockers[:5]:
            print(f"    - {blocker.op_type}: {blocker.reason[:50]}...")
        print()
    
    # ==========================================================================
    # Step 2: Extract transformations from all model pairs
    # ==========================================================================
    if verbose:
        print("[Step 2/5] Extracting transformations from model pairs...")
    
    extractor = TransformationExtractor(verbose=verbose)
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return db
    
    # Find all model directories
    model_dirs = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            # Check for Original/Modified subdirectories
            original_dir = item / "Original"
            modified_dir = item / "Modified"
            
            # Handle case variations
            if not original_dir.exists():
                original_dir = item / "original"
            if not modified_dir.exists():
                modified_dir = item / "modified"
            
            if original_dir.exists() and modified_dir.exists():
                model_dirs.append((item.name, original_dir, modified_dir))
    
    if verbose:
        print(f"  Found {len(model_dirs)} model pairs")
    
    successful = 0
    failed = 0
    
    for model_name, original_dir, modified_dir in model_dirs:
        # Find ONNX files
        original_files = list(original_dir.glob("*.onnx"))
        modified_files = list(modified_dir.glob("*.onnx"))
        
        if not original_files or not modified_files:
            if verbose:
                print(f"  SKIP: {model_name} (missing ONNX files)")
            failed += 1
            continue
        
        original_path = str(original_files[0])
        modified_path = str(modified_files[0])
        
        try:
            if verbose:
                print(f"  Processing: {model_name}...")
            
            record = extractor.extract_transformations(
                original_path, modified_path, model_name
            )
            db.add_transformation_record(record)
            successful += 1
            
            if verbose:
                print(f"    -> {len(record.transformations)} transformations extracted")
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {model_name}: {str(e)[:100]}")
            failed += 1
            continue
        
        # Clear cache periodically to manage memory
        if (successful + failed) % 5 == 0:
            extractor.clear_cache()
    
    if verbose:
        print()
        print(f"  Extraction complete: {successful} succeeded, {failed} failed")
        print(f"  Total transformations: {db.total_transformations}")
        print()
    
    # ==========================================================================
    # Step 3: Generate templates from common patterns
    # ==========================================================================
    if verbose:
        print("[Step 3/5] Generating surgery templates...")
    
    template_generator = TemplateGenerator(min_occurrences=2, min_confidence=0.6)
    templates = template_generator.generate_templates(db)
    
    for template in templates:
        db.add_surgery_template(template)
    
    if verbose:
        print(f"  Generated {len(templates)} templates")
        for template in templates[:5]:
            print(f"    - {template.name} (confidence: {template.confidence:.2f})")
        print()
    
    # ==========================================================================
    # Step 4: Update blocker statistics
    # ==========================================================================
    if verbose:
        print("[Step 4/5] Updating compilation blocker statistics...")
    
    blocker_analyzer = BlockerAnalyzer(db)
    updated_blockers = blocker_analyzer.update_blocker_statistics()
    
    # Replace blockers with updated ones
    db.compilation_blockers = updated_blockers
    
    if verbose:
        print(f"  Updated {len(updated_blockers)} blocker entries")
        for blocker in updated_blockers[:5]:
            if blocker.occurrence_count > 0:
                print(f"    - {blocker.op_type}: {blocker.occurrence_count} occurrences")
        print()
    
    # ==========================================================================
    # Step 5: Extract strategies from transformation patterns
    # ==========================================================================
    if verbose:
        print("[Step 5/6] Extracting strategies from transformation patterns...")
    
    strategy_extractor = StrategyExtractor(verbose=verbose)
    
    # Build model pairs list from successful extractions
    model_pairs_for_strategy = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            original_dir = item / "Original"
            modified_dir = item / "Modified"
            if not original_dir.exists():
                original_dir = item / "original"
            if not modified_dir.exists():
                modified_dir = item / "modified"
            
            if original_dir.exists() and modified_dir.exists():
                original_files = list(original_dir.glob("*.onnx"))
                modified_files = list(modified_dir.glob("*.onnx"))
                if original_files and modified_files:
                    model_pairs_for_strategy.append(
                        (item.name, str(original_files[0]), str(modified_files[0]))
                    )
    
    learned_strategies = strategy_extractor.extract_strategies(db, model_pairs_for_strategy)
    
    if verbose:
        print(f"  Learned {len(learned_strategies)} strategy patterns")
        for strat in learned_strategies:
            print(f"    - {strat['name']} (confidence: {strat['confidence']:.2f})")
        print()
    
    # ==========================================================================
    # Step 6: Save database to JSON files
    # ==========================================================================
    if verbose:
        print("[Step 6/6] Saving database to JSON files...")
    
    # Main database
    main_db_path = os.path.join(output_dir, "surgery_database.json")
    db.save(main_db_path)
    
    # Templates (separate file for quick access)
    templates_path = os.path.join(output_dir, "surgery_templates.json")
    db.save_templates(templates_path)
    
    # Blocker index (separate file for quick lookups)
    blocker_path = os.path.join(output_dir, "blocker_index.json")
    db.save_blocker_index(blocker_path)
    
    # Learned strategies (separate file for strategy database integration)
    if learned_strategies:
        strategies_path = os.path.join(output_dir, "learned_strategies.json")
        strategy_extractor.save_strategies(learned_strategies, strategies_path)
    else:
        strategies_path = None
    
    if verbose:
        print(f"  Saved: {main_db_path}")
        print(f"  Saved: {templates_path}")
        print(f"  Saved: {blocker_path}")
        if strategies_path:
            print(f"  Saved: {strategies_path}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if verbose:
        print()
        print("=" * 70)
        print("BUILD COMPLETE")
        print("=" * 70)
        stats = db.get_statistics()
        print(f"Database version: {stats['version']}")
        print(f"Total models: {stats['total_models']}")
        print(f"Total transformations: {stats['total_transformations']}")
        print(f"Total templates: {stats['total_templates']}")
        print(f"Blockers cataloged: {stats['total_blockers_cataloged']}")
        print()
        print("Models by category:")
        for cat, count in stats.get('models_by_category', {}).items():
            print(f"  {cat}: {count}")
        print()
        print("Top transformations by action:")
        for action, count in sorted(
            stats.get('transformations_by_action', {}).items(),
            key=lambda x: -x[1]
        )[:5]:
            print(f"  {action}: {count}")
        print()
        print(f"Duration: {duration:.1f} seconds")
        print("=" * 70)
    
    return db


def validate_database(db: SurgeryDatabase, verbose: bool = True) -> bool:
    """
    Validate the built database for completeness.
    
    Checks:
    - Has transformation records
    - Has templates
    - Has blocker definitions
    - Can query by op_type
    - Can export for LLM
    """
    issues = []
    
    # Check transformation records
    if db.total_models == 0:
        issues.append("No models loaded")
    
    if db.total_transformations == 0:
        issues.append("No transformations extracted")
    
    # Check templates
    if len(db.surgery_templates) == 0:
        issues.append("No templates generated (may be expected for small datasets)")
    
    # Check blockers
    if len(db.compilation_blockers) == 0:
        issues.append("No blocker definitions")
    
    # Test query functionality
    try:
        result = db.find_transformations_by_op_type("Einsum")
        if verbose:
            print(f"Query test (Einsum): {len(result)} results")
    except Exception as e:
        issues.append(f"Query failed: {e}")
    
    # Test LLM export
    try:
        export = db.export_for_llm("Einsum", "Transformer")
        if verbose:
            print(f"LLM export test: {len(export)} characters")
    except Exception as e:
        issues.append(f"LLM export failed: {e}")
    
    if issues:
        if verbose:
            print("\nValidation issues:")
            for issue in issues:
                print(f"  - {issue}")
        return False
    
    if verbose:
        print("\nValidation passed!")
    return True


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Surgery Database from Raw ONNX Model Pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build from dataset directory
    python scripts/build_surgery_db.py --dataset dataset/ --output rag_data/
    
    # Build with verbose output
    python scripts/build_surgery_db.py --dataset dataset/ --verbose
    
    # Validate existing database
    python scripts/build_surgery_db.py --validate rag_data/surgery_database.json
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        help='Path to dataset directory containing Original/Modified model pairs'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='rag_data',
        help='Output directory for JSON files (default: rag_data)'
    )
    
    parser.add_argument(
        '--validate', '-V',
        help='Path to existing database JSON to validate'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    if args.validate:
        # Validate existing database
        print(f"Validating database: {args.validate}")
        db = SurgeryDatabase.load(args.validate)
        success = validate_database(db, verbose=True)
        sys.exit(0 if success else 1)
    
    if not args.dataset:
        parser.error("--dataset is required for building the database")
    
    # Build database
    db = build_surgery_database(
        dataset_dir=args.dataset,
        output_dir=args.output,
        verbose=verbose
    )
    
    # Validate
    validate_database(db, verbose=verbose)


if __name__ == "__main__":
    main()
