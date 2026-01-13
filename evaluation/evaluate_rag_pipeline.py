#!/usr/bin/env python3
"""
Evaluate RAG-Enhanced Advisory Pipeline on Test Set.

This script:
1. Builds knowledge base from training data only
2. Runs advisory system on test models
3. Compares suggestions with ground truth (modified models)
4. Generates evaluation metrics
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import onnx
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.knowledge_base import KnowledgeBaseBuilder
from production.production_pipeline import AdvisoryPipeline, AdvisoryConfig
from core_analysis.onnx_analyzer import ONNXAnalyzer
from suggestion_pipeline.suggestion_applicator import SuggestionApplicator
from evaluation.model_comparator import ModelComparator


@dataclass
class EvaluationResult:
    """Result for a single test model."""
    model_name: str
    original_path: str
    modified_path: str
    
    # Analysis results
    suggestions_count: int
    critical_count: int
    high_count: int
    
    # Ground truth analysis
    gt_blockers: List[str]
    gt_operations: List[str]
    
    # Suggestion quality (simple comparison)
    suggestions_match_gt: bool
    confidence_scores: List[float]
    
    # Detailed comparison (simple)
    detected_blockers: List[str]
    missed_blockers: List[str]
    false_positives: List[str]
    
    # Enhanced comparison (structural)
    structural_comparison: Optional[Dict] = None
    suggested_modified_path: Optional[str] = None
    applied_suggestions_count: int = 0  # Kept for backward compatibility
    failed_suggestions_count: int = 0
    # New tracking fields
    transformed_suggestions_count: int = 0
    attempted_suggestions_count: int = 0
    skipped_suggestions_count: int = 0
    transformation_effectiveness: float = 0.0  # transformed / attempted
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


class RAGPipelineEvaluator:
    """Evaluates RAG pipeline on test set."""
    
    # API call budget
    MAX_BATCH_SIZE = 10  # Must match rag_suggestion_generator.py
    MAX_RETRIES = 3  # Must match rag_suggestion_generator.py
    
    def __init__(
        self,
        train_test_split_path: str = "rag_data/train_test_split.json",
        kb_path: str = "knowledge_base_train.json",
        pdf_path: str = "ONNX Graph Surgery for Model SDK.pdf",
        dataset_dir: str = "dataset",
        api_key: Optional[str] = None,
        use_enhanced_comparison: bool = True
    ):
        self.train_test_split_path = train_test_split_path
        self.kb_path = kb_path
        self.pdf_path = pdf_path
        self.dataset_dir = dataset_dir
        self.api_key = api_key
        self.use_enhanced_comparison = use_enhanced_comparison
        
        # Initialize components
        self.applicator = SuggestionApplicator() if use_enhanced_comparison else None
        self.comparator = ModelComparator() if use_enhanced_comparison else None
        
        # Load train/test split
        with open(train_test_split_path) as f:
            self.split_data = json.load(f)
        
        self.train_models = set(self.split_data['train_models'])
        self.test_models = set(self.split_data['test_models'])
        
        print(f"Train models: {len(self.train_models)}")
        print(f"Test models: {len(self.test_models)}")
        if use_enhanced_comparison:
            print(f"Enhanced comparison: ENABLED")
    
    def estimate_api_calls(self) -> Dict[str, Any]:
        """Estimate API calls needed for evaluation (dry run)."""
        print("\n" + "="*70)
        print("API CALL ESTIMATION (DRY RUN)")
        print("="*70)
        
        dataset_path = Path(self.dataset_dir)
        estimates = {}
        total_suggestions = 0
        total_batches = 0
        
        for model_name in sorted(self.test_models):
            # Find model directory
            model_dir = None
            for d in dataset_path.iterdir():
                if d.is_dir() and d.name == model_name:
                    model_dir = d
                    break
            
            if not model_dir:
                continue
            
            # Find original model
            original_dir = model_dir / "original"
            if not original_dir.exists():
                original_dir = model_dir / "Original"
            
            original_files = list(original_dir.glob("*.onnx"))
            if not original_files:
                continue
            
            original_path = str(original_files[0])
            
            # Quick analysis to count issues
            try:
                analyzer = ONNXAnalyzer()
                analysis = analyzer.analyze(original_path)  # Pass path, not model
                
                # Count suggestions based on blockers
                # Note: actual suggestions ~2x blockers (multiple nodes per blocker type)
                blocker_count = len(analysis.compilation_blockers)
                suggestion_count = blocker_count * 2  # Empirical multiplier
                
                # Calculate batches (grouped by op_type, max 10 per batch)
                batch_count = max(1, (suggestion_count + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE)
                
                estimates[model_name] = {
                    'suggestions': suggestion_count,
                    'batches': batch_count,
                    'max_api_calls': batch_count * (1 + self.MAX_RETRIES)
                }
                
                total_suggestions += suggestion_count
                total_batches += batch_count
                
            except Exception as e:
                estimates[model_name] = {'error': str(e)}
        
        # Print estimates
        print(f"\n{'Model':<35} {'Suggestions':>12} {'Batches':>10} {'Max Calls':>10}")
        print("-" * 70)
        for model_name, est in estimates.items():
            if 'error' in est:
                print(f"{model_name:<35} {'Error':>12}")
            else:
                print(f"{model_name:<35} {est['suggestions']:>12} {est['batches']:>10} {est['max_api_calls']:>10}")
        
        print("-" * 70)
        max_calls = total_batches * (1 + self.MAX_RETRIES)
        print(f"{'TOTAL':<35} {total_suggestions:>12} {total_batches:>10} {max_calls:>10}")
        print()
        print(f"Estimated API calls: {total_batches} - {max_calls}")
        print(f"With 250 daily quota: {'✓ Should fit' if max_calls <= 250 else '✗ May exceed quota'}")
        
        if max_calls > 250:
            # Suggest which models to skip
            sorted_models = sorted(estimates.items(), key=lambda x: x[1].get('batches', 0), reverse=True)
            print(f"\nTo reduce API calls, consider excluding largest models:")
            cumulative = 0
            for model, est in sorted_models:
                if 'batches' in est:
                    cumulative += est['batches'] * (1 + self.MAX_RETRIES)
                    remaining = max_calls - est['max_api_calls']
                    print(f"  - Exclude {model}: saves {est['max_api_calls']} calls, remaining: {remaining}")
                    if remaining <= 250:
                        break
        
        return {
            'estimates': estimates,
            'total_suggestions': total_suggestions,
            'total_batches': total_batches,
            'max_api_calls': max_calls,
            'within_quota': max_calls <= 250
        }
    
    def build_training_kb(self) -> str:
        """Build knowledge base from training data only."""
        print("\n" + "="*70)
        print("STEP 1: Building Knowledge Base from Training Data")
        print("="*70)
        
        builder = KnowledgeBaseBuilder(
            api_key=self.api_key,
            use_gemini_enhancement=True  # Enable Gemini enhancement for better KB quality
        )
        kb = builder.build(
            pdf_path=self.pdf_path,
            dataset_dir=self.dataset_dir,
            train_test_split_path=self.train_test_split_path,
            use_train_only=True
        )
        
        kb.save(self.kb_path)
        print(f"\nKnowledge base saved to: {self.kb_path}")
        print(f"Total chunks: {len(kb.chunks)}")
        
        # Print chunk breakdown
        sources = {}
        for chunk in kb.chunks:
            sources[chunk.source] = sources.get(chunk.source, 0) + 1
        print("\nChunks by source:")
        for source, count in sources.items():
            print(f"  {source}: {count}")
        
        return self.kb_path
    
    def analyze_test_models(self, use_rag: bool = True) -> List[EvaluationResult]:
        """Analyze test models and generate suggestions."""
        print("\n" + "="*70)
        print("STEP 2: Analyzing Test Models")
        print("="*70)
        
        config = AdvisoryConfig(
            output_dir="evaluation_output",
            report_format="json",
            use_rag=use_rag,
            kb_path=self.kb_path
        )
        
        pipeline = AdvisoryPipeline(config=config, api_key=self.api_key)
        
        results = []
        dataset_path = Path(self.dataset_dir)
        
        for model_name in sorted(self.test_models):
            print(f"\nProcessing: {model_name}")
            
            # Find model directory
            model_dir = None
            for d in dataset_path.iterdir():
                if d.is_dir() and d.name == model_name:
                    model_dir = d
                    break
            
            if not model_dir:
                print(f"  Warning: Model directory not found for {model_name}")
                continue
            
            # Find original model
            original_dir = model_dir / "original"
            if not original_dir.exists():
                original_dir = model_dir / "Original"
            
            original_files = list(original_dir.glob("*.onnx"))
            if not original_files:
                print(f"  Warning: No original model found for {model_name}")
                continue
            
            original_path = str(original_files[0])
            
            # Find modified model (ground truth)
            modified_dir = model_dir / "modified"
            if not modified_dir.exists():
                modified_dir = model_dir / "Modified"
            
            modified_files = list(modified_dir.glob("*.onnx"))
            if not modified_files:
                print(f"  Warning: No modified model found for {model_name}")
                continue
            
            modified_path = str(modified_files[0])
            
            # Analyze with advisory pipeline
            try:
                result = pipeline.analyze_model(original_path, output_name=f"{model_name}_test")
                
                # Load suggestions
                json_path = Path(config.output_dir) / f"{model_name}_test_analysis.json"
                if json_path.exists():
                    with open(json_path) as f:
                        suggestion_data = json.load(f)
                    
                    # Analyze ground truth
                    gt_analysis = self._analyze_ground_truth(original_path, modified_path)
                    
                    # Compare suggestions with ground truth
                    eval_result = self._compare_with_ground_truth(
                        model_name,
                        original_path,
                        modified_path,
                        suggestion_data,
                        gt_analysis
                    )
                    
                    # Enhanced comparison: Apply suggestions and compare models
                    if self.use_enhanced_comparison and self.applicator and self.comparator:
                        try:
                            enhanced_result = self._enhanced_comparison(
                                model_name,
                                original_path,
                                modified_path,
                                suggestion_data,
                                eval_result
                            )
                            eval_result = enhanced_result
                            if eval_result.structural_comparison:
                                print(f"  ✓ Enhanced comparison: {eval_result.structural_comparison.get('overall_similarity', 0):.3f} similarity")
                            else:
                                print(f"  ⚠ Enhanced comparison: No comparison data available")
                        except Exception as e:
                            print(f"  ⚠ Enhanced comparison failed: {e}")
                    
                    results.append(eval_result)
                    print(f"  ✓ Analyzed: {eval_result.suggestions_count} suggestions")
                else:
                    print(f"  ✗ Failed: No analysis file generated")
            
            except Exception as e:
                print(f"  ✗ Error analyzing {model_name}: {e}")
        
        return results
    
    def _analyze_ground_truth(
        self,
        original_path: str,
        modified_path: str
    ) -> Dict:
        """Analyze ground truth modified model."""
        try:
            orig_model = onnx.load(original_path)
            mod_model = onnx.load(modified_path)
            
            orig_ops = [n.op_type for n in orig_model.graph.node]
            mod_ops = [n.op_type for n in mod_model.graph.node]
            
            # Find removed operations
            orig_op_counts = Counter(orig_ops)
            mod_op_counts = Counter(mod_ops)
            
            removed_ops = []
            for op, count in orig_op_counts.items():
                if op not in mod_op_counts or mod_op_counts[op] < count:
                    removed_ops.append(op)
            
            # Find added operations
            added_ops = []
            for op, count in mod_op_counts.items():
                if op not in orig_op_counts or orig_op_counts[op] < count:
                    added_ops.append(op)
            
            return {
                'removed_ops': removed_ops,
                'added_ops': added_ops,
                'orig_op_count': len(orig_ops),
                'mod_op_count': len(mod_ops)
            }
        
        except Exception as e:
            print(f"    Error analyzing ground truth: {e}")
            return {'removed_ops': [], 'added_ops': [], 'orig_op_count': 0, 'mod_op_count': 0}
    
    def _compare_with_ground_truth(
        self,
        model_name: str,
        original_path: str,
        modified_path: str,
        suggestion_data: Dict,
        gt_analysis: Dict
    ) -> EvaluationResult:
        """Compare suggestions with ground truth."""
        suggestions = suggestion_data.get('suggestions', [])
        
        # Extract detected blockers from suggestions
        detected_blockers = []
        for sug in suggestions:
            if sug.get('priority') in ['critical', 'high']:
                op_type = sug.get('location', {}).get('op_type', '')
                if op_type:
                    detected_blockers.append(op_type)
        
        # Ground truth blockers (operations that were removed/modified)
        gt_blockers = gt_analysis.get('removed_ops', [])
        
        # Find matches and misses
        detected_set = set(detected_blockers)
        gt_set = set(gt_blockers)
        
        matches = detected_set & gt_set
        missed = gt_set - detected_set
        false_positives = detected_set - gt_set
        
        # Confidence scores
        confidences = [s.get('confidence', 0.0) for s in suggestions]
        
        return EvaluationResult(
            model_name=model_name,
            original_path=original_path,
            modified_path=modified_path,
            suggestions_count=len(suggestions),
            critical_count=sum(1 for s in suggestions if s.get('priority') == 'critical'),
            high_count=sum(1 for s in suggestions if s.get('priority') == 'high'),
            gt_blockers=gt_blockers,
            gt_operations=gt_analysis.get('added_ops', []),
            suggestions_match_gt=len(matches) > 0,
            confidence_scores=confidences,
            detected_blockers=list(detected_blockers),
            missed_blockers=list(missed),
            false_positives=list(false_positives),
            structural_comparison=None,
            suggested_modified_path=None,
            applied_suggestions_count=0,
            failed_suggestions_count=0
        )
    
    def _enhanced_comparison(
        self,
        model_name: str,
        original_path: str,
        ground_truth_path: str,
        suggestion_data: Dict,
        base_result: EvaluationResult
    ) -> EvaluationResult:
        """
        Enhanced comparison: Apply suggestions and compare resulting models.
        
        This creates a "suggested-modified" model by applying suggestions,
        then compares it structurally with the ground truth modified model.
        """
        suggestions = suggestion_data.get('suggestions', [])
        
        # Step 1: Apply suggestions to create suggested-modified model
        output_dir = Path("evaluation_output") / "suggested_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        suggested_model_path = str(output_dir / f"{model_name}_suggested.onnx")
        
        print(f"    Applying {len(suggestions)} suggestions...")
        try:
            suggested_model = self.applicator.apply_suggestions(
                original_path,
                suggestions,
                output_path=suggested_model_path
            )
            
            base_result.applied_suggestions_count = self.applicator.applied_count  # Backward compatibility
            base_result.failed_suggestions_count = self.applicator.failed_count
            base_result.transformed_suggestions_count = self.applicator.transformed_count
            base_result.attempted_suggestions_count = self.applicator.attempted_count
            base_result.skipped_suggestions_count = self.applicator.skipped_count
            # Calculate transformation effectiveness
            if self.applicator.attempted_count > 0:
                base_result.transformation_effectiveness = (
                    self.applicator.transformed_count / self.applicator.attempted_count
                )
            base_result.suggested_modified_path = suggested_model_path
            
        except Exception as e:
            print(f"    Error applying suggestions: {e}")
            return base_result
        
        # Step 2: Compare suggested-modified with ground truth
        print(f"    Comparing suggested model with ground truth...")
        try:
            # Load models from paths
            suggested_model = onnx.load(suggested_model_path)
            ground_truth_model = onnx.load(ground_truth_path)
            original_model = onnx.load(original_path) if original_path else None
            
            # Compare models (pass loaded ModelProto objects, not paths)
            comparison = self.comparator.compare_models(
                suggested_model,
                ground_truth_model,
                original_model=original_model
            )
            
            base_result.structural_comparison = comparison
            
        except Exception as e:
            print(f"    Error comparing models: {e}")
            return base_result
        
        return base_result
    
    def generate_evaluation_report(
        self,
        results: List[EvaluationResult],
        output_path: str = "evaluation_output/evaluation_report.json"
    ) -> Dict:
        """Generate comprehensive evaluation report."""
        print("\n" + "="*70)
        print("STEP 3: Generating Evaluation Report")
        print("="*70)
        
        # Add timestamp to output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_obj = Path(output_path)
        timestamped_path = output_path_obj.parent / f"{output_path_obj.stem}_{timestamp}{output_path_obj.suffix}"
        
        total_models = len(results)
        total_suggestions = sum(r.suggestions_count for r in results)
        total_gt_blockers = sum(len(r.gt_blockers) for r in results)
        
        # Calculate metrics (simple comparison)
        total_detected = sum(len(r.detected_blockers) for r in results)
        total_missed = sum(len(r.missed_blockers) for r in results)
        total_false_positives = sum(len(r.false_positives) for r in results)
        total_matches = sum(len(set(r.detected_blockers) & set(r.gt_blockers)) for r in results)
        
        precision = total_matches / total_detected if total_detected > 0 else 0.0
        recall = total_matches / total_gt_blockers if total_gt_blockers > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_confidence = sum(
            sum(r.confidence_scores) / len(r.confidence_scores) if r.confidence_scores else 0.0
            for r in results
        ) / total_models if total_models > 0 else 0.0
        
        # Enhanced metrics (structural comparison + transformation accuracy)
        enhanced_metrics = {}
        if any(r.structural_comparison for r in results):
            structural_results = [r for r in results if r.structural_comparison]
            
            # Extract transformation accuracy metrics
            transformation_scores = []
            critical_area_matches = []
            location_based_matches = []
            combined_matches = []
            for r in structural_results:
                if r.structural_comparison and 'transformation_accuracy' in r.structural_comparison:
                    ta = r.structural_comparison['transformation_accuracy']
                    transformation_scores.append(ta.get('transformation_score', 0.0))
                    critical_area_matches.append(ta.get('critical_areas_match', 0.0))
                    location_based_matches.append(ta.get('location_based_critical_areas_match', 0.0))
                    combined_matches.append(ta.get('combined_critical_areas_match', 0.0))
            
            enhanced_metrics = {
                'models_with_structural_comparison': len(structural_results),
                'avg_structural_similarity': sum(
                    r.structural_comparison.get('overall_similarity', 0.0) if r.structural_comparison else 0.0
                    for r in structural_results
                ) / len(structural_results) if structural_results else 0.0,
                'avg_operation_similarity': sum(
                    (r.structural_comparison.get('operation_comparison', {}) if r.structural_comparison else {}).get('jaccard_similarity', 0.0)
                    for r in structural_results
                ) / len(structural_results) if structural_results else 0.0,
                # Transformation accuracy metrics (CRITICAL)
                'avg_transformation_accuracy': sum(transformation_scores) / len(transformation_scores) if transformation_scores else 0.0,
                'avg_critical_areas_match': sum(critical_area_matches) / len(critical_area_matches) if critical_area_matches else 0.0,
                # Location-based metrics (NEW)
                'avg_location_based_critical_areas_match': sum(location_based_matches) / len(location_based_matches) if location_based_matches else 0.0,
                'avg_combined_critical_areas_match': sum(combined_matches) / len(combined_matches) if combined_matches else 0.0,
                # Suggestion application metrics
                'total_applied_suggestions': sum(r.applied_suggestions_count for r in results),  # Backward compatibility
                'total_failed_suggestions': sum(r.failed_suggestions_count for r in results),
                'total_transformed_suggestions': sum(r.transformed_suggestions_count for r in results),
                'total_attempted_suggestions': sum(r.attempted_suggestions_count for r in results),
                'total_skipped_suggestions': sum(r.skipped_suggestions_count for r in results),
                'suggestion_application_rate': sum(r.applied_suggestions_count for r in results) / (
                    sum(r.applied_suggestions_count + r.failed_suggestions_count for r in results)
                    if sum(r.applied_suggestions_count + r.failed_suggestions_count for r in results) > 0 else 1
                ),
                'transformation_effectiveness': sum(r.transformation_effectiveness for r in results) / len(results) if results else 0.0,
                'transformation_rate': (
                    sum(r.transformed_suggestions_count for r in results) / sum(r.attempted_suggestions_count for r in results)
                    if sum(r.attempted_suggestions_count for r in results) > 0 else 0.0
                )
            }
        
        report = {
            'summary': {
                'total_test_models': total_models,
                'total_suggestions': total_suggestions,
                'total_gt_blockers': total_gt_blockers,
                'total_detected': total_detected,
                'total_missed': total_missed,
                'total_false_positives': total_false_positives,
                'total_matches': total_matches,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'avg_confidence': avg_confidence,
                **enhanced_metrics  # Add enhanced metrics if available
            },
            'per_model_results': [r.to_dict() for r in results],
            'train_models': list(self.train_models),
            'test_models': list(self.test_models)
        }
        
        # Add timestamp to report metadata
        report['evaluation_timestamp'] = datetime.now().isoformat()
        report['evaluation_timestamp_readable'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save report with timestamped filename
        Path(timestamped_path).parent.mkdir(parents=True, exist_ok=True)
        with open(timestamped_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save to original path (without timestamp) for compatibility
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"  Test Models: {total_models}")
        print(f"  Total Suggestions: {total_suggestions}")
        print(f"  Ground Truth Blockers: {total_gt_blockers}")
        print(f"  Detected: {total_detected}")
        print(f"  Matches: {total_matches}")
        print(f"  Missed: {total_missed}")
        print(f"  False Positives: {total_false_positives}")
        print(f"\nMetrics (Simple Comparison):")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1_score:.3f}")
        print(f"  Avg Confidence: {avg_confidence:.3f}")
        
        if enhanced_metrics:
            print(f"\nMetrics (Critical Areas Match - PRIMARY METRIC):")
            print(f"  ✅ Type-Based Match: {enhanced_metrics.get('avg_critical_areas_match', 0):.3f} (Operation type matching)")
            print(f"  📍 Location-Based Match: {enhanced_metrics.get('avg_location_based_critical_areas_match', 0):.3f} (Exact node location matching)")
            print(f"  🎯 Combined Match: {enhanced_metrics.get('avg_combined_critical_areas_match', 0):.3f} (60% type + 40% location)")
            print(f"  📊 Avg Transformation Accuracy: {enhanced_metrics.get('avg_transformation_accuracy', 0):.3f} (Includes count accuracy & penalties)")
            print(f"  🔧 Transformation Effectiveness: {enhanced_metrics.get('transformation_effectiveness', 0):.3f}")
            print(f"  📈 Transformed/Attempted: {enhanced_metrics.get('transformation_rate', 0):.3f}")
            print(f"\nMetrics (Structural Comparison):")
            print(f"  Models with structural comparison: {enhanced_metrics.get('models_with_structural_comparison', 0)}")
            print(f"  Avg Structural Similarity: {enhanced_metrics.get('avg_structural_similarity', 0):.3f}")
            print(f"  Avg Operation Similarity: {enhanced_metrics.get('avg_operation_similarity', 0):.3f}")
            print(f"  Applied Suggestions: {enhanced_metrics.get('total_applied_suggestions', 0)}")
            print(f"  Failed Suggestions: {enhanced_metrics.get('total_failed_suggestions', 0)}")
            print(f"  Skipped Suggestions: {enhanced_metrics.get('total_skipped_suggestions', 0)}")
            print(f"  Application Rate: {enhanced_metrics.get('suggestion_application_rate', 0):.3f}")
        
        print(f"\nReport saved to: {timestamped_path}")
        print(f"  (Also saved to: {output_path} for compatibility)")
        
        return report


def main():
    """Main evaluation workflow."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Evaluate RAG pipeline on test set')
    parser.add_argument('--split', default='rag_data/train_test_split.json',
                       help='Path to train_test_split.json')
    parser.add_argument('--kb', default='knowledge_base_train.json',
                       help='Output path for training KB')
    parser.add_argument('--pdf', default='ONNX Graph Surgery for Model SDK.pdf',
                       help='Path to PDF documentation')
    parser.add_argument('--dataset', default='dataset',
                       help='Dataset directory')
    parser.add_argument('--no-rag', action='store_true',
                       help='Disable RAG (baseline comparison)')
    parser.add_argument('--skip-kb-build', action='store_true',
                       help='Skip KB building (use existing)')
    parser.add_argument('--no-enhanced', action='store_true',
                       help='Disable enhanced structural comparison')
    
    args = parser.parse_args()
    
    # Get API key from env or config
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        try:
            from config import GEMINI_API_KEY
            if GEMINI_API_KEY and GEMINI_API_KEY != "your-api-key-here":
                api_key = GEMINI_API_KEY
        except ImportError:
            pass
    
    evaluator = RAGPipelineEvaluator(
        train_test_split_path=args.split,
        kb_path=args.kb,
        pdf_path=args.pdf,
        dataset_dir=args.dataset,
        api_key=api_key,
        use_enhanced_comparison=not args.no_enhanced
    )
    
    # Step 1: Build KB from training data
    if not args.skip_kb_build:
        evaluator.build_training_kb()
    else:
        print(f"Using existing KB: {args.kb}")
    
    # Step 2: Analyze test models
    results = evaluator.analyze_test_models(use_rag=not args.no_rag)
    
    # Step 3: Generate evaluation report
    evaluator.generate_evaluation_report(results)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

