#!/usr/bin/env python3
"""
Evaluation system to compare RAG-generated rules vs ground truth modified models.
"""

import os
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
from difference_extractor import extract_differences, ModelDiff
from feature_extractor import FeatureExtractor


class RuleEvaluator:
    """Evaluates RAG-generated rules against ground truth."""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def evaluate_model(
        self,
        model_name: str,
        rag_rules_file: str,
        ground_truth_map_file: str
    ) -> Dict[str, Any]:
        """
        Evaluate RAG rules for a single model.
        
        Args:
            model_name: Name of the model
            rag_rules_file: Path to RAG-generated rules JSON
            ground_truth_map_file: Path to ground truth modified model map
        
        Returns:
            Evaluation metrics dictionary
        """
        # Load RAG rules
        with open(rag_rules_file, 'r') as f:
            rag_result = json.load(f)
        
        # Extract ground truth differences
        ground_truth_diff = extract_differences(ground_truth_map_file)
        ground_truth_features = self.feature_extractor.extract_features(ground_truth_diff)
        
        # Extract what RAG predicted
        rag_rules = rag_result.get('generated_rules', [])
        rag_features = rag_result.get('features', {})
        
        # Compare
        metrics = self._compute_metrics(
            rag_rules,
            rag_features,
            ground_truth_diff,
            ground_truth_features
        )
        
        return {
            'model_name': model_name,
            'metrics': metrics,
            'rag_rules_count': len(rag_rules),
            'ground_truth_changes': {
                'node_count_diff': ground_truth_diff.node_count_diff,
                'added_nodes': len(ground_truth_diff.added_nodes),
                'removed_nodes': len(ground_truth_diff.removed_nodes),
                'modified_nodes': len(ground_truth_diff.modified_nodes),
                'op_type_changes': dict(ground_truth_diff.op_type_changes)
            },
            'rag_rules': rag_rules,
            'ground_truth_patterns': ground_truth_features.get('transformation_patterns', [])
        }
    
    def _compute_metrics(
        self,
        rag_rules: List[Dict],
        rag_features: Dict,
        ground_truth_diff: ModelDiff,
        ground_truth_features: Dict
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        metrics = {}
        
        # 1. Pattern matching: Do RAG rules match ground truth patterns?
        rag_patterns = set(rag_features.get('transformation_patterns', []))
        gt_patterns = set(ground_truth_features.get('transformation_patterns', []))
        
        if gt_patterns:
            pattern_precision = len(rag_patterns & gt_patterns) / len(rag_patterns) if rag_patterns else 0
            pattern_recall = len(rag_patterns & gt_patterns) / len(gt_patterns)
            pattern_f1 = 2 * pattern_precision * pattern_recall / (pattern_precision + pattern_recall) if (pattern_precision + pattern_recall) > 0 else 0
        else:
            pattern_precision = pattern_recall = pattern_f1 = 0
        
        metrics['pattern_precision'] = pattern_precision
        metrics['pattern_recall'] = pattern_recall
        metrics['pattern_f1'] = pattern_f1
        metrics['matched_patterns'] = list(rag_patterns & gt_patterns)
        metrics['missed_patterns'] = list(gt_patterns - rag_patterns)
        metrics['extra_patterns'] = list(rag_patterns - gt_patterns)
        
        # 2. Operation type changes: Do rules address actual changes?
        gt_op_changes = set(ground_truth_diff.op_type_changes.keys())
        rag_op_mentions = set()
        
        for rule in rag_rules:
            rule_text = f"{rule.get('name', '')} {rule.get('transformation', '')} {rule.get('condition', '')}"
            for op_change in gt_op_changes:
                if op_change.replace('->', ' ').lower() in rule_text.lower():
                    rag_op_mentions.add(op_change)
        
        if gt_op_changes:
            op_coverage = len(rag_op_mentions) / len(gt_op_changes)
        else:
            op_coverage = 1.0 if not rag_rules else 0.0
        
        metrics['operation_coverage'] = op_coverage
        metrics['covered_operations'] = list(rag_op_mentions)
        metrics['missed_operations'] = list(gt_op_changes - rag_op_mentions)
        
        # 3. Compilation issues: Do rules address the right issues?
        rag_issues = set(rag_features.get('compilation_issues_fixed', []))
        gt_issues = set(ground_truth_features.get('compilation_issues_fixed', []))
        
        if gt_issues:
            issue_precision = len(rag_issues & gt_issues) / len(rag_issues) if rag_issues else 0
            issue_recall = len(rag_issues & gt_issues) / len(gt_issues)
            issue_f1 = 2 * issue_precision * issue_recall / (issue_precision + issue_recall) if (issue_precision + issue_recall) > 0 else 0
        else:
            issue_precision = issue_recall = issue_f1 = 0
        
        metrics['issue_precision'] = issue_precision
        metrics['issue_recall'] = issue_recall
        metrics['issue_f1'] = issue_f1
        metrics['matched_issues'] = list(rag_issues & gt_issues)
        metrics['missed_issues'] = list(gt_issues - rag_issues)
        
        # 4. Rule quality: Are rules actionable?
        actionable_rules = sum(
            1 for rule in rag_rules
            if rule.get('condition') and rule.get('transformation') and rule.get('steps')
        )
        metrics['actionable_rules_ratio'] = actionable_rules / len(rag_rules) if rag_rules else 0
        
        # 5. Overall score (weighted average)
        metrics['overall_score'] = (
            pattern_f1 * 0.3 +
            op_coverage * 0.3 +
            issue_f1 * 0.2 +
            metrics['actionable_rules_ratio'] * 0.2
        )
        
        return metrics
    
    def evaluate_all(
        self,
        test_models: List[str],
        rag_results_dir: str,
        map_dataset_dir: str
    ) -> Dict[str, Any]:
        """Evaluate all test models."""
        results = []
        
        for model_name in test_models:
            rag_rules_file = os.path.join(rag_results_dir, f"rules_{model_name.replace('/', '_')}.json")
            ground_truth_file = os.path.join(map_dataset_dir, f"{model_name}.txt")
            
            if not os.path.exists(rag_rules_file):
                print(f"Warning: RAG rules not found for {model_name}")
                continue
            
            if not os.path.exists(ground_truth_file):
                print(f"Warning: Ground truth not found for {model_name}")
                continue
            
            print(f"Evaluating {model_name}...")
            try:
                eval_result = self.evaluate_model(model_name, rag_rules_file, ground_truth_file)
                results.append(eval_result)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Compute aggregate metrics
        aggregate = self._compute_aggregate_metrics(results)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate,
            'test_models_count': len(results)
        }
    
    def _compute_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate metrics across all test models."""
        if not results:
            return {}
        
        metrics_list = [r['metrics'] for r in results]
        
        aggregate = {
            'avg_pattern_f1': sum(m['pattern_f1'] for m in metrics_list) / len(metrics_list),
            'avg_operation_coverage': sum(m['operation_coverage'] for m in metrics_list) / len(metrics_list),
            'avg_issue_f1': sum(m['issue_f1'] for m in metrics_list) / len(metrics_list),
            'avg_actionable_rules_ratio': sum(m['actionable_rules_ratio'] for m in metrics_list) / len(metrics_list),
            'avg_overall_score': sum(m['overall_score'] for m in metrics_list) / len(metrics_list),
            'avg_rules_per_model': sum(r['rag_rules_count'] for r in results) / len(results)
        }
        
        return aggregate
    
    def generate_report(self, evaluation_results: Dict, output_file: str):
        """Generate a detailed evaluation report."""
        report_lines = [
            "=" * 80,
            "RAG Pipeline Evaluation Report",
            "=" * 80,
            "",
            f"Test Models Evaluated: {evaluation_results['test_models_count']}",
            "",
            "=" * 80,
            "Aggregate Metrics",
            "=" * 80,
        ]
        
        agg = evaluation_results['aggregate_metrics']
        report_lines.extend([
            f"Average Pattern F1 Score: {agg['avg_pattern_f1']:.3f}",
            f"Average Operation Coverage: {agg['avg_operation_coverage']:.3f}",
            f"Average Issue F1 Score: {agg['avg_issue_f1']:.3f}",
            f"Average Actionable Rules Ratio: {agg['avg_actionable_rules_ratio']:.3f}",
            f"Average Overall Score: {agg['avg_overall_score']:.3f}",
            f"Average Rules per Model: {agg['avg_rules_per_model']:.1f}",
            "",
            "=" * 80,
            "Individual Model Results",
            "=" * 80,
        ])
        
        for result in evaluation_results['individual_results']:
            model_name = result['model_name']
            metrics = result['metrics']
            
            report_lines.extend([
                f"\nModel: {model_name}",
                f"  Overall Score: {metrics['overall_score']:.3f}",
                f"  Pattern F1: {metrics['pattern_f1']:.3f}",
                f"  Operation Coverage: {metrics['operation_coverage']:.3f}",
                f"  Issue F1: {metrics['issue_f1']:.3f}",
                f"  Rules Generated: {result['rag_rules_count']}",
                f"  Matched Patterns: {len(metrics['matched_patterns'])}/{len(metrics['matched_patterns']) + len(metrics['missed_patterns'])}",
            ])
            
            if metrics['missed_patterns']:
                report_lines.append(f"  Missed Patterns: {', '.join(metrics['missed_patterns'][:3])}")
        
        report_text = "\n".join(report_lines)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nEvaluation report saved to {output_file}")
        print("\n" + report_text)
        
        return report_text


if __name__ == "__main__":
    import argparse
    from train_test_split import load_train_test_split
    
    parser = argparse.ArgumentParser(description='Evaluate RAG pipeline on test set')
    parser.add_argument('--split-file', default='rag_data/train_test_split.json')
    parser.add_argument('--rag-results-dir', default='rag_data')
    parser.add_argument('--map-dataset-dir', default='map_dataset')
    parser.add_argument('--output', default='rag_data/evaluation_results.json')
    parser.add_argument('--report', default='rag_data/evaluation_report.txt')
    
    args = parser.parse_args()
    
    # Load test models
    _, test_models = load_train_test_split(args.split_file)
    print(f"Evaluating {len(test_models)} test models")
    
    # Evaluate
    evaluator = RuleEvaluator()
    results = evaluator.evaluate_all(
        test_models,
        args.rag_results_dir,
        args.map_dataset_dir
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    evaluator.generate_report(results, args.report)

