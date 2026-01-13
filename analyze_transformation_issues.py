#!/usr/bin/env python3
"""
Analyze evaluation report to identify specific transformation accuracy issues.
This helps prioritize code improvements.
"""

import json
from collections import defaultdict, Counter
from pathlib import Path


def analyze_transformation_issues(report_path: str = "evaluation_output/evaluation_report.json"):
    """Analyze evaluation report and extract transformation accuracy issues."""
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("=" * 80)
    print("TRANSFORMATION ACCURACY ISSUE ANALYSIS")
    print("=" * 80)
    
    # Aggregate issues across all models
    all_removal_mismatches_gt = Counter()  # GT removed but we didn't
    all_addition_mismatches_gt = Counter()  # GT added but we didn't
    all_removal_mismatches_suggested = Counter()  # We removed but GT didn't
    all_addition_mismatches_suggested = Counter()  # We added but GT didn't
    removal_count_issues = []  # Operations with count mismatches
    addition_count_issues = []
    
    models_with_issues = []
    
    for model_result in report.get('per_model_results', []):
        model_name = model_result.get('model_name', 'Unknown')
        structural = model_result.get('structural_comparison', {})
        ta = structural.get('transformation_accuracy', {})
        
        if not ta:
            continue
        
        # Collect mismatches
        removal_mismatches_gt = ta.get('removal_mismatches_gt', [])
        addition_mismatches_gt = ta.get('addition_mismatches_gt', [])
        removal_mismatches_suggested = ta.get('removal_mismatches_suggested', [])
        addition_mismatches_suggested = ta.get('addition_mismatches_suggested', [])
        
        removal_count_acc = ta.get('removal_count_accuracy', 1.0)
        addition_count_acc = ta.get('addition_count_accuracy', 1.0)
        
        # Count issues
        for op in removal_mismatches_gt:
            all_removal_mismatches_gt[op] += 1
        for op in addition_mismatches_gt:
            all_addition_mismatches_gt[op] += 1
        for op in removal_mismatches_suggested:
            all_removal_mismatches_suggested[op] += 1
        for op in addition_mismatches_suggested:
            all_addition_mismatches_suggested[op] += 1
        
        # Track count accuracy issues
        if removal_count_acc < 0.8:
            expected = ta.get('expected_removals', [])
            actual = ta.get('actual_removals', [])
            removal_count_issues.append({
                'model': model_name,
                'accuracy': removal_count_acc,
                'expected': Counter(expected),
                'actual': Counter(actual)
            })
        
        if addition_count_acc < 0.8:
            expected = ta.get('expected_additions', [])
            actual = ta.get('actual_additions', [])
            addition_count_issues.append({
                'model': model_name,
                'accuracy': addition_count_acc,
                'expected': Counter(expected),
                'actual': Counter(actual)
            })
        
        # Track models with issues
        if (removal_mismatches_gt or addition_mismatches_gt or 
            removal_mismatches_suggested or addition_mismatches_suggested or
            removal_count_acc < 0.8 or addition_count_acc < 0.8):
            models_with_issues.append({
                'model': model_name,
                'transformation_score': ta.get('transformation_score', 0),
                'critical_areas_match': ta.get('critical_areas_match', 0),
                'removal_mismatches_gt': removal_mismatches_gt,
                'addition_mismatches_gt': addition_mismatches_gt,
                'removal_mismatches_suggested': removal_mismatches_suggested,
                'addition_mismatches_suggested': addition_mismatches_suggested,
                'removal_count_accuracy': removal_count_acc,
                'addition_count_accuracy': addition_count_acc
            })
    
    # Print summary
    print(f"\nTotal Models Analyzed: {len(report.get('per_model_results', []))}")
    print(f"Models with Issues: {len(models_with_issues)}")
    
    # Priority 1: Missing removals (GT removed but we didn't)
    print("\n" + "=" * 80)
    print("PRIORITY 1: MISSING REMOVALS (GT removed but we didn't)")
    print("=" * 80)
    if all_removal_mismatches_gt:
        print("\nOperations GT removed but we didn't (need handlers):")
        for op, count in all_removal_mismatches_gt.most_common():
            print(f"  - {op}: {count} model(s)")
            print(f"    → Need handler to remove {op} when GT pattern suggests it")
    else:
        print("✅ No missing removals!")
    
    # Priority 2: Missing additions (GT added but we didn't)
    print("\n" + "=" * 80)
    print("PRIORITY 2: MISSING ADDITIONS (GT added but we didn't)")
    print("=" * 80)
    if all_addition_mismatches_gt:
        print("\nOperations GT added but we didn't (need handlers):")
        for op, count in all_addition_mismatches_gt.most_common():
            print(f"  - {op}: {count} model(s)")
            print(f"    → Need handler to add {op} when GT pattern suggests it")
    else:
        print("✅ No missing additions!")
    
    # Priority 3: Wrong removals (we removed but GT didn't)
    print("\n" + "=" * 80)
    print("PRIORITY 3: WRONG REMOVALS (we removed but GT didn't)")
    print("=" * 80)
    if all_removal_mismatches_suggested:
        print("\nOperations we removed but GT didn't (handlers too aggressive):")
        for op, count in all_removal_mismatches_suggested.most_common():
            print(f"  - {op}: {count} model(s)")
            print(f"    → Handler for {op} is too aggressive, needs better logic")
    else:
        print("✅ No wrong removals!")
    
    # Priority 4: Wrong additions (we added but GT didn't)
    print("\n" + "=" * 80)
    print("PRIORITY 4: WRONG ADDITIONS (we added but GT didn't)")
    print("=" * 80)
    if all_addition_mismatches_suggested:
        print("\nOperations we added but GT didn't (handlers too aggressive):")
        for op, count in all_addition_mismatches_suggested.most_common():
            print(f"  - {op}: {count} model(s)")
            print(f"    → Handler for {op} is too aggressive, needs better logic")
    else:
        print("✅ No wrong additions!")
    
    # Priority 5: Count accuracy issues
    print("\n" + "=" * 80)
    print("PRIORITY 5: COUNT ACCURACY ISSUES")
    print("=" * 80)
    if removal_count_issues:
        print(f"\nRemoval count accuracy issues ({len(removal_count_issues)} models):")
        for issue in removal_count_issues[:5]:  # Show top 5
            print(f"\n  Model: {issue['model']}")
            print(f"  Accuracy: {issue['accuracy']:.2%}")
            print(f"  Expected: {dict(issue['expected'])}")
            print(f"  Actual: {dict(issue['actual'])}")
            # Find mismatches
            all_ops = set(issue['expected'].keys()) | set(issue['actual'].keys())
            for op in all_ops:
                exp_count = issue['expected'].get(op, 0)
                act_count = issue['actual'].get(op, 0)
                if exp_count != act_count:
                    print(f"    ⚠️ {op}: Expected {exp_count}, Got {act_count}")
    
    if addition_count_issues:
        print(f"\nAddition count accuracy issues ({len(addition_count_issues)} models):")
        for issue in addition_count_issues[:5]:  # Show top 5
            print(f"\n  Model: {issue['model']}")
            print(f"  Accuracy: {issue['accuracy']:.2%}")
            print(f"  Expected: {dict(issue['expected'])}")
            print(f"  Actual: {dict(issue['actual'])}")
            # Find mismatches
            all_ops = set(issue['expected'].keys()) | set(issue['actual'].keys())
            for op in all_ops:
                exp_count = issue['expected'].get(op, 0)
                act_count = issue['actual'].get(op, 0)
                if exp_count != act_count:
                    print(f"    ⚠️ {op}: Expected {exp_count}, Got {act_count}")
    
    # Code improvement recommendations
    print("\n" + "=" * 80)
    print("CODE IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # 1. Missing handlers
    if all_addition_mismatches_gt:
        for op in all_addition_mismatches_gt.keys():
            recommendations.append({
                'priority': 'HIGH',
                'type': 'add_handler',
                'operation': op,
                'action': f"Implement _add_{op.lower()} handler in suggestion_applicator.py",
                'reason': f"GT adds {op} in {all_addition_mismatches_gt[op]} model(s) but we don't"
            })
    
    if all_removal_mismatches_gt:
        for op in all_removal_mismatches_gt.keys():
            recommendations.append({
                'priority': 'HIGH',
                'type': 'improve_handler',
                'operation': op,
                'action': f"Improve _remove_{op.lower()} handler to match GT patterns",
                'reason': f"GT removes {op} in {all_removal_mismatches_gt[op]} model(s) but we don't"
            })
    
    # 2. Too aggressive handlers
    if all_removal_mismatches_suggested:
        for op in all_removal_mismatches_suggested.keys():
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'fix_handler',
                'operation': op,
                'action': f"Make _remove_{op.lower()} handler less aggressive",
                'reason': f"We remove {op} in {all_removal_mismatches_suggested[op]} model(s) but GT doesn't"
            })
    
    # 3. Count accuracy
    if removal_count_issues or addition_count_issues:
        recommendations.append({
            'priority': 'MEDIUM',
            'type': 'count_tracking',
            'operation': 'all',
            'action': "Add operation count tracking to handlers",
            'reason': "Handlers need to match GT operation counts precisely"
        })
    
    # Print recommendations
    print("\nRecommended Code Improvements:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['action']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Type: {rec['type']}")
    
    # Save detailed report
    output_file = Path("evaluation_output/transformation_issues_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    analysis = {
        'summary': {
            'total_models': len(report.get('per_model_results', [])),
            'models_with_issues': len(models_with_issues),
            'avg_transformation_accuracy': report.get('summary', {}).get('avg_transformation_accuracy', 0),
            'avg_critical_areas_match': report.get('summary', {}).get('avg_critical_areas_match', 0)
        },
        'issues': {
            'missing_removals': dict(all_removal_mismatches_gt),
            'missing_additions': dict(all_addition_mismatches_gt),
            'wrong_removals': dict(all_removal_mismatches_suggested),
            'wrong_additions': dict(all_addition_mismatches_suggested)
        },
        'count_accuracy_issues': {
            'removal': removal_count_issues,
            'addition': addition_count_issues
        },
        'models_with_issues': models_with_issues,
        'recommendations': recommendations
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Detailed analysis saved to: {output_file}")
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze transformation accuracy issues')
    parser.add_argument('--report', default='evaluation_output/evaluation_report.json',
                       help='Path to evaluation report JSON')
    
    args = parser.parse_args()
    analyze_transformation_issues(args.report)

