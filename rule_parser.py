#!/usr/bin/env python3
"""
Parse generated rules into executable operations for ONNX graph surgery.
"""

from typing import Dict, List, Callable, Optional
import re
from dataclasses import dataclass


@dataclass
class ExecutableRule:
    """A rule that can be executed on an ONNX model."""
    name: str
    condition_checker: Callable  # Function to check if rule applies
    transformer: Callable  # Function to apply transformation
    priority: int  # Lower = higher priority
    description: str


class RuleParser:
    """Parse text rules into executable operations."""
    
    def __init__(self):
        # Map rule patterns to transformation functions
        self.rule_patterns = {
            'einsum': self._parse_einsum_replacement,
            'dynamic.*shape': self._parse_shape_resolution,
            '4d.*tensor': self._parse_4d_maintenance,
            'slice.*reshape': self._parse_slice_replacement,
            'reshape.*transpose': self._parse_reshape_transpose_replacement,
        }
    
    def parse_rule(self, rule: Dict) -> Optional[ExecutableRule]:
        """
        Parse a rule dictionary into an ExecutableRule.
        
        Args:
            rule: Dictionary with 'name', 'condition', 'transformation', 'steps', 'benefit'
        
        Returns:
            ExecutableRule or None if parsing fails
        """
        name = rule.get('name', '')
        condition = rule.get('condition', '')
        transformation = rule.get('transformation', '')
        steps = rule.get('steps', '')
        
        # Determine rule type and parse accordingly
        rule_text = f"{name} {condition} {transformation}".lower()
        
        for pattern, parser_func in self.rule_patterns.items():
            if re.search(pattern, rule_text):
                try:
                    return parser_func(rule)
                except Exception as e:
                    print(f"Warning: Failed to parse rule '{name}': {e}")
                    continue
        
        # Default: generic rule
        return self._parse_generic_rule(rule)
    
    def _parse_einsum_replacement(self, rule: Dict) -> ExecutableRule:
        """Parse Einsum replacement rule."""
        def condition_checker(model):
            """Check if model has Einsum operations."""
            import onnx
            for node in model.graph.node:
                if node.op_type == 'Einsum':
                    return True
            return False
        
        def transformer(model):
            """Replace Einsum with MatMul + Reshape."""
            # This is a placeholder - actual implementation would use ONNX graph surgery
            # See ONNX Graph Surgery PDF for Einsum decomposition patterns
            print(f"Applying: {rule['name']}")
            print(f"  Steps: {rule.get('steps', 'N/A')}")
            # TODO: Implement actual ONNX graph surgery
            return model
        
        return ExecutableRule(
            name=rule['name'],
            condition_checker=condition_checker,
            transformer=transformer,
            priority=1,  # High priority
            description=rule.get('transformation', '')
        )
    
    def _parse_shape_resolution(self, rule: Dict) -> ExecutableRule:
        """Parse dynamic shape resolution rule."""
        def condition_checker(model):
            """Check if model has dynamic shapes."""
            # Check for unknown dimensions in value_info
            for value_info in model.graph.value_info:
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.dim_value == 0 and not dim.dim_param:
                        return True
            return False
        
        def transformer(model):
            """Add shape inference operations."""
            print(f"Applying: {rule['name']}")
            # TODO: Implement shape resolution
            return model
        
        return ExecutableRule(
            name=rule['name'],
            condition_checker=condition_checker,
            transformer=transformer,
            priority=2,
            description=rule.get('transformation', '')
        )
    
    def _parse_4d_maintenance(self, rule: Dict) -> ExecutableRule:
        """Parse 4D tensor maintenance rule."""
        def condition_checker(model):
            """Check if model has non-4D tensors."""
            # Check tensor dimensions
            for value_info in model.graph.value_info:
                shape = value_info.type.tensor_type.shape
                dim_count = len([d for d in shape.dim if d.dim_value > 0])
                if dim_count != 4 and dim_count > 0:
                    return True
            return False
        
        def transformer(model):
            """Add Reshape operations to maintain 4D."""
            print(f"Applying: {rule['name']}")
            # TODO: Implement 4D maintenance
            return model
        
        return ExecutableRule(
            name=rule['name'],
            condition_checker=condition_checker,
            transformer=transformer,
            priority=3,
            description=rule.get('transformation', '')
        )
    
    def _parse_slice_replacement(self, rule: Dict) -> ExecutableRule:
        """Parse Slice replacement rule (from PDF: YoloV8 case study)."""
        def condition_checker(model):
            """Check if model has Reshape/Transpose that could be replaced."""
            for node in model.graph.node:
                if node.op_type in ['Reshape', 'Transpose']:
                    # Check if it's in a pattern that could use Slice
                    return True
            return False
        
        def transformer(model):
            """Replace Reshape/Transpose with Slice operations."""
            print(f"Applying: {rule['name']}")
            # TODO: Implement Slice replacement (see PDF YoloV8 case)
            return model
        
        return ExecutableRule(
            name=rule['name'],
            condition_checker=condition_checker,
            transformer=transformer,
            priority=4,
            description=rule.get('transformation', '')
        )
    
    def _parse_reshape_transpose_replacement(self, rule: Dict) -> ExecutableRule:
        """Parse Reshape/Transpose replacement rule."""
        return self._parse_slice_replacement(rule)  # Similar pattern
    
    def _parse_generic_rule(self, rule: Dict) -> ExecutableRule:
        """Parse a generic rule when no specific pattern matches."""
        def condition_checker(model):
            """Generic condition - always returns True (apply if rule says so)."""
            return True
        
        def transformer(model):
            """Generic transformer - logs the rule."""
            print(f"Applying generic rule: {rule['name']}")
            print(f"  Condition: {rule.get('condition', 'N/A')}")
            print(f"  Transformation: {rule.get('transformation', 'N/A')}")
            return model
        
        return ExecutableRule(
            name=rule['name'],
            condition_checker=condition_checker,
            transformer=transformer,
            priority=10,  # Low priority
            description=rule.get('transformation', '')
        )
    
    def parse_rules(self, rules: List[Dict]) -> List[ExecutableRule]:
        """Parse a list of rules."""
        executable_rules = []
        for rule in rules:
            parsed = self.parse_rule(rule)
            if parsed:
                executable_rules.append(parsed)
        
        # Sort by priority
        executable_rules.sort(key=lambda r: r.priority)
        return executable_rules

