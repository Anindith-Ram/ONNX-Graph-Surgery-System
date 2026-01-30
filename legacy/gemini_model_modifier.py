#!/usr/bin/env python3
"""
Use Gemini 3 Pro to actually modify ONNX models based on generated rules.
This is the core RAG application - using LLM to perform graph surgery.
"""

import onnx
import json
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from onnx import helper, checker, shape_inference


class GeminiModelModifier:
    """Use Gemini to modify ONNX models based on rules."""
    
    def __init__(self, api_key: str):
        """Initialize with Gemini API."""
        genai.configure(api_key=api_key)
        self.model = None
        self.model_name = None
        
        # Try to initialize Gemini model
        for model_name in ['gemini-3-pro-preview']:
            try:
                test_model = genai.GenerativeModel(model_name)
                test_model.generate_content("test", generation_config={"max_output_tokens": 1})
                self.model = test_model
                self.model_name = model_name
                print(f"Using Gemini model for graph surgery: {model_name}")
                break
            except Exception as e:
                continue
    
    def modify_model_with_rules(
        self,
        model: onnx.ModelProto,
        rules: List[Dict],
        retrieved_examples: Optional[str] = None
    ) -> onnx.ModelProto:
        """
        Use Gemini to modify ONNX model based on rules.
        
        This is the core RAG application - Gemini:
        1. Understands the model structure
        2. Applies the generated rules
        3. Performs graph surgery
        4. Returns modified model
        
        Args:
            model: Original ONNX model
            rules: List of rules to apply
            retrieved_examples: Similar examples from vector store
        
        Returns:
            Modified ONNX model
        """
        if not self.model:
            print("Warning: No Gemini model available, returning original model")
            return model
        
        # Extract model structure for Gemini
        model_structure = self._extract_model_structure(model)
        
        # Create prompt for Gemini
        prompt = self._create_modification_prompt(
            model_structure,
            rules,
            retrieved_examples
        )
        
        try:
            # Get Gemini's modification plan
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for precise modifications
                    "max_output_tokens": 8192
                }
            )
            
            modification_plan = response.text
            
            # Parse and apply modifications
            modified_model = self._apply_gemini_modifications(
                model,
                modification_plan,
                rules
            )
            
            return modified_model
            
        except Exception as e:
            print(f"Error in Gemini model modification: {e}")
            return model
    
    def _extract_model_structure(self, model: onnx.ModelProto) -> Dict[str, Any]:
        """Extract model structure for Gemini to understand."""
        structure = {
            'nodes': [],
            'inputs': [],
            'outputs': [],
            'initializers': []
        }
        
        # Extract nodes
        for i, node in enumerate(model.graph.node):
            structure['nodes'].append({
                'id': i,
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {attr.name: str(attr) for attr in node.attribute}
            })
        
        # Extract inputs
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else (d.dim_param if d.dim_param else '?') 
                    for d in inp.type.tensor_type.shape.dim]
            structure['inputs'].append({
                'name': inp.name,
                'shape': shape
            })
        
        # Extract outputs
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else (d.dim_param if d.dim_param else '?') 
                    for d in out.type.tensor_type.shape.dim]
            structure['outputs'].append({
                'name': out.name,
                'shape': shape
            })
        
        return structure
    
    def _create_modification_prompt(
        self,
        model_structure: Dict,
        rules: List[Dict],
        retrieved_examples: Optional[str]
    ) -> str:
        """Create prompt for Gemini to modify the model."""
        
        rules_text = "\n".join([
            f"RULE {i+1}: {rule.get('name', 'Unknown')}\n"
            f"  Condition: {rule.get('condition', 'N/A')}\n"
            f"  Transformation: {rule.get('transformation', 'N/A')}\n"
            f"  Steps: {rule.get('steps', 'N/A')}\n"
            for i, rule in enumerate(rules)
        ])
        
        prompt = f"""You are an expert in ONNX graph surgery for hardware compilation.

TASK: Modify the ONNX model structure to apply the given rules and make it compilable.

CURRENT MODEL STRUCTURE:
{json.dumps(model_structure, indent=2)}

RULES TO APPLY:
{rules_text}

{"SIMILAR SUCCESSFUL EXAMPLES:" + retrieved_examples if retrieved_examples else ""}

ONNX GRAPH SURGERY PRINCIPLES:
1. Non-4D tensors must be reshaped to 4D throughout for MLA compatibility
2. Unsupported operators (Einsum, Complex, etc.) must be replaced with supported equivalents
3. Reshape/Transpose that prevent MLA mapping should be replaced with Slice/Concat patterns
4. Dynamic/unknown shapes must be resolved to concrete dimensions
5. Maintain mathematical correctness - outputs should match (within numerical precision)

INSTRUCTIONS:
1. Analyze the current model structure
2. Identify which rules apply to which nodes
3. Generate a detailed modification plan that specifies:
   - Which nodes to remove
   - Which nodes to add (with full specifications: op_type, inputs, outputs, attributes)
   - Which nodes to modify
   - How to reconnect the graph

Format your response as JSON:
{{
  "modifications": [
    {{
      "action": "remove|add|modify",
      "node_id": <node id or null for new nodes>,
      "node_name": "<name>",
      "op_type": "<operation type>",
      "inputs": ["<input1>", "<input2>"],
      "outputs": ["<output1>"],
      "attributes": {{"attr_name": "attr_value"}},
      "rule_applied": "<rule name>"
    }}
  ],
  "explanation": "<brief explanation of changes>"
}}

Generate the modification plan now:"""
        
        return prompt
    
    def _apply_gemini_modifications(
        self,
        model: onnx.ModelProto,
        modification_plan: str,
        rules: List[Dict]
    ) -> onnx.ModelProto:
        """Parse Gemini's modification plan and apply to ONNX model."""
        try:
            # Extract JSON from response
            json_start = modification_plan.find('{')
            json_end = modification_plan.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print("Warning: Could not find JSON in Gemini response")
                return model
            
            plan_json = json.loads(modification_plan[json_start:json_end])
            modifications = plan_json.get('modifications', [])
            
            print(f"Gemini generated {len(modifications)} modifications")
            print(f"Explanation: {plan_json.get('explanation', 'N/A')}")
            
            # Create a copy of the model
            modified_model = onnx.ModelProto()
            modified_model.CopyFrom(model)
            
            # Apply modifications in reverse order (to preserve indices)
            # First, collect all modifications
            nodes_to_remove = []
            nodes_to_add = []
            nodes_to_modify = []
            
            for mod in modifications:
                action = mod.get('action', '').lower()
                
                if action == 'remove':
                    nodes_to_remove.append(mod)
                elif action == 'add':
                    nodes_to_add.append(mod)
                elif action == 'modify':
                    nodes_to_modify.append(mod)
            
            # Apply removals (from end to preserve indices)
            for mod in sorted(nodes_to_remove, key=lambda x: x.get('node_id', 0), reverse=True):
                node_id = mod.get('node_id')
                if node_id is not None and node_id < len(modified_model.graph.node):
                    node_name = modified_model.graph.node[node_id].name
                    print(f"  Removing node {node_id}: {node_name}")
                    modified_model.graph.node.remove(modified_model.graph.node[node_id])
            
            # Apply modifications
            for mod in nodes_to_modify:
                node_id = mod.get('node_id')
                if node_id is not None and node_id < len(modified_model.graph.node):
                    node = modified_model.graph.node[node_id]
                    print(f"  Modifying node {node_id}: {node.name}")
                    
                    # Update operation type
                    if 'op_type' in mod:
                        node.op_type = mod['op_type']
                    
                    # Update inputs/outputs
                    if 'inputs' in mod:
                        node.input[:] = mod['inputs']
                    if 'outputs' in mod:
                        node.output[:] = mod['outputs']
            
            # Apply additions
            for mod in nodes_to_add:
                op_type = mod.get('op_type', 'Identity')
                inputs = mod.get('inputs', [])
                outputs = mod.get('outputs', [])
                node_name = mod.get('node_name', f"{op_type}_{len(modified_model.graph.node)}")
                attributes = mod.get('attributes', {})
                
                # Create attributes list
                attr_list = []
                for attr_name, attr_value in attributes.items():
                    # Simple attribute creation (can be enhanced)
                    if isinstance(attr_value, (int, float)):
                        attr_list.append(helper.make_attribute(attr_name, attr_value))
                
                # Create new node
                new_node = helper.make_node(
                    op_type,
                    inputs,
                    outputs,
                    name=node_name,
                    **{k: v for k, v in attributes.items() if isinstance(v, (int, float, str))}
                )
                
                print(f"  Adding node: {node_name} ({op_type})")
                modified_model.graph.node.append(new_node)
            
            # Validate and infer shapes
            try:
                modified_model = shape_inference.infer_shapes(modified_model)
                checker.check_model(modified_model)
                print("  ✓ Modified model is valid")
            except Exception as e:
                print(f"  ⚠ Validation warning: {e}")
            
            return modified_model
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini JSON: {e}")
            print(f"Response was: {modification_plan[:500]}")
            return model
        except Exception as e:
            print(f"Error applying modifications: {e}")
            import traceback
            traceback.print_exc()
            return model
    
    def modify_model_iterative(
        self,
        model: onnx.ModelProto,
        rules: List[Dict],
        retrieved_examples: Optional[str] = None,
        max_iterations: int = 3
    ) -> onnx.ModelProto:
        """
        Iteratively modify model, validating after each iteration.
        
        This allows Gemini to fix issues that arise from previous modifications.
        """
        current_model = model
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Modify model
            modified = self.modify_model_with_rules(
                current_model,
                rules,
                retrieved_examples
            )
            
            # Check if model is valid
            try:
                checker.check_model(modified)
                print(f"✓ Model valid after iteration {iteration + 1}")
                return modified
            except Exception as e:
                print(f"⚠ Model invalid: {e}")
                if iteration < max_iterations - 1:
                    print("  Retrying with error feedback...")
                    # Add error to context for next iteration
                    rules.append({
                        'name': 'Fix Validation Error',
                        'condition': 'Model validation failed',
                        'transformation': f'Fix: {str(e)}',
                        'steps': 'Resolve validation errors'
                    })
                    current_model = modified
                else:
                    print("  Max iterations reached, returning last attempt")
                    return modified
        
        return current_model

