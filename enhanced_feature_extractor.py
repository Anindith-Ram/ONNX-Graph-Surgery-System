#!/usr/bin/env python3
"""
Enhanced feature extractor using Gemini 3 for better semantic understanding.
"""

import os
import json
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from feature_extractor import FeatureExtractor
from difference_extractor import ModelDiff


class EnhancedFeatureExtractor(FeatureExtractor):
    """Enhanced feature extractor with Gemini 3 for semantic analysis."""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        super().__init__()
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        self.gemini_model_name = None
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            # Try Gemini 3 Pro first, then fallback to other models
            self._initialize_gemini_model()
    
    def _initialize_gemini_model(self):
        """Initialize Gemini model with fallback logic."""
        model_names = ['gemini-3-pro', 'gemini-3.0-pro', 'gemini-3-pro-preview', 
                      'gemini-1.5-pro', 'gemini-pro', 'gemini-2.0-flash-exp']
        
        for i, model_name in enumerate(model_names):
            try:
                # Test if model can be initialized
                test_model = genai.GenerativeModel(model_name)
                # Try a simple test call to verify it works
                try:
                    test_model.generate_content("test", generation_config={"max_output_tokens": 1})
                    self.gemini_model = test_model
                    self.gemini_model_name = model_name
                    print(f"Using Gemini model: {model_name}")
                    return
                except Exception as e:
                    # Model initialized but doesn't work, try next
                    if i < len(model_names) - 1:
                        print(f"Model {model_name} not available, trying next...")
                        continue
                    else:
                        print(f"Warning: Could not use any Gemini model, will use base features only. Last error: {e}")
                        self.gemini_model = None
                        self.gemini_model_name = None
                        return
            except Exception as e:
                # Model can't be initialized, try next
                if i < len(model_names) - 1:
                    continue
                else:
                    print(f"Warning: Could not initialize any Gemini model, will use base features only. Last error: {e}")
                    self.gemini_model = None
                    self.gemini_model_name = None
                    return
    
    def extract_features(self, diff: ModelDiff, use_gemini: bool = True) -> Dict[str, Any]:
        """Extract features with optional Gemini enhancement."""
        # Get base features
        base_features = super().extract_features(diff)
        
        # Enhance with Gemini if available
        if use_gemini and self.gemini_model:
            enhanced_features = self._enhance_with_gemini(diff, base_features)
            base_features.update(enhanced_features)
        
        return base_features
    
    def _enhance_with_gemini(self, diff: ModelDiff, base_features: Dict) -> Dict[str, Any]:
        """Use Gemini to enhance feature extraction with semantic understanding."""
        if not self.gemini_model:
            return {}
        
        # Get current model index to know which to try next
        model_names = ['gemini-3-pro', 'gemini-3.0-pro', 'gemini-3-pro-preview', 
                      'gemini-1.5-pro', 'gemini-pro', 'gemini-2.0-flash-exp']
        current_index = model_names.index(self.gemini_model_name) if self.gemini_model_name in model_names else -1
        
        try:
            # Create prompt for Gemini
            prompt = self._create_analysis_prompt(diff, base_features)
            
            # Get Gemini analysis
            response = self.gemini_model.generate_content(prompt)
            analysis = response.text
            
            # Parse Gemini response
            enhanced = self._parse_gemini_analysis(analysis, base_features)
            
            return {
                'gemini_analysis': analysis,
                'enhanced_key_changes': enhanced.get('key_changes', base_features['key_changes']),
                'enhanced_patterns': enhanced.get('patterns', base_features['transformation_patterns']),
                'semantic_insights': enhanced.get('insights', []),
                'compilation_priorities': enhanced.get('priorities', [])
            }
        except Exception as e:
            # If this model fails, try to reinitialize with next model
            print(f"Warning: Gemini enhancement failed with {self.gemini_model_name}: {e}")
            if current_index >= 0 and current_index < len(model_names) - 1:
                print(f"Attempting to fallback to next model...")
                # Try remaining models
                for next_model_name in model_names[current_index + 1:]:
                    try:
                        test_model = genai.GenerativeModel(next_model_name)
                        test_model.generate_content("test", generation_config={"max_output_tokens": 1})
                        self.gemini_model = test_model
                        self.gemini_model_name = next_model_name
                        print(f"Switched to fallback model: {next_model_name}")
                        # Retry with new model
                        prompt = self._create_analysis_prompt(diff, base_features)
                        response = self.gemini_model.generate_content(prompt)
                        analysis = response.text
                        enhanced = self._parse_gemini_analysis(analysis, base_features)
                        return {
                            'gemini_analysis': analysis,
                            'enhanced_key_changes': enhanced.get('key_changes', base_features['key_changes']),
                            'enhanced_patterns': enhanced.get('patterns', base_features['transformation_patterns']),
                            'semantic_insights': enhanced.get('insights', []),
                            'compilation_priorities': enhanced.get('priorities', [])
                        }
                    except Exception as e2:
                        continue
                print(f"All fallback models failed, using base features only")
            return {}
    
    def _create_analysis_prompt(self, diff: ModelDiff, base_features: Dict) -> str:
        """Create prompt for Gemini analysis."""
        prompt = f"""You are an expert in ONNX model optimization and hardware compilation.

Analyze the following model transformation that made a previously non-compilable model compilable:

MODEL: {diff.model_name}

CHANGES SUMMARY:
- Node count: {len(diff.added_nodes) + len(diff.removed_nodes) + len(diff.modified_nodes)} total changes
- Added nodes: {len(diff.added_nodes)}
- Removed nodes: {len(diff.removed_nodes)}
- Modified nodes: {len(diff.modified_nodes)}
- Operation type changes: {dict(diff.op_type_changes)}
- Shape modifications: {len(diff.shape_changes)} nodes

KEY OPERATIONS:
- Removed operations: {[n.op_type for n in diff.removed_nodes[:10]]}
- Added operations: {[n.op_type for n in diff.added_nodes[:10]]}

BASE ANALYSIS:
{base_features['summary']}

TASK:
1. Identify the MOST CRITICAL changes that enabled compilation
2. Explain WHY these changes were necessary for hardware compilation
3. Identify patterns that could be generalized to other models
4. Prioritize fixes by compilation impact

Format your response as:
CRITICAL_CHANGES:
- [Change 1]: [Why it matters]
- [Change 2]: [Why it matters]

PATTERNS:
- [Pattern 1]: [Description]
- [Pattern 2]: [Description]

INSIGHTS:
- [Insight 1]
- [Insight 2]

PRIORITIES:
- [Priority 1]: [Reason]
- [Priority 2]: [Reason]
"""
        return prompt
    
    def _parse_gemini_analysis(self, analysis: str, base_features: Dict) -> Dict:
        """Parse Gemini's structured response."""
        enhanced = {
            'key_changes': [],
            'patterns': [],
            'insights': [],
            'priorities': []
        }
        
        current_section = None
        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('CRITICAL_CHANGES:'):
                current_section = 'key_changes'
            elif line.startswith('PATTERNS:'):
                current_section = 'patterns'
            elif line.startswith('INSIGHTS:'):
                current_section = 'insights'
            elif line.startswith('PRIORITIES:'):
                current_section = 'priorities'
            elif line.startswith('-') and current_section:
                item = line[1:].strip()
                if current_section in enhanced:
                    enhanced[current_section].append(item)
        
        return enhanced


def extract_all_features_enhanced(
    map_dataset_dir: str,
    train_models: List[str],
    gemini_api_key: Optional[str] = None,
    output_file: str = None
) -> List[Dict[str, Any]]:
    """Extract enhanced features for training models only."""
    from difference_extractor import extract_differences
    
    extractor = EnhancedFeatureExtractor(gemini_api_key)
    features = []
    
    for model_name in train_models:
        map_file = os.path.join(map_dataset_dir, f"{model_name}.txt")
        if not os.path.exists(map_file):
            print(f"Warning: {map_file} not found, skipping")
            continue
        
        try:
            print(f"Extracting features for {model_name}...")
            diff = extract_differences(map_file)
            feat = extractor.extract_features(diff, use_gemini=True)
            features.append(feat)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"\nExtracted features for {len(features)} models")
        print(f"Saved to {output_file}")
    
    return features


if __name__ == "__main__":
    import argparse
    from train_test_split import load_train_test_split
    
    parser = argparse.ArgumentParser(description='Extract enhanced features with Gemini')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var, or use config.py)', default=None)
    parser.add_argument('--map-dataset-dir', default='map_dataset')
    parser.add_argument('--split-file', default='rag_data/train_test_split.json')
    parser.add_argument('--output', default='rag_data/features_train.json')
    
    args = parser.parse_args()
    
    # Get API key from: command line > environment variable > config file
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        try:
            from config import GEMINI_API_KEY
            api_key = GEMINI_API_KEY
        except ImportError:
            pass
    
    if not api_key:
        print("Error: Gemini API key required")
        print("  Options:")
        print("  1. Use --api-key flag")
        print("  2. Set GEMINI_API_KEY environment variable")
        print("  3. Add GEMINI_API_KEY to config.py")
        exit(1)
    
    args.api_key = api_key
    
    # Load train/test split
    train_models, _ = load_train_test_split(args.split_file)
    print(f"Extracting features for {len(train_models)} training models")
    
    # Extract features
    features = extract_all_features_enhanced(
        args.map_dataset_dir,
        train_models,
        args.api_key,
        args.output
    )

