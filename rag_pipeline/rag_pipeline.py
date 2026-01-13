#!/usr/bin/env python3
"""
RAG Pipeline for ONNX Model Compilation Rules Generation.
Uses vector embeddings to retrieve similar transformation patterns and
Gemini 3 Pro to generate compilation rules.
"""

import os
import sys
import json
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import google.generativeai as genai

sys.path.insert(0, str(Path(__file__).parent.parent))
from core_analysis.feature_extractor import FeatureExtractor, extract_all_features
from core_analysis.difference_extractor import ModelDiff


class VectorStore:
    """Simple in-memory vector store using embeddings."""
    
    def __init__(self, embedding_model=None):
        self.embeddings = []
        self.metadata = []
        self.embedding_model = embedding_model
    
    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        """Add documents to the store."""
        if self.embedding_model:
            embeddings = self.embedding_model.embed_documents(texts)
        else:
            # Simple TF-IDF-like representation as fallback
            embeddings = [self._simple_embed(text) for text in texts]
        
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadatas)
    
    def _simple_embed(self, text: str) -> List[float]:
        """Simple embedding using word frequencies."""
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalize
        total = sum(word_freq.values())
        if total == 0:
            return [0.0] * 100
        
        # Create a simple vector (up to 100 dimensions)
        vector = [word_freq.get(f"word_{i}", 0) / total for i in range(100)]
        return vector
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if self.embedding_model:
            query_embedding = self.embedding_model.embed_query(query)
        else:
            query_embedding = self._simple_embed(query)
        
        similarities = [
            self.similarity(query_embedding, emb) 
            for emb in self.embeddings
        ]
        
        # Get top k
        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'metadata': self.metadata[idx],
                'similarity': similarities[idx],
                'text': self.metadata[idx].get('text', '')
            })
        
        return results


class RAGPipeline:
    """RAG Pipeline for generating compilation rules."""
    
    def __init__(self, gemini_api_key: str, vector_store: Optional[VectorStore] = None):
        """Initialize the RAG pipeline."""
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.model_name = None
        # Initialize vector store FIRST (before Gemini initialization)
        self.vector_store = vector_store or VectorStore()
        self.feature_extractor = FeatureExtractor()
        # Try Gemini 3 Pro first, then fallback to other models
        self._initialize_gemini_model()
    
    def _initialize_gemini_model(self):
        """Initialize Gemini model with fallback logic."""
        model_names = ['gemini-3-pro-preview', 'gemini-1.5-pro', 'gemini-pro', 'gemini-2.0-flash-exp']
        
        for i, model_name in enumerate(model_names):
            try:
                # Test if model can be initialized and used
                test_model = genai.GenerativeModel(model_name)
                # Try a simple test call to verify it works
                try:
                    test_model.generate_content("test", generation_config={"max_output_tokens": 1})
                    self.model = test_model
                    self.model_name = model_name
                    print(f"RAG Pipeline using Gemini model: {model_name}")
                    return
                except Exception as e:
                    # Model initialized but doesn't work, try next
                    if i < len(model_names) - 1:
                        print(f"Model {model_name} not available, trying next...")
                        continue
                    else:
                        print(f"Warning: Could not use any Gemini model. Last error: {e}")
                        self.model = None
                        self.model_name = None
                        return
            except Exception as e:
                # Model can't be initialized, try next
                if i < len(model_names) - 1:
                    continue
                else:
                    print(f"Warning: Could not initialize any Gemini model. Last error: {e}")
                    self.model = None
                    self.model_name = None
                    return
    
    def build_knowledge_base(self, features_file: str, train_models: List[str] = None):
        """Build the knowledge base from extracted features (training models only)."""
        with open(features_file, 'r') as f:
            all_features = json.load(f)
        
        # Filter to training models only if specified
        if train_models:
            features = [f for f in all_features if f['model_name'] in train_models]
            print(f"Filtering to {len(features)} training models (out of {len(all_features)} total)")
        else:
            features = all_features
        
        documents = []
        metadatas = []
        
        for feat in features:
            # Create a document text from features
            doc_text = self._create_document_text(feat)
            documents.append(doc_text)
            metadatas.append({
                'model_name': feat['model_name'],
                'summary': feat.get('summary', ''),
                'key_changes': feat.get('key_changes', []),
                'patterns': feat.get('transformation_patterns', []),
                'text': doc_text
            })
        
        self.vector_store.add_documents(documents, metadatas)
        print(f"Built knowledge base with {len(documents)} training documents")
    
    def _create_document_text(self, features: Dict) -> str:
        """Create a searchable text document from features."""
        # Handle both enhanced and base features
        key_changes = features.get('enhanced_key_changes') or features.get('key_changes', [])
        patterns = features.get('enhanced_patterns') or features.get('transformation_patterns', [])
        issues = features.get('compilation_issues_fixed', [])
        fixes = features.get('recommended_fixes', [])
        
        parts = [
            f"Model: {features['model_name']}",
            f"Summary: {features.get('summary', '')}",
            f"Key Changes: {', '.join(key_changes) if key_changes else 'None'}",
            f"Compilation Issues Fixed: {', '.join(issues) if issues else 'None'}",
            f"Transformation Patterns: {', '.join(patterns) if patterns else 'None'}",
            f"Recommended Fixes: {', '.join(fixes) if fixes else 'None'}"
        ]
        
        # Add Gemini insights if available
        if 'semantic_insights' in features:
            parts.append(f"Semantic Insights: {', '.join(features['semantic_insights'])}")
        
        return "\n".join(parts)
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context from the knowledge base."""
        results = self.vector_store.search(query, top_k=top_k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"Example {i} ({result['metadata']['model_name']}):\n"
                f"  Changes: {', '.join(result['metadata']['key_changes'][:3])}\n"
                f"  Patterns: {', '.join(result['metadata']['patterns'][:2])}\n"
                f"  Similarity: {result['similarity']:.3f}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_rules(self, model_diff: ModelDiff, query: str = None) -> Dict[str, Any]:
        """Generate compilation rules for a model."""
        # Extract features
        features = self.feature_extractor.extract_features(model_diff)
        
        # Create query if not provided
        if not query:
            query = f"Model compilation issues: {', '.join(features['compilation_issues_fixed'])}"
        
        # Retrieve similar examples
        context = self.retrieve_context(query, top_k=5)
        
        # Create prompt for Gemini
        prompt = self._create_rule_generation_prompt(features, context)
        
        # Generate rules using Gemini
        if not self.model:
            print("No Gemini model available, using fallback rules")
            rules_text = self._fallback_rules(features)
        else:
            model_names = ['gemini-3-pro', 'gemini-3.0-pro', 'gemini-3-pro-preview',
                          'gemini-1.5-pro', 'gemini-pro', 'gemini-2.0-flash-exp']
            current_index = model_names.index(self.model_name) if self.model_name in model_names else -1
            
            try:
                response = self.model.generate_content(prompt)
                rules_text = response.text
            except Exception as e:
                print(f"Error generating rules with {self.model_name}: {e}")
                # Try to fallback to next model
                if current_index >= 0 and current_index < len(model_names) - 1:
                    print(f"Attempting to fallback to next model...")
                    # Try remaining models
                    for next_model_name in model_names[current_index + 1:]:
                        try:
                            test_model = genai.GenerativeModel(next_model_name)
                            test_model.generate_content("test", generation_config={"max_output_tokens": 1})
                            self.model = test_model
                            self.model_name = next_model_name
                            print(f"Switched to fallback model: {next_model_name}")
                            # Retry with new model
                            response = self.model.generate_content(prompt)
                            rules_text = response.text
                            break
                        except Exception as e2:
                            continue
                    else:
                        # All models failed
                        print(f"All models failed, using fallback rules")
                        rules_text = self._fallback_rules(features)
                else:
                    rules_text = self._fallback_rules(features)
        
        # Parse and structure rules
        rules = self._parse_rules(rules_text, features)
        
        return {
            'model_name': model_diff.model_name,
            'features': features,
            'retrieved_context': context,
            'generated_rules': rules,
            'rules_text': rules_text
        }
    
    def _create_rule_generation_prompt(self, features: Dict, context: str) -> str:
        """Create a prompt for rule generation based on ONNX Graph Surgery best practices."""
        prompt = f"""You are an expert in ONNX model optimization and compilation for edge devices (MLA/CVU/APU backends).

Based on ONNX Graph Surgery best practices, the goal is to modify models so they can be compiled entirely to MLA (single LM file) rather than split across multiple backends.

KEY PRINCIPLES FROM ONNX GRAPH SURGERY:
1. Non-4D tensors must be reshaped to 4D throughout the model for MLA compatibility
2. Unsupported operators (Einsum, Complex, DynamicSlice, etc.) must be replaced with supported equivalents
3. Reshape/Transpose operations that prevent MLA mapping should be replaced with Slice/Concat patterns
4. Dynamic/unknown shapes must be resolved to concrete dimensions
5. Use divide-and-conquer approach: split model first, then modify each part
6. Data reshuffling operations (Reshape, Slice, Concat, Transpose) that don't change math processing should produce identical outputs

TASK: Generate specific, actionable rules for rewriting ONNX models to make them compilable on hardware accelerators.

CURRENT MODEL ANALYSIS:
Model: {features['model_name']}
Summary: {features['summary']}
Key Changes Made: {', '.join(features['key_changes'])}
Compilation Issues Fixed: {', '.join(features['compilation_issues_fixed'])}
Transformation Patterns: {', '.join(features['transformation_patterns'])}

SIMILAR SUCCESSFUL TRANSFORMATIONS:
{context}

INSTRUCTIONS:
1. Analyze the current model's compilation issues
2. Review the similar successful transformations
3. Generate 5-10 specific, actionable rules that can be automatically applied
4. Each rule should specify:
   - WHEN to apply it (condition/pattern to match)
   - WHAT to change (specific operation/pattern)
   - HOW to change it (transformation steps)
   - WHY it helps (compilation benefit)

Format your response as:
RULE 1: [Rule Name]
- Condition: [When to apply]
- Transformation: [What to change]
- Steps: [How to change]
- Benefit: [Why it helps]

RULE 2: [Rule Name]
...

Generate rules now:"""
        return prompt
    
    def _parse_rules(self, rules_text: str, features: Dict) -> List[Dict]:
        """Parse generated rules into structured format."""
        rules = []
        
        # Simple parsing - split by "RULE"
        rule_sections = rules_text.split('RULE')
        
        for section in rule_sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            rule = {
                'name': lines[0].strip(': '),
                'condition': '',
                'transformation': '',
                'steps': '',
                'benefit': ''
            }
            
            current_field = None
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('- Condition:'):
                    current_field = 'condition'
                    rule['condition'] = line.replace('- Condition:', '').strip()
                elif line.startswith('- Transformation:'):
                    current_field = 'transformation'
                    rule['transformation'] = line.replace('- Transformation:', '').strip()
                elif line.startswith('- Steps:'):
                    current_field = 'steps'
                    rule['steps'] = line.replace('- Steps:', '').strip()
                elif line.startswith('- Benefit:'):
                    current_field = 'benefit'
                    rule['benefit'] = line.replace('- Benefit:', '').strip()
                elif current_field and line:
                    rule[current_field] += ' ' + line
            
            if rule['name']:
                rules.append(rule)
        
        return rules
    
    def _fallback_rules(self, features: Dict) -> str:
        """Generate fallback rules if LLM fails."""
        rules = []
        
        if 'Einsum' in str(features['key_changes']):
            rules.append("RULE 1: Replace Einsum with MatMul\n- Condition: Model contains Einsum operations\n- Transformation: Decompose Einsum into MatMul + Reshape\n- Steps: 1) Analyze Einsum equation 2) Replace with equivalent MatMul 3) Add Reshape for correct output shape\n- Benefit: Einsum often unsupported on edge devices")
        
        if features['shape_modifications'] > 0:
            rules.append("RULE 2: Resolve Dynamic Shapes\n- Condition: Model has unknown/dynamic tensor dimensions\n- Transformation: Add shape inference operations\n- Steps: 1) Identify dynamic dimensions 2) Add Reshape/Transpose to fix shapes 3) Use concrete dimensions\n- Benefit: Dynamic shapes prevent compilation")
        
        return '\n\n'.join(rules) if rules else "No specific rules identified"


def main():
    """Main function to run the RAG pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate compilation rules using RAG')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--model-map', help='Path to a specific model map file')
    parser.add_argument('--build-kb', action='store_true', help='Build knowledge base from features')
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent
    rag_data_dir = base_dir / "rag_data"
    rag_data_dir.mkdir(exist_ok=True)
    
    features_file = rag_data_dir / "features.json"
    kb_file = rag_data_dir / "knowledge_base.pkl"
    
    # Initialize pipeline
    pipeline = RAGPipeline(args.api_key)
    
    if args.build_kb:
        # Extract features and build knowledge base
        print("Extracting features...")
        from feature_extractor import extract_all_features
        map_dataset_dir = base_dir / "map_dataset"
        features = extract_all_features(str(map_dataset_dir), str(features_file))
        
        print("Building knowledge base...")
        pipeline.build_knowledge_base(str(features_file))
        
        # Save vector store
        with open(kb_file, 'wb') as f:
            pickle.dump(pipeline.vector_store, f)
        print(f"Knowledge base saved to {kb_file}")
    
    else:
        # Load knowledge base
        if kb_file.exists():
            with open(kb_file, 'rb') as f:
                pipeline.vector_store = pickle.load(f)
            print(f"Loaded knowledge base from {kb_file}")
        else:
            print("Knowledge base not found. Run with --build-kb first.")
            return
        
        # Generate rules for a model
        if args.model_map:
            from difference_extractor import extract_differences
            diff = extract_differences(args.model_map)
            result = pipeline.generate_rules(diff)
            
            # Save results
            output_file = rag_data_dir / f"rules_{diff.model_name}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\nGenerated rules for {diff.model_name}")
            print(f"Rules saved to {output_file}")
            print("\nGenerated Rules:")
            for i, rule in enumerate(result['generated_rules'], 1):
                print(f"\n{i}. {rule['name']}")
                print(f"   Condition: {rule['condition']}")
                print(f"   Transformation: {rule['transformation']}")


if __name__ == "__main__":
    main()

