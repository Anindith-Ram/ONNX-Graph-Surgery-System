#!/usr/bin/env python3
"""
Checkpoint Manager for Multi-Day Processing.

Manages checkpointing and resuming of processing across days.
Saves state after quota exhaustion and allows resuming next day.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class CheckpointManager:
    """Manages checkpointing for multi-day processing."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model_name: str,
        phase: str,
        processed_suggestions: List[Any],
        remaining_suggestions: List[Any],
        api_calls_used: int,
        timestamp: Optional[str] = None,
        additional_data: Optional[Dict] = None
    ):
        """
        Save checkpoint with current state.
        
        Args:
            model_name: Name of the model being processed
            phase: Phase identifier ("phase1" or "phase2")
            processed_suggestions: List of processed Suggestion objects
            remaining_suggestions: List of remaining Suggestion objects
            api_calls_used: Number of API calls used so far
            timestamp: Optional timestamp (defaults to now)
            additional_data: Optional additional data to store
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Convert suggestions to dicts if they have to_dict method
        processed_dicts = []
        for s in processed_suggestions:
            if hasattr(s, 'to_dict'):
                processed_dicts.append(s.to_dict())
            elif isinstance(s, dict):
                processed_dicts.append(s)
            else:
                # Try to serialize as dict
                try:
                    processed_dicts.append({
                        'id': getattr(s, 'id', None),
                        'priority': str(getattr(s, 'priority', '')),
                        'confidence': getattr(s, 'confidence', 0.0),
                        'issue': getattr(s, 'issue', ''),
                        'suggestion': getattr(s, 'suggestion', ''),
                    })
                except:
                    pass
        
        remaining_dicts = []
        for s in remaining_suggestions:
            if hasattr(s, 'to_dict'):
                remaining_dicts.append(s.to_dict())
            elif isinstance(s, dict):
                remaining_dicts.append(s)
            else:
                try:
                    remaining_dicts.append({
                        'id': getattr(s, 'id', None),
                        'priority': str(getattr(s, 'priority', '')),
                        'confidence': getattr(s, 'confidence', 0.0),
                        'issue': getattr(s, 'issue', ''),
                        'suggestion': getattr(s, 'suggestion', ''),
                    })
                except:
                    pass
        
        checkpoint = {
            'model_name': model_name,
            'phase': phase,
            'processed_suggestions': processed_dicts,
            'remaining_suggestions': remaining_dicts,
            'api_calls_used': api_calls_used,
            'timestamp': timestamp,
            'date': datetime.now().date().isoformat(),
            'total_suggestions': len(processed_dicts) + len(remaining_dicts),
            'processed_count': len(processed_dicts),
            'remaining_count': len(remaining_dicts)
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        checkpoint_file = self.checkpoint_dir / f"{model_name}_{phase}.json"
        
        # Atomic write: write to temp file, then rename
        temp_file = checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            temp_file.replace(checkpoint_file)
            print(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def load_checkpoint(self, model_name: str, phase: str) -> Optional[Dict]:
        """
        Load checkpoint if exists.
        
        Args:
            model_name: Name of the model
            phase: Phase identifier
            
        Returns:
            Checkpoint dictionary or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{model_name}_{phase}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                return checkpoint
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load checkpoint {checkpoint_file}: {e}")
                return None
        return None
    
    def clear_checkpoint(self, model_name: str, phase: str):
        """
        Clear checkpoint after successful completion.
        
        Args:
            model_name: Name of the model
            phase: Phase identifier
        """
        checkpoint_file = self.checkpoint_dir / f"{model_name}_{phase}.json"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"Checkpoint cleared: {checkpoint_file}")
            except Exception as e:
                print(f"Warning: Failed to clear checkpoint: {e}")
    
    def list_checkpoints(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        List all checkpoints.
        
        Args:
            model_name: Optional filter by model name
            
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    if model_name is None or checkpoint.get('model_name') == model_name:
                        checkpoints.append({
                            'file': str(checkpoint_file),
                            'model_name': checkpoint.get('model_name'),
                            'phase': checkpoint.get('phase'),
                            'date': checkpoint.get('date'),
                            'processed_count': checkpoint.get('processed_count', 0),
                            'remaining_count': checkpoint.get('remaining_count', 0),
                            'timestamp': checkpoint.get('timestamp')
                        })
            except:
                pass
        return checkpoints

