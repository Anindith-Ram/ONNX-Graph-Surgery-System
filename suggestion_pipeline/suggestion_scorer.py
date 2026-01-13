#!/usr/bin/env python3
"""
Suggestion Scorer for Prioritization.

Scores suggestions based on priority, confidence, location importance,
and impact to determine which suggestions should get individual enhancement.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from suggestion_generator import Suggestion, SuggestionLocation, Priority
else:
    # Import at runtime to avoid circular dependencies
    pass


class SuggestionScorer:
    """Score suggestions for prioritization."""
    
    PRIORITY_WEIGHTS = {
        'CRITICAL': 10.0,
        'critical': 10.0,
        'HIGH': 5.0,
        'high': 5.0,
        'MEDIUM': 2.0,
        'medium': 2.0,
        'LOW': 1.0,
        'low': 1.0,
        'INFO': 0.5,
        'info': 0.5
    }
    
    def calculate_impact_score(self, suggestion) -> float:
        """
        Calculate impact score for prioritization.
        
        Args:
            suggestion: Suggestion object with priority, confidence, location
            
        Returns:
            Float impact score (higher = more important)
        """
        # Get priority weight
        priority_value = suggestion.priority
        if hasattr(priority_value, 'value'):
            priority_str = priority_value.value
        else:
            priority_str = str(priority_value)
        
        priority_weight = self.PRIORITY_WEIGHTS.get(priority_str, 1.0)
        
        # Base score from priority and confidence
        base_score = priority_weight * suggestion.confidence
        
        # Location importance multiplier
        location_mult = self._get_location_multiplier(suggestion.location)
        
        # Dependent count (more dependents = more impact)
        dependent_count = len(suggestion.location.successors) if hasattr(suggestion.location, 'successors') else 0
        dependent_mult = 1.0 + (dependent_count * 0.1)
        
        return base_score * location_mult * dependent_mult
    
    def _get_location_multiplier(self, location) -> float:
        """
        Get multiplier based on graph position.
        
        Args:
            location: SuggestionLocation object
            
        Returns:
            Float multiplier (1.0 = normal, >1.0 = more important)
        """
        if not hasattr(location, 'graph_position') or location.graph_position is None:
            return 1.0
        
        # Near outputs (critical path) = higher multiplier
        if location.graph_position > 0.8:
            return 1.5
        elif location.graph_position < 0.2:
            return 1.2
        else:
            return 1.0
    
    def rank_suggestions(self, suggestions: list) -> list:
        """
        Rank suggestions by impact score.
        
        Args:
            suggestions: List of Suggestion objects
            
        Returns:
            List of tuples (suggestion, impact_score) sorted by score (descending)
        """
        scored = [(s, self.calculate_impact_score(s)) for s in suggestions]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

