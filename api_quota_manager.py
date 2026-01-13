#!/usr/bin/env python3
"""
API Quota Manager for Daily Request Tracking.

Manages daily API quota limits and tracks usage across days.
Automatically resets quota at midnight.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class APIQuotaManager:
    """Manages daily API quota and tracking."""
    
    DAILY_LIMIT = 250
    
    def __init__(self, quota_file: str = "api_quota.json", daily_limit: Optional[int] = None):
        """
        Initialize API quota manager.
        
        Args:
            quota_file: Path to quota state file
            daily_limit: Override daily limit (default: 250)
        """
        self.quota_file = Path(quota_file)
        if daily_limit is not None:
            self.DAILY_LIMIT = daily_limit
        self._load_quota_state()
    
    def _load_quota_state(self):
        """Load quota state from file."""
        if self.quota_file.exists():
            try:
                with open(self.quota_file, 'r') as f:
                    state = json.load(f)
                    self.last_date = state.get('last_date')
                    self.used_today = state.get('used_today', 0)
            except (json.JSONDecodeError, KeyError):
                # Corrupted file, reset
                self.last_date = None
                self.used_today = 0
        else:
            self.last_date = None
            self.used_today = 0
        
        # Reset if new day
        today = datetime.now().date().isoformat()
        if self.last_date != today:
            if self.last_date is not None:
                print(f"New day detected. Resetting quota (was {self.used_today}/{self.DAILY_LIMIT} on {self.last_date})")
            self.used_today = 0
            self.last_date = today
            self._save_quota_state()
    
    def can_make_request(self, count: int = 1) -> bool:
        """
        Check if we can make N requests within quota.
        
        Args:
            count: Number of requests to check
            
        Returns:
            True if quota allows, False otherwise
        """
        # Check if new day (reset if needed)
        today = datetime.now().date().isoformat()
        if self.last_date != today:
            self._load_quota_state()
        
        return (self.used_today + count) <= self.DAILY_LIMIT
    
    def record_request(self, count: int = 1):
        """
        Record API request(s) and save state.
        
        Args:
            count: Number of requests to record
        """
        # Check if new day (reset if needed)
        today = datetime.now().date().isoformat()
        if self.last_date != today:
            self._load_quota_state()
        
        self.used_today += count
        self._save_quota_state()
    
    def get_remaining_quota(self) -> int:
        """
        Get remaining quota for today.
        
        Returns:
            Number of requests remaining
        """
        # Check if new day (reset if needed)
        today = datetime.now().date().isoformat()
        if self.last_date != today:
            self._load_quota_state()
        
        return max(0, self.DAILY_LIMIT - self.used_today)
    
    def get_usage_stats(self) -> dict:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        return {
            'used_today': self.used_today,
            'daily_limit': self.DAILY_LIMIT,
            'remaining': self.get_remaining_quota(),
            'last_date': self.last_date,
            'percentage_used': (self.used_today / self.DAILY_LIMIT * 100) if self.DAILY_LIMIT > 0 else 0
        }
    
    def _save_quota_state(self):
        """Save quota state to file."""
        state = {
            'last_date': self.last_date,
            'used_today': self.used_today,
            'daily_limit': self.DAILY_LIMIT
        }
        
        # Atomic write: write to temp file, then rename
        temp_file = self.quota_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            temp_file.replace(self.quota_file)
        except Exception as e:
            print(f"Warning: Failed to save quota state: {e}")
            if temp_file.exists():
                temp_file.unlink()

