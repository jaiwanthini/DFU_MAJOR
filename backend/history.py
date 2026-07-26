"""
==========================================================
Smart Insole DFU Risk Prediction
Prediction History Manager

In-memory store for keeping the latest N predictions.
Provides endpoint data for dashboard trending graphs.
==========================================================
"""

from datetime import datetime
from collections import deque
from typing import List, Dict, Any

class HistoryManager:
    """
    Maintains an in-memory deque of the last N predictions.
    No database is required for this phase of the project.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the history manager.
        
        Parameters
        ----------
        max_size : int
            Maximum number of predictions to store before older ones 
            are automatically dropped (FIFO). Defaults to 100.
        """
        self.max_size = max_size
        self.records = deque(maxlen=max_size)

    def add(self, prediction: Dict[str, Any]) -> None:
        """
        Stores a new prediction in history.
        Automatically injects an ISO 8601 server timestamp if missing.
        """
        if not isinstance(prediction, dict):
            raise TypeError("Prediction must be a dictionary.")
            
        # Copy to avoid mutating the live dictionary reference
        record = prediction.copy()
        
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()
            
        self.records.append(record)

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Returns the entire prediction history.
        
        Returns
        -------
        List[Dict[str, Any]]
            Chronological list of predictions (oldest to newest).
        """
        return [record.copy() for record in self.records]

    def get_latest(self) -> Dict[str, Any]:
        """
        Returns the most recent prediction.
        
        Returns
        -------
        Dict[str, Any]
            The latest prediction dictionary, or an empty dict if none exist.
        """
        if self.records:
            return self.records[-1].copy()
        return {}
        
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the most recent N predictions.
        
        Parameters
        ----------
        count : int
            The number of recent records to retrieve.
            
        Returns
        -------
        List[Dict[str, Any]]
            Chronological list of the requested recent predictions.
        """
        if count <= 0:
            return []
            
        # Safe slicing: if count > len(self.records), Python returns all available items
        recent_records = list(self.records)[-count:]
        return [record.copy() for record in recent_records]

    def size(self) -> int:
        """
        Returns the current number of stored predictions.
        
        Returns
        -------
        int
            The current history size.
        """
        return len(self.records)

    def clear(self) -> None:
        """
        Wipes the current history.
        Useful when switching patient profiles or resetting the dashboard.
        """
        self.records.clear()


# ==========================================================
# Singleton Instance
# ==========================================================
# This singleton is imported directly by routes.py
history_manager = HistoryManager(max_size=100)
