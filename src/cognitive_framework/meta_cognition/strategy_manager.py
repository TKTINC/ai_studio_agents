"""
Strategy Manager Module for TAAT Cognitive Framework.

This module implements strategy management capabilities for registering,
retrieving, and managing different cognitive strategies.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime

class StrategyManager:
    """
    Strategy Manager for TAAT Cognitive Framework.
    
    Manages the registration, retrieval, and organization of different
    cognitive strategies used by the agent.
    """
    
    def __init__(self):
        """Initialize the strategy manager."""
        self.strategies = {}
        self.strategy_functions = {}
        self.logger = logging.getLogger("StrategyManager")
    
    def register_strategy(self,
                         strategy_id: str,
                         name: str,
                         description: str,
                         function: Callable,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            name: Human-readable name of the strategy
            description: Description of what the strategy does
            function: Function implementing the strategy
            metadata: Additional metadata about the strategy
        """
        self.strategies[strategy_id] = {
            "name": name,
            "description": description,
            "metadata": metadata or {},
            "registered_at": datetime.now()
        }
        
        self.strategy_functions[strategy_id] = function
        
        self.logger.info(f"Registered strategy {strategy_id}: {name}")
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a strategy by ID.
        
        Args:
            strategy_id: ID of the strategy to retrieve
            
        Returns:
            Strategy information or None if not found
        """
        if strategy_id not in self.strategies:
            return None
        
        return self.strategies[strategy_id]
    
    def get_strategy_function(self, strategy_id: str) -> Optional[Callable]:
        """
        Get a strategy function by ID.
        
        Args:
            strategy_id: ID of the strategy function to retrieve
            
        Returns:
            Strategy function or None if not found
        """
        if strategy_id not in self.strategy_functions:
            return None
        
        return self.strategy_functions[strategy_id]
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available strategies.
        
        Returns:
            Dictionary of available strategies
        """
        return self.strategies
    
    def get_strategy_functions(self) -> Dict[str, Callable]:
        """
        Get all strategy functions.
        
        Returns:
            Dictionary of strategy functions
        """
        return self.strategy_functions
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy.
        
        Args:
            strategy_id: ID of the strategy to remove
            
        Returns:
            True if successful, False if strategy not found
        """
        if strategy_id not in self.strategies:
            return False
        
        del self.strategies[strategy_id]
        
        if strategy_id in self.strategy_functions:
            del self.strategy_functions[strategy_id]
        
        self.logger.info(f"Removed strategy {strategy_id}")
        
        return True
    
    def update_strategy_metadata(self,
                               strategy_id: str,
                               metadata: Dict[str, Any]) -> bool:
        """
        Update strategy metadata.
        
        Args:
            strategy_id: ID of the strategy to update
            metadata: New metadata to merge with existing metadata
            
        Returns:
            True if successful, False if strategy not found
        """
        if strategy_id not in self.strategies:
            return False
        
        # Merge new metadata with existing metadata
        self.strategies[strategy_id]["metadata"].update(metadata)
        self.strategies[strategy_id]["updated_at"] = datetime.now()
        
        self.logger.info(f"Updated metadata for strategy {strategy_id}")
        
        return True
    
    def get_strategies_by_tag(self, tag: str) -> Dict[str, Dict[str, Any]]:
        """
        Get strategies by tag.
        
        Args:
            tag: Tag to filter strategies by
            
        Returns:
            Dictionary of strategies with the specified tag
        """
        filtered_strategies = {}
        
        for strategy_id, strategy in self.strategies.items():
            tags = strategy.get("metadata", {}).get("tags", [])
            
            if tag in tags:
                filtered_strategies[strategy_id] = strategy
        
        return filtered_strategies
    
    def get_strategy_count(self) -> int:
        """
        Get the number of registered strategies.
        
        Returns:
            Number of registered strategies
        """
        return len(self.strategies)
