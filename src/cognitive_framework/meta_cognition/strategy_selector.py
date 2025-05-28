"""
Strategy Selector Module for TAAT Cognitive Framework.

This module implements strategy selection capabilities for choosing
the most appropriate strategy based on context and performance history.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import random

class StrategySelector:
    """
    Strategy Selector for TAAT Cognitive Framework.
    
    Selects the most appropriate strategy based on context, performance history,
    and other relevant factors.
    """
    
    def __init__(self):
        """Initialize the strategy selector."""
        self.strategy_priorities = {}
        self.strategy_performance = {}
        self.selection_history = []
        self.logger = logging.getLogger("StrategySelector")
    
    def add_strategy(self, 
                    strategy_id: str, 
                    priority: float = 0.5) -> None:
        """
        Add a strategy to the selector.
        
        Args:
            strategy_id: ID of the strategy to add
            priority: Initial priority of the strategy (0.0 to 1.0)
        """
        self.strategy_priorities[strategy_id] = priority
        
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                "success_count": 0,
                "failure_count": 0,
                "total_reward": 0.0,
                "execution_count": 0,
                "last_executed": None,
                "contexts": []
            }
        
        self.logger.info(f"Added strategy {strategy_id} with priority {priority}")
    
    def select_strategy(self, 
                       context: Dict[str, Any],
                       available_strategies: List[str]) -> Tuple[str, float]:
        """
        Select the most appropriate strategy based on context.
        
        Args:
            context: Current context
            available_strategies: List of available strategy IDs
            
        Returns:
            Tuple of (selected strategy ID, confidence)
        """
        # Filter available strategies
        valid_strategies = [
            strategy_id for strategy_id in available_strategies
            if strategy_id in self.strategy_priorities
        ]
        
        if not valid_strategies:
            # No valid strategies available
            return None, 0.0
        
        # Calculate scores for each strategy
        strategy_scores = {}
        
        for strategy_id in valid_strategies:
            # Base score from priority
            base_score = self.strategy_priorities[strategy_id]
            
            # Context similarity score
            context_score = self._calculate_context_similarity(strategy_id, context)
            
            # Performance score
            performance_score = self._calculate_performance_score(strategy_id)
            
            # Combine scores
            strategy_scores[strategy_id] = (
                0.4 * base_score + 
                0.3 * context_score + 
                0.3 * performance_score
            )
        
        # Select strategy with highest score
        selected_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        strategy_id = selected_strategy[0]
        confidence = selected_strategy[1]
        
        # Record selection
        self.selection_history.append({
            "timestamp": datetime.now(),
            "selected_strategy": strategy_id,
            "confidence": confidence,
            "context": context,
            "available_strategies": available_strategies,
            "scores": strategy_scores
        })
        
        self.logger.info(f"Selected strategy {strategy_id} with confidence {confidence}")
        
        return strategy_id, confidence
    
    def update_strategy_performance(self, 
                                  strategy_id: str, 
                                  success: bool, 
                                  reward: float = 0.0,
                                  context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update strategy performance based on execution results.
        
        Args:
            strategy_id: ID of the strategy
            success: Whether the strategy execution was successful
            reward: Reward value for the execution
            context: Context in which the strategy was executed
        """
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                "success_count": 0,
                "failure_count": 0,
                "total_reward": 0.0,
                "execution_count": 0,
                "last_executed": None,
                "contexts": []
            }
        
        # Update performance metrics
        performance = self.strategy_performance[strategy_id]
        performance["execution_count"] += 1
        performance["last_executed"] = datetime.now()
        performance["total_reward"] += reward
        
        if success:
            performance["success_count"] += 1
        else:
            performance["failure_count"] += 1
        
        # Store context
        if context:
            performance["contexts"].append({
                "timestamp": datetime.now(),
                "context": context,
                "success": success,
                "reward": reward
            })
            
            # Limit stored contexts
            max_contexts = 10
            if len(performance["contexts"]) > max_contexts:
                performance["contexts"] = performance["contexts"][-max_contexts:]
        
        # Update priority based on performance
        if strategy_id in self.strategy_priorities:
            success_rate = performance["success_count"] / performance["execution_count"]
            avg_reward = performance["total_reward"] / performance["execution_count"]
            
            # Adjust priority (with some randomness to encourage exploration)
            new_priority = (
                0.7 * success_rate + 
                0.2 * avg_reward + 
                0.1 * random.random()
            )
            
            # Ensure priority is in valid range
            new_priority = max(0.1, min(0.9, new_priority))
            
            self.strategy_priorities[strategy_id] = new_priority
        
        self.logger.info(f"Updated performance for strategy {strategy_id}: success={success}, reward={reward}")
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Performance metrics or None if strategy not found
        """
        if strategy_id not in self.strategy_performance:
            return None
        
        return self.strategy_performance[strategy_id]
    
    def get_selection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get strategy selection history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of selection history entries
        """
        # Sort by timestamp
        sorted_history = sorted(
            self.selection_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def reset_priorities(self) -> None:
        """Reset all strategy priorities to default value."""
        for strategy_id in self.strategy_priorities:
            self.strategy_priorities[strategy_id] = 0.5
        
        self.logger.info("Reset all strategy priorities")
    
    def _calculate_context_similarity(self, 
                                    strategy_id: str, 
                                    current_context: Dict[str, Any]) -> float:
        """
        Calculate similarity between current context and historical contexts.
        
        Args:
            strategy_id: ID of the strategy
            current_context: Current context
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if strategy_id not in self.strategy_performance:
            return 0.5  # Default score for unknown strategies
        
        historical_contexts = self.strategy_performance[strategy_id].get("contexts", [])
        
        if not historical_contexts or not current_context:
            return 0.5  # Default score if no historical contexts or current context
        
        # Find most similar historical context
        max_similarity = 0.0
        
        for historical_entry in historical_contexts:
            historical_context = historical_entry.get("context", {})
            
            if not historical_context:
                continue
            
            # Calculate similarity
            similarity = self._calculate_dict_similarity(current_context, historical_context)
            
            # Weight by success
            if historical_entry.get("success", False):
                similarity *= 1.2
            
            max_similarity = max(max_similarity, similarity)
        
        return min(1.0, max_similarity)
    
    def _calculate_dict_similarity(self, 
                                 dict1: Dict[str, Any], 
                                 dict2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Get all keys
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        if not all_keys:
            return 0.0
        
        # Count matching keys and values
        matching_keys = 0
        matching_values = 0
        
        for key in all_keys:
            if key in dict1 and key in dict2:
                matching_keys += 1
                
                if dict1[key] == dict2[key]:
                    matching_values += 1
        
        # Calculate similarity
        key_similarity = matching_keys / len(all_keys)
        value_similarity = matching_values / len(all_keys) if all_keys else 0.0
        
        # Combine similarities
        return 0.7 * key_similarity + 0.3 * value_similarity
    
    def _calculate_performance_score(self, strategy_id: str) -> float:
        """
        Calculate performance score for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        if strategy_id not in self.strategy_performance:
            return 0.5  # Default score for unknown strategies
        
        performance = self.strategy_performance[strategy_id]
        
        if performance["execution_count"] == 0:
            return 0.5  # Default score for never-executed strategies
        
        # Calculate success rate
        success_rate = performance["success_count"] / performance["execution_count"]
        
        # Calculate average reward
        avg_reward = performance["total_reward"] / performance["execution_count"]
        
        # Combine metrics
        return 0.7 * success_rate + 0.3 * avg_reward
