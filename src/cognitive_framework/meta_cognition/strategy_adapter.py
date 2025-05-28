"""
Strategy Adapter Module for TAAT Cognitive Framework.

This module implements strategy adaptation capabilities for dynamically
adjusting strategy parameters based on performance and context.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import random

class StrategyAdapter:
    """
    Strategy Adapter for TAAT Cognitive Framework.
    
    Adapts strategy parameters based on performance feedback and context,
    enabling dynamic optimization of strategy execution.
    """
    
    def __init__(self):
        """Initialize the strategy adapter."""
        self.strategy_parameters = {}
        self.parameter_constraints = {}
        self.adaptation_history = {}
        self.logger = logging.getLogger("StrategyAdapter")
    
    def register_strategy_parameters(self,
                                   strategy_id: str,
                                   parameters: Dict[str, Any],
                                   parameter_constraints: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Register parameters for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            parameters: Initial parameter values
            parameter_constraints: Optional constraints for parameters
        """
        self.strategy_parameters[strategy_id] = parameters.copy()
        
        if parameter_constraints:
            self.parameter_constraints[strategy_id] = parameter_constraints.copy()
        else:
            self.parameter_constraints[strategy_id] = {}
        
        # Initialize adaptation history
        if strategy_id not in self.adaptation_history:
            self.adaptation_history[strategy_id] = []
        
        self.logger.info(f"Registered parameters for strategy {strategy_id}")
    
    def adapt_strategy(self,
                      strategy_id: str,
                      performance_data: Dict[str, Any],
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adapt strategy parameters based on performance data.
        
        Args:
            strategy_id: ID of the strategy
            performance_data: Performance data for the strategy
            context: Optional context in which the strategy was executed
            
        Returns:
            Adapted parameters
        """
        if strategy_id not in self.strategy_parameters:
            self.logger.warning(f"Strategy {strategy_id} not found in adapter")
            return {}
        
        # Get current parameters
        current_params = self.strategy_parameters[strategy_id].copy()
        
        # Get constraints
        constraints = self.parameter_constraints.get(strategy_id, {})
        
        # Determine adaptation direction based on performance
        success = performance_data.get("success", False)
        metrics = performance_data.get("metrics", {})
        
        # Adapt parameters
        adapted_params = self._adapt_parameters(
            strategy_id, current_params, constraints, success, metrics, context
        )
        
        # Apply constraints
        for param_name, param_value in adapted_params.items():
            if param_name in constraints:
                param_constraints = constraints[param_name]
                
                if "min" in param_constraints and param_value < param_constraints["min"]:
                    adapted_params[param_name] = param_constraints["min"]
                
                if "max" in param_constraints and param_value > param_constraints["max"]:
                    adapted_params[param_name] = param_constraints["max"]
                
                if "allowed_values" in param_constraints:
                    if param_value not in param_constraints["allowed_values"]:
                        # Find closest allowed value
                        if isinstance(param_value, (int, float)):
                            allowed_values = param_constraints["allowed_values"]
                            closest_value = min(allowed_values, key=lambda x: abs(x - param_value))
                            adapted_params[param_name] = closest_value
        
        # Update strategy parameters
        self.strategy_parameters[strategy_id] = adapted_params
        
        # Record adaptation
        self.adaptation_history[strategy_id].append({
            "timestamp": datetime.now(),
            "previous_params": current_params,
            "adapted_params": adapted_params,
            "performance_data": performance_data,
            "context": context
        })
        
        # Limit history size
        max_history = 10
        if len(self.adaptation_history[strategy_id]) > max_history:
            self.adaptation_history[strategy_id] = self.adaptation_history[strategy_id][-max_history:]
        
        self.logger.info(f"Adapted parameters for strategy {strategy_id}")
        
        return adapted_params
    
    def get_strategy_parameters(self,
                              strategy_id: str,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get parameters for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            context: Optional context for context-specific parameters
            
        Returns:
            Strategy parameters
        """
        if strategy_id not in self.strategy_parameters:
            return {}
        
        # Get base parameters
        params = self.strategy_parameters[strategy_id].copy()
        
        # Apply context-specific adjustments if needed
        if context:
            params = self._adjust_for_context(strategy_id, params, context)
        
        return params
    
    def get_adaptation_history(self,
                             strategy_id: str,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get adaptation history for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            limit: Maximum number of history entries to return
            
        Returns:
            List of adaptation history entries
        """
        if strategy_id not in self.adaptation_history:
            return []
        
        # Sort by timestamp
        sorted_history = sorted(
            self.adaptation_history[strategy_id],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def reset_parameters(self,
                        strategy_id: str,
                        parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Reset parameters for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            parameters: Optional new parameters (None to use initial parameters)
            
        Returns:
            True if successful, False if strategy not found
        """
        if strategy_id not in self.strategy_parameters:
            return False
        
        if parameters:
            self.strategy_parameters[strategy_id] = parameters.copy()
        else:
            # Try to get initial parameters from history
            if strategy_id in self.adaptation_history and self.adaptation_history[strategy_id]:
                initial_entry = self.adaptation_history[strategy_id][0]
                self.strategy_parameters[strategy_id] = initial_entry["previous_params"].copy()
        
        self.logger.info(f"Reset parameters for strategy {strategy_id}")
        
        return True
    
    def _adapt_parameters(self,
                        strategy_id: str,
                        current_params: Dict[str, Any],
                        constraints: Dict[str, Dict[str, Any]],
                        success: bool,
                        metrics: Dict[str, Any],
                        context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adapt parameters based on performance.
        
        Args:
            strategy_id: ID of the strategy
            current_params: Current parameter values
            constraints: Parameter constraints
            success: Whether the strategy execution was successful
            metrics: Performance metrics
            context: Optional context
            
        Returns:
            Adapted parameters
        """
        adapted_params = current_params.copy()
        
        # Determine adaptation strength
        adaptation_strength = 0.1  # Default
        
        if success:
            # Smaller adjustments for successful strategies
            adaptation_strength = 0.05
        else:
            # Larger adjustments for unsuccessful strategies
            adaptation_strength = 0.2
        
        # Adapt numerical parameters
        for param_name, param_value in current_params.items():
            if isinstance(param_value, (int, float)):
                # Determine adaptation direction
                if success:
                    # If successful, make small adjustments in the same direction
                    if self._parameter_improved(strategy_id, param_name, metrics):
                        direction = self._get_last_direction(strategy_id, param_name)
                    else:
                        # If not improved, try the opposite direction
                        direction = -self._get_last_direction(strategy_id, param_name)
                else:
                    # If unsuccessful, try a different direction
                    direction = random.choice([-1, 1])
                
                # Calculate adjustment
                if isinstance(param_value, int):
                    adjustment = max(1, int(param_value * adaptation_strength))
                    new_value = param_value + (direction * adjustment)
                else:
                    adjustment = param_value * adaptation_strength
                    new_value = param_value + (direction * adjustment)
                
                adapted_params[param_name] = new_value
        
        return adapted_params
    
    def _adjust_for_context(self,
                          strategy_id: str,
                          params: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters based on context.
        
        Args:
            strategy_id: ID of the strategy
            params: Base parameters
            context: Current context
            
        Returns:
            Context-adjusted parameters
        """
        adjusted_params = params.copy()
        
        # Find similar contexts in adaptation history
        if strategy_id in self.adaptation_history:
            similar_entries = []
            
            for entry in self.adaptation_history[strategy_id]:
                entry_context = entry.get("context")
                
                if entry_context and self._context_similarity(context, entry_context) > 0.7:
                    similar_entries.append(entry)
            
            # If similar contexts found, use parameters from successful executions
            successful_entries = [
                entry for entry in similar_entries
                if entry.get("performance_data", {}).get("success", False)
            ]
            
            if successful_entries:
                # Use parameters from most recent successful execution in similar context
                successful_entries.sort(key=lambda x: x["timestamp"], reverse=True)
                adjusted_params = successful_entries[0]["adapted_params"].copy()
        
        return adjusted_params
    
    def _context_similarity(self,
                          context1: Dict[str, Any],
                          context2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two contexts.
        
        Args:
            context1: First context
            context2: Second context
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Get all keys
        all_keys = set(context1.keys()) | set(context2.keys())
        
        if not all_keys:
            return 0.0
        
        # Count matching keys and values
        matching_keys = 0
        matching_values = 0
        
        for key in all_keys:
            if key in context1 and key in context2:
                matching_keys += 1
                
                if context1[key] == context2[key]:
                    matching_values += 1
        
        # Calculate similarity
        key_similarity = matching_keys / len(all_keys)
        value_similarity = matching_values / len(all_keys) if all_keys else 0.0
        
        # Combine similarities
        return 0.7 * key_similarity + 0.3 * value_similarity
    
    def _parameter_improved(self,
                          strategy_id: str,
                          param_name: str,
                          current_metrics: Dict[str, Any]) -> bool:
        """
        Determine if a parameter change improved performance.
        
        Args:
            strategy_id: ID of the strategy
            param_name: Name of the parameter
            current_metrics: Current performance metrics
            
        Returns:
            True if performance improved, False otherwise
        """
        if strategy_id not in self.adaptation_history or not self.adaptation_history[strategy_id]:
            return True
        
        # Get previous metrics
        previous_entry = self.adaptation_history[strategy_id][-1]
        previous_metrics = previous_entry.get("performance_data", {}).get("metrics", {})
        
        if not previous_metrics:
            return True
        
        # Compare key metrics
        for metric_name in ["accuracy", "efficiency", "speed"]:
            if metric_name in current_metrics and metric_name in previous_metrics:
                if current_metrics[metric_name] > previous_metrics[metric_name]:
                    return True
        
        return False
    
    def _get_last_direction(self,
                          strategy_id: str,
                          param_name: str) -> int:
        """
        Get the direction of the last parameter change.
        
        Args:
            strategy_id: ID of the strategy
            param_name: Name of the parameter
            
        Returns:
            Direction of change (1 for increase, -1 for decrease)
        """
        if strategy_id not in self.adaptation_history or not self.adaptation_history[strategy_id]:
            return random.choice([-1, 1])
        
        # Get last two entries
        if len(self.adaptation_history[strategy_id]) >= 2:
            current_entry = self.adaptation_history[strategy_id][-1]
            previous_entry = self.adaptation_history[strategy_id][-2]
            
            current_value = current_entry.get("adapted_params", {}).get(param_name)
            previous_value = previous_entry.get("adapted_params", {}).get(param_name)
            
            if current_value is not None and previous_value is not None:
                if current_value > previous_value:
                    return 1
                elif current_value < previous_value:
                    return -1
        
        return random.choice([-1, 1])
