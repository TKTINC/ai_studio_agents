"""
Strategy Coordinator Module for TAAT Cognitive Framework.

This module implements strategy coordination capabilities for managing
the execution and coordination of multiple cognitive strategies.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import logging
from datetime import datetime

class StrategyCoordinator:
    """
    Strategy Coordinator for TAAT Cognitive Framework.
    
    Coordinates the execution and interaction of multiple cognitive strategies,
    ensuring proper sequencing, conflict resolution, and resource allocation.
    """
    
    def __init__(self):
        """Initialize the strategy coordinator."""
        self.execution_history = []
        self.active_strategies = {}
        self.registered_strategies = {}
        self.strategy_manager = None
        self.logger = logging.getLogger("StrategyCoordinator")
    
    def connect_strategy_manager(self, strategy_manager):
        """
        Connect to a strategy manager.
        
        Args:
            strategy_manager: Strategy manager instance
        """
        self.strategy_manager = strategy_manager
        self.logger.info("Connected to strategy manager")
    
    def register_strategy(self, strategy_id: str, priority: float = 0.5):
        """
        Register a strategy with the coordinator.
        
        Args:
            strategy_id: ID of the strategy to register
            priority: Priority of the strategy (0.0 to 1.0)
        """
        self.registered_strategies[strategy_id] = {
            "priority": priority,
            "registered_at": datetime.now(),
            "active": False
        }
        
        self.logger.info(f"Registered strategy {strategy_id} with priority {priority}")
    
    def activate_strategy(self, strategy_id: str) -> bool:
        """
        Activate a registered strategy.
        
        Args:
            strategy_id: ID of the strategy to activate
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_id not in self.registered_strategies:
            return False
        
        # Mark strategy as active
        self.registered_strategies[strategy_id]["active"] = True
        
        # Add to active strategies
        self.active_strategies[strategy_id] = {
            "activated_at": datetime.now(),
            "priority": self.registered_strategies[strategy_id]["priority"]
        }
        
        self.logger.info(f"Activated strategy {strategy_id}")
        
        return True
    
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Deactivate an active strategy.
        
        Args:
            strategy_id: ID of the strategy to deactivate
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_id not in self.active_strategies:
            return False
        
        # Mark strategy as inactive
        if strategy_id in self.registered_strategies:
            self.registered_strategies[strategy_id]["active"] = False
        
        # Remove from active strategies
        del self.active_strategies[strategy_id]
        
        self.logger.info(f"Deactivated strategy {strategy_id}")
        
        return True
    
    def coordinate_execution(self, 
                           context: Dict[str, Any], 
                           strategy_functions: Dict[str, Callable]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Coordinate the execution of multiple strategies.
        
        Args:
            context: Context for strategy execution
            strategy_functions: Dictionary mapping strategy IDs to execution functions
            
        Returns:
            List of tuples containing strategy ID and execution results
        """
        timestamp = datetime.now()
        
        # Initialize results
        results = []
        
        # Get active strategies
        active_strategy_ids = list(self.active_strategies.keys())
        
        # Sort by priority
        active_strategy_ids.sort(
            key=lambda sid: self.active_strategies[sid].get("priority", 0.0),
            reverse=True
        )
        
        # Execute strategies
        for strategy_id in active_strategy_ids:
            if strategy_id not in strategy_functions:
                continue
            
            # Get strategy function
            strategy_function = strategy_functions[strategy_id]
            
            # Execute strategy
            try:
                # Prepare execution context
                execution_context = self._prepare_execution_context(context, {})
                
                # Execute strategy
                result = strategy_function(execution_context)
                
                # Add to results
                results.append((strategy_id, result))
                
                # Record execution
                execution_record = {
                    "strategy_id": strategy_id,
                    "timestamp": timestamp,
                    "context": context,
                    "result": result
                }
                
                self.execution_history.append(execution_record)
                
                self.logger.info(f"Executed strategy {strategy_id}")
            
            except Exception as e:
                # Handle execution error
                error_result = {
                    "error": str(e)
                }
                
                # Add to results
                results.append((strategy_id, error_result))
                
                self.logger.error(f"Error executing strategy {strategy_id}: {str(e)}")
        
        # Limit history size
        max_history = 100
        if len(self.execution_history) > max_history:
            self.execution_history = self.execution_history[-max_history:]
        
        return results
    
    def execute_strategy(self,
                        strategy_id: str,
                        context: Dict[str, Any],
                        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a strategy.
        
        Args:
            strategy_id: ID of the strategy to execute
            context: Context for strategy execution
            parameters: Optional parameters for strategy execution
            
        Returns:
            Execution results
        """
        timestamp = datetime.now()
        
        # Check if strategy manager is available
        if not self.strategy_manager:
            return {
                "success": False,
                "error": "Strategy manager not connected",
                "timestamp": timestamp
            }
        
        # Get strategy from manager
        strategy = self.strategy_manager.get_strategy(strategy_id)
        
        if not strategy:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} not found",
                "timestamp": timestamp
            }
        
        # Check if strategy is already active
        if strategy_id in self.active_strategies:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} is already active",
                "timestamp": timestamp
            }
        
        # Mark strategy as active
        self.active_strategies[strategy_id] = {
            "start_time": timestamp,
            "context": context,
            "parameters": parameters
        }
        
        # Execute strategy
        try:
            # Prepare execution context
            execution_context = self._prepare_execution_context(context, parameters)
            
            # Execute strategy
            result = strategy["execute"](execution_context)
            
            # Add execution metadata
            result["timestamp"] = timestamp
            result["strategy_id"] = strategy_id
            result["execution_time"] = (datetime.now() - timestamp).total_seconds()
            
            # Ensure success flag is present
            if "success" not in result:
                result["success"] = True
            
            # Record execution
            execution_record = {
                "strategy_id": strategy_id,
                "timestamp": timestamp,
                "context": context,
                "parameters": parameters,
                "result": result
            }
            
            self.execution_history.append(execution_record)
            
            # Limit history size
            max_history = 100
            if len(self.execution_history) > max_history:
                self.execution_history = self.execution_history[-max_history:]
            
            self.logger.info(f"Executed strategy {strategy_id} with result: {result['success']}")
        
        except Exception as e:
            # Handle execution error
            result = {
                "success": False,
                "error": str(e),
                "timestamp": timestamp,
                "strategy_id": strategy_id,
                "execution_time": (datetime.now() - timestamp).total_seconds()
            }
            
            self.logger.error(f"Error executing strategy {strategy_id}: {str(e)}")
        
        finally:
            # Remove strategy from active strategies
            if strategy_id in self.active_strategies:
                del self.active_strategies[strategy_id]
        
        return result
    
    def execute_strategy_sequence(self,
                                sequence: List[Dict[str, Any]],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a sequence of strategies.
        
        Args:
            sequence: List of strategy execution specifications
            context: Context for strategy execution
            
        Returns:
            Execution results
        """
        timestamp = datetime.now()
        
        # Initialize results
        results = {
            "success": True,
            "timestamp": timestamp,
            "sequence_results": [],
            "failed_at": None
        }
        
        # Execute strategies in sequence
        for i, strategy_spec in enumerate(sequence):
            strategy_id = strategy_spec.get("strategy_id")
            parameters = strategy_spec.get("parameters")
            
            if not strategy_id:
                results["success"] = False
                results["error"] = f"Missing strategy_id at position {i}"
                results["failed_at"] = i
                break
            
            # Update context with previous results if available
            if results["sequence_results"]:
                previous_result = results["sequence_results"][-1]
                context["previous_result"] = previous_result
            
            # Execute strategy
            strategy_result = self.execute_strategy(
                strategy_id=strategy_id,
                context=context,
                parameters=parameters
            )
            
            # Add to sequence results
            results["sequence_results"].append(strategy_result)
            
            # Check for failure
            if not strategy_result.get("success", False):
                results["success"] = False
                results["error"] = f"Strategy {strategy_id} failed: {strategy_result.get('error', 'Unknown error')}"
                results["failed_at"] = i
                break
        
        # Calculate total execution time
        results["execution_time"] = (datetime.now() - timestamp).total_seconds()
        
        return results
    
    def get_active_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get currently active strategies.
        
        Returns:
            Dictionary of active strategies
        """
        return self.active_strategies
    
    def get_registered_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered strategies.
        
        Returns:
            Dictionary of registered strategies
        """
        return self.registered_strategies
    
    def get_execution_history(self, 
                            strategy_id: Optional[str] = None, 
                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get execution history.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            limit: Maximum number of history entries to return
            
        Returns:
            List of execution history entries
        """
        if strategy_id:
            filtered_history = [
                execution for execution in self.execution_history
                if execution["strategy_id"] == strategy_id
            ]
            
            # Sort by timestamp
            filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return filtered_history[:limit]
        else:
            # Sort by timestamp
            sorted_history = sorted(
                self.execution_history,
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            return sorted_history[:limit]
    
    def cancel_strategy(self, strategy_id: str) -> bool:
        """
        Cancel an active strategy.
        
        Args:
            strategy_id: ID of the strategy to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_id not in self.active_strategies:
            return False
        
        # Remove strategy from active strategies
        del self.active_strategies[strategy_id]
        
        self.logger.info(f"Cancelled strategy {strategy_id}")
        
        return True
    
    def cancel_all_strategies(self) -> int:
        """
        Cancel all active strategies.
        
        Returns:
            Number of strategies cancelled
        """
        count = len(self.active_strategies)
        
        # Clear active strategies
        self.active_strategies = {}
        
        self.logger.info(f"Cancelled {count} active strategies")
        
        return count
    
    def _prepare_execution_context(self, 
                                 context: Dict[str, Any], 
                                 parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare execution context for a strategy.
        
        Args:
            context: Base context
            parameters: Strategy parameters
            
        Returns:
            Prepared execution context
        """
        # Create a copy of the context
        execution_context = dict(context)
        
        # Add parameters to context
        if parameters:
            execution_context["parameters"] = parameters
        
        # Add execution metadata
        execution_context["execution_timestamp"] = datetime.now()
        
        return execution_context
