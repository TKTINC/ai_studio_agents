"""
Procedural Memory System for AI Studio Agents.

This module implements procedural memory capabilities for agents,
allowing them to store, retrieve, execute, and learn from action sequences and procedures.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
import json
import os
import uuid
import logging
import time

# Set up logging
logger = logging.getLogger(__name__)


class ProceduralMemory:
    """
    Procedural Memory system for AI Studio Agents.
    
    Stores action sequences, skills, and procedures, allowing agents to:
    - Store and retrieve complex action sequences
    - Execute stored procedures with parameter passing
    - Learn and adapt procedures based on outcomes
    - Maintain version history with rollback capabilities
    - Track execution statistics for optimization
    """
    
    def __init__(self, agent_id: str, storage_path: Optional[str] = None, max_versions: int = 5):
        """
        Initialize the procedural memory system.
        
        Args:
            agent_id: Unique identifier for the agent
            storage_path: Path to store persistent memory (None for in-memory only)
            max_versions: Maximum number of previous versions to store per procedure
        """
        self.agent_id = agent_id
        self.storage_path = storage_path
        self.max_versions = max_versions
        self.procedures: Dict[str, Dict[str, Any]] = {}
        self.function_registry: Dict[str, Callable] = {}
        
        # Create storage directory if needed
        if storage_path:
            os.makedirs(os.path.join(storage_path, agent_id, "procedural"), exist_ok=True)
            self._load_from_disk()
    
    def register_function(self, func_name: str, func: Callable) -> None:
        """
        Register a function that can be called by procedures.
        
        Args:
            func_name: Name to register the function under
            func: The callable function
        """
        self.function_registry[func_name] = func
    
    def store_procedure(self, 
                       procedure_id: str, 
                       steps: List[Any], 
                       parameters: Optional[Dict[str, Any]] = None, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new procedure or update an existing one.
        
        Args:
            procedure_id: Unique identifier for the procedure
            steps: List of steps that make up the procedure
            parameters: Default parameters for the procedure
            metadata: Additional information about the procedure
            
        Returns:
            The procedure ID
        """
        timestamp = datetime.now().isoformat()
        
        if procedure_id in self.procedures:
            # Version control - store previous versions
            current = self.procedures[procedure_id]
            if "versions" not in current:
                current["versions"] = []
            
            # Add current as a version
            current_version = {
                "steps": current["steps"],
                "parameters": current["parameters"],
                "metadata": current["metadata"],
                "updated": current["updated"]
            }
            current["versions"].insert(0, current_version)
            
            # Trim versions if needed
            if len(current["versions"]) > self.max_versions:
                current["versions"] = current["versions"][:self.max_versions]
            
            # Update with new content
            current["steps"] = steps
            current["parameters"] = parameters or current["parameters"]
            current["metadata"] = metadata or current["metadata"]
            current["updated"] = timestamp
        else:
            # Create new procedure
            self.procedures[procedure_id] = {
                "id": procedure_id,
                "steps": steps,
                "parameters": parameters or {},
                "metadata": metadata or {},
                "created": timestamp,
                "updated": timestamp,
                "versions": [],
                "execution_stats": {
                    "success_count": 0,
                    "failure_count": 0,
                    "last_execution": None,
                    "average_duration": 0
                }
            }
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return procedure_id
    
    def retrieve_procedure(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a procedure by ID.
        
        Args:
            procedure_id: ID of the procedure to retrieve
            
        Returns:
            The procedure or None if not found
        """
        return self.procedures.get(procedure_id)
    
    def list_procedures(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all procedures, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of procedures
        """
        if category is None:
            return list(self.procedures.values())
        
        return [p for p in self.procedures.values() 
                if p.get("metadata", {}).get("category") == category]
    
    def execute_procedure(self, 
                         procedure_id: str, 
                         context: Optional[Dict[str, Any]] = None, 
                         parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a stored procedure with given context and parameters.
        
        Args:
            procedure_id: ID of the procedure to execute
            context: Execution context (e.g., agent state)
            parameters: Parameters to pass to the procedure
            
        Returns:
            Execution results
        """
        if procedure_id not in self.procedures:
            return {"success": False, "error": "Procedure not found"}
        
        procedure = self.procedures[procedure_id]
        start_time = datetime.now()
        
        try:
            # Merge default parameters with provided ones
            merged_params = procedure["parameters"].copy()
            if parameters:
                merged_params.update(parameters)
            
            # Execute steps
            results = []
            for i, step in enumerate(procedure["steps"]):
                step_start = time.time()
                
                # Each step can be a function reference or a callable
                if isinstance(step, dict) and "function" in step:
                    # Function reference with parameters
                    func_name = step["function"]
                    func_params = step.get("parameters", {})
                    
                    # Resolve function
                    func = self._resolve_function(func_name)
                    if func is None:
                        step_result = {
                            "success": False, 
                            "error": f"Function '{func_name}' not found"
                        }
                    else:
                        try:
                            # Execute function with merged parameters
                            step_result = func(
                                context=context, 
                                **{**func_params, **merged_params}
                            )
                        except Exception as e:
                            step_result = {
                                "success": False,
                                "error": f"Function execution error: {str(e)}"
                            }
                elif callable(step):
                    # Direct callable
                    try:
                        step_result = step(context=context, **merged_params)
                    except Exception as e:
                        step_result = {
                            "success": False,
                            "error": f"Callable execution error: {str(e)}"
                        }
                else:
                    step_result = {
                        "success": False,
                        "error": f"Invalid step type: {type(step)}"
                    }
                
                step_duration = time.time() - step_start
                
                # Add metadata to result
                if isinstance(step_result, dict):
                    step_result["step_index"] = i
                    step_result["step_duration"] = step_duration
                else:
                    # Wrap non-dict results
                    step_result = {
                        "success": True,
                        "result": step_result,
                        "step_index": i,
                        "step_duration": step_duration
                    }
                
                results.append(step_result)
                
                # Stop execution if a step fails
                if isinstance(step_result, dict) and step_result.get("success") is False:
                    break
            
            # Determine overall success
            overall_success = all(
                r.get("success", False) if isinstance(r, dict) else True 
                for r in results
            )
            
            # Update execution stats
            duration = (datetime.now() - start_time).total_seconds()
            self._update_execution_stats(procedure_id, overall_success, duration)
            
            return {
                "success": overall_success,
                "results": results,
                "duration": duration,
                "procedure_id": procedure_id
            }
            
        except Exception as e:
            # Update execution stats
            duration = (datetime.now() - start_time).total_seconds()
            self._update_execution_stats(procedure_id, False, duration)
            
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "procedure_id": procedure_id
            }
    
    def rollback_procedure(self, procedure_id: str, version_index: int = 0) -> bool:
        """
        Rollback a procedure to a previous version.
        
        Args:
            procedure_id: ID of the procedure to rollback
            version_index: Index of the version to rollback to (0 is most recent)
            
        Returns:
            True if successful, False otherwise
        """
        if procedure_id not in self.procedures:
            return False
            
        procedure = self.procedures[procedure_id]
        if "versions" not in procedure or version_index >= len(procedure["versions"]):
            return False
            
        # Get the version to rollback to
        version = procedure["versions"][version_index]
        
        # Add current as a new version
        current_version = {
            "steps": procedure["steps"],
            "parameters": procedure["parameters"],
            "metadata": procedure["metadata"],
            "updated": procedure["updated"]
        }
        
        # Replace current with version
        procedure["steps"] = version["steps"]
        procedure["parameters"] = version["parameters"]
        procedure["metadata"] = version["metadata"]
        procedure["updated"] = datetime.now().isoformat()
        
        # Update versions list
        procedure["versions"].insert(0, current_version)
        if len(procedure["versions"]) > self.max_versions:
            procedure["versions"] = procedure["versions"][:self.max_versions]
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return True
    
    def learn_from_outcome(self, 
                          procedure_id: str, 
                          outcome_data: Dict[str, Any], 
                          update_procedure: bool = True) -> bool:
        """
        Update a procedure based on execution outcomes.
        
        Args:
            procedure_id: ID of the procedure to update
            outcome_data: Data about the execution outcome
            update_procedure: Whether to automatically update the procedure
            
        Returns:
            True if successful, False otherwise
        """
        if procedure_id not in self.procedures:
            return False
            
        # Store the outcome in metadata
        procedure = self.procedures[procedure_id]
        if "learning_outcomes" not in procedure["metadata"]:
            procedure["metadata"]["learning_outcomes"] = []
            
        procedure["metadata"]["learning_outcomes"].append({
            "timestamp": datetime.now().isoformat(),
            "data": outcome_data
        })
        
        # If automatic updates are enabled, modify the procedure
        if update_procedure and "update_strategy" in outcome_data:
            strategy = outcome_data["update_strategy"]
            
            if strategy.get("type") == "parameter_adjustment":
                # Adjust parameters based on outcome
                adjustments = strategy.get("adjustments", {})
                for param_name, adjustment in adjustments.items():
                    if param_name in procedure["parameters"]:
                        current_value = procedure["parameters"][param_name]
                        
                        # Apply adjustment based on type
                        if isinstance(current_value, (int, float)) and isinstance(adjustment, (int, float)):
                            procedure["parameters"][param_name] = current_value + adjustment
                        else:
                            procedure["parameters"][param_name] = adjustment
            
            elif strategy.get("type") == "step_replacement":
                # Replace specific steps
                replacements = strategy.get("replacements", [])
                for replacement in replacements:
                    step_index = replacement.get("index")
                    new_step = replacement.get("step")
                    
                    if step_index is not None and new_step is not None and 0 <= step_index < len(procedure["steps"]):
                        procedure["steps"][step_index] = new_step
            
            elif strategy.get("type") == "step_insertion":
                # Insert new steps
                insertions = strategy.get("insertions", [])
                # Sort by index descending to avoid shifting positions
                insertions.sort(key=lambda x: x.get("index", 0), reverse=True)
                
                for insertion in insertions:
                    step_index = insertion.get("index", len(procedure["steps"]))
                    new_step = insertion.get("step")
                    
                    if new_step is not None and 0 <= step_index <= len(procedure["steps"]):
                        procedure["steps"].insert(step_index, new_step)
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return True
    
    def delete_procedure(self, procedure_id: str) -> bool:
        """
        Delete a procedure.
        
        Args:
            procedure_id: ID of the procedure to delete
            
        Returns:
            True if successful, False if procedure not found
        """
        if procedure_id in self.procedures:
            del self.procedures[procedure_id]
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
                
            return True
        return False
    
    def clear(self) -> None:
        """Clear all procedures."""
        self.procedures = {}
        if self.storage_path:
            self._save_to_disk()
    
    def _update_execution_stats(self, procedure_id: str, success: bool, duration: float) -> None:
        """
        Update execution statistics for a procedure.
        
        Args:
            procedure_id: ID of the procedure
            success: Whether execution was successful
            duration: Execution duration in seconds
        """
        if procedure_id not in self.procedures:
            return
            
        stats = self.procedures[procedure_id]["execution_stats"]
        
        # Update success/failure counts
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
            
        # Update last execution time
        stats["last_execution"] = datetime.now().isoformat()
        
        # Update average duration
        total_executions = stats["success_count"] + stats["failure_count"]
        if total_executions == 1:
            stats["average_duration"] = duration
        else:
            stats["average_duration"] = (
                (stats["average_duration"] * (total_executions - 1) + duration) / 
                total_executions
            )
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
    
    def _resolve_function(self, func_name: str) -> Optional[Callable]:
        """
        Resolve a function reference to a callable.
        
        Args:
            func_name: Name of the function to resolve
            
        Returns:
            The callable function or None if not found
        """
        return self.function_registry.get(func_name)
    
    def _save_to_disk(self) -> None:
        """Save procedures to disk for persistence."""
        if not self.storage_path:
            return
            
        file_path = os.path.join(self.storage_path, self.agent_id, "procedural", "procedures.json")
        
        # Convert procedures to a serializable format
        serializable_procedures = {}
        for proc_id, procedure in self.procedures.items():
            # Deep copy to avoid modifying the original
            serializable_proc = json.loads(json.dumps(procedure))
            
            # Handle non-serializable steps (callables)
            for i, step in enumerate(procedure["steps"]):
                if callable(step):
                    # Replace callable with a reference
                    func_name = getattr(step, "__name__", f"anonymous_func_{i}")
                    serializable_proc["steps"][i] = {
                        "function": func_name,
                        "is_callable_reference": True
                    }
            
            serializable_procedures[proc_id] = serializable_proc
        
        try:
            with open(file_path, 'w') as f:
                json.dump(serializable_procedures, f)
        except Exception as e:
            logger.error(f"Error saving procedural memory: {e}")
    
    def _load_from_disk(self) -> None:
        """Load procedures from disk."""
        if not self.storage_path:
            return
            
        file_path = os.path.join(self.storage_path, self.agent_id, "procedural", "procedures.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    self.procedures = json.load(f)
                    
                # Note: Callable steps will need to be re-registered after loading
                # as they cannot be serialized
                
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading procedural memory: {e}")
                # Handle corrupted file
                self.procedures = {}
