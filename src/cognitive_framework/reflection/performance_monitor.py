"""
Performance Monitor Module for TAAT Cognitive Framework.

This module implements performance monitoring capabilities for tracking
and analyzing the performance of various operations and strategies.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class PerformanceMonitor:
    """
    Performance Monitor for TAAT Cognitive Framework.
    
    Tracks and analyzes the performance of operations and strategies,
    providing metrics and insights for optimization.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.operations = {}
        self.metrics_history = []
        self.logger = logging.getLogger("PerformanceMonitor")
    
    def start_tracking(self, 
                      operation_id: str, 
                      operation_type: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start tracking an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation being tracked
            metadata: Additional metadata about the operation
        """
        self.operations[operation_id] = {
            "start_time": datetime.now(),
            "operation_type": operation_type,
            "metadata": metadata or {},
            "status": "in_progress"
        }
        self.logger.info(f"Started tracking operation {operation_id} of type {operation_type}")
    
    def end_tracking(self, 
                    operation_id: str, 
                    status: str, 
                    result: Optional[Dict[str, Any]] = None,
                    error_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        End tracking an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            status: Final status of the operation (success, failure, etc.)
            result: Result data from the operation
            error_details: Optional details about any errors that occurred
            
        Returns:
            Performance metrics for the operation
        """
        if operation_id not in self.operations:
            self.logger.warning(f"Operation {operation_id} not found in tracking")
            return {"error": "Operation not found"}
        
        end_time = datetime.now()
        start_time = self.operations[operation_id]["start_time"]
        duration = (end_time - start_time).total_seconds()
        
        operation_update = {
            "end_time": end_time,
            "duration": duration,
            "status": status,
            "result": result
        }
        
        # Add error details if provided
        if error_details:
            operation_update["error_details"] = error_details
        
        self.operations[operation_id].update(operation_update)
        
        metrics = self._calculate_metrics(operation_id)
        self.metrics_history.append({
            "timestamp": end_time,
            "operation_id": operation_id,
            "metrics": metrics
        })
        
        self.logger.info(f"Ended tracking operation {operation_id} with status {status}")
        
        return metrics
    
    def get_metrics(self, operation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            operation_id: Optional operation ID to get metrics for
            
        Returns:
            Performance metrics
        """
        if operation_id:
            if operation_id not in self.operations:
                return {"error": "Operation not found"}
            
            return {
                "operation": self.operations[operation_id],
                "metrics": self._calculate_metrics(operation_id)
            }
        else:
            # Return overall metrics
            return {
                "operations": self.operations,
                "overall_metrics": self._calculate_overall_metrics(),
                "history": self.metrics_history[-10:]  # Last 10 entries
            }
    
    def get_operation_history(self, 
                            operation_type: Optional[str] = None, 
                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get operation history.
        
        Args:
            operation_type: Optional type of operations to filter by
            limit: Maximum number of operations to return
            
        Returns:
            List of operations
        """
        history = []
        
        for op_id, op_data in self.operations.items():
            if operation_type and op_data["operation_type"] != operation_type:
                continue
            
            history.append({
                "operation_id": op_id,
                **op_data
            })
        
        # Sort by end time if available, otherwise start time
        history.sort(key=lambda x: x.get("end_time", x["start_time"]), reverse=True)
        
        return history[:limit]
    
    def clear_history(self, 
                     operation_type: Optional[str] = None, 
                     before_timestamp: Optional[datetime] = None) -> None:
        """
        Clear operation history.
        
        Args:
            operation_type: Optional type of operations to clear
            before_timestamp: Optional timestamp to clear operations before
        """
        if operation_type is None and before_timestamp is None:
            # Clear all history
            self.operations = {}
            self.metrics_history = []
            self.logger.info("Cleared all operation history")
            return
        
        # Filter operations to keep
        operations_to_keep = {}
        
        for op_id, op_data in self.operations.items():
            keep = True
            
            if operation_type and op_data["operation_type"] == operation_type:
                keep = False
            
            if before_timestamp and op_data["start_time"] < before_timestamp:
                keep = False
            
            if keep:
                operations_to_keep[op_id] = op_data
        
        # Filter metrics history to keep
        metrics_to_keep = []
        
        for metric in self.metrics_history:
            keep = True
            
            if operation_type and self.operations.get(metric["operation_id"], {}).get("operation_type") == operation_type:
                keep = False
            
            if before_timestamp and metric["timestamp"] < before_timestamp:
                keep = False
            
            if keep:
                metrics_to_keep.append(metric)
        
        self.operations = operations_to_keep
        self.metrics_history = metrics_to_keep
        
        self.logger.info(f"Cleared operation history with filters: operation_type={operation_type}, before_timestamp={before_timestamp}")
    
    def _calculate_metrics(self, operation_id: str) -> Dict[str, Any]:
        """
        Calculate metrics for an operation.
        
        Args:
            operation_id: Operation ID to calculate metrics for
            
        Returns:
            Metrics for the operation
        """
        operation = self.operations[operation_id]
        
        metrics = {
            "duration": operation.get("duration", 0),
            "status": operation.get("status", "unknown"),
            "operation_type": operation["operation_type"]
        }
        
        # Add type-specific metrics
        if operation["operation_type"] == "strategy_execution":
            metrics.update(self._calculate_strategy_metrics(operation))
        elif operation["operation_type"] == "memory_retrieval":
            metrics.update(self._calculate_memory_metrics(operation))
        
        return metrics
    
    def _calculate_strategy_metrics(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics for a strategy execution operation.
        
        Args:
            operation: Operation data
            
        Returns:
            Strategy-specific metrics
        """
        result = operation.get("result", {})
        
        return {
            "success": result.get("success", False),
            "confidence": result.get("confidence", 0.0),
            "execution_time": operation.get("duration", 0)
        }
    
    def _calculate_memory_metrics(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics for a memory retrieval operation.
        
        Args:
            operation: Operation data
            
        Returns:
            Memory-specific metrics
        """
        result = operation.get("result", {})
        
        return {
            "retrieval_count": len(result.get("memories", [])),
            "retrieval_time": operation.get("duration", 0),
            "relevance_score": result.get("relevance_score", 0.0)
        }
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """
        Calculate overall performance metrics.
        
        Returns:
            Overall metrics
        """
        if not self.operations:
            return {"operation_count": 0}
        
        # Count operations by type and status
        operation_counts = {}
        status_counts = {}
        total_duration = 0
        
        for op_data in self.operations.values():
            op_type = op_data["operation_type"]
            status = op_data.get("status", "unknown")
            
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if "duration" in op_data:
                total_duration += op_data["duration"]
        
        # Calculate success rate
        success_count = status_counts.get("success", 0)
        total_count = len(self.operations)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        # Calculate average duration
        avg_duration = total_duration / total_count if total_count > 0 else 0
        
        return {
            "operation_count": total_count,
            "operation_counts_by_type": operation_counts,
            "status_counts": status_counts,
            "success_rate": success_rate,
            "average_duration": avg_duration
        }
