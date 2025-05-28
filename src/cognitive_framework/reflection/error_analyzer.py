"""
Error Analyzer Module for TAAT Cognitive Framework.

This module implements error analysis capabilities for diagnosing
and understanding errors that occur during agent operation.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid

class ErrorAnalyzer:
    """
    Error Analyzer for TAAT Cognitive Framework.
    
    Analyzes errors that occur during agent operation, providing insights
    into their causes and potential solutions.
    """
    
    def __init__(self):
        """Initialize the error analyzer."""
        self.error_history = []
        self.error_patterns = {}
        self.logger = logging.getLogger("ErrorAnalyzer")
    
    def analyze_error(self, 
                     error: Any, 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error and its context.
        
        Args:
            error: Error object or message
            context: Context in which the error occurred
            
        Returns:
            Error analysis result
        """
        timestamp = datetime.now()
        
        # Convert error to string if it's not already
        error_message = str(error)
        
        # Create error data structure
        error_data = {
            "error": error_message,
            "context": context,
            "timestamp": timestamp,
            "error_id": f"err_{timestamp.timestamp()}_" + str(uuid.uuid4())[:8]
        }
        
        # Perform analysis
        analysis = self._analyze_error_message(error_message, context)
        error_data["analysis"] = analysis
        
        # Store in error history
        self.error_history.append(error_data)
        
        # Update error patterns
        self._update_error_patterns(error_message, analysis)
        
        self.logger.info(f"Analyzed error: {error_message[:50]}...")
        
        return error_data
    
    def get_error_history(self, 
                         limit: int = 10, 
                         error_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get error history.
        
        Args:
            limit: Maximum number of errors to return
            error_type: Optional error type to filter by
            
        Returns:
            List of error records
        """
        if error_type:
            filtered_history = [
                error for error in self.error_history
                if error_type.lower() in error["analysis"].get("error_type", "").lower()
            ]
            
            # Sort by timestamp
            filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return filtered_history[:limit]
        else:
            # Sort by timestamp
            sorted_history = sorted(
                self.error_history,
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            return sorted_history[:limit]
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """
        Get identified error patterns.
        
        Returns:
            Dictionary of error patterns and their frequencies
        """
        return {
            "patterns": self.error_patterns,
            "total_errors": len(self.error_history),
            "unique_patterns": len(self.error_patterns)
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Error statistics
        """
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count errors by type
        error_types = {}
        for error in self.error_history:
            error_type = error["analysis"].get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Count errors by severity
        error_severities = {}
        for error in self.error_history:
            severity = error["analysis"].get("severity", "unknown")
            error_severities[severity] = error_severities.get(severity, 0) + 1
        
        # Calculate time distribution
        now = datetime.now()
        time_distribution = {
            "last_hour": 0,
            "last_day": 0,
            "last_week": 0,
            "older": 0
        }
        
        for error in self.error_history:
            time_diff = (now - error["timestamp"]).total_seconds()
            
            if time_diff <= 3600:  # 1 hour
                time_distribution["last_hour"] += 1
            elif time_diff <= 86400:  # 1 day
                time_distribution["last_day"] += 1
            elif time_diff <= 604800:  # 1 week
                time_distribution["last_week"] += 1
            else:
                time_distribution["older"] += 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "error_severities": error_severities,
            "time_distribution": time_distribution,
            "most_recent": self.error_history[-1]["timestamp"] if self.error_history else None,
            "most_common_type": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history = []
        self.logger.info("Cleared error history")
    
    def _analyze_error_message(self, 
                             error_message: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error message.
        
        Args:
            error_message: Error message to analyze
            context: Context in which the error occurred
            
        Returns:
            Analysis result
        """
        # Determine error type
        error_type = "unknown"
        if "not found" in error_message.lower() or "missing" in error_message.lower():
            error_type = "not_found"
        elif "permission" in error_message.lower() or "access" in error_message.lower():
            error_type = "permission"
        elif "timeout" in error_message.lower():
            error_type = "timeout"
        elif "syntax" in error_message.lower():
            error_type = "syntax"
        elif "value" in error_message.lower():
            error_type = "value"
        elif "type" in error_message.lower():
            error_type = "type"
        elif "memory" in error_message.lower():
            error_type = "memory"
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            error_type = "network"
        
        # Determine severity
        severity = "medium"
        if error_type in ["memory", "network"]:
            severity = "high"
        elif error_type in ["syntax", "type", "value"]:
            severity = "low"
        
        # Generate potential causes
        potential_causes = []
        
        if error_type == "not_found":
            potential_causes.append("Resource does not exist")
            potential_causes.append("Incorrect path or identifier")
        elif error_type == "permission":
            potential_causes.append("Insufficient permissions")
            potential_causes.append("Authentication failure")
        elif error_type == "timeout":
            potential_causes.append("Operation took too long")
            potential_causes.append("Resource unavailable")
        elif error_type == "syntax":
            potential_causes.append("Invalid syntax in code or query")
        elif error_type == "value":
            potential_causes.append("Invalid value provided")
        elif error_type == "type":
            potential_causes.append("Type mismatch in operation")
        elif error_type == "memory":
            potential_causes.append("Insufficient memory")
            potential_causes.append("Memory leak")
        elif error_type == "network":
            potential_causes.append("Network connectivity issues")
            potential_causes.append("Service unavailable")
        
        # Generate potential solutions
        potential_solutions = []
        
        if error_type == "not_found":
            potential_solutions.append("Verify resource exists")
            potential_solutions.append("Check path or identifier")
        elif error_type == "permission":
            potential_solutions.append("Check permissions")
            potential_solutions.append("Verify authentication")
        elif error_type == "timeout":
            potential_solutions.append("Increase timeout value")
            potential_solutions.append("Optimize operation")
        elif error_type == "syntax":
            potential_solutions.append("Fix syntax errors")
        elif error_type == "value":
            potential_solutions.append("Provide valid value")
        elif error_type == "type":
            potential_solutions.append("Ensure type compatibility")
        elif error_type == "memory":
            potential_solutions.append("Optimize memory usage")
            potential_solutions.append("Increase memory allocation")
        elif error_type == "network":
            potential_solutions.append("Check network connectivity")
            potential_solutions.append("Verify service status")
        
        # Check for context-specific insights
        context_insights = []
        
        for key, value in context.items():
            if key == "operation" and value == "memory_retrieval" and error_type == "not_found":
                context_insights.append("Memory retrieval failed, possibly due to incorrect memory ID")
            elif key == "operation" and value == "strategy_execution" and error_type == "timeout":
                context_insights.append("Strategy execution timed out, consider optimizing or using a simpler strategy")
        
        return {
            "error_type": error_type,
            "severity": severity,
            "potential_causes": potential_causes,
            "potential_solutions": potential_solutions,
            "context_insights": context_insights
        }
    
    def _update_error_patterns(self, 
                             error_message: str, 
                             analysis: Dict[str, Any]) -> None:
        """
        Update error patterns based on a new error.
        
        Args:
            error_message: Error message
            analysis: Error analysis
        """
        error_type = analysis.get("error_type", "unknown")
        
        # Extract key parts of the error message
        words = error_message.lower().split()
        key_parts = [word for word in words if len(word) > 3 and word.isalpha()]
        
        if not key_parts:
            return
        
        # Create pattern key
        pattern_key = f"{error_type}:{':'.join(key_parts[:3])}"
        
        # Update pattern count
        if pattern_key in self.error_patterns:
            self.error_patterns[pattern_key]["count"] += 1
            self.error_patterns[pattern_key]["last_seen"] = datetime.now()
        else:
            self.error_patterns[pattern_key] = {
                "error_type": error_type,
                "pattern": " ".join(key_parts[:5]),
                "count": 1,
                "first_seen": datetime.now(),
                "last_seen": datetime.now()
            }
