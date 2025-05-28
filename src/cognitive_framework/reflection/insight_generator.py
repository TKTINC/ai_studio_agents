"""
Insight Generator Module for TAAT Cognitive Framework.

This module implements insight generation capabilities for extracting
meaningful insights from agent experiences and errors.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class InsightGenerator:
    """
    Insight Generator for TAAT Cognitive Framework.
    
    Generates insights from agent experiences, errors, and performance data
    to improve future decision-making and strategy selection.
    """
    
    def __init__(self):
        """Initialize the insight generator."""
        self.insights = []
        self.error_analyzer = None
        self.logger = logging.getLogger("InsightGenerator")
    
    def connect_error_analyzer(self, error_analyzer):
        """
        Connect to an error analyzer.
        
        Args:
            error_analyzer: Error analyzer instance
        """
        self.error_analyzer = error_analyzer
        self.logger.info("Connected to error analyzer")
    
    def generate_insights(self, data_source: str, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights from source data.
        
        Args:
            data_source: Source of the data
            data: Data to generate insights from
            context: Context in which the data was generated
            
        Returns:
            Generated insights
        """
        timestamp = datetime.now()
        
        # Generate insight based on data source
        if data_source == "error":
            insight = self._generate_error_insight(data, context)
        elif data_source == "experience":
            insight = self._generate_experience_insight(data, context)
        elif data_source == "performance":
            insight = self._generate_performance_insight(data, context)
        else:
            insight = self._generate_generic_insight(data_source, data, context)
        
        # Create result
        result = {
            "source": data_source,
            "timestamp": timestamp,
            "insights": [insight],
            "context": context
        }
        
        # Store insight
        self.insights.append(result)
        
        # Limit insights size
        max_insights = 100
        if len(self.insights) > max_insights:
            self.insights = self.insights[-max_insights:]
        
        self.logger.info(f"Generated insights from {data_source}")
        
        return result
    
    def generate_insight(self,
                        source_type: str,
                        source_data: Dict[str, Any],
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single insight from source data.
        
        Args:
            source_type: Type of source data (e.g., 'error', 'experience', 'performance')
            source_data: Source data to generate insight from
            context: Context in which the insight is generated
            
        Returns:
            Generated insight
        """
        timestamp = datetime.now()
        
        # Generate insight based on source type
        if source_type == "error":
            insight = self._generate_error_insight(source_data, context)
        elif source_type == "experience":
            insight = self._generate_experience_insight(source_data, context)
        elif source_type == "performance":
            insight = self._generate_performance_insight(source_data, context)
        else:
            insight = self._generate_generic_insight(source_type, source_data, context)
        
        # Add metadata
        insight.update({
            "timestamp": timestamp,
            "source_type": source_type,
            "context_summary": self._summarize_context(context)
        })
        
        # Store insight
        self.insights.append({
            "source": source_type,
            "timestamp": timestamp,
            "insights": [insight],
            "context": context
        })
        
        # Limit insights size
        max_insights = 100
        if len(self.insights) > max_insights:
            self.insights = self.insights[-max_insights:]
        
        self.logger.info(f"Generated {insight['type']} insight: {insight['summary']}")
        
        return insight
    
    def get_insights(self, 
                   insight_type: Optional[str] = None, 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get insights.
        
        Args:
            insight_type: Optional insight type to filter by
            limit: Maximum number of insights to return
            
        Returns:
            List of insights
        """
        if insight_type:
            filtered_insights = [
                insight for insight in self.insights
                if any(i.get("type") == insight_type for i in insight.get("insights", []))
            ]
            
            # Sort by timestamp
            filtered_insights.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return filtered_insights[:limit]
        else:
            # Sort by timestamp
            sorted_insights = sorted(
                self.insights,
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            return sorted_insights[:limit]
    
    def get_recent_insights(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent insights.
        
        Args:
            limit: Maximum number of insights to return
            
        Returns:
            List of recent insights
        """
        # Sort by timestamp
        sorted_insights = sorted(
            self.insights,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return sorted_insights[:limit]
    
    def get_insights_for_context(self, 
                               context: Dict[str, Any], 
                               limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get insights relevant to a context.
        
        Args:
            context: Context to find relevant insights for
            limit: Maximum number of insights to return
            
        Returns:
            List of relevant insights
        """
        # Calculate relevance scores for all insights
        relevance_scores = []
        
        for insight in self.insights:
            insight_context = insight.get("context", {})
            
            if not insight_context:
                continue
            
            # Calculate context similarity
            similarity = self._calculate_context_similarity(context, insight_context)
            
            relevance_scores.append((insight, similarity))
        
        # Sort by relevance score
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top insights
        return [insight for insight, _ in relevance_scores[:limit]]
    
    def _generate_error_insight(self, 
                              error_data: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insight from error data.
        
        Args:
            error_data: Error data
            context: Context in which the error occurred
            
        Returns:
            Generated insight
        """
        # Get error analysis if error analyzer is available
        error_analysis = None
        if self.error_analyzer:
            error_analysis = self.error_analyzer.analyze_error(error_data, context)
        
        # Extract error information
        error_type = error_data.get("type", "unknown")
        error_message = error_data.get("message", "")
        error_severity = error_data.get("severity", "medium")
        
        # Generate insight
        insight = {
            "type": "error_insight",
            "error_type": error_type,
            "severity": error_severity,
            "summary": f"Error of type {error_type} occurred: {error_message}",
            "analysis": error_analysis,
            "recommendations": self._generate_error_recommendations(error_data, error_analysis)
        }
        
        return insight
    
    def _generate_experience_insight(self, 
                                   experience_data: Dict[str, Any], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insight from experience data.
        
        Args:
            experience_data: Experience data
            context: Context in which the experience occurred
            
        Returns:
            Generated insight
        """
        # Extract experience information
        experience_type = experience_data.get("type", "unknown")
        experience_outcome = experience_data.get("outcome", {})
        experience_success = experience_outcome.get("success", False)
        
        # Generate insight
        if experience_success:
            insight_type = "success_pattern"
            summary = f"Successful {experience_type} experience identified"
        else:
            insight_type = "failure_pattern"
            summary = f"Unsuccessful {experience_type} experience analyzed"
        
        insight = {
            "type": insight_type,
            "experience_type": experience_type,
            "success": experience_success,
            "summary": summary,
            "patterns": self._extract_patterns(experience_data, context),
            "recommendations": self._generate_experience_recommendations(experience_data, experience_success)
        }
        
        return insight
    
    def _generate_performance_insight(self, 
                                    performance_data: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insight from performance data.
        
        Args:
            performance_data: Performance data
            context: Context in which the performance was measured
            
        Returns:
            Generated insight
        """
        # Extract performance information
        metrics = performance_data.get("metrics", {})
        
        # Identify significant metrics
        significant_metrics = {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                # Check if metric is significantly good or bad
                if metric_name in ["accuracy", "success_rate", "precision", "recall", "f1"]:
                    if metric_value > 0.8:
                        significant_metrics[metric_name] = {"value": metric_value, "assessment": "excellent"}
                    elif metric_value < 0.5:
                        significant_metrics[metric_name] = {"value": metric_value, "assessment": "poor"}
                elif metric_name in ["error_rate", "latency", "resource_usage"]:
                    if metric_value < 0.2:
                        significant_metrics[metric_name] = {"value": metric_value, "assessment": "excellent"}
                    elif metric_value > 0.5:
                        significant_metrics[metric_name] = {"value": metric_value, "assessment": "poor"}
        
        # Generate insight
        if significant_metrics:
            # Determine overall assessment
            excellent_count = sum(1 for m in significant_metrics.values() if m["assessment"] == "excellent")
            poor_count = sum(1 for m in significant_metrics.values() if m["assessment"] == "poor")
            
            if excellent_count > poor_count:
                insight_type = "performance_strength"
                summary = "Performance strengths identified in key metrics"
            elif poor_count > excellent_count:
                insight_type = "performance_weakness"
                summary = "Performance weaknesses identified in key metrics"
            else:
                insight_type = "mixed_performance"
                summary = "Mixed performance observed across metrics"
        else:
            insight_type = "normal_performance"
            summary = "Performance within normal parameters"
        
        insight = {
            "type": insight_type,
            "summary": summary,
            "significant_metrics": significant_metrics,
            "recommendations": self._generate_performance_recommendations(performance_data, significant_metrics)
        }
        
        return insight
    
    def _generate_generic_insight(self, 
                                source_type: str, 
                                source_data: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insight from generic source data.
        
        Args:
            source_type: Type of source data
            source_data: Source data
            context: Context
            
        Returns:
            Generated insight
        """
        # Generate generic insight
        insight = {
            "type": "generic_insight",
            "source_type": source_type,
            "summary": f"Insight generated from {source_type} data",
            "key_observations": self._extract_key_observations(source_data),
            "recommendations": []
        }
        
        return insight
    
    def _generate_error_recommendations(self, 
                                      error_data: Dict[str, Any], 
                                      error_analysis: Optional[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on error data.
        
        Args:
            error_data: Error data
            error_analysis: Error analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Extract error information
        error_type = error_data.get("type", "unknown")
        
        # Generate recommendations based on error type
        if error_type == "validation_error":
            recommendations.append("Implement stronger input validation")
            recommendations.append("Add pre-condition checks before operations")
        elif error_type == "timeout_error":
            recommendations.append("Implement timeout handling and retry logic")
            recommendations.append("Consider optimizing performance-critical operations")
        elif error_type == "resource_error":
            recommendations.append("Implement resource usage monitoring")
            recommendations.append("Consider resource-efficient alternatives")
        elif error_type == "logic_error":
            recommendations.append("Review and refine decision logic")
            recommendations.append("Add additional safeguards for edge cases")
        
        # Add recommendations from error analysis if available
        if error_analysis and "recommendations" in error_analysis:
            analysis_recommendations = error_analysis["recommendations"]
            if isinstance(analysis_recommendations, list):
                recommendations.extend(analysis_recommendations)
        
        return recommendations
    
    def _generate_experience_recommendations(self, 
                                           experience_data: Dict[str, Any], 
                                           success: bool) -> List[str]:
        """
        Generate recommendations based on experience data.
        
        Args:
            experience_data: Experience data
            success: Whether the experience was successful
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Extract experience information
        experience_type = experience_data.get("type", "unknown")
        
        # Generate recommendations based on experience type and success
        if success:
            recommendations.append(f"Reinforce successful {experience_type} patterns")
            recommendations.append("Consider expanding application to similar contexts")
        else:
            recommendations.append(f"Review and refine {experience_type} approach")
            recommendations.append("Consider alternative strategies for similar contexts")
        
        return recommendations
    
    def _generate_performance_recommendations(self, 
                                            performance_data: Dict[str, Any], 
                                            significant_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on performance data.
        
        Args:
            performance_data: Performance data
            significant_metrics: Significant metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on significant metrics
        for metric_name, metric_info in significant_metrics.items():
            assessment = metric_info["assessment"]
            value = metric_info["value"]
            
            if assessment == "poor":
                if metric_name in ["accuracy", "success_rate", "precision", "recall", "f1"]:
                    recommendations.append(f"Improve {metric_name} (currently {value:.2f})")
                elif metric_name in ["error_rate", "latency", "resource_usage"]:
                    recommendations.append(f"Reduce {metric_name} (currently {value:.2f})")
            elif assessment == "excellent":
                recommendations.append(f"Maintain excellent {metric_name} performance")
        
        return recommendations
    
    def _extract_patterns(self, 
                        experience_data: Dict[str, Any], 
                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract patterns from experience data.
        
        Args:
            experience_data: Experience data
            context: Context
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        # Extract experience information
        experience_type = experience_data.get("type", "unknown")
        experience_outcome = experience_data.get("outcome", {})
        experience_success = experience_outcome.get("success", False)
        
        # Identify context patterns
        context_pattern = {
            "type": "context_pattern",
            "description": f"Context pattern for {experience_type} experiences",
            "context_factors": self._extract_key_context_factors(context),
            "success_correlation": "positive" if experience_success else "negative"
        }
        
        patterns.append(context_pattern)
        
        # Identify sequence patterns if available
        if "sequence" in experience_data:
            sequence = experience_data["sequence"]
            
            sequence_pattern = {
                "type": "sequence_pattern",
                "description": f"Sequence pattern for {experience_type} experiences",
                "sequence_length": len(sequence),
                "key_steps": self._extract_key_steps(sequence),
                "success_correlation": "positive" if experience_success else "negative"
            }
            
            patterns.append(sequence_pattern)
        
        return patterns
    
    def _extract_key_context_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key factors from context.
        
        Args:
            context: Context
            
        Returns:
            Dictionary of key context factors
        """
        key_factors = {}
        
        # Extract key context elements
        if "type" in context:
            key_factors["type"] = context["type"]
        
        if "domain" in context:
            key_factors["domain"] = context["domain"]
        
        if "complexity" in context:
            key_factors["complexity"] = context["complexity"]
        
        if "urgency" in context:
            key_factors["urgency"] = context["urgency"]
        
        return key_factors
    
    def _extract_key_steps(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract key steps from a sequence.
        
        Args:
            sequence: Sequence of steps
            
        Returns:
            List of key steps
        """
        key_steps = []
        
        # Extract key steps (first, last, and any with significant outcomes)
        if sequence:
            # Add first step
            first_step = sequence[0]
            key_steps.append({
                "position": "first",
                "type": first_step.get("type", "unknown"),
                "outcome": first_step.get("outcome", {})
            })
            
            # Add last step
            last_step = sequence[-1]
            key_steps.append({
                "position": "last",
                "type": last_step.get("type", "unknown"),
                "outcome": last_step.get("outcome", {})
            })
            
            # Add significant intermediate steps
            for i, step in enumerate(sequence[1:-1], 1):
                outcome = step.get("outcome", {})
                
                # Check if step has significant outcome
                if outcome.get("significant", False) or outcome.get("success", False) is False:
                    key_steps.append({
                        "position": i,
                        "type": step.get("type", "unknown"),
                        "outcome": outcome
                    })
        
        return key_steps
    
    def _extract_key_observations(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract key observations from data.
        
        Args:
            data: Data to extract observations from
            
        Returns:
            List of key observations
        """
        observations = []
        
        # Extract observations based on data content
        for key, value in data.items():
            if isinstance(value, dict) and "significant" in value and value["significant"]:
                observations.append(f"Significant {key}: {value}")
            elif isinstance(value, (int, float)) and key in ["accuracy", "error_rate", "performance"]:
                observations.append(f"{key.capitalize()}: {value}")
        
        return observations
    
    def _calculate_context_similarity(self, 
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
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of context.
        
        Args:
            context: Context to summarize
            
        Returns:
            Context summary
        """
        summary = {}
        
        # Extract key context elements
        if "type" in context:
            summary["type"] = context["type"]
        
        if "domain" in context:
            summary["domain"] = context["domain"]
        
        if "complexity" in context:
            summary["complexity"] = context["complexity"]
        
        if "urgency" in context:
            summary["urgency"] = context["urgency"]
        
        # Summarize numerical data
        numerical_summary = {}
        for key, value in context.items():
            if isinstance(value, (int, float)) and key not in summary:
                numerical_summary[key] = value
        
        if numerical_summary:
            summary["numerical_data"] = numerical_summary
        
        return summary
