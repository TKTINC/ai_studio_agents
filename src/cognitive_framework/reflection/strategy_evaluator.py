"""
Strategy Evaluator Module for TAAT Cognitive Framework.

This module implements strategy evaluation capabilities for assessing
the performance and effectiveness of cognitive strategies.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class StrategyEvaluator:
    """
    Strategy Evaluator for TAAT Cognitive Framework.
    
    Evaluates the performance and effectiveness of cognitive strategies
    to inform strategy selection and adaptation.
    """
    
    def __init__(self):
        """Initialize the strategy evaluator."""
        self.evaluations = {}
        self.performance_monitor = None
        self.logger = logging.getLogger("StrategyEvaluator")
    
    def connect_performance_monitor(self, performance_monitor):
        """
        Connect to a performance monitor.
        
        Args:
            performance_monitor: Performance monitor instance
        """
        self.performance_monitor = performance_monitor
        self.logger.info("Connected to performance monitor")
    
    def evaluate_strategy(self, strategy_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a strategy based on performance data.
        
        Args:
            strategy_id: ID of the strategy to evaluate
            performance_data: Performance data for evaluation
            
        Returns:
            Evaluation results
        """
        timestamp = datetime.now()
        
        # Extract performance metrics
        success = performance_data.get("success", False)
        metrics = performance_data.get("metrics", {})
        context = performance_data.get("context", {})
        
        # Calculate evaluation scores
        effectiveness_score = self._calculate_effectiveness_score(success, metrics)
        efficiency_score = self._calculate_efficiency_score(metrics)
        context_fit_score = self._calculate_context_fit_score(context, metrics)
        
        # Calculate overall score
        overall_score = 0.5 * effectiveness_score + 0.3 * efficiency_score + 0.2 * context_fit_score
        
        # Create evaluation result
        evaluation = {
            "strategy_id": strategy_id,
            "timestamp": timestamp,
            "success": success,
            "scores": {
                "effectiveness": effectiveness_score,
                "efficiency": efficiency_score,
                "context_fit": context_fit_score,
                "overall": overall_score
            },
            "metrics": metrics,
            "context": context
        }
        
        # Store evaluation
        if strategy_id not in self.evaluations:
            self.evaluations[strategy_id] = []
        
        self.evaluations[strategy_id].append(evaluation)
        
        # Limit evaluations size
        max_evaluations = 10
        if len(self.evaluations[strategy_id]) > max_evaluations:
            self.evaluations[strategy_id] = self.evaluations[strategy_id][-max_evaluations:]
        
        self.logger.info(f"Evaluated strategy {strategy_id} with overall score {overall_score:.2f}")
        
        return evaluation
    
    def get_evaluation_history(self, strategy_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get evaluation history for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            limit: Maximum number of evaluations to return
            
        Returns:
            List of evaluations
        """
        if strategy_id not in self.evaluations:
            return []
        
        # Sort by timestamp
        sorted_evaluations = sorted(
            self.evaluations[strategy_id],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return sorted_evaluations[:limit]
    
    def get_strategy_evaluations(self, strategy_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get evaluations for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            limit: Maximum number of evaluations to return
            
        Returns:
            List of evaluations
        """
        return self.get_evaluation_history(strategy_id, limit)
    
    def get_strategy_performance_trend(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get performance trend for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Performance trend data
        """
        if strategy_id not in self.evaluations or not self.evaluations[strategy_id]:
            return {
                "strategy_id": strategy_id,
                "trend": "unknown",
                "data_points": 0
            }
        
        evaluations = self.evaluations[strategy_id]
        
        # Sort by timestamp
        sorted_evaluations = sorted(
            evaluations,
            key=lambda x: x["timestamp"]
        )
        
        # Extract overall scores
        scores = [evaluation["scores"]["overall"] for evaluation in sorted_evaluations]
        
        # Calculate trend
        if len(scores) < 2:
            trend = "stable"
        else:
            # Calculate average change
            changes = [scores[i] - scores[i-1] for i in range(1, len(scores))]
            avg_change = sum(changes) / len(changes)
            
            if avg_change > 0.05:
                trend = "improving"
            elif avg_change < -0.05:
                trend = "declining"
            else:
                trend = "stable"
        
        # Calculate statistics
        avg_score = sum(scores) / len(scores)
        latest_score = scores[-1]
        
        return {
            "strategy_id": strategy_id,
            "trend": trend,
            "data_points": len(scores),
            "average_score": avg_score,
            "latest_score": latest_score,
            "score_history": scores
        }
    
    def compare_strategies(self, strategy_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            "timestamp": datetime.now(),
            "strategies": {},
            "ranking": []
        }
        
        # Collect data for each strategy
        for strategy_id in strategy_ids:
            if strategy_id not in self.evaluations or not self.evaluations[strategy_id]:
                comparison["strategies"][strategy_id] = {
                    "status": "no_data",
                    "average_score": 0.0,
                    "success_rate": 0.0,
                    "data_points": 0
                }
                continue
            
            evaluations = self.evaluations[strategy_id]
            
            # Calculate average score
            avg_score = sum(e["scores"]["overall"] for e in evaluations) / len(evaluations)
            
            # Calculate success rate
            success_count = sum(1 for e in evaluations if e["success"])
            success_rate = success_count / len(evaluations)
            
            # Store strategy data
            comparison["strategies"][strategy_id] = {
                "status": "evaluated",
                "average_score": avg_score,
                "success_rate": success_rate,
                "data_points": len(evaluations)
            }
        
        # Rank strategies
        ranked_strategies = sorted(
            [
                (strategy_id, data["average_score"])
                for strategy_id, data in comparison["strategies"].items()
                if data["status"] == "evaluated"
            ],
            key=lambda x: x[1],
            reverse=True
        )
        
        comparison["ranking"] = [
            {
                "rank": i + 1,
                "strategy_id": strategy_id,
                "score": score
            }
            for i, (strategy_id, score) in enumerate(ranked_strategies)
        ]
        
        return comparison
    
    def get_best_strategy_for_context(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get the best strategy for a given context.
        
        Args:
            context: Context to find the best strategy for
            
        Returns:
            Best strategy or None if no strategies available
        """
        if not self.evaluations:
            return None
        
        # Calculate context similarity scores for each strategy
        similarity_scores = []
        
        for strategy_id, evaluations in self.evaluations.items():
            if not evaluations:
                continue
            
            # Get most recent evaluation
            latest_evaluation = max(evaluations, key=lambda x: x["timestamp"])
            
            # Calculate context similarity
            similarity = self._calculate_context_similarity(context, latest_evaluation["context"])
            
            # Get overall score
            overall_score = latest_evaluation["scores"]["overall"]
            
            # Calculate combined score
            combined_score = 0.7 * similarity + 0.3 * overall_score
            
            similarity_scores.append((strategy_id, combined_score, similarity, overall_score))
        
        if not similarity_scores:
            return None
        
        # Sort by combined score
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get best strategy
        best_strategy_id, combined_score, similarity, overall_score = similarity_scores[0]
        
        return {
            "strategy_id": best_strategy_id,
            "combined_score": combined_score,
            "context_similarity": similarity,
            "performance_score": overall_score
        }
    
    def _calculate_effectiveness_score(self, success: bool, metrics: Dict[str, Any]) -> float:
        """
        Calculate effectiveness score.
        
        Args:
            success: Whether the strategy execution was successful
            metrics: Performance metrics
            
        Returns:
            Effectiveness score (0.0 to 1.0)
        """
        # Base score on success
        base_score = 0.5 if success else 0.0
        
        # Add score based on metrics
        metric_score = 0.0
        
        if "accuracy" in metrics:
            metric_score += 0.2 * metrics["accuracy"]
        
        if "precision" in metrics:
            metric_score += 0.1 * metrics["precision"]
        
        if "recall" in metrics:
            metric_score += 0.1 * metrics["recall"]
        
        if "f1" in metrics:
            metric_score += 0.1 * metrics["f1"]
        
        # Calculate final score
        return base_score + metric_score
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate efficiency score.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Efficiency score (0.0 to 1.0)
        """
        efficiency_score = 0.5  # Default score
        
        if "speed" in metrics:
            efficiency_score += 0.2 * metrics["speed"]
        
        if "resource_usage" in metrics:
            # Lower resource usage is better
            resource_score = 1.0 - metrics["resource_usage"]
            efficiency_score += 0.2 * resource_score
        
        if "latency" in metrics:
            # Lower latency is better
            latency_score = 1.0 - metrics["latency"]
            efficiency_score += 0.1 * latency_score
        
        # Ensure score is within range
        return max(0.0, min(1.0, efficiency_score))
    
    def _calculate_context_fit_score(self, context: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """
        Calculate context fit score.
        
        Args:
            context: Context in which the strategy was executed
            metrics: Performance metrics
            
        Returns:
            Context fit score (0.0 to 1.0)
        """
        # Default score
        fit_score = 0.5
        
        # Check if context has complexity
        if "complexity" in context:
            complexity = context["complexity"]
            
            # Check if metrics has complexity_handling
            if "complexity_handling" in metrics:
                complexity_handling = metrics["complexity_handling"]
                
                # Calculate fit based on complexity and handling
                if complexity <= 0.3 and complexity_handling >= 0.7:
                    fit_score += 0.1
                elif complexity >= 0.7 and complexity_handling >= 0.7:
                    fit_score += 0.2
                elif complexity >= 0.7 and complexity_handling <= 0.3:
                    fit_score -= 0.2
            else:
                # Penalize for high complexity without handling metric
                if complexity >= 0.7:
                    fit_score -= 0.1
        
        # Check if context has urgency
        if "urgency" in context:
            urgency = context["urgency"]
            
            # Check if metrics has speed
            if "speed" in metrics:
                speed = metrics["speed"]
                
                # Calculate fit based on urgency and speed
                if urgency >= 0.7 and speed >= 0.7:
                    fit_score += 0.2
                elif urgency >= 0.7 and speed <= 0.3:
                    fit_score -= 0.2
        
        # Ensure score is within range
        return max(0.0, min(1.0, fit_score))
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
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
