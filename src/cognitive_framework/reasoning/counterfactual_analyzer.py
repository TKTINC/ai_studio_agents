"""
Counterfactual Analyzer for AI Studio Agents.

This module implements advanced reasoning capabilities for counterfactual
analysis, what-if scenarios, and alternative outcome evaluation.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import numpy as np
import time
import json
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


class CounterfactualAnalyzer:
    """
    Performs counterfactual analysis and what-if scenario evaluation.
    
    Key Features:
    - Counterfactual generation
    - Alternative scenario evaluation
    - Decision path analysis
    - Causal inference
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the counterfactual analyzer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.max_counterfactuals = self.config.get("max_counterfactuals", 5)
        self.min_impact_threshold = self.config.get("min_impact_threshold", 0.1)
        
        logger.info("CounterfactualAnalyzer initialized")

    def generate_counterfactuals(self, 
                               scenario: Dict[str, Any], 
                               factors: List[Dict[str, Any]], 
                               outcome_function: Callable) -> Dict[str, Any]:
        """
        Generate counterfactual scenarios by varying input factors.
        
        Args:
            scenario: Base scenario details
            factors: List of factors that can be varied
            outcome_function: Function to evaluate outcomes for each counterfactual
            
        Returns:
            Dictionary containing generated counterfactuals.
        """
        logger.info("Generating counterfactuals for scenario")
        
        if not factors:
            return {
                "error": "No factors provided for counterfactual generation",
                "success": False
            }
        
        # Evaluate base scenario
        try:
            base_outcome = outcome_function(scenario)
        except Exception as e:
            return {
                "error": f"Error evaluating base scenario: {str(e)}",
                "success": False
            }
        
        # Generate counterfactuals by varying each factor
        counterfactuals = []
        
        for factor in factors:
            factor_name = factor.get("name")
            current_value = factor.get("current_value")
            
            if factor_name is None or current_value is None:
                continue
            
            # Get alternative values
            alternative_values = factor.get("alternative_values", [])
            
            # If no alternatives provided, generate some
            if not alternative_values and "min_value" in factor and "max_value" in factor:
                min_val = factor["min_value"]
                max_val = factor["max_value"]
                
                # Generate 3 alternative values
                step = (max_val - min_val) / 4
                alternative_values = [min_val, min_val + step, min_val + 2*step, max_val]
                
                # Remove current value if present
                if current_value in alternative_values:
                    alternative_values.remove(current_value)
            
            # Generate counterfactual for each alternative value
            for alt_value in alternative_values:
                # Create counterfactual scenario
                cf_scenario = self._create_counterfactual_scenario(scenario, factor_name, alt_value)
                
                # Evaluate counterfactual
                try:
                    cf_outcome = outcome_function(cf_scenario)
                    
                    # Calculate impact
                    impact = self._calculate_impact(base_outcome, cf_outcome)
                    
                    if abs(impact) >= self.min_impact_threshold:
                        counterfactuals.append({
                            "factor_changed": factor_name,
                            "original_value": current_value,
                            "counterfactual_value": alt_value,
                            "base_outcome": base_outcome,
                            "counterfactual_outcome": cf_outcome,
                            "impact": impact,
                            "scenario": cf_scenario
                        })
                except Exception as e:
                    logger.error(f"Error evaluating counterfactual for {factor_name}={alt_value}: {str(e)}")
        
        # Sort counterfactuals by impact
        counterfactuals.sort(key=lambda x: abs(x["impact"]), reverse=True)
        
        # Limit number of counterfactuals
        if len(counterfactuals) > self.max_counterfactuals:
            counterfactuals = counterfactuals[:self.max_counterfactuals]
        
        return {
            "base_scenario": scenario,
            "base_outcome": base_outcome,
            "counterfactuals": counterfactuals,
            "counterfactual_count": len(counterfactuals),
            "success": True
        }

    def analyze_decision_paths(self, 
                              decision_tree: Dict[str, Any], 
                              actual_path: List[str], 
                              outcome: Any) -> Dict[str, Any]:
        """
        Analyze alternative decision paths and their potential outcomes.
        
        Args:
            decision_tree: Tree structure of decision points and outcomes
            actual_path: List of decision points taken
            outcome: Actual outcome achieved
            
        Returns:
            Dictionary containing decision path analysis.
        """
        logger.info("Analyzing decision paths")
        
        if not decision_tree:
            return {
                "error": "No decision tree provided",
                "success": False
            }
        
        if not actual_path:
            return {
                "error": "No actual path provided",
                "success": False
            }
        
        # Validate actual path
        valid_path = self._validate_decision_path(decision_tree, actual_path)
        if not valid_path:
            return {
                "error": "Invalid decision path",
                "success": False
            }
        
        # Find all alternative paths
        all_paths = self._find_all_paths(decision_tree)
        
        # Remove actual path from alternatives
        alternative_paths = [path for path in all_paths if path["path"] != actual_path]
        
        # Calculate path metrics
        path_metrics = self._calculate_path_metrics(decision_tree, actual_path, alternative_paths)
        
        # Find optimal path
        optimal_path = max(all_paths, key=lambda x: x["expected_value"])
        
        # Calculate regret (difference between optimal and actual)
        regret = optimal_path["expected_value"] - path_metrics["actual_expected_value"]
        
        # Find critical decision points
        critical_points = self._identify_critical_points(decision_tree, actual_path, alternative_paths)
        
        return {
            "actual_path": actual_path,
            "actual_outcome": outcome,
            "actual_expected_value": path_metrics["actual_expected_value"],
            "optimal_path": optimal_path["path"],
            "optimal_expected_value": optimal_path["expected_value"],
            "regret": regret,
            "alternative_paths": alternative_paths,
            "critical_decision_points": critical_points,
            "success": True
        }

    def evaluate_causal_factors(self, 
                              outcome: Any, 
                              factors: List[Dict[str, Any]], 
                              historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate causal factors contributing to an outcome.
        
        Args:
            outcome: Outcome to analyze
            factors: List of potential causal factors
            historical_data: Historical data for causal inference
            
        Returns:
            Dictionary containing causal factor evaluation.
        """
        logger.info("Evaluating causal factors")
        
        if not factors:
            return {
                "error": "No factors provided for causal evaluation",
                "success": False
            }
        
        if not historical_data:
            return {
                "error": "No historical data provided for causal inference",
                "success": False
            }
        
        # Extract factor values from historical data
        factor_data = {}
        outcome_data = []
        
        for entry in historical_data:
            # Extract outcome
            entry_outcome = entry.get("outcome")
            if entry_outcome is None:
                continue
                
            outcome_data.append(entry_outcome)
            
            # Extract factor values
            for factor in factors:
                factor_name = factor.get("name")
                if factor_name is None:
                    continue
                    
                factor_value = entry.get(factor_name)
                if factor_value is None:
                    continue
                    
                if factor_name not in factor_data:
                    factor_data[factor_name] = []
                    
                factor_data[factor_name].append(factor_value)
        
        # Calculate correlations
        correlations = {}
        for factor_name, values in factor_data.items():
            if len(values) != len(outcome_data):
                continue
                
            correlation = self._calculate_correlation(values, outcome_data)
            correlations[factor_name] = correlation
        
        # Calculate causal strength using simple heuristics
        causal_strengths = {}
        for factor_name, correlation in correlations.items():
            # Simple heuristic: higher correlation suggests stronger causality
            # In a real system, use more sophisticated causal inference methods
            causal_strength = abs(correlation)
            causal_strengths[factor_name] = causal_strength
        
        # Sort factors by causal strength
        sorted_factors = sorted(causal_strengths.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare factor analysis
        factor_analysis = []
        for factor_name, strength in sorted_factors:
            direction = "positive" if correlations[factor_name] > 0 else "negative"
            
            factor_analysis.append({
                "factor": factor_name,
                "causal_strength": strength,
                "correlation": correlations[factor_name],
                "direction": direction,
                "significance": strength > 0.3  # Simple threshold for significance
            })
        
        return {
            "outcome": outcome,
            "factor_count": len(factors),
            "historical_data_points": len(historical_data),
            "factor_analysis": factor_analysis,
            "success": True
        }

    def compare_alternative_strategies(self, 
                                     strategies: List[Dict[str, Any]], 
                                     scenarios: List[Dict[str, Any]], 
                                     evaluation_function: Callable) -> Dict[str, Any]:
        """
        Compare alternative strategies across multiple scenarios.
        
        Args:
            strategies: List of strategies to compare
            scenarios: List of scenarios to evaluate against
            evaluation_function: Function to evaluate strategy in each scenario
            
        Returns:
            Dictionary containing strategy comparison.
        """
        logger.info("Comparing %d strategies across %d scenarios", len(strategies), len(scenarios))
        
        if not strategies:
            return {
                "error": "No strategies provided for comparison",
                "success": False
            }
        
        if not scenarios:
            return {
                "error": "No scenarios provided for evaluation",
                "success": False
            }
        
        # Evaluate each strategy in each scenario
        strategy_results = []
        
        for strategy in strategies:
            strategy_name = strategy.get("name", "Unnamed Strategy")
            
            scenario_evaluations = []
            total_score = 0
            success_count = 0
            
            for scenario in scenarios:
                scenario_name = scenario.get("name", "Unnamed Scenario")
                probability = scenario.get("probability", 1.0 / len(scenarios))
                
                try:
                    evaluation = evaluation_function(strategy, scenario)
                    success = evaluation.get("success", False)
                    score = evaluation.get("score", 0)
                    
                    if success:
                        success_count += 1
                        
                    weighted_score = probability * score
                    total_score += weighted_score
                    
                    scenario_evaluations.append({
                        "scenario": scenario_name,
                        "probability": probability,
                        "success": success,
                        "score": score,
                        "weighted_score": weighted_score,
                        "details": evaluation
                    })
                except Exception as e:
                    logger.error(f"Error evaluating strategy {strategy_name} in scenario {scenario_name}: {str(e)}")
                    scenario_evaluations.append({
                        "scenario": scenario_name,
                        "probability": probability,
                        "success": False,
                        "score": 0,
                        "weighted_score": 0,
                        "error": str(e)
                    })
            
            # Calculate strategy metrics
            success_rate = success_count / len(scenarios)
            
            # Calculate score variance
            scores = [eval["score"] for eval in scenario_evaluations]
            mean_score = sum(scores) / len(scores)
            score_variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            score_std_dev = score_variance ** 0.5
            
            strategy_results.append({
                "strategy": strategy_name,
                "expected_value": total_score,
                "success_rate": success_rate,
                "score_variance": score_variance,
                "score_std_dev": score_std_dev,
                "coefficient_of_variation": score_std_dev / mean_score if mean_score != 0 else float('inf'),
                "scenario_evaluations": scenario_evaluations
            })
        
        # Sort strategies by expected value
        strategy_results.sort(key=lambda x: x["expected_value"], reverse=True)
        
        # Identify optimal strategy
        optimal_strategy = strategy_results[0] if strategy_results else None
        
        # Calculate regret for each strategy
        if optimal_strategy:
            optimal_value = optimal_strategy["expected_value"]
            
            for result in strategy_results:
                result["regret"] = optimal_value - result["expected_value"]
        
        return {
            "strategy_count": len(strategies),
            "scenario_count": len(scenarios),
            "strategy_results": strategy_results,
            "optimal_strategy": optimal_strategy["strategy"] if optimal_strategy else None,
            "success": True
        }

    def _create_counterfactual_scenario(self, 
                                      base_scenario: Dict[str, Any], 
                                      factor_name: str, 
                                      new_value: Any) -> Dict[str, Any]:
        """
        Create a counterfactual scenario by changing a single factor.
        
        Args:
            base_scenario: Base scenario to modify
            factor_name: Name of factor to change
            new_value: New value for the factor
            
        Returns:
            Modified scenario.
        """
        # Create deep copy of base scenario
        cf_scenario = json.loads(json.dumps(base_scenario))
        
        # Update factor value
        if "." in factor_name:
            # Handle nested factors
            parts = factor_name.split(".")
            current = cf_scenario
            
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = new_value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        else:
            # Handle top-level factors
            cf_scenario[factor_name] = new_value
        
        return cf_scenario

    def _calculate_impact(self, base_outcome: Any, cf_outcome: Any) -> float:
        """
        Calculate impact of counterfactual change.
        
        Args:
            base_outcome: Outcome of base scenario
            cf_outcome: Outcome of counterfactual scenario
            
        Returns:
            Impact score.
        """
        # Handle different outcome types
        if isinstance(base_outcome, (int, float)) and isinstance(cf_outcome, (int, float)):
            # Numeric outcomes
            if base_outcome != 0:
                return (cf_outcome - base_outcome) / abs(base_outcome)
            else:
                return cf_outcome
        
        elif isinstance(base_outcome, dict) and isinstance(cf_outcome, dict):
            # Dictionary outcomes
            if "value" in base_outcome and "value" in cf_outcome:
                base_value = base_outcome["value"]
                cf_value = cf_outcome["value"]
                
                if isinstance(base_value, (int, float)) and isinstance(cf_value, (int, float)):
                    if base_value != 0:
                        return (cf_value - base_value) / abs(base_value)
                    else:
                        return cf_value
            
            # If no numeric value found, use success as impact
            base_success = base_outcome.get("success", False)
            cf_success = cf_outcome.get("success", False)
            
            if base_success == cf_success:
                return 0
            elif cf_success:
                return 1  # Positive impact
            else:
                return -1  # Negative impact
        
        else:
            # Boolean outcomes
            if base_outcome == cf_outcome:
                return 0
            elif cf_outcome:
                return 1  # Positive impact
            else:
                return -1  # Negative impact

    def _validate_decision_path(self, decision_tree: Dict[str, Any], path: List[str]) -> bool:
        """
        Validate if a decision path exists in the tree.
        
        Args:
            decision_tree: Decision tree structure
            path: Decision path to validate
            
        Returns:
            True if path is valid, False otherwise.
        """
        current_node = decision_tree
        
        for decision in path:
            if "children" not in current_node:
                return False
                
            if decision not in current_node["children"]:
                return False
                
            current_node = current_node["children"][decision]
        
        return True

    def _find_all_paths(self, decision_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find all possible paths in a decision tree.
        
        Args:
            decision_tree: Decision tree structure
            
        Returns:
            List of all paths with their expected values.
        """
        paths = []
        
        def traverse(node, current_path, probability=1.0):
            if "children" not in node:
                # Leaf node
                expected_value = node.get("value", 0)
                paths.append({
                    "path": current_path,
                    "expected_value": expected_value,
                    "probability": probability
                })
                return
            
            # Branch node
            for decision, child in node["children"].items():
                decision_prob = child.get("probability", 1.0 / len(node["children"]))
                traverse(child, current_path + [decision], probability * decision_prob)
        
        traverse(decision_tree, [])
        return paths

    def _calculate_path_metrics(self, 
                              decision_tree: Dict[str, Any], 
                              actual_path: List[str], 
                              alternative_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics for actual and alternative paths.
        
        Args:
            decision_tree: Decision tree structure
            actual_path: Actual decision path taken
            alternative_paths: Alternative paths
            
        Returns:
            Dictionary containing path metrics.
        """
        # Find actual path in tree
        current_node = decision_tree
        actual_probability = 1.0
        
        for decision in actual_path:
            if "children" not in current_node or decision not in current_node["children"]:
                break
                
            child = current_node["children"][decision]
            decision_prob = child.get("probability", 1.0 / len(current_node["children"]))
            actual_probability *= decision_prob
            
            current_node = child
        
        # Get expected value of actual path
        actual_expected_value = current_node.get("value", 0)
        
        return {
            "actual_expected_value": actual_expected_value,
            "actual_probability": actual_probability
        }

    def _identify_critical_points(self, 
                                decision_tree: Dict[str, Any], 
                                actual_path: List[str], 
                                alternative_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify critical decision points where alternatives would be better.
        
        Args:
            decision_tree: Decision tree structure
            actual_path: Actual decision path taken
            alternative_paths: Alternative paths
            
        Returns:
            List of critical decision points.
        """
        critical_points = []
        
        # Find actual path expected value
        actual_metrics = self._calculate_path_metrics(decision_tree, actual_path, alternative_paths)
        actual_value = actual_metrics["actual_expected_value"]
        
        # Check each decision point
        for i in range(len(actual_path)):
            # Find paths that differ only at this decision point
            prefix = actual_path[:i]
            actual_decision = actual_path[i]
            
            alternatives_at_point = []
            
            for alt_path_info in alternative_paths:
                alt_path = alt_path_info["path"]
                
                if len(alt_path) <= i:
                    continue
                    
                if alt_path[:i] == prefix and alt_path[i] != actual_decision:
                    alternatives_at_point.append({
                        "decision": alt_path[i],
                        "path": alt_path,
                        "expected_value": alt_path_info["expected_value"]
                    })
            
            # Find best alternative
            if alternatives_at_point:
                best_alternative = max(alternatives_at_point, key=lambda x: x["expected_value"])
                
                # Check if better than actual
                if best_alternative["expected_value"] > actual_value:
                    critical_points.append({
                        "decision_index": i,
                        "actual_decision": actual_decision,
                        "better_alternative": best_alternative["decision"],
                        "improvement": best_alternative["expected_value"] - actual_value,
                        "alternative_path": best_alternative["path"]
                    })
        
        # Sort by potential improvement
        critical_points.sort(key=lambda x: x["improvement"], reverse=True)
        
        return critical_points

    def _calculate_correlation(self, values_a: List[float], values_b: List[float]) -> float:
        """
        Calculate Pearson correlation between two series.
        
        Args:
            values_a: First series of values
            values_b: Second series of values
            
        Returns:
            Correlation coefficient.
        """
        # Ensure equal length
        min_length = min(len(values_a), len(values_b))
        values_a = values_a[:min_length]
        values_b = values_b[:min_length]
        
        if min_length < 2:
            return 0
        
        # Calculate means
        mean_a = sum(values_a) / min_length
        mean_b = sum(values_b) / min_length
        
        # Calculate covariance and variances
        covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b)) / min_length
        variance_a = sum((a - mean_a) ** 2 for a in values_a) / min_length
        variance_b = sum((b - mean_b) ** 2 for b in values_b) / min_length
        
        # Calculate Pearson correlation
        if variance_a > 0 and variance_b > 0:
            return covariance / ((variance_a * variance_b) ** 0.5)
        else:
            return 0
