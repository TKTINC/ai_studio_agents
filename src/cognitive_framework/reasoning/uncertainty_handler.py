"""
Uncertainty Handler for AI Studio Agents.

This module implements advanced reasoning capabilities for handling
uncertainty in decision making and risk assessment.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import numpy as np
import time
import json
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


class UncertaintyHandler:
    """
    Handles uncertainty in decision making and risk assessment.
    
    Key Features:
    - Probabilistic reasoning
    - Confidence estimation
    - Scenario analysis
    - Risk quantification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the uncertainty handler.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default confidence thresholds
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }
        
        # Override with config if provided
        if "confidence_thresholds" in self.config:
            self.confidence_thresholds.update(self.config["confidence_thresholds"])
            
        logger.info("UncertaintyHandler initialized with confidence thresholds: %s", 
                   self.confidence_thresholds)

    def estimate_confidence(self, 
                           prediction: Dict[str, Any], 
                           evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate confidence level for a prediction based on available evidence.
        
        Args:
            prediction: Prediction details
            evidence: List of evidence supporting or contradicting the prediction
            
        Returns:
            Dictionary containing confidence estimation.
        """
        logger.info("Estimating confidence for prediction")
        
        if not evidence:
            return {
                "prediction": prediction,
                "confidence": 0.5,  # Default medium confidence with no evidence
                "confidence_level": "medium",
                "reasoning": "No evidence provided, using default medium confidence",
                "success": True
            }
        
        # Calculate base confidence from evidence
        total_weight = 0
        weighted_confidence = 0
        
        for item in evidence:
            # Extract evidence details
            relevance = item.get("relevance", 0.5)
            reliability = item.get("reliability", 0.5)
            support = item.get("support", 0)  # -1 to 1, negative means contradicts
            
            # Calculate weight for this evidence
            weight = relevance * reliability
            total_weight += weight
            
            # Calculate contribution to confidence
            # Map support from [-1, 1] to [0, 1]
            confidence_contribution = (support + 1) / 2
            weighted_confidence += weight * confidence_contribution
        
        # Normalize confidence
        if total_weight > 0:
            confidence = weighted_confidence / total_weight
        else:
            confidence = 0.5  # Default medium confidence
        
        # Determine confidence level
        if confidence >= self.confidence_thresholds["high"]:
            confidence_level = "high"
        elif confidence >= self.confidence_thresholds["medium"]:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Generate reasoning
        supporting_evidence = [e for e in evidence if e.get("support", 0) > 0]
        contradicting_evidence = [e for e in evidence if e.get("support", 0) < 0]
        
        reasoning = f"Based on {len(supporting_evidence)} supporting and {len(contradicting_evidence)} contradicting pieces of evidence. "
        
        if supporting_evidence:
            reasoning += f"Key supporting evidence: {supporting_evidence[0].get('description', 'N/A')}. "
            
        if contradicting_evidence:
            reasoning += f"Key contradicting evidence: {contradicting_evidence[0].get('description', 'N/A')}."
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "supporting_evidence_count": len(supporting_evidence),
            "contradicting_evidence_count": len(contradicting_evidence),
            "reasoning": reasoning,
            "success": True
        }

    def generate_scenarios(self, 
                          base_scenario: Dict[str, Any], 
                          uncertain_factors: List[Dict[str, Any]], 
                          num_scenarios: int = 3) -> Dict[str, Any]:
        """
        Generate multiple scenarios based on uncertain factors.
        
        Args:
            base_scenario: Base scenario details
            uncertain_factors: List of factors with uncertainty ranges
            num_scenarios: Number of scenarios to generate
            
        Returns:
            Dictionary containing generated scenarios.
        """
        logger.info("Generating %d scenarios", num_scenarios)
        
        if not uncertain_factors:
            return {
                "error": "No uncertain factors provided",
                "success": False
            }
        
        scenarios = []
        
        # Always include base scenario
        scenarios.append({
            "name": "Base Scenario",
            "description": "Expected case based on current information",
            "factors": {factor.get("name"): factor.get("base_value") for factor in uncertain_factors if "name" in factor and "base_value" in factor},
            "probability": 0.5  # Base scenario has highest probability
        })
        
        # Generate optimistic scenario
        optimistic_factors = {}
        for factor in uncertain_factors:
            name = factor.get("name")
            if name and "optimistic_value" in factor:
                optimistic_factors[name] = factor["optimistic_value"]
            elif name and "base_value" in factor and "uncertainty" in factor:
                # Calculate optimistic as base + uncertainty
                direction = factor.get("direction", 1)  # 1 means higher is better, -1 means lower is better
                if direction > 0:
                    optimistic_factors[name] = factor["base_value"] + factor["uncertainty"]
                else:
                    optimistic_factors[name] = factor["base_value"] - factor["uncertainty"]
        
        scenarios.append({
            "name": "Optimistic Scenario",
            "description": "Best case scenario based on favorable outcomes",
            "factors": optimistic_factors,
            "probability": 0.25  # Optimistic scenario has medium probability
        })
        
        # Generate pessimistic scenario
        pessimistic_factors = {}
        for factor in uncertain_factors:
            name = factor.get("name")
            if name and "pessimistic_value" in factor:
                pessimistic_factors[name] = factor["pessimistic_value"]
            elif name and "base_value" in factor and "uncertainty" in factor:
                # Calculate pessimistic as base - uncertainty
                direction = factor.get("direction", 1)  # 1 means higher is better, -1 means lower is better
                if direction > 0:
                    pessimistic_factors[name] = factor["base_value"] - factor["uncertainty"]
                else:
                    pessimistic_factors[name] = factor["base_value"] + factor["uncertainty"]
        
        scenarios.append({
            "name": "Pessimistic Scenario",
            "description": "Worst case scenario based on unfavorable outcomes",
            "factors": pessimistic_factors,
            "probability": 0.25  # Pessimistic scenario has medium probability
        })
        
        # Generate additional scenarios if requested
        if num_scenarios > 3:
            for i in range(num_scenarios - 3):
                # Generate random scenario between optimistic and pessimistic
                random_factors = {}
                for factor in uncertain_factors:
                    name = factor.get("name")
                    if name and "base_value" in factor and "uncertainty" in factor:
                        # Generate random value within uncertainty range
                        uncertainty = factor["uncertainty"]
                        base_value = factor["base_value"]
                        random_value = base_value + (np.random.random() * 2 - 1) * uncertainty
                        random_factors[name] = random_value
                
                scenarios.append({
                    "name": f"Alternative Scenario {i+1}",
                    "description": f"Random scenario with mixed factor outcomes",
                    "factors": random_factors,
                    "probability": (1.0 - 0.5 - 0.25 - 0.25) / (num_scenarios - 3)  # Distribute remaining probability
                })
        
        return {
            "base_scenario": base_scenario,
            "scenarios": scenarios,
            "uncertain_factors": uncertain_factors,
            "success": True
        }

    def quantify_risk(self, 
                     scenarios: List[Dict[str, Any]], 
                     impact_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Quantify risk based on multiple scenarios and their probabilities.
        
        Args:
            scenarios: List of scenarios with probabilities
            impact_function: Function to calculate impact from scenario factors
            
        Returns:
            Dictionary containing risk quantification.
        """
        logger.info("Quantifying risk based on %d scenarios", len(scenarios))
        
        if not scenarios:
            return {
                "error": "No scenarios provided",
                "success": False
            }
        
        # Default impact function if none provided
        if impact_function is None:
            # Simple average of factor values
            impact_function = lambda factors: sum(factors.values()) / len(factors) if factors else 0
        
        # Calculate expected value and variance
        total_probability = 0
        expected_value = 0
        values = []
        probabilities = []
        
        for scenario in scenarios:
            probability = scenario.get("probability", 1.0 / len(scenarios))
            factors = scenario.get("factors", {})
            
            # Calculate impact for this scenario
            try:
                impact = impact_function(factors)
            except Exception as e:
                logger.error("Error calculating impact: %s", str(e))
                impact = 0
            
            total_probability += probability
            expected_value += probability * impact
            values.append(impact)
            probabilities.append(probability)
        
        # Normalize probabilities if they don't sum to 1
        if total_probability > 0 and abs(total_probability - 1.0) > 1e-6:
            probabilities = [p / total_probability for p in probabilities]
            
            # Recalculate expected value
            expected_value = sum(p * v for p, v in zip(probabilities, values))
        
        # Calculate variance and standard deviation
        variance = sum(p * ((v - expected_value) ** 2) for p, v in zip(probabilities, values))
        std_deviation = variance ** 0.5
        
        # Calculate value at risk (VaR) at 95% confidence
        # Sort values in ascending order
        sorted_indices = np.argsort(values)
        sorted_values = [values[i] for i in sorted_indices]
        sorted_probabilities = [probabilities[i] for i in sorted_indices]
        
        # Calculate cumulative probabilities
        cumulative_probability = 0
        var_95 = None
        
        for i, (value, probability) in enumerate(zip(sorted_values, sorted_probabilities)):
            cumulative_probability += probability
            if cumulative_probability >= 0.05:  # 5% worst case
                var_95 = value
                break
        
        if var_95 is None and sorted_values:
            var_95 = sorted_values[0]  # Worst case if not enough scenarios
        
        # Calculate conditional value at risk (CVaR) / Expected Shortfall
        cvar_95 = None
        if var_95 is not None:
            # Find all scenarios worse than VaR
            worse_scenarios = [(v, p) for v, p in zip(values, probabilities) if v <= var_95]
            
            if worse_scenarios:
                worse_values, worse_probs = zip(*worse_scenarios)
                total_worse_prob = sum(worse_probs)
                
                if total_worse_prob > 0:
                    # Normalize probabilities
                    normalized_probs = [p / total_worse_prob for p in worse_probs]
                    cvar_95 = sum(v * p for v, p in zip(worse_values, normalized_probs))
        
        return {
            "expected_value": expected_value,
            "variance": variance,
            "standard_deviation": std_deviation,
            "coefficient_of_variation": std_deviation / expected_value if expected_value != 0 else float('inf'),
            "value_at_risk_95": var_95,
            "conditional_value_at_risk_95": cvar_95,
            "scenarios_evaluated": len(scenarios),
            "success": True
        }

    def calculate_decision_robustness(self, 
                                    decision: Dict[str, Any], 
                                    scenarios: List[Dict[str, Any]], 
                                    evaluation_function: Callable) -> Dict[str, Any]:
        """
        Calculate how robust a decision is across multiple scenarios.
        
        Args:
            decision: Decision details
            scenarios: List of scenarios to evaluate against
            evaluation_function: Function to evaluate decision in each scenario
            
        Returns:
            Dictionary containing robustness analysis.
        """
        logger.info("Calculating robustness for decision across %d scenarios", len(scenarios))
        
        if not scenarios:
            return {
                "error": "No scenarios provided",
                "success": False
            }
        
        # Evaluate decision in each scenario
        scenario_results = []
        success_count = 0
        weighted_score = 0
        
        for scenario in scenarios:
            scenario_name = scenario.get("name", "Unnamed Scenario")
            probability = scenario.get("probability", 1.0 / len(scenarios))
            factors = scenario.get("factors", {})
            
            # Evaluate decision in this scenario
            try:
                evaluation = evaluation_function(decision, factors)
                success = evaluation.get("success", False)
                score = evaluation.get("score", 0)
                
                if success:
                    success_count += 1
                    
                weighted_score += probability * score
                
                scenario_results.append({
                    "scenario": scenario_name,
                    "probability": probability,
                    "success": success,
                    "score": score,
                    "details": evaluation
                })
            except Exception as e:
                logger.error("Error evaluating scenario %s: %s", scenario_name, str(e))
                scenario_results.append({
                    "scenario": scenario_name,
                    "probability": probability,
                    "success": False,
                    "score": 0,
                    "error": str(e)
                })
        
        # Calculate robustness metrics
        success_rate = success_count / len(scenarios)
        
        # Calculate score variance
        scores = [result["score"] for result in scenario_results]
        mean_score = sum(scores) / len(scores)
        score_variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        score_std_dev = score_variance ** 0.5
        
        # Determine robustness level
        if success_rate >= 0.8 and score_std_dev / mean_score < 0.2:
            robustness_level = "high"
        elif success_rate >= 0.5 and score_std_dev / mean_score < 0.5:
            robustness_level = "medium"
        else:
            robustness_level = "low"
        
        return {
            "decision": decision,
            "success_rate": success_rate,
            "weighted_score": weighted_score,
            "score_variance": score_variance,
            "score_std_dev": score_std_dev,
            "robustness_level": robustness_level,
            "scenario_results": scenario_results,
            "success": True
        }

    def bayesian_update(self, 
                       prior_beliefs: Dict[str, float], 
                       new_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update beliefs using Bayesian inference based on new evidence.
        
        Args:
            prior_beliefs: Dictionary of hypotheses and their prior probabilities
            new_evidence: List of evidence with likelihoods for each hypothesis
            
        Returns:
            Dictionary containing updated beliefs.
        """
        logger.info("Updating beliefs with Bayesian inference")
        
        if not prior_beliefs:
            return {
                "error": "No prior beliefs provided",
                "success": False
            }
            
        if not new_evidence:
            return {
                "prior_beliefs": prior_beliefs,
                "posterior_beliefs": prior_beliefs,
                "evidence_applied": 0,
                "success": True
            }
        
        # Start with prior beliefs
        posterior_beliefs = prior_beliefs.copy()
        
        # Process each piece of evidence
        evidence_applied = 0
        
        for evidence in new_evidence:
            # Extract evidence details
            likelihoods = evidence.get("likelihoods", {})
            
            if not likelihoods:
                continue
                
            evidence_applied += 1
            
            # Calculate normalization factor
            normalization = 0
            for hypothesis, prior in posterior_beliefs.items():
                likelihood = likelihoods.get(hypothesis, 0.5)  # Default to 0.5 if not specified
                normalization += prior * likelihood
            
            # Update beliefs
            if normalization > 0:
                for hypothesis in posterior_beliefs:
                    likelihood = likelihoods.get(hypothesis, 0.5)
                    posterior_beliefs[hypothesis] = (posterior_beliefs[hypothesis] * likelihood) / normalization
        
        # Find most likely hypothesis
        most_likely = max(posterior_beliefs.items(), key=lambda x: x[1])
        
        return {
            "prior_beliefs": prior_beliefs,
            "posterior_beliefs": posterior_beliefs,
            "most_likely_hypothesis": most_likely[0],
            "most_likely_probability": most_likely[1],
            "evidence_applied": evidence_applied,
            "success": True
        }
