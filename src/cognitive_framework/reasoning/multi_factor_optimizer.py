"""
Multi-Factor Optimizer for AI Studio Agents.

This module implements advanced reasoning capabilities for optimizing
decisions based on multiple factors and constraints.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import numpy as np
import time
import json
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


class MultiFactorOptimizer:
    """
    Optimizes decisions based on multiple factors and constraints.
    
    Key Features:
    - Multi-objective optimization
    - Constraint satisfaction
    - Utility maximization
    - Risk-adjusted decision making
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-factor optimizer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default weights for different factors
        self.default_weights = {
            "expected_return": 0.4,
            "risk": 0.3,
            "liquidity": 0.1,
            "market_impact": 0.1,
            "transaction_cost": 0.1
        }
        
        # Override with config if provided
        self.weights = self.config.get("weights", self.default_weights)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
            
        logger.info("MultiFactorOptimizer initialized with weights: %s", self.weights)

    def optimize_decision(self, 
                         options: List[Dict[str, Any]], 
                         constraints: Optional[List[Dict[str, Any]]] = None,
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize a decision based on multiple factors and constraints.
        
        Args:
            options: List of decision options with factor values
            constraints: List of constraints to apply
            context: Additional context for optimization
            
        Returns:
            Dictionary containing optimization results.
        """
        logger.info("Optimizing decision with %d options", len(options))
        
        if not options:
            return {
                "error": "No options provided for optimization",
                "success": False
            }
            
        constraints = constraints or []
        context = context or {}
        
        # Apply constraints to filter options
        valid_options = self._apply_constraints(options, constraints)
        
        if not valid_options:
            return {
                "error": "No options satisfy the constraints",
                "filtered_options": options,
                "constraints": constraints,
                "success": False
            }
            
        # Calculate utility scores for each option
        scored_options = self._calculate_utility(valid_options, context)
        
        # Sort by score (descending)
        sorted_options = sorted(scored_options, key=lambda x: x["score"], reverse=True)
        
        # Return results
        return {
            "optimal_option": sorted_options[0],
            "all_options": sorted_options,
            "weights_used": self.weights,
            "success": True
        }

    def _apply_constraints(self, 
                          options: List[Dict[str, Any]], 
                          constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply constraints to filter options.
        
        Args:
            options: List of decision options
            constraints: List of constraints to apply
            
        Returns:
            List of options that satisfy all constraints.
        """
        valid_options = []
        
        for option in options:
            valid = True
            
            for constraint in constraints:
                factor = constraint.get("factor")
                operator = constraint.get("operator")
                value = constraint.get("value")
                
                if factor is None or operator is None or value is None:
                    continue
                    
                if factor not in option:
                    valid = False
                    break
                    
                option_value = option[factor]
                
                if operator == "eq" and option_value != value:
                    valid = False
                    break
                elif operator == "neq" and option_value == value:
                    valid = False
                    break
                elif operator == "gt" and option_value <= value:
                    valid = False
                    break
                elif operator == "lt" and option_value >= value:
                    valid = False
                    break
                elif operator == "gte" and option_value < value:
                    valid = False
                    break
                elif operator == "lte" and option_value > value:
                    valid = False
                    break
                elif operator == "in" and option_value not in value:
                    valid = False
                    break
                elif operator == "nin" and option_value in value:
                    valid = False
                    break
            
            if valid:
                valid_options.append(option)
                
        return valid_options

    def _calculate_utility(self, 
                          options: List[Dict[str, Any]], 
                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate utility scores for each option.
        
        Args:
            options: List of decision options
            context: Additional context for scoring
            
        Returns:
            List of options with utility scores.
        """
        scored_options = []
        
        # Get min/max values for normalization
        factor_ranges = {}
        for factor in self.weights.keys():
            values = [option.get(factor, 0) for option in options if factor in option]
            if values:
                factor_ranges[factor] = {
                    "min": min(values),
                    "max": max(values)
                }
            else:
                factor_ranges[factor] = {
                    "min": 0,
                    "max": 1  # Avoid division by zero
                }
        
        # Calculate utility scores
        for option in options:
            score = 0
            factor_scores = {}
            
            for factor, weight in self.weights.items():
                if factor in option:
                    # Get raw value
                    value = option[factor]
                    
                    # Normalize to [0, 1] range
                    factor_min = factor_ranges[factor]["min"]
                    factor_max = factor_ranges[factor]["max"]
                    
                    if factor_max > factor_min:
                        normalized_value = (value - factor_min) / (factor_max - factor_min)
                    else:
                        normalized_value = 0.5  # Default if all values are the same
                    
                    # Invert for factors where lower is better
                    if factor in ["risk", "market_impact", "transaction_cost"]:
                        normalized_value = 1 - normalized_value
                    
                    # Apply weight
                    factor_score = normalized_value * weight
                    factor_scores[factor] = factor_score
                    score += factor_score
            
            # Add score to option
            option_copy = option.copy()
            option_copy["score"] = score
            option_copy["factor_scores"] = factor_scores
            scored_options.append(option_copy)
        
        return scored_options

    def optimize_portfolio(self, 
                          assets: List[Dict[str, Any]], 
                          constraints: Optional[Dict[str, Any]] = None,
                          objective: str = "sharpe") -> Dict[str, Any]:
        """
        Optimize a portfolio allocation based on multiple factors.
        
        Args:
            assets: List of assets with return and risk characteristics
            constraints: Portfolio constraints
            objective: Optimization objective (sharpe, return, risk, etc.)
            
        Returns:
            Dictionary containing optimization results.
        """
        logger.info("Optimizing portfolio with %d assets", len(assets))
        
        if not assets:
            return {
                "error": "No assets provided for optimization",
                "success": False
            }
            
        constraints = constraints or {}
        
        # Extract asset data
        asset_names = [asset.get("name", f"Asset_{i}") for i, asset in enumerate(assets)]
        expected_returns = np.array([asset.get("expected_return", 0) for asset in assets])
        
        # Build covariance matrix from individual risks and correlations
        n_assets = len(assets)
        cov_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            risk_i = assets[i].get("risk", 0.1)
            cov_matrix[i, i] = risk_i ** 2  # Variance on diagonal
            
            for j in range(i + 1, n_assets):
                risk_j = assets[j].get("risk", 0.1)
                correlation = assets[i].get("correlations", {}).get(assets[j].get("name", ""), 0)
                
                # Covariance = correlation * std_dev_i * std_dev_j
                cov_ij = correlation * risk_i * risk_j
                cov_matrix[i, j] = cov_ij
                cov_matrix[j, i] = cov_ij  # Symmetric
        
        # Generate random portfolios
        num_portfolios = self.config.get("num_portfolios", 10000)
        results = []
        
        min_allocation = constraints.get("min_allocation", 0)
        max_allocation = constraints.get("max_allocation", 1)
        
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1
            
            # Apply min/max constraints
            if min_allocation > 0 or max_allocation < 1:
                valid_weights = True
                for w in weights:
                    if w < min_allocation or w > max_allocation:
                        valid_weights = False
                        break
                        
                if not valid_weights:
                    continue
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            results.append({
                "weights": weights.tolist(),
                "return": portfolio_return,
                "risk": portfolio_risk,
                "sharpe_ratio": sharpe_ratio
            })
        
        # Sort based on objective
        if objective == "sharpe":
            results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
        elif objective == "return":
            results.sort(key=lambda x: x["return"], reverse=True)
        elif objective == "risk":
            results.sort(key=lambda x: x["risk"])
        
        # Return best portfolio
        if not results:
            return {
                "error": "No valid portfolios found with given constraints",
                "success": False
            }
            
        best_portfolio = results[0]
        
        # Format weights with asset names
        weights_dict = {}
        for i, asset_name in enumerate(asset_names):
            weights_dict[asset_name] = best_portfolio["weights"][i]
        
        return {
            "optimal_weights": weights_dict,
            "expected_return": best_portfolio["return"],
            "expected_risk": best_portfolio["risk"],
            "sharpe_ratio": best_portfolio["sharpe_ratio"],
            "num_portfolios_evaluated": len(results),
            "success": True
        }

    def optimize_trade_execution(self, 
                               trade: Dict[str, Any], 
                               market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize trade execution strategy based on market conditions.
        
        Args:
            trade: Trade details (asset, size, direction, etc.)
            market_conditions: Current market conditions
            
        Returns:
            Dictionary containing optimization results.
        """
        logger.info("Optimizing trade execution for %s", trade.get("asset", "unknown"))
        
        # Extract trade details
        asset = trade.get("asset", "")
        size = trade.get("size", 0)
        direction = trade.get("direction", "buy")
        urgency = trade.get("urgency", "normal")
        
        if not asset or size <= 0:
            return {
                "error": "Invalid trade details",
                "success": False
            }
        
        # Extract market conditions
        volatility = market_conditions.get("volatility", "medium")
        liquidity = market_conditions.get("liquidity", "medium")
        spread = market_conditions.get("spread", 0.01)
        
        # Define execution strategies
        strategies = [
            {
                "name": "Market Order",
                "description": "Immediate execution at market price",
                "expected_price_impact": self._calculate_price_impact(size, liquidity, "market"),
                "expected_completion_time": 1,  # minutes
                "expected_slippage": self._calculate_slippage(size, volatility, spread, "market"),
                "risk": "high" if size > 1000 else "medium"
            },
            {
                "name": "TWAP",
                "description": "Time-Weighted Average Price over specified period",
                "expected_price_impact": self._calculate_price_impact(size, liquidity, "twap"),
                "expected_completion_time": 60,  # minutes
                "expected_slippage": self._calculate_slippage(size, volatility, spread, "twap"),
                "risk": "medium"
            },
            {
                "name": "VWAP",
                "description": "Volume-Weighted Average Price over specified period",
                "expected_price_impact": self._calculate_price_impact(size, liquidity, "vwap"),
                "expected_completion_time": 120,  # minutes
                "expected_slippage": self._calculate_slippage(size, volatility, spread, "vwap"),
                "risk": "medium"
            },
            {
                "name": "Iceberg",
                "description": "Gradually reveal order in small chunks",
                "expected_price_impact": self._calculate_price_impact(size, liquidity, "iceberg"),
                "expected_completion_time": 90,  # minutes
                "expected_slippage": self._calculate_slippage(size, volatility, spread, "iceberg"),
                "risk": "low"
            },
            {
                "name": "Adaptive",
                "description": "Dynamically adjust to market conditions",
                "expected_price_impact": self._calculate_price_impact(size, liquidity, "adaptive"),
                "expected_completion_time": 75,  # minutes
                "expected_slippage": self._calculate_slippage(size, volatility, spread, "adaptive"),
                "risk": "low"
            }
        ]
        
        # Convert risk levels to numeric scores
        risk_scores = {"low": 0.2, "medium": 0.5, "high": 0.8}
        for strategy in strategies:
            strategy["risk_score"] = risk_scores.get(strategy["risk"], 0.5)
        
        # Define constraints based on urgency
        constraints = []
        if urgency == "high":
            constraints.append({
                "factor": "expected_completion_time",
                "operator": "lte",
                "value": 30
            })
        elif urgency == "low":
            constraints.append({
                "factor": "risk_score",
                "operator": "lte",
                "value": 0.3
            })
        
        # Optimize strategy selection
        result = self.optimize_decision(strategies, constraints)
        
        if not result.get("success", False):
            # If constraints are too strict, retry without constraints
            result = self.optimize_decision(strategies)
        
        # Add trade details to result
        if result.get("success", False):
            result["trade"] = trade
            result["market_conditions"] = market_conditions
            
            # Add execution plan
            optimal_strategy = result["optimal_option"]
            result["execution_plan"] = {
                "strategy": optimal_strategy["name"],
                "description": optimal_strategy["description"],
                "expected_completion_time": optimal_strategy["expected_completion_time"],
                "expected_slippage": optimal_strategy["expected_slippage"],
                "expected_price_impact": optimal_strategy["expected_price_impact"],
                "risk_level": optimal_strategy["risk"]
            }
        
        return result

    def _calculate_price_impact(self, size: float, liquidity: str, strategy: str) -> float:
        """
        Calculate expected price impact based on order size, liquidity, and strategy.
        
        Args:
            size: Order size
            liquidity: Market liquidity (high, medium, low)
            strategy: Execution strategy
            
        Returns:
            Expected price impact as a percentage.
        """
        # Base impact factors by liquidity
        liquidity_factors = {
            "high": 0.0001,
            "medium": 0.0005,
            "low": 0.002
        }
        
        # Strategy impact multipliers
        strategy_multipliers = {
            "market": 1.0,
            "twap": 0.6,
            "vwap": 0.5,
            "iceberg": 0.4,
            "adaptive": 0.3
        }
        
        base_factor = liquidity_factors.get(liquidity, 0.0005)
        strategy_multiplier = strategy_multipliers.get(strategy, 1.0)
        
        # Square root model for price impact
        impact = base_factor * strategy_multiplier * (size ** 0.5)
        
        return impact

    def _calculate_slippage(self, size: float, volatility: str, spread: float, strategy: str) -> float:
        """
        Calculate expected slippage based on order size, volatility, spread, and strategy.
        
        Args:
            size: Order size
            volatility: Market volatility (high, medium, low)
            spread: Bid-ask spread
            strategy: Execution strategy
            
        Returns:
            Expected slippage as a percentage.
        """
        # Volatility factors
        volatility_factors = {
            "high": 0.003,
            "medium": 0.001,
            "low": 0.0003
        }
        
        # Strategy slippage multipliers
        strategy_multipliers = {
            "market": 1.0,
            "twap": 0.7,
            "vwap": 0.6,
            "iceberg": 0.5,
            "adaptive": 0.4
        }
        
        vol_factor = volatility_factors.get(volatility, 0.001)
        strategy_multiplier = strategy_multipliers.get(strategy, 1.0)
        
        # Slippage model: half spread + volatility component + size component
        slippage = (spread / 2) + (vol_factor * strategy_multiplier) + (0.0001 * size ** 0.3)
        
        return slippage
