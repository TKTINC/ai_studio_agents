"""
MENTOR-specific perception module implementation.

This module extends the base perception module with MENTOR-specific
functionality for understanding user investment philosophy and preferences.
"""

from typing import Any, Dict, Optional
from ...agent_core.perception.perception import BasePerceptionModule


class MentorPerceptionModule(BasePerceptionModule):
    """
    MENTOR-specific perception module.
    
    Extends the base perception module with specialized functionality
    for understanding user investment philosophy, preferences, and behavior patterns.
    """
    
    def __init__(self):
        """Initialize the MENTOR perception module."""
        super().__init__()
        # Register MENTOR-specific processors
        self.register_processor("user_profile", self._process_user_profile)
        self.register_processor("investment_preferences", self._process_investment_preferences)
        self.register_processor("behavior_patterns", self._process_behavior_patterns)
    
    async def _process_user_profile(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user profile data.
        
        Args:
            input_data: User profile data to process
            
        Returns:
            Processed user profile data
        """
        return {
            "type": "user_profile",
            "content": f"User profile for {input_data.get('user_id', 'unknown')}",
            "metadata": {
                "user_id": input_data.get("user_id", ""),
                "age": input_data.get("age", 0),
                "income_bracket": input_data.get("income_bracket", ""),
                "net_worth": input_data.get("net_worth", 0.0),
                "education": input_data.get("education", ""),
                "occupation": input_data.get("occupation", ""),
                "investment_experience": input_data.get("investment_experience", ""),
                "risk_tolerance": self._assess_risk_tolerance(input_data)
            }
        }
    
    async def _process_investment_preferences(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process investment preferences data.
        
        Args:
            input_data: Investment preferences data to process
            
        Returns:
            Processed investment preferences data
        """
        return {
            "type": "investment_preferences",
            "content": f"Investment preferences for {input_data.get('user_id', 'unknown')}",
            "metadata": {
                "user_id": input_data.get("user_id", ""),
                "time_horizon": input_data.get("time_horizon", ""),
                "preferred_assets": input_data.get("preferred_assets", []),
                "excluded_sectors": input_data.get("excluded_sectors", []),
                "esg_preferences": input_data.get("esg_preferences", {}),
                "tax_considerations": input_data.get("tax_considerations", {}),
                "liquidity_needs": input_data.get("liquidity_needs", ""),
                "investment_philosophy": self._extract_investment_philosophy(input_data)
            }
        }
    
    async def _process_behavior_patterns(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user behavior pattern data.
        
        Args:
            input_data: Behavior pattern data to process
            
        Returns:
            Processed behavior pattern data
        """
        return {
            "type": "behavior_patterns",
            "content": f"Behavior patterns for {input_data.get('user_id', 'unknown')}",
            "metadata": {
                "user_id": input_data.get("user_id", ""),
                "trading_frequency": input_data.get("trading_frequency", ""),
                "reaction_to_volatility": input_data.get("reaction_to_volatility", ""),
                "decision_making_speed": input_data.get("decision_making_speed", ""),
                "information_sources": input_data.get("information_sources", []),
                "past_mistakes": input_data.get("past_mistakes", []),
                "behavioral_biases": self._identify_behavioral_biases(input_data)
            }
        }
    
    def _assess_risk_tolerance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess user's risk tolerance from profile data.
        
        Args:
            data: User profile data
            
        Returns:
            Risk tolerance assessment
        """
        # This is a simplified implementation
        # In a real system, this would use a more sophisticated risk assessment model
        
        # Extract relevant factors
        age = data.get("age", 35)
        investment_experience = data.get("investment_experience", "moderate")
        questionnaire_score = data.get("risk_questionnaire_score", 50)
        
        # Simple scoring system
        score = 0
        
        # Age factor (younger = higher risk tolerance)
        if age < 30:
            score += 30
        elif age < 40:
            score += 25
        elif age < 50:
            score += 20
        elif age < 60:
            score += 15
        else:
            score += 10
            
        # Experience factor
        if investment_experience == "extensive":
            score += 30
        elif investment_experience == "moderate":
            score += 20
        else:
            score += 10
            
        # Questionnaire factor
        score += questionnaire_score // 2
        
        # Determine category
        category = "moderate"
        if score > 70:
            category = "aggressive"
        elif score > 50:
            category = "growth"
        elif score > 30:
            category = "moderate"
        else:
            category = "conservative"
            
        return {
            "score": score,
            "category": category,
            "max_drawdown_tolerance": f"{(score // 10)}%",
            "volatility_tolerance": "high" if score > 70 else "medium" if score > 40 else "low"
        }
    
    def _extract_investment_philosophy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract user's investment philosophy from preferences data.
        
        Args:
            data: Investment preferences data
            
        Returns:
            Investment philosophy assessment
        """
        # This is a simplified implementation
        # In a real system, this would use NLP to analyze user statements
        
        # Extract relevant factors
        time_horizon = data.get("time_horizon", "medium")
        preferred_assets = data.get("preferred_assets", [])
        statements = data.get("philosophy_statements", [])
        
        # Determine primary approach
        approach = "balanced"
        if "value" in statements or "dividend" in preferred_assets:
            approach = "value"
        elif "growth" in statements or "technology" in preferred_assets:
            approach = "growth"
        elif "index" in statements or "ETF" in preferred_assets:
            approach = "passive"
        
        # Determine time perspective
        perspective = "medium-term"
        if time_horizon == "long":
            perspective = "long-term"
        elif time_horizon == "short":
            perspective = "short-term"
            
        return {
            "primary_approach": approach,
            "time_perspective": perspective,
            "key_values": self._extract_key_values(statements),
            "investment_heroes": data.get("investment_heroes", []),
            "guiding_principles": data.get("guiding_principles", [])
        }
    
    def _extract_key_values(self, statements: List[str]) -> List[str]:
        """
        Extract key investment values from user statements.
        
        Args:
            statements: List of user statements about investing
            
        Returns:
            List of key values
        """
        # This is a simplified implementation
        # In a real system, this would use NLP to extract values
        
        values = []
        value_keywords = {
            "safety": ["safety", "security", "protection", "conservative"],
            "growth": ["growth", "appreciation", "expansion"],
            "income": ["income", "dividend", "yield", "cash flow"],
            "sustainability": ["sustainable", "responsible", "ESG", "ethical"],
            "innovation": ["innovation", "disruptive", "cutting-edge", "technology"]
        }
        
        for statement in statements:
            statement = statement.lower()
            for value, keywords in value_keywords.items():
                if any(keyword in statement for keyword in keywords) and value not in values:
                    values.append(value)
                    
        return values
    
    def _identify_behavioral_biases(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Identify potential behavioral biases from user behavior data.
        
        Args:
            data: User behavior data
            
        Returns:
            Dictionary of biases and their likelihood scores
        """
        # This is a simplified implementation
        # In a real system, this would use a more sophisticated behavioral analysis
        
        biases = {}
        
        # Check for loss aversion
        if data.get("reaction_to_volatility") == "sell_quickly" or data.get("risk_aversion") == "high":
            biases["loss_aversion"] = 0.8
            
        # Check for recency bias
        if data.get("decision_making_factors", {}).get("recent_performance", 0) > 7:
            biases["recency_bias"] = 0.7
            
        # Check for confirmation bias
        if data.get("information_diversity", 0) < 3:
            biases["confirmation_bias"] = 0.6
            
        # Check for overconfidence
        if data.get("confidence_level", 0) > 8 and data.get("investment_experience") != "extensive":
            biases["overconfidence"] = 0.75
            
        # Check for herd mentality
        if "social_media" in data.get("information_sources", []) and data.get("independent_thinking", 0) < 5:
            biases["herd_mentality"] = 0.65
            
        return biases
