"""
Trader Model Module for TAAT Cognitive Framework.

This module implements trader modeling capabilities for understanding
and predicting trader behaviors, strategies, and decision patterns.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid

class TraderModel:
    """
    Trader Model for TAAT Cognitive Framework.
    
    Models trader behaviors, strategies, and decision patterns to enable
    better market understanding and prediction.
    """
    
    def __init__(self):
        """Initialize the trader model."""
        self.trader_types = {}
        self.trader_profiles = {}
        self.market_interactions = {}
        self.logger = logging.getLogger("TraderModel")
    
    def add_trader_type(self,
                       type_id: str,
                       characteristics: Dict[str, Any],
                       typical_behaviors: List[Dict[str, Any]]) -> None:
        """
        Add a trader type.
        
        Args:
            type_id: Unique identifier for the trader type
            characteristics: Characteristics of the trader type
            typical_behaviors: Typical behaviors of the trader type
        """
        self.trader_types[type_id] = {
            "characteristics": characteristics,
            "typical_behaviors": typical_behaviors,
            "created_at": datetime.now()
        }
        
        self.logger.info(f"Added trader type {type_id}")
    
    def add_trader_profile(self,
                          trader_id: str,
                          profile_data: Dict[str, Any],
                          trader_type: str) -> None:
        """
        Add a trader profile.
        
        Args:
            trader_id: Unique identifier for the trader
            profile_data: Profile data for the trader
            trader_type: Type of the trader
        """
        self.trader_profiles[trader_id] = {
            "profile_data": profile_data,
            "trader_type": trader_type,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Initialize market interactions
        if trader_id not in self.market_interactions:
            self.market_interactions[trader_id] = []
        
        self.logger.info(f"Added trader profile for {trader_id} of type {trader_type}")
    
    def get_trader_profile(self, trader_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trader profile.
        
        Args:
            trader_id: ID of the trader
            
        Returns:
            Trader profile or None if not found
        """
        if trader_id not in self.trader_profiles:
            return None
        
        return self.trader_profiles[trader_id]
    
    def update_trader_profile(self,
                            trader_id: str,
                            profile_data: Dict[str, Any]) -> bool:
        """
        Update a trader profile.
        
        Args:
            trader_id: ID of the trader
            profile_data: New profile data
            
        Returns:
            True if successful, False if trader not found
        """
        if trader_id not in self.trader_profiles:
            return False
        
        # Update profile data
        self.trader_profiles[trader_id]["profile_data"].update(profile_data)
        self.trader_profiles[trader_id]["updated_at"] = datetime.now()
        
        self.logger.info(f"Updated profile for trader {trader_id}")
        
        return True
    
    def record_market_interaction(self,
                                trader_id: str,
                                interaction_data: Dict[str, Any]) -> str:
        """
        Record a market interaction for a trader.
        
        Args:
            trader_id: ID of the trader
            interaction_data: Data about the interaction
            
        Returns:
            Interaction ID
        """
        # Create trader if not exists
        if trader_id not in self.trader_profiles:
            self.add_trader_profile(
                trader_id=trader_id,
                profile_data={"name": f"Trader {trader_id}"},
                trader_type="unknown"
            )
        
        # Initialize market interactions
        if trader_id not in self.market_interactions:
            self.market_interactions[trader_id] = []
        
        # Create interaction record
        interaction_id = f"interaction_{datetime.now().timestamp()}_{str(uuid.uuid4())[:8]}"
        
        interaction = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now(),
            "data": interaction_data
        }
        
        # Add to interactions
        self.market_interactions[trader_id].append(interaction)
        
        # Limit interactions history
        max_interactions = 100
        if len(self.market_interactions[trader_id]) > max_interactions:
            self.market_interactions[trader_id] = self.market_interactions[trader_id][-max_interactions:]
        
        self.logger.info(f"Recorded market interaction for trader {trader_id}")
        
        return interaction_id
    
    def get_trader_interactions(self,
                              trader_id: str,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get market interactions for a trader.
        
        Args:
            trader_id: ID of the trader
            limit: Maximum number of interactions to return
            
        Returns:
            List of market interactions
        """
        if trader_id not in self.market_interactions:
            return []
        
        # Sort by timestamp
        sorted_interactions = sorted(
            self.market_interactions[trader_id],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return sorted_interactions[:limit]
    
    def identify_trader_strategy(self, trader_id: str) -> List[Dict[str, Any]]:
        """
        Identify strategies used by a trader.
        
        Args:
            trader_id: ID of the trader
            
        Returns:
            List of identified strategies
        """
        if trader_id not in self.market_interactions:
            return []
        
        interactions = self.market_interactions[trader_id]
        
        if len(interactions) < 3:
            return []
        
        identified_strategies = []
        
        # Check for trend following
        trend_following_count = 0
        for interaction in interactions:
            data = interaction.get("data", {})
            
            if data.get("type") == "buy" and data.get("reason") == "uptrend":
                trend_following_count += 1
            elif data.get("type") == "sell" and data.get("reason") == "downtrend":
                trend_following_count += 1
        
        if trend_following_count >= 2:
            identified_strategies.append({
                "strategy": "trend_following",
                "confidence": min(0.9, 0.5 + (0.1 * trend_following_count)),
                "evidence_count": trend_following_count
            })
        
        # Check for contrarian
        contrarian_count = 0
        for interaction in interactions:
            data = interaction.get("data", {})
            
            if data.get("type") == "buy" and data.get("reason") == "oversold":
                contrarian_count += 1
            elif data.get("type") == "sell" and data.get("reason") == "overbought":
                contrarian_count += 1
        
        if contrarian_count >= 2:
            identified_strategies.append({
                "strategy": "contrarian",
                "confidence": min(0.9, 0.5 + (0.1 * contrarian_count)),
                "evidence_count": contrarian_count
            })
        
        # Check for stop loss usage
        stop_loss_count = 0
        for interaction in interactions:
            data = interaction.get("data", {})
            
            if data.get("type") == "sell" and data.get("reason") == "stop_loss":
                stop_loss_count += 1
        
        if stop_loss_count >= 1:
            identified_strategies.append({
                "strategy": "stop_loss_user",
                "confidence": min(0.9, 0.5 + (0.2 * stop_loss_count)),
                "evidence_count": stop_loss_count
            })
        
        return identified_strategies
    
    def get_trader_type_distribution(self) -> Dict[str, int]:
        """
        Get distribution of trader types.
        
        Returns:
            Dictionary of trader type counts
        """
        type_counts = {}
        
        for trader_id, profile in self.trader_profiles.items():
            trader_type = profile.get("trader_type", "unknown")
            type_counts[trader_type] = type_counts.get(trader_type, 0) + 1
        
        return type_counts
    
    def get_trader_behavior_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about trader behaviors.
        
        Returns:
            Dictionary of behavior statistics
        """
        if not self.market_interactions:
            return {"total_interactions": 0}
        
        # Count interaction types
        interaction_types = {}
        total_interactions = 0
        
        for trader_id, interactions in self.market_interactions.items():
            total_interactions += len(interactions)
            
            for interaction in interactions:
                data = interaction.get("data", {})
                interaction_type = data.get("type", "unknown")
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
        
        # Calculate buy/sell ratio
        buy_count = interaction_types.get("buy", 0)
        sell_count = interaction_types.get("sell", 0)
        
        buy_sell_ratio = buy_count / sell_count if sell_count > 0 else float('inf')
        
        return {
            "total_interactions": total_interactions,
            "interaction_types": interaction_types,
            "buy_sell_ratio": buy_sell_ratio,
            "trader_count": len(self.trader_profiles),
            "active_trader_count": len(self.market_interactions)
        }
