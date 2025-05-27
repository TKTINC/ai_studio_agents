"""
TAAT-specific perception module implementation.

This module extends the base perception module with TAAT-specific
functionality for monitoring social media and identifying trade signals.
"""

from typing import Any, Dict, Optional

# Use absolute imports instead of relative imports
from src.agent_core.perception.perception import BasePerceptionModule


class TaatPerceptionModule(BasePerceptionModule):
    """
    TAAT-specific perception module.
    
    Extends the base perception module with specialized functionality
    for monitoring social media and identifying trade signals.
    """
    
    def __init__(self):
        """Initialize the TAAT perception module."""
        super().__init__()
        # Register TAAT-specific processors
        self.register_processor("social_media", self._process_social_media_input)
        self.register_processor("trade_signal", self._process_trade_signal)
    
    async def _process_social_media_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process social media input data.
        
        Args:
            input_data: Social media data to process
            
        Returns:
            Processed input data with extracted trade signals
        """
        # Extract relevant information from social media post
        return {
            "type": "social_media",
            "content": input_data.get("text", ""),
            "metadata": {
                "platform": input_data.get("platform", "unknown"),
                "user": input_data.get("user", "unknown"),
                "timestamp": input_data.get("timestamp", ""),
                "potential_signal": self._detect_trade_signal(input_data.get("text", ""))
            }
        }
    
    async def _process_trade_signal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process explicit trade signal input.
        
        Args:
            input_data: Trade signal data to process
            
        Returns:
            Processed trade signal data
        """
        return {
            "type": "trade_signal",
            "content": input_data.get("description", ""),
            "metadata": {
                "symbol": input_data.get("symbol", ""),
                "action": input_data.get("action", ""),
                "price": input_data.get("price", None),
                "stop_loss": input_data.get("stop_loss", None),
                "take_profit": input_data.get("take_profit", None),
                "confidence": input_data.get("confidence", 0.0)
            }
        }
    
    def _detect_trade_signal(self, text: str) -> Dict[str, Any]:
        """
        Detect if a social media post contains a trade signal.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with signal detection results
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP
        
        # Check for common trading keywords
        trading_keywords = ["buy", "sell", "long", "short", "entry", "exit", "target", "stop"]
        contains_keywords = any(keyword in text.lower() for keyword in trading_keywords)
        
        # Check for ticker symbols (simplified)
        contains_ticker = "$" in text and any(c.isupper() for c in text)
        
        return {
            "is_signal": contains_keywords and contains_ticker,
            "confidence": 0.7 if (contains_keywords and contains_ticker) else 0.3,
            "keywords_detected": [kw for kw in trading_keywords if kw in text.lower()]
        }
