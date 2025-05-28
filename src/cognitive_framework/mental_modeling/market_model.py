"""
Market Model Module for TAAT Cognitive Framework.

This module implements market modeling capabilities for understanding
and predicting market behaviors, trends, and patterns.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import statistics

class MarketModel:
    """
    Market Model for TAAT Cognitive Framework.
    
    Models market behaviors, trends, and patterns to enable better
    understanding and prediction of market movements.
    """
    
    def __init__(self, market_id: str):
        """
        Initialize the market model.
        
        Args:
            market_id: Unique identifier for the market
        """
        self.market_id = market_id
        self.price_history = []
        self.volume_history = []
        self.sentiment_history = []
        self.market_events = []
        self.logger = logging.getLogger("MarketModel")
    
    def add_price_data(self,
                      timestamp: datetime,
                      price_data: Dict[str, float],
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add price data to the market model.
        
        Args:
            timestamp: Timestamp of the price data
            price_data: Price data (open, high, low, close, etc.)
            metadata: Optional metadata about the price data
        """
        price_entry = {
            "timestamp": timestamp,
            "data": price_data,
            "metadata": metadata or {}
        }
        
        self.price_history.append(price_entry)
        
        # Limit history size
        max_history = 1000
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
        
        self.logger.info(f"Added price data for market {self.market_id} at {timestamp}")
    
    def add_volume_data(self,
                       timestamp: datetime,
                       volume: float,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add volume data to the market model.
        
        Args:
            timestamp: Timestamp of the volume data
            volume: Trading volume
            metadata: Optional metadata about the volume data
        """
        volume_entry = {
            "timestamp": timestamp,
            "volume": volume,
            "metadata": metadata or {}
        }
        
        self.volume_history.append(volume_entry)
        
        # Limit history size
        max_history = 1000
        if len(self.volume_history) > max_history:
            self.volume_history = self.volume_history[-max_history:]
        
        self.logger.info(f"Added volume data for market {self.market_id} at {timestamp}")
    
    def add_sentiment_data(self,
                         timestamp: datetime,
                         sentiment: float,
                         source: str,
                         confidence: float = 0.5) -> None:
        """
        Add sentiment data to the market model.
        
        Args:
            timestamp: Timestamp of the sentiment data
            sentiment: Sentiment value (-1.0 to 1.0)
            source: Source of the sentiment data
            confidence: Confidence in the sentiment data (0.0 to 1.0)
        """
        sentiment_entry = {
            "timestamp": timestamp,
            "sentiment": sentiment,
            "source": source,
            "confidence": confidence
        }
        
        self.sentiment_history.append(sentiment_entry)
        
        # Limit history size
        max_history = 100
        if len(self.sentiment_history) > max_history:
            self.sentiment_history = self.sentiment_history[-max_history:]
        
        self.logger.info(f"Added sentiment data for market {self.market_id} at {timestamp}")
    
    def add_market_event(self,
                        timestamp: datetime,
                        event_type: str,
                        event_data: Dict[str, Any]) -> None:
        """
        Add a market event to the model.
        
        Args:
            timestamp: Timestamp of the event
            event_type: Type of event
            event_data: Data about the event
        """
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "data": event_data
        }
        
        self.market_events.append(event)
        
        # Limit history size
        max_events = 100
        if len(self.market_events) > max_events:
            self.market_events = self.market_events[-max_events:]
        
        self.logger.info(f"Added {event_type} event for market {self.market_id} at {timestamp}")
    
    def get_price_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get price history.
        
        Args:
            limit: Maximum number of price entries to return
            
        Returns:
            List of price entries
        """
        # Sort by timestamp
        sorted_history = sorted(
            self.price_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def get_volume_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get volume history.
        
        Args:
            limit: Maximum number of volume entries to return
            
        Returns:
            List of volume entries
        """
        # Sort by timestamp
        sorted_history = sorted(
            self.volume_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def get_sentiment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get sentiment history.
        
        Args:
            limit: Maximum number of sentiment entries to return
            
        Returns:
            List of sentiment entries
        """
        # Sort by timestamp
        sorted_history = sorted(
            self.sentiment_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def get_market_events(self,
                         event_type: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get market events.
        
        Args:
            event_type: Optional type of events to filter by
            limit: Maximum number of events to return
            
        Returns:
            List of market events
        """
        if event_type:
            filtered_events = [
                event for event in self.market_events
                if event["type"] == event_type
            ]
            
            # Sort by timestamp
            filtered_events.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return filtered_events[:limit]
        else:
            # Sort by timestamp
            sorted_events = sorted(
                self.market_events,
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            return sorted_events[:limit]
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current market state.
        
        Returns:
            Current market state
        """
        state = {
            "market_id": self.market_id,
            "timestamp": datetime.now()
        }
        
        # Get current price
        if self.price_history:
            latest_price = max(self.price_history, key=lambda x: x["timestamp"])
            state["current_price"] = latest_price["data"].get("close", 0.0)
            state["latest_price_data"] = latest_price["data"]
        
        # Get current volume
        if self.volume_history:
            latest_volume = max(self.volume_history, key=lambda x: x["timestamp"])
            state["current_volume"] = latest_volume["volume"]
        
        # Get current sentiment
        if self.sentiment_history:
            latest_sentiment = max(self.sentiment_history, key=lambda x: x["timestamp"])
            state["current_sentiment"] = latest_sentiment["sentiment"]
            state["sentiment_source"] = latest_sentiment["source"]
        
        # Get recent events
        recent_events = self.get_market_events(limit=5)
        if recent_events:
            state["recent_events"] = recent_events
        
        return state
    
    def calculate_technical_indicators(self) -> Dict[str, Any]:
        """
        Calculate technical indicators.
        
        Returns:
            Dictionary of technical indicators
        """
        indicators = {}
        
        if not self.price_history:
            return indicators
        
        # Sort price history by timestamp
        sorted_prices = sorted(self.price_history, key=lambda x: x["timestamp"])
        
        # Extract close prices
        close_prices = [entry["data"].get("close", 0.0) for entry in sorted_prices if "close" in entry["data"]]
        
        if not close_prices:
            return indicators
        
        # Simple Moving Averages
        if len(close_prices) >= 20:
            indicators["sma_20"] = sum(close_prices[-20:]) / 20
        
        if len(close_prices) >= 50:
            indicators["sma_50"] = sum(close_prices[-50:]) / 50
        
        if len(close_prices) >= 200:
            indicators["sma_200"] = sum(close_prices[-200:]) / 200
        
        # Relative Strength Index (RSI)
        if len(close_prices) >= 15:
            price_changes = [close_prices[i] - close_prices[i-1] for i in range(1, len(close_prices))]
            gains = [change for change in price_changes[-14:] if change > 0]
            losses = [abs(change) for change in price_changes[-14:] if change < 0]
            
            avg_gain = sum(gains) / 14 if gains else 0
            avg_loss = sum(losses) / 14 if losses else 0
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                indicators["rsi_14"] = 100 - (100 / (1 + rs))
            else:
                indicators["rsi_14"] = 100
        
        # Volatility
        if len(close_prices) >= 20:
            returns = [(close_prices[i] / close_prices[i-1]) - 1 for i in range(1, len(close_prices))]
            indicators["volatility_20"] = statistics.stdev(returns[-20:]) if len(returns) >= 20 else 0
        
        return indicators
    
    def detect_market_regime(self) -> Dict[str, Any]:
        """
        Detect current market regime.
        
        Returns:
            Market regime information
        """
        regime = {
            "timestamp": datetime.now(),
            "market_id": self.market_id
        }
        
        if not self.price_history:
            regime["regime"] = "unknown"
            return regime
        
        # Sort price history by timestamp
        sorted_prices = sorted(self.price_history, key=lambda x: x["timestamp"])
        
        # Extract close prices
        close_prices = [entry["data"].get("close", 0.0) for entry in sorted_prices if "close" in entry["data"]]
        
        if not close_prices or len(close_prices) < 20:
            regime["regime"] = "unknown"
            return regime
        
        # Calculate indicators
        indicators = self.calculate_technical_indicators()
        
        # Determine trend
        if len(close_prices) >= 20:
            current_price = close_prices[-1]
            sma_20 = indicators.get("sma_20", 0)
            
            if current_price > sma_20 * 1.05:
                regime["trend"] = "strong_uptrend"
            elif current_price > sma_20:
                regime["trend"] = "uptrend"
            elif current_price < sma_20 * 0.95:
                regime["trend"] = "strong_downtrend"
            elif current_price < sma_20:
                regime["trend"] = "downtrend"
            else:
                regime["trend"] = "sideways"
        
        # Determine volatility regime
        if "volatility_20" in indicators:
            volatility = indicators["volatility_20"]
            
            if volatility > 0.03:
                regime["volatility"] = "high"
            elif volatility > 0.01:
                regime["volatility"] = "medium"
            else:
                regime["volatility"] = "low"
        
        # Determine sentiment regime
        if self.sentiment_history:
            recent_sentiments = [entry["sentiment"] for entry in self.get_sentiment_history(limit=10)]
            avg_sentiment = sum(recent_sentiments) / len(recent_sentiments)
            
            if avg_sentiment > 0.3:
                regime["sentiment"] = "bullish"
            elif avg_sentiment < -0.3:
                regime["sentiment"] = "bearish"
            else:
                regime["sentiment"] = "neutral"
        
        # Determine overall regime
        if "trend" in regime and "volatility" in regime:
            if regime["trend"] in ["strong_uptrend", "uptrend"] and regime["volatility"] == "low":
                regime["regime"] = "bull_market"
            elif regime["trend"] in ["strong_downtrend", "downtrend"] and regime["volatility"] == "high":
                regime["regime"] = "bear_market"
            elif regime["trend"] == "sideways" and regime["volatility"] == "low":
                regime["regime"] = "consolidation"
            elif regime["volatility"] == "high":
                regime["regime"] = "volatile"
            else:
                regime["regime"] = "mixed"
        else:
            regime["regime"] = "unknown"
        
        return regime
