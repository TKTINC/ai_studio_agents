"""
Shared modules for market data processing.

This module provides common functionality for fetching, processing,
and analyzing market data across different agents.
"""

from typing import Dict, List, Any, Optional
import datetime


class MarketDataProcessor:
    """
    Shared market data processing functionality.
    
    Provides methods for fetching, normalizing, and analyzing market data
    that can be used by multiple agents.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the market data processor.
        
        Args:
            api_keys: Dictionary of API keys for different data providers
        """
        self.api_keys = api_keys or {}
        self.data_cache = {}
        self.cache_expiry = {}
    
    async def fetch_market_data(self, symbol: str, data_type: str = "price", 
                               timeframe: str = "1d", limit: int = 100) -> Dict[str, Any]:
        """
        Fetch market data for a specific symbol.
        
        Args:
            symbol: The ticker symbol to fetch data for
            data_type: Type of data to fetch (price, volume, etc.)
            timeframe: Timeframe for the data (1m, 5m, 1h, 1d, etc.)
            limit: Number of data points to fetch
            
        Returns:
            Dictionary containing the fetched market data
        """
        # Check cache first
        cache_key = f"{symbol}_{data_type}_{timeframe}_{limit}"
        if cache_key in self.data_cache:
            # Check if cache is still valid
            if datetime.datetime.now() < self.cache_expiry.get(cache_key, datetime.datetime.min):
                return self.data_cache[cache_key]
        
        # In a real implementation, this would connect to market data APIs
        # For now, we'll return simulated data
        
        # Simulate fetching data
        data = self._simulate_market_data(symbol, data_type, timeframe, limit)
        
        # Cache the data with a 5-minute expiry
        self.data_cache[cache_key] = data
        self.cache_expiry[cache_key] = datetime.datetime.now() + datetime.timedelta(minutes=5)
        
        return data
    
    def _simulate_market_data(self, symbol: str, data_type: str, 
                             timeframe: str, limit: int) -> Dict[str, Any]:
        """
        Simulate market data for testing purposes.
        
        Args:
            symbol: The ticker symbol
            data_type: Type of data
            timeframe: Timeframe for the data
            limit: Number of data points
            
        Returns:
            Simulated market data
        """
        import random
        
        # Generate timestamps
        now = datetime.datetime.now()
        timestamps = []
        
        if timeframe == "1m":
            delta = datetime.timedelta(minutes=1)
        elif timeframe == "5m":
            delta = datetime.timedelta(minutes=5)
        elif timeframe == "1h":
            delta = datetime.timedelta(hours=1)
        else:  # Default to 1d
            delta = datetime.timedelta(days=1)
        
        for i in range(limit):
            timestamps.append((now - (limit - i - 1) * delta).isoformat())
        
        # Generate price data
        base_price = random.uniform(50, 200)
        prices = []
        for _ in range(limit):
            change = random.uniform(-2, 2)
            base_price += change
            prices.append(round(base_price, 2))
        
        # Generate volume data
        volumes = [int(random.uniform(100000, 1000000)) for _ in range(limit)]
        
        return {
            "symbol": symbol,
            "data_type": data_type,
            "timeframe": timeframe,
            "timestamps": timestamps,
            "prices": prices if data_type == "price" else None,
            "volumes": volumes if data_type == "volume" else None,
            "data": prices if data_type == "price" else volumes
        }
    
    def calculate_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate technical indicators from market data.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Dictionary of calculated technical indicators
        """
        prices = data.get("prices", [])
        if not prices or len(prices) < 20:
            return {"error": "Insufficient data for technical analysis"}
        
        # Calculate simple moving averages
        sma_20 = self._calculate_sma(prices, 20)
        sma_50 = self._calculate_sma(prices, 50) if len(prices) >= 50 else None
        sma_200 = self._calculate_sma(prices, 200) if len(prices) >= 200 else None
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices, 14) if len(prices) >= 14 else None
        
        # Calculate MACD
        macd = self._calculate_macd(prices) if len(prices) >= 26 else None
        
        return {
            "sma": {
                "20": sma_20,
                "50": sma_50,
                "200": sma_200
            },
            "rsi": rsi,
            "macd": macd
        }
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of price data
            period: Period for the moving average
            
        Returns:
            Calculated SMA value
        """
        if len(prices) < period:
            return None
        
        return sum(prices[-period:]) / period
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: List of price data
            period: Period for RSI calculation
            
        Returns:
            Calculated RSI value
        """
        if len(prices) <= period:
            return None
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Calculate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gains and losses
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: List of price data
            
        Returns:
            Dictionary with MACD values
        """
        # Calculate EMAs
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        if ema_12 is None or ema_26 is None:
            return None
        
        # Calculate MACD line
        macd_line = ema_12 - ema_26
        
        # Calculate signal line (9-day EMA of MACD line)
        # In a real implementation, this would calculate the EMA of the MACD line
        signal_line = macd_line * 0.9  # Simplified for this example
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            "macd_line": round(macd_line, 2),
            "signal_line": round(signal_line, 2),
            "histogram": round(histogram, 2)
        }
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of price data
            period: Period for EMA calculation
            
        Returns:
            Calculated EMA value
        """
        if len(prices) < period:
            return None
        
        # Start with SMA
        ema = sum(prices[:period]) / period
        
        # Calculate multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate EMA
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
