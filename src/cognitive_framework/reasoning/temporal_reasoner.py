"""
Temporal Reasoner for AI Studio Agents.

This module implements advanced reasoning capabilities for temporal
analysis, forecasting, and pattern detection in time series data.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import numpy as np
import time
import json
from collections import defaultdict
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)


class TemporalReasoner:
    """
    Performs temporal reasoning, forecasting, and pattern detection.
    
    Key Features:
    - Time series analysis
    - Temporal pattern recognition
    - Trend detection and forecasting
    - Seasonality analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the temporal reasoner.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_forecast_horizon = self.config.get("default_forecast_horizon", 10)
        self.min_pattern_length = self.config.get("min_pattern_length", 3)
        self.significance_threshold = self.config.get("significance_threshold", 0.05)
        
        logger.info("TemporalReasoner initialized")

    def analyze_time_series(self, 
                           time_series: List[Dict[str, Any]], 
                           analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a time series for trends, patterns, and statistics.
        
        Args:
            time_series: List of data points with timestamp and value
            analysis_types: Types of analysis to perform
            
        Returns:
            Dictionary containing analysis results.
        """
        logger.info("Analyzing time series with %d points", len(time_series))
        
        if not time_series:
            return {
                "error": "No time series data provided",
                "success": False
            }
        
        # Default analysis types
        analysis_types = analysis_types or ["basic_stats", "trend", "seasonality", "outliers"]
        
        # Extract timestamps and values
        try:
            # Sort by timestamp
            sorted_series = sorted(time_series, key=lambda x: x.get("timestamp", 0))
            
            timestamps = [point.get("timestamp", i) for i, point in enumerate(sorted_series)]
            values = [point.get("value", 0) for point in sorted_series]
            
            # Convert timestamps to relative time if they're datetime strings
            if isinstance(timestamps[0], str):
                try:
                    datetime_objects = [datetime.fromisoformat(ts) for ts in timestamps]
                    # Convert to seconds since first timestamp
                    first_dt = datetime_objects[0]
                    timestamps = [(dt - first_dt).total_seconds() for dt in datetime_objects]
                except ValueError:
                    # If not ISO format, keep as is
                    pass
        except Exception as e:
            return {
                "error": f"Error processing time series data: {str(e)}",
                "success": False
            }
        
        results = {}
        
        # Basic statistics
        if "basic_stats" in analysis_types:
            results["basic_stats"] = self._calculate_basic_stats(values)
        
        # Trend analysis
        if "trend" in analysis_types:
            results["trend"] = self._analyze_trend(timestamps, values)
        
        # Seasonality analysis
        if "seasonality" in analysis_types:
            results["seasonality"] = self._analyze_seasonality(timestamps, values)
        
        # Outlier detection
        if "outliers" in analysis_types:
            results["outliers"] = self._detect_outliers(timestamps, values)
        
        # Change point detection
        if "change_points" in analysis_types:
            results["change_points"] = self._detect_change_points(timestamps, values)
        
        return {
            "time_series_length": len(time_series),
            "analysis_types": analysis_types,
            "results": results,
            "success": True
        }

    def forecast_time_series(self, 
                            time_series: List[Dict[str, Any]], 
                            horizon: Optional[int] = None, 
                            method: str = "auto") -> Dict[str, Any]:
        """
        Forecast future values of a time series.
        
        Args:
            time_series: List of data points with timestamp and value
            horizon: Number of steps to forecast
            method: Forecasting method to use
            
        Returns:
            Dictionary containing forecast results.
        """
        logger.info("Forecasting time series with method: %s", method)
        
        if not time_series:
            return {
                "error": "No time series data provided",
                "success": False
            }
        
        # Set horizon
        horizon = horizon or self.default_forecast_horizon
        
        # Extract timestamps and values
        try:
            # Sort by timestamp
            sorted_series = sorted(time_series, key=lambda x: x.get("timestamp", 0))
            
            timestamps = [point.get("timestamp", i) for i, point in enumerate(sorted_series)]
            values = [point.get("value", 0) for point in sorted_series]
            
            # Check if timestamps are datetime strings
            timestamp_format = None
            if isinstance(timestamps[0], str):
                try:
                    datetime_objects = [datetime.fromisoformat(ts) for ts in timestamps]
                    timestamp_format = "iso"
                    
                    # Calculate average time delta for forecasting timestamps
                    time_deltas = [(datetime_objects[i] - datetime_objects[i-1]).total_seconds() 
                                  for i in range(1, len(datetime_objects))]
                    avg_delta = sum(time_deltas) / len(time_deltas) if time_deltas else 86400  # Default to 1 day
                    
                    # Convert to seconds since first timestamp for calculations
                    first_dt = datetime_objects[0]
                    numeric_timestamps = [(dt - first_dt).total_seconds() for dt in datetime_objects]
                    
                except ValueError:
                    # If not ISO format, keep as is
                    numeric_timestamps = timestamps
            else:
                numeric_timestamps = timestamps
        except Exception as e:
            return {
                "error": f"Error processing time series data: {str(e)}",
                "success": False
            }
        
        # Select forecasting method
        if method == "auto":
            # Simple heuristic to choose method
            if len(time_series) >= 30:
                method = "arima"
            else:
                method = "linear"
        
        # Generate forecast
        forecast_values = []
        forecast_timestamps = []
        prediction_intervals = []
        
        if method == "linear":
            # Simple linear regression
            forecast_result = self._forecast_linear(numeric_timestamps, values, horizon)
            forecast_values = forecast_result["values"]
            forecast_timestamps = forecast_result["timestamps"]
            prediction_intervals = forecast_result["intervals"]
            
        elif method == "exponential_smoothing":
            # Exponential smoothing
            forecast_result = self._forecast_exponential_smoothing(numeric_timestamps, values, horizon)
            forecast_values = forecast_result["values"]
            forecast_timestamps = forecast_result["timestamps"]
            prediction_intervals = forecast_result["intervals"]
            
        elif method == "arima":
            # ARIMA model (simplified)
            forecast_result = self._forecast_arima(numeric_timestamps, values, horizon)
            forecast_values = forecast_result["values"]
            forecast_timestamps = forecast_result["timestamps"]
            prediction_intervals = forecast_result["intervals"]
            
        else:
            return {
                "error": f"Unknown forecasting method: {method}",
                "success": False
            }
        
        # Convert forecast timestamps back to original format if needed
        if timestamp_format == "iso":
            forecast_datetime = [first_dt + timedelta(seconds=ts) for ts in forecast_timestamps]
            forecast_timestamps = [dt.isoformat() for dt in forecast_datetime]
        
        # Prepare forecast points
        forecast_points = []
        for i in range(len(forecast_values)):
            point = {
                "timestamp": forecast_timestamps[i],
                "value": forecast_values[i],
                "lower_bound": prediction_intervals[i][0],
                "upper_bound": prediction_intervals[i][1]
            }
            forecast_points.append(point)
        
        return {
            "original_series_length": len(time_series),
            "forecast_horizon": horizon,
            "method": method,
            "forecast": forecast_points,
            "success": True
        }

    def detect_temporal_patterns(self, 
                               time_series: List[Dict[str, Any]], 
                               pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect temporal patterns in a time series.
        
        Args:
            time_series: List of data points with timestamp and value
            pattern_types: Types of patterns to detect
            
        Returns:
            Dictionary containing detected patterns.
        """
        logger.info("Detecting temporal patterns")
        
        if not time_series:
            return {
                "error": "No time series data provided",
                "success": False
            }
        
        # Default pattern types
        pattern_types = pattern_types or ["cycles", "trends", "anomalies", "regime_shifts"]
        
        # Extract timestamps and values
        try:
            # Sort by timestamp
            sorted_series = sorted(time_series, key=lambda x: x.get("timestamp", 0))
            
            timestamps = [point.get("timestamp", i) for i, point in enumerate(sorted_series)]
            values = [point.get("value", 0) for point in sorted_series]
        except Exception as e:
            return {
                "error": f"Error processing time series data: {str(e)}",
                "success": False
            }
        
        patterns = {}
        
        # Detect cycles/periodicity
        if "cycles" in pattern_types:
            patterns["cycles"] = self._detect_cycles(timestamps, values)
        
        # Detect trends
        if "trends" in pattern_types:
            patterns["trends"] = self._detect_trends(timestamps, values)
        
        # Detect anomalies
        if "anomalies" in pattern_types:
            patterns["anomalies"] = self._detect_anomalies(timestamps, values)
        
        # Detect regime shifts
        if "regime_shifts" in pattern_types:
            patterns["regime_shifts"] = self._detect_regime_shifts(timestamps, values)
        
        return {
            "time_series_length": len(time_series),
            "pattern_types": pattern_types,
            "patterns": patterns,
            "success": True
        }

    def compare_time_series(self, 
                           series_a: List[Dict[str, Any]], 
                           series_b: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare two time series for similarities and differences.
        
        Args:
            series_a: First time series
            series_b: Second time series
            
        Returns:
            Dictionary containing comparison results.
        """
        logger.info("Comparing two time series")
        
        if not series_a or not series_b:
            return {
                "error": "Both time series must contain data",
                "success": False
            }
        
        # Extract values
        try:
            # Sort by timestamp
            sorted_a = sorted(series_a, key=lambda x: x.get("timestamp", 0))
            sorted_b = sorted(series_b, key=lambda x: x.get("timestamp", 0))
            
            values_a = [point.get("value", 0) for point in sorted_a]
            values_b = [point.get("value", 0) for point in sorted_b]
        except Exception as e:
            return {
                "error": f"Error processing time series data: {str(e)}",
                "success": False
            }
        
        # Basic statistics comparison
        stats_a = self._calculate_basic_stats(values_a)
        stats_b = self._calculate_basic_stats(values_b)
        
        # Calculate correlation
        correlation = self._calculate_correlation(values_a, values_b)
        
        # Calculate similarity metrics
        similarity = self._calculate_similarity(values_a, values_b)
        
        # Determine lead/lag relationship
        lead_lag = self._calculate_lead_lag(values_a, values_b)
        
        return {
            "series_a_length": len(series_a),
            "series_b_length": len(series_b),
            "stats_a": stats_a,
            "stats_b": stats_b,
            "correlation": correlation,
            "similarity": similarity,
            "lead_lag": lead_lag,
            "success": True
        }

    def _calculate_basic_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a series of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary containing basic statistics.
        """
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "std_dev": None
            }
        
        # Calculate statistics
        count = len(values)
        mean = sum(values) / count
        sorted_values = sorted(values)
        median = sorted_values[count // 2] if count % 2 == 1 else (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2
        min_val = min(values)
        max_val = max(values)
        
        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in values) / count
        std_dev = variance ** 0.5
        
        return {
            "count": count,
            "mean": mean,
            "median": median,
            "min": min_val,
            "max": max_val,
            "std_dev": std_dev,
            "range": max_val - min_val
        }

    def _analyze_trend(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """
        Analyze trend in a time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary containing trend analysis.
        """
        if len(timestamps) != len(values) or len(timestamps) < 2:
            return {
                "trend_detected": False,
                "reason": "Insufficient data points"
            }
        
        # Linear regression
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_xx = sum(x * x for x in timestamps)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_mean = sum_y / n
        ss_total = sum((y - y_mean) ** 2 for y in values)
        ss_residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(timestamps, values))
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        # Determine trend direction and strength
        if abs(slope) < 1e-6:
            trend_direction = "flat"
            trend_strength = "none"
        else:
            trend_direction = "upward" if slope > 0 else "downward"
            
            if abs(r_squared) > 0.7:
                trend_strength = "strong"
            elif abs(r_squared) > 0.3:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
        
        return {
            "trend_detected": abs(r_squared) > 0.1,
            "direction": trend_direction,
            "strength": trend_strength,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared
        }

    def _analyze_seasonality(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """
        Analyze seasonality in a time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary containing seasonality analysis.
        """
        if len(timestamps) != len(values) or len(timestamps) < 4:
            return {
                "seasonality_detected": False,
                "reason": "Insufficient data points"
            }
        
        # Calculate time differences
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_diff = sum(time_diffs) / len(time_diffs)
        
        # Check if timestamps are regularly spaced
        regular_spacing = all(abs(diff - avg_diff) / avg_diff < 0.1 for diff in time_diffs)
        
        # Simple autocorrelation for seasonality detection
        max_lag = min(len(values) // 2, 50)  # Maximum lag to consider
        
        autocorrelations = []
        for lag in range(1, max_lag + 1):
            # Calculate autocorrelation at this lag
            mean = sum(values) / len(values)
            numerator = sum((values[i] - mean) * (values[i - lag] - mean) for i in range(lag, len(values)))
            denominator = sum((value - mean) ** 2 for value in values)
            
            if denominator > 0:
                autocorr = numerator / denominator
            else:
                autocorr = 0
                
            autocorrelations.append((lag, autocorr))
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorrelations) - 1):
            lag, autocorr = autocorrelations[i]
            prev_autocorr = autocorrelations[i-1][1]
            next_autocorr = autocorrelations[i+1][1]
            
            if autocorr > prev_autocorr and autocorr > next_autocorr and autocorr > 0.2:
                peaks.append((lag, autocorr))
        
        # Sort peaks by correlation strength
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        if peaks:
            # Convert lag to actual time period
            top_period_lag = peaks[0][0]
            top_period_time = top_period_lag * avg_diff
            
            # Determine seasonality strength
            top_correlation = peaks[0][1]
            
            if top_correlation > 0.7:
                seasonality_strength = "strong"
            elif top_correlation > 0.4:
                seasonality_strength = "moderate"
            else:
                seasonality_strength = "weak"
                
            return {
                "seasonality_detected": True,
                "top_period_lag": top_period_lag,
                "top_period_time": top_period_time,
                "top_correlation": top_correlation,
                "strength": seasonality_strength,
                "all_periods": [(lag, corr) for lag, corr in peaks],
                "regular_spacing": regular_spacing
            }
        else:
            return {
                "seasonality_detected": False,
                "reason": "No significant periodic patterns detected",
                "regular_spacing": regular_spacing
            }

    def _detect_outliers(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """
        Detect outliers in a time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary containing outlier detection results.
        """
        if len(timestamps) != len(values) or len(timestamps) < 4:
            return {
                "outliers_detected": False,
                "reason": "Insufficient data points"
            }
        
        # Calculate mean and standard deviation
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        # Define threshold for outliers (Z-score)
        threshold = 3.0
        
        # Detect outliers
        outliers = []
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            z_score = abs(value - mean) / std_dev if std_dev > 0 else 0
            
            if z_score > threshold:
                outliers.append({
                    "index": i,
                    "timestamp": timestamp,
                    "value": value,
                    "z_score": z_score
                })
        
        return {
            "outliers_detected": len(outliers) > 0,
            "outlier_count": len(outliers),
            "outliers": outliers,
            "mean": mean,
            "std_dev": std_dev,
            "threshold": threshold
        }

    def _detect_change_points(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """
        Detect change points in a time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary containing change point detection results.
        """
        if len(timestamps) != len(values) or len(timestamps) < 10:
            return {
                "change_points_detected": False,
                "reason": "Insufficient data points"
            }
        
        # Simple change point detection using sliding window
        window_size = max(5, len(values) // 10)
        
        change_points = []
        for i in range(window_size, len(values) - window_size):
            # Calculate statistics for windows before and after current point
            before_window = values[i - window_size:i]
            after_window = values[i:i + window_size]
            
            before_mean = sum(before_window) / len(before_window)
            after_mean = sum(after_window) / len(after_window)
            
            before_std = (sum((x - before_mean) ** 2 for x in before_window) / len(before_window)) ** 0.5
            after_std = (sum((x - after_mean) ** 2 for x in after_window) / len(after_window)) ** 0.5
            
            # Calculate change score
            mean_diff = abs(after_mean - before_mean)
            pooled_std = ((before_std ** 2 + after_std ** 2) / 2) ** 0.5
            
            change_score = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Check if significant change
            if change_score > 2.0:  # Threshold for significance
                change_points.append({
                    "index": i,
                    "timestamp": timestamps[i],
                    "value": values[i],
                    "change_score": change_score,
                    "before_mean": before_mean,
                    "after_mean": after_mean,
                    "mean_diff": mean_diff
                })
        
        # Merge nearby change points
        merged_change_points = []
        if change_points:
            current_point = change_points[0]
            
            for i in range(1, len(change_points)):
                if change_points[i]["index"] - current_point["index"] <= window_size:
                    # Keep the one with higher change score
                    if change_points[i]["change_score"] > current_point["change_score"]:
                        current_point = change_points[i]
                else:
                    merged_change_points.append(current_point)
                    current_point = change_points[i]
            
            merged_change_points.append(current_point)
        
        return {
            "change_points_detected": len(merged_change_points) > 0,
            "change_point_count": len(merged_change_points),
            "change_points": merged_change_points,
            "window_size": window_size
        }

    def _detect_cycles(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """
        Detect cycles in a time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary containing cycle detection results.
        """
        # Use seasonality analysis for cycle detection
        seasonality_result = self._analyze_seasonality(timestamps, values)
        
        if seasonality_result.get("seasonality_detected", False):
            cycles = []
            
            for period_lag, correlation in seasonality_result.get("all_periods", []):
                if correlation > 0.3:  # Only include significant cycles
                    avg_diff = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
                    period_time = period_lag * avg_diff
                    
                    cycles.append({
                        "period_lag": period_lag,
                        "period_time": period_time,
                        "correlation": correlation,
                        "strength": "strong" if correlation > 0.7 else "moderate" if correlation > 0.4 else "weak"
                    })
            
            return {
                "cycles_detected": True,
                "cycle_count": len(cycles),
                "cycles": cycles,
                "dominant_cycle": cycles[0] if cycles else None
            }
        else:
            return {
                "cycles_detected": False,
                "reason": seasonality_result.get("reason", "No significant cycles detected")
            }

    def _detect_trends(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """
        Detect multiple trends in a time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary containing trend detection results.
        """
        # First detect change points
        change_points_result = self._detect_change_points(timestamps, values)
        
        if not change_points_result.get("change_points_detected", False):
            # If no change points, analyze the whole series
            trend_result = self._analyze_trend(timestamps, values)
            
            if trend_result.get("trend_detected", False):
                return {
                    "trends_detected": True,
                    "trend_count": 1,
                    "trends": [
                        {
                            "start_index": 0,
                            "end_index": len(values) - 1,
                            "start_timestamp": timestamps[0],
                            "end_timestamp": timestamps[-1],
                            "direction": trend_result["direction"],
                            "strength": trend_result["strength"],
                            "slope": trend_result["slope"],
                            "r_squared": trend_result["r_squared"]
                        }
                    ]
                }
            else:
                return {
                    "trends_detected": False,
                    "reason": "No significant trends detected"
                }
        else:
            # Analyze trends between change points
            change_points = change_points_result["change_points"]
            segment_boundaries = [0] + [cp["index"] for cp in change_points] + [len(values) - 1]
            
            trends = []
            for i in range(len(segment_boundaries) - 1):
                start_idx = segment_boundaries[i]
                end_idx = segment_boundaries[i + 1]
                
                # Need at least 3 points for trend analysis
                if end_idx - start_idx < 3:
                    continue
                
                segment_timestamps = timestamps[start_idx:end_idx + 1]
                segment_values = values[start_idx:end_idx + 1]
                
                trend_result = self._analyze_trend(segment_timestamps, segment_values)
                
                if trend_result.get("trend_detected", False):
                    trends.append({
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "start_timestamp": timestamps[start_idx],
                        "end_timestamp": timestamps[end_idx],
                        "direction": trend_result["direction"],
                        "strength": trend_result["strength"],
                        "slope": trend_result["slope"],
                        "r_squared": trend_result["r_squared"]
                    })
            
            return {
                "trends_detected": len(trends) > 0,
                "trend_count": len(trends),
                "trends": trends
            }

    def _detect_anomalies(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """
        Detect anomalies in a time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary containing anomaly detection results.
        """
        # Combine outlier detection with contextual analysis
        outliers_result = self._detect_outliers(timestamps, values)
        
        if not outliers_result.get("outliers_detected", False):
            return {
                "anomalies_detected": False,
                "reason": "No anomalies detected"
            }
        
        # Analyze each outlier for contextual anomaly
        anomalies = []
        for outlier in outliers_result["outliers"]:
            index = outlier["index"]
            
            # Define context window
            window_size = min(5, len(values) // 10)
            start_idx = max(0, index - window_size)
            end_idx = min(len(values) - 1, index + window_size)
            
            # Skip if at the boundaries
            if index - start_idx < 2 or end_idx - index < 2:
                continue
            
            # Check if the point breaks a pattern
            before_values = values[start_idx:index]
            after_values = values[index+1:end_idx+1]
            
            # Simple check: does the point deviate from local trend?
            if len(before_values) >= 2 and len(after_values) >= 2:
                # Linear extrapolation from before points
                x_before = list(range(len(before_values)))
                slope_before, intercept_before = np.polyfit(x_before, before_values, 1)
                expected_value = slope_before * len(before_values) + intercept_before
                
                # Check if actual value deviates significantly
                actual_value = values[index]
                local_std = np.std(before_values + after_values)
                
                if local_std > 0 and abs(actual_value - expected_value) > 2 * local_std:
                    anomalies.append({
                        "index": index,
                        "timestamp": timestamps[index],
                        "value": actual_value,
                        "expected_value": expected_value,
                        "deviation": actual_value - expected_value,
                        "z_score": outlier["z_score"],
                        "type": "contextual"
                    })
            
        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies
        }

    def _detect_regime_shifts(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """
        Detect regime shifts in a time series.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary containing regime shift detection results.
        """
        # Use change points as potential regime shifts
        change_points_result = self._detect_change_points(timestamps, values)
        
        if not change_points_result.get("change_points_detected", False):
            return {
                "regime_shifts_detected": False,
                "reason": "No significant regime shifts detected"
            }
        
        # Analyze each segment for different statistical properties
        change_points = change_points_result["change_points"]
        segment_boundaries = [0] + [cp["index"] for cp in change_points] + [len(values) - 1]
        
        regimes = []
        for i in range(len(segment_boundaries) - 1):
            start_idx = segment_boundaries[i]
            end_idx = segment_boundaries[i + 1]
            
            # Need at least 5 points for regime analysis
            if end_idx - start_idx < 5:
                continue
            
            segment_values = values[start_idx:end_idx + 1]
            
            # Calculate regime statistics
            stats = self._calculate_basic_stats(segment_values)
            
            # Analyze trend within regime
            segment_timestamps = timestamps[start_idx:end_idx + 1]
            trend = self._analyze_trend(segment_timestamps, segment_values)
            
            regimes.append({
                "start_index": start_idx,
                "end_index": end_idx,
                "start_timestamp": timestamps[start_idx],
                "end_timestamp": timestamps[end_idx],
                "length": end_idx - start_idx + 1,
                "mean": stats["mean"],
                "std_dev": stats["std_dev"],
                "trend_direction": trend.get("direction", "flat"),
                "trend_strength": trend.get("strength", "none")
            })
        
        # Check if regimes are significantly different
        regime_shifts = []
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            
            # Calculate difference in means
            mean_diff = abs(next_regime["mean"] - current_regime["mean"])
            pooled_std = ((current_regime["std_dev"] ** 2 + next_regime["std_dev"] ** 2) / 2) ** 0.5
            
            # Calculate effect size (Cohen's d)
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Check if significant shift
            if effect_size > 0.8 or current_regime["trend_direction"] != next_regime["trend_direction"]:
                shift_index = current_regime["end_index"]
                
                regime_shifts.append({
                    "index": shift_index,
                    "timestamp": timestamps[shift_index],
                    "from_regime": i,
                    "to_regime": i + 1,
                    "effect_size": effect_size,
                    "mean_change": mean_diff,
                    "trend_change": current_regime["trend_direction"] != next_regime["trend_direction"]
                })
        
        return {
            "regime_shifts_detected": len(regime_shifts) > 0,
            "regime_count": len(regimes),
            "shift_count": len(regime_shifts),
            "regimes": regimes,
            "shifts": regime_shifts
        }

    def _forecast_linear(self, 
                        timestamps: List[float], 
                        values: List[float], 
                        horizon: int) -> Dict[str, Any]:
        """
        Generate forecast using linear regression.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            horizon: Number of steps to forecast
            
        Returns:
            Dictionary containing forecast results.
        """
        # Perform linear regression
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_xx = sum(x * x for x in timestamps)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate prediction interval
        y_pred = [slope * x + intercept for x in timestamps]
        residuals = [y - yp for y, yp in zip(values, y_pred)]
        residual_std = (sum(r ** 2 for r in residuals) / (n - 2)) ** 0.5 if n > 2 else 1.0
        
        # Generate forecast timestamps
        last_timestamp = timestamps[-1]
        time_diff = (timestamps[-1] - timestamps[0]) / (n - 1) if n > 1 else 1.0
        forecast_timestamps = [last_timestamp + (i + 1) * time_diff for i in range(horizon)]
        
        # Generate forecast values
        forecast_values = [slope * x + intercept for x in forecast_timestamps]
        
        # Generate prediction intervals (95% confidence)
        t_value = 1.96  # Approximation of t-distribution for large samples
        prediction_intervals = []
        
        for i, x in enumerate(forecast_timestamps):
            # Distance from mean of x
            x_mean = sum_x / n
            x_dist = (x - x_mean) ** 2
            
            # Standard error of prediction
            se_pred = residual_std * (1 + 1/n + (x_dist / (sum_xx - (sum_x ** 2) / n))) ** 0.5
            
            # Prediction interval
            lower = forecast_values[i] - t_value * se_pred
            upper = forecast_values[i] + t_value * se_pred
            
            prediction_intervals.append((lower, upper))
        
        return {
            "values": forecast_values,
            "timestamps": forecast_timestamps,
            "intervals": prediction_intervals,
            "method": "linear"
        }

    def _forecast_exponential_smoothing(self, 
                                       timestamps: List[float], 
                                       values: List[float], 
                                       horizon: int) -> Dict[str, Any]:
        """
        Generate forecast using exponential smoothing.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            horizon: Number of steps to forecast
            
        Returns:
            Dictionary containing forecast results.
        """
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        
        # Initialize level
        level = values[0]
        smoothed = [level]
        
        # Apply smoothing
        for i in range(1, len(values)):
            level = alpha * values[i] + (1 - alpha) * level
            smoothed.append(level)
        
        # Generate forecast timestamps
        last_timestamp = timestamps[-1]
        time_diff = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) > 1 else 1.0
        forecast_timestamps = [last_timestamp + (i + 1) * time_diff for i in range(horizon)]
        
        # Generate forecast values (constant)
        forecast_values = [level] * horizon
        
        # Calculate prediction intervals
        residuals = [values[i] - smoothed[i-1] for i in range(1, len(values))]
        residual_std = (sum(r ** 2 for r in residuals) / len(residuals)) ** 0.5 if residuals else 1.0
        
        # Generate prediction intervals (95% confidence)
        t_value = 1.96
        prediction_intervals = []
        
        for i in range(horizon):
            # Increasing uncertainty with horizon
            uncertainty = residual_std * (1 + i * 0.1)
            
            lower = forecast_values[i] - t_value * uncertainty
            upper = forecast_values[i] + t_value * uncertainty
            
            prediction_intervals.append((lower, upper))
        
        return {
            "values": forecast_values,
            "timestamps": forecast_timestamps,
            "intervals": prediction_intervals,
            "method": "exponential_smoothing"
        }

    def _forecast_arima(self, 
                       timestamps: List[float], 
                       values: List[float], 
                       horizon: int) -> Dict[str, Any]:
        """
        Generate forecast using ARIMA model (simplified).
        
        Args:
            timestamps: List of timestamps
            values: List of values
            horizon: Number of steps to forecast
            
        Returns:
            Dictionary containing forecast results.
        """
        # This is a simplified ARIMA implementation
        # In a real system, use statsmodels or another library
        
        # Difference the series (first-order differencing)
        diff_values = [values[i] - values[i-1] for i in range(1, len(values))]
        
        # Apply AR(1) model to differenced series
        ar_coef = 0.7  # Autoregressive coefficient
        
        # Generate forecast timestamps
        last_timestamp = timestamps[-1]
        time_diff = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) > 1 else 1.0
        forecast_timestamps = [last_timestamp + (i + 1) * time_diff for i in range(horizon)]
        
        # Generate forecast
        forecast_values = []
        last_value = values[-1]
        last_diff = diff_values[-1]
        
        for _ in range(horizon):
            # Forecast next difference
            next_diff = ar_coef * last_diff
            
            # Add difference to last value
            next_value = last_value + next_diff
            
            forecast_values.append(next_value)
            
            # Update for next iteration
            last_value = next_value
            last_diff = next_diff
        
        # Calculate prediction intervals
        residuals = [diff_values[i] - ar_coef * diff_values[i-1] for i in range(1, len(diff_values))]
        residual_std = (sum(r ** 2 for r in residuals) / len(residuals)) ** 0.5 if residuals else 1.0
        
        # Generate prediction intervals (95% confidence)
        t_value = 1.96
        prediction_intervals = []
        
        for i in range(horizon):
            # Increasing uncertainty with horizon
            uncertainty = residual_std * (i + 1) ** 0.5
            
            lower = forecast_values[i] - t_value * uncertainty
            upper = forecast_values[i] + t_value * uncertainty
            
            prediction_intervals.append((lower, upper))
        
        return {
            "values": forecast_values,
            "timestamps": forecast_timestamps,
            "intervals": prediction_intervals,
            "method": "arima"
        }

    def _calculate_correlation(self, values_a: List[float], values_b: List[float]) -> Dict[str, float]:
        """
        Calculate correlation between two series.
        
        Args:
            values_a: First series of values
            values_b: Second series of values
            
        Returns:
            Dictionary containing correlation metrics.
        """
        # Ensure equal length by truncating the longer series
        min_length = min(len(values_a), len(values_b))
        values_a = values_a[:min_length]
        values_b = values_b[:min_length]
        
        if min_length < 2:
            return {
                "pearson": None,
                "significance": None,
                "relationship": "unknown"
            }
        
        # Calculate means
        mean_a = sum(values_a) / min_length
        mean_b = sum(values_b) / min_length
        
        # Calculate covariance and variances
        covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b)) / min_length
        variance_a = sum((a - mean_a) ** 2 for a in values_a) / min_length
        variance_b = sum((b - mean_b) ** 2 for b in values_b) / min_length
        
        # Calculate Pearson correlation
        if variance_a > 0 and variance_b > 0:
            pearson = covariance / ((variance_a * variance_b) ** 0.5)
        else:
            pearson = 0
        
        # Determine relationship
        if abs(pearson) < 0.3:
            relationship = "weak"
        elif abs(pearson) < 0.7:
            relationship = "moderate"
        else:
            relationship = "strong"
            
        if pearson > 0:
            relationship += " positive"
        elif pearson < 0:
            relationship += " negative"
        else:
            relationship = "none"
        
        # Calculate significance (simplified)
        t_stat = pearson * ((min_length - 2) / (1 - pearson ** 2)) ** 0.5
        significance = abs(t_stat) > 1.96  # Approximation for 95% confidence
        
        return {
            "pearson": pearson,
            "significance": significance,
            "relationship": relationship
        }

    def _calculate_similarity(self, values_a: List[float], values_b: List[float]) -> Dict[str, float]:
        """
        Calculate similarity metrics between two series.
        
        Args:
            values_a: First series of values
            values_b: Second series of values
            
        Returns:
            Dictionary containing similarity metrics.
        """
        # Ensure equal length by truncating the longer series
        min_length = min(len(values_a), len(values_b))
        values_a = values_a[:min_length]
        values_b = values_b[:min_length]
        
        if min_length < 2:
            return {
                "euclidean_distance": None,
                "normalized_distance": None,
                "similarity_score": None
            }
        
        # Calculate Euclidean distance
        euclidean_distance = sum((a - b) ** 2 for a, b in zip(values_a, values_b)) ** 0.5
        
        # Normalize series
        mean_a = sum(values_a) / min_length
        mean_b = sum(values_b) / min_length
        std_a = (sum((a - mean_a) ** 2 for a in values_a) / min_length) ** 0.5
        std_b = (sum((b - mean_b) ** 2 for b in values_b) / min_length) ** 0.5
        
        if std_a > 0 and std_b > 0:
            norm_a = [(a - mean_a) / std_a for a in values_a]
            norm_b = [(b - mean_b) / std_b for b in values_b]
            
            # Calculate normalized distance
            normalized_distance = sum((a - b) ** 2 for a, b in zip(norm_a, norm_b)) ** 0.5
            
            # Calculate similarity score (0 to 1, higher is more similar)
            similarity_score = 1 / (1 + normalized_distance / min_length ** 0.5)
        else:
            normalized_distance = None
            similarity_score = 0 if mean_a != mean_b else 1
        
        return {
            "euclidean_distance": euclidean_distance,
            "normalized_distance": normalized_distance,
            "similarity_score": similarity_score
        }

    def _calculate_lead_lag(self, values_a: List[float], values_b: List[float]) -> Dict[str, Any]:
        """
        Calculate lead/lag relationship between two series.
        
        Args:
            values_a: First series of values
            values_b: Second series of values
            
        Returns:
            Dictionary containing lead/lag analysis.
        """
        if len(values_a) < 5 or len(values_b) < 5:
            return {
                "relationship_detected": False,
                "reason": "Insufficient data points"
            }
        
        # Calculate cross-correlation at different lags
        max_lag = min(len(values_a), len(values_b)) // 3
        
        # Normalize series
        mean_a = sum(values_a) / len(values_a)
        mean_b = sum(values_b) / len(values_b)
        std_a = (sum((a - mean_a) ** 2 for a in values_a) / len(values_a)) ** 0.5
        std_b = (sum((b - mean_b) ** 2 for b in values_b) / len(values_b)) ** 0.5
        
        if std_a == 0 or std_b == 0:
            return {
                "relationship_detected": False,
                "reason": "One or both series have zero variance"
            }
        
        norm_a = [(a - mean_a) / std_a for a in values_a]
        norm_b = [(b - mean_b) / std_b for b in values_b]
        
        # Calculate cross-correlation for different lags
        correlations = []
        
        # B lags A (A leads B)
        for lag in range(1, max_lag + 1):
            if lag >= len(norm_a) or lag >= len(norm_b):
                continue
                
            # Correlation with B lagging A
            corr = sum(norm_a[i] * norm_b[i - lag] for i in range(lag, min(len(norm_a), len(norm_b)))) / (min(len(norm_a), len(norm_b)) - lag)
            correlations.append(("a_leads", lag, corr))
        
        # A lags B (B leads A)
        for lag in range(1, max_lag + 1):
            if lag >= len(norm_a) or lag >= len(norm_b):
                continue
                
            # Correlation with A lagging B
            corr = sum(norm_a[i - lag] * norm_b[i] for i in range(lag, min(len(norm_a), len(norm_b)))) / (min(len(norm_a), len(norm_b)) - lag)
            correlations.append(("b_leads", lag, corr))
        
        # Find maximum correlation
        if not correlations:
            return {
                "relationship_detected": False,
                "reason": "Could not calculate correlations"
            }
            
        max_corr = max(correlations, key=lambda x: abs(x[2]))
        relationship, lag, correlation = max_corr
        
        # Check if correlation is significant
        if abs(correlation) < 0.3:
            return {
                "relationship_detected": False,
                "reason": "No significant lead/lag relationship detected"
            }
        
        # Determine lead/lag relationship
        if relationship == "a_leads":
            lead_series = "A"
            lag_series = "B"
        else:
            lead_series = "B"
            lag_series = "A"
        
        return {
            "relationship_detected": True,
            "lead_series": lead_series,
            "lag_series": lag_series,
            "lag_periods": lag,
            "correlation": correlation,
            "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.5 else "weak"
        }
