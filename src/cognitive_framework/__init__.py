"""
Cognitive Framework Package for TAAT.

This package contains the cognitive framework components for the TAAT agent,
including reflection, meta-cognition, mental modeling, and cognitive controller.
"""

# Import cognitive framework components for easier access
from .reflection.performance_monitor import PerformanceMonitor
from .reflection.strategy_evaluator import StrategyEvaluator
from .reflection.error_analyzer import ErrorAnalyzer
from .reflection.insight_generator import InsightGenerator

from .meta_cognition.strategy_manager import StrategyManager
from .meta_cognition.strategy_selector import StrategySelector
from .meta_cognition.strategy_adapter import StrategyAdapter
from .meta_cognition.strategy_coordinator import StrategyCoordinator

from .mental_modeling.user_model import UserModel
from .mental_modeling.trader_model import TraderModel
from .mental_modeling.market_model import MarketModel
from .mental_modeling.model_updater import ModelUpdater

from .cognitive_controller import CognitiveController
