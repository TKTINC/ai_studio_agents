"""
Cognitive Controller Module for TAAT Cognitive Framework.

This module serves as the central coordinator for the cognitive framework,
integrating reflection, meta-cognition, and mental modeling systems with
the agent core and memory systems.
"""

from typing import Dict, List, Any, Optional, Tuple, Type
from datetime import datetime
import logging
from collections import defaultdict

# Import reflection modules
from cognitive_framework.reflection.performance_monitor import PerformanceMonitor
from cognitive_framework.reflection.strategy_evaluator import StrategyEvaluator
from cognitive_framework.reflection.error_analyzer import ErrorAnalyzer
from cognitive_framework.reflection.insight_generator import InsightGenerator

# Import meta-cognition modules
from cognitive_framework.meta_cognition.strategy_manager import StrategyManager
from cognitive_framework.meta_cognition.strategy_selector import StrategySelector
from cognitive_framework.meta_cognition.strategy_adapter import StrategyAdapter
from cognitive_framework.meta_cognition.strategy_coordinator import StrategyCoordinator

# Import mental modeling modules
from cognitive_framework.mental_modeling.user_model import UserModel
from cognitive_framework.mental_modeling.trader_model import TraderModel
from cognitive_framework.mental_modeling.market_model import MarketModel
from cognitive_framework.mental_modeling.model_updater import ModelUpdater

# Import memory system modules
from memory_systems.episodic import EpisodicMemory
from memory_systems.semantic import SemanticMemory
from memory_systems.procedural import ProceduralMemory
from memory_systems.advanced_retrieval import AdvancedRetrieval
from memory_systems.memory_consolidation import MemoryConsolidation
from memory_systems.cognitive_integration import CognitiveIntegration

# Import agent core modules
from agent_core.memory.memory_manager import MemoryManager
from agent_core.perception.perception import Perception
from agent_core.cognition.cognition import Cognition
from agent_core.action.action import Action


class CognitiveController:
    """
    Central coordinator for the TAAT cognitive framework.
    
    The CognitiveController integrates all cognitive framework components,
    manages their interactions, and coordinates with the agent core and
    memory systems to enable advanced cognitive capabilities.
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CognitiveController with all required components.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Optional configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.logger = logging.getLogger("CognitiveController")
        
        # Initialize reflection system
        self.performance_monitor = PerformanceMonitor()
        self.strategy_evaluator = StrategyEvaluator()
        self.error_analyzer = ErrorAnalyzer()
        self.insight_generator = InsightGenerator()
        
        # Initialize meta-cognition system
        self.strategy_manager = StrategyManager()
        self.strategy_selector = StrategySelector()
        self.strategy_adapter = StrategyAdapter()
        self.strategy_coordinator = StrategyCoordinator()
        
        # Initialize mental modeling system
        self.user_model = UserModel(user_id=f"user_{agent_id}")
        self.trader_model = TraderModel()
        self.market_model = MarketModel(market_id=f"market_{agent_id}")
        self.model_updater = ModelUpdater()
        
        # Initialize memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        self.advanced_retrieval = AdvancedRetrieval()
        self.memory_consolidation = MemoryConsolidation()
        self.cognitive_integration = CognitiveIntegration()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            procedural_memory=self.procedural_memory
        )
        
        # Initialize agent core components
        self.perception = Perception()
        self.cognition = Cognition()
        self.action = Action()
        
        # Register models with model updater
        self._register_models()
        
        # Register strategies with strategy manager
        self._register_strategies()
        
        # Connect components
        self._connect_components()
        
        self.logger.info(f"CognitiveController initialized for agent {agent_id}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the cognitive framework.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Dictionary containing processing results
        """
        timestamp = datetime.now()
        
        # Track performance
        self.performance_monitor.start_tracking(
            operation_id=f"process_{timestamp.timestamp()}",
            operation_type="process_input",
            metadata={"input_type": input_data.get("type")}
        )
        
        try:
            # 1. Perception: Process input through perception module
            perception_result = self.perception.process(input_data)
            
            # 2. Memory: Store and retrieve relevant memories
            memory_result = self._process_memory(perception_result)
            
            # 3. Mental Modeling: Update mental models
            modeling_result = self._update_mental_models(perception_result, memory_result)
            
            # 4. Meta-Cognition: Select and coordinate strategies
            strategy_result = self._select_strategies(perception_result, memory_result, modeling_result)
            
            # 5. Cognition: Process through cognition module
            cognition_input = {
                "perception": perception_result,
                "memory": memory_result,
                "modeling": modeling_result,
                "strategy": strategy_result
            }
            cognition_result = self.cognition.process(cognition_input)
            
            # 6. Action: Generate actions
            action_result = self.action.generate(cognition_result)
            
            # 7. Reflection: Evaluate performance and generate insights
            reflection_result = self._reflect_on_process(
                input_data, 
                perception_result,
                memory_result,
                modeling_result,
                strategy_result,
                cognition_result,
                action_result
            )
            
            # Combine results
            result = {
                "timestamp": timestamp,
                "input": input_data,
                "perception": perception_result,
                "memory": memory_result,
                "modeling": modeling_result,
                "strategy": strategy_result,
                "cognition": cognition_result,
                "action": action_result,
                "reflection": reflection_result,
                "status": "success"
            }
            
            # End performance tracking
            self.performance_monitor.end_tracking(
                operation_id=f"process_{timestamp.timestamp()}",
                status="success"
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            
            # Analyze error
            error_analysis = self.error_analyzer.analyze_error(
                error=str(e),
                context={
                    "input": input_data,
                    "timestamp": timestamp
                }
            )
            
            # End performance tracking
            self.performance_monitor.end_tracking(
                operation_id=f"process_{timestamp.timestamp()}",
                status="error",
                error_details=str(e)
            )
            
            return {
                "timestamp": timestamp,
                "input": input_data,
                "status": "error",
                "error": str(e),
                "error_analysis": error_analysis
            }
    
    def learn_from_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from feedback to improve future performance.
        
        Args:
            feedback_data: Feedback data to learn from
            
        Returns:
            Dictionary containing learning results
        """
        timestamp = datetime.now()
        
        # Extract relevant information
        feedback_type = feedback_data.get("type", "general")
        feedback_value = feedback_data.get("value")
        feedback_context = feedback_data.get("context", {})
        
        # Track performance
        self.performance_monitor.start_tracking(
            operation_id=f"learn_{timestamp.timestamp()}",
            operation_type="learn_from_feedback",
            metadata={"feedback_type": feedback_type}
        )
        
        try:
            # 1. Update user model with feedback
            if "user" in feedback_context:
                user_feedback = {
                    "feedback_type": feedback_type,
                    "feedback_value": feedback_value,
                    "context": feedback_context
                }
                self.user_model.record_feedback(
                    interaction_id=feedback_context.get("interaction_id", f"interaction_{timestamp.timestamp()}"),
                    feedback_type=feedback_type,
                    rating=feedback_data.get("rating"),
                    comments=feedback_data.get("comments")
                )
            
            # 2. Update strategy evaluations
            if "strategy" in feedback_context:
                strategy_id = feedback_context["strategy"].get("id")
                if strategy_id:
                    self.strategy_evaluator.evaluate_strategy(
                        strategy_id=strategy_id,
                        performance_data={
                            "success": feedback_value > 0 if isinstance(feedback_value, (int, float)) else True,
                            "feedback": feedback_value,
                            "context": feedback_context
                        }
                    )
                    
                    # Adapt strategy based on feedback
                    self.strategy_adapter.adapt_strategy(
                        strategy_id=strategy_id,
                        performance_data={
                            "success": feedback_value > 0 if isinstance(feedback_value, (int, float)) else True,
                            "feedback": feedback_value,
                            "context": feedback_context
                        }
                    )
            
            # 3. Store feedback in episodic memory
            memory_id = self.episodic_memory.store(
                memory_type="feedback",
                content=feedback_data,
                metadata={
                    "timestamp": timestamp,
                    "feedback_type": feedback_type
                }
            )
            
            # 4. Generate insights from feedback
            insights = self.insight_generator.generate_insights(
                data_source="feedback",
                data=feedback_data,
                context=feedback_context
            )
            
            # 5. Update procedural memory with learned procedures
            if insights and "procedural_updates" in insights:
                for procedure in insights["procedural_updates"]:
                    self.procedural_memory.store_procedure(
                        name=procedure["name"],
                        steps=procedure["steps"],
                        context=procedure["context"],
                        metadata={
                            "source": "feedback_learning",
                            "timestamp": timestamp
                        }
                    )
            
            # End performance tracking
            self.performance_monitor.end_tracking(
                operation_id=f"learn_{timestamp.timestamp()}",
                status="success"
            )
            
            return {
                "timestamp": timestamp,
                "feedback": feedback_data,
                "memory_id": memory_id,
                "insights": insights,
                "status": "success"
            }
        
        except Exception as e:
            self.logger.error(f"Error learning from feedback: {e}")
            
            # Analyze error
            error_analysis = self.error_analyzer.analyze_error(
                error=str(e),
                context={
                    "feedback": feedback_data,
                    "timestamp": timestamp
                }
            )
            
            # End performance tracking
            self.performance_monitor.end_tracking(
                operation_id=f"learn_{timestamp.timestamp()}",
                status="error",
                error_details=str(e)
            )
            
            return {
                "timestamp": timestamp,
                "feedback": feedback_data,
                "status": "error",
                "error": str(e),
                "error_analysis": error_analysis
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the cognitive framework.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.performance_monitor.get_metrics()
    
    def get_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent insights generated by the cognitive framework.
        
        Args:
            limit: Maximum number of insights to return
            
        Returns:
            List of insight records
        """
        return self.insight_generator.get_recent_insights(limit)
    
    def get_user_model(self) -> Dict[str, Any]:
        """
        Get the current user model.
        
        Returns:
            Dictionary representation of the user model
        """
        return self.user_model.to_dict()
    
    def get_market_model(self) -> Dict[str, Any]:
        """
        Get the current market model.
        
        Returns:
            Dictionary representation of the market model
        """
        return self.market_model.to_dict()
    
    def get_trader_model(self) -> Dict[str, Any]:
        """
        Get the current trader model.
        
        Returns:
            Dictionary representation of the trader model
        """
        return self.trader_model.to_dict()
    
    def get_active_strategies(self) -> List[Dict[str, Any]]:
        """
        Get currently active strategies.
        
        Returns:
            List of active strategy records
        """
        active_strategies = self.strategy_coordinator.get_active_strategies()
        
        result = []
        for strategy_id, info in active_strategies.items():
            strategy_info = self.strategy_manager.get_strategy(strategy_id)
            if strategy_info:
                result.append({
                    "id": strategy_id,
                    "name": strategy_info.get("name"),
                    "description": strategy_info.get("description"),
                    "activated_at": info.get("activated_at"),
                    "status": info.get("status")
                })
        
        return result
    
    def _process_memory(self, perception_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory operations based on perception results.
        
        Args:
            perception_result: Results from perception processing
            
        Returns:
            Dictionary containing memory processing results
        """
        # Store perception in episodic memory
        episodic_id = self.episodic_memory.store(
            memory_type="perception",
            content=perception_result,
            metadata={
                "timestamp": datetime.now(),
                "source": "perception"
            }
        )
        
        # Extract concepts for semantic memory
        concepts = perception_result.get("concepts", [])
        semantic_ids = []
        
        for concept in concepts:
            semantic_id = self.semantic_memory.store_concept(
                name=concept.get("name"),
                attributes=concept.get("attributes", {}),
                relationships=concept.get("relationships", [])
            )
            semantic_ids.append(semantic_id)
        
        # Retrieve relevant memories using advanced retrieval
        query = {
            "content": perception_result.get("content", {}),
            "context": perception_result.get("context", {})
        }
        
        retrieved_memories = self.advanced_retrieval.retrieve(
            query=query,
            memory_types=["episodic", "semantic", "procedural"],
            max_results=10
        )
        
        # Consolidate memories if needed
        consolidation_result = self.memory_consolidation.consolidate_memories(
            recent_memories=[episodic_id],
            retrieved_memories=retrieved_memories.get("memory_ids", [])
        )
        
        # Integrate with cognitive processes
        integration_result = self.cognitive_integration.integrate(
            perception=perception_result,
            memories=retrieved_memories,
            consolidation=consolidation_result
        )
        
        return {
            "episodic_id": episodic_id,
            "semantic_ids": semantic_ids,
            "retrieved_memories": retrieved_memories,
            "consolidation": consolidation_result,
            "integration": integration_result
        }
    
    def _update_mental_models(self, perception_result: Dict[str, Any], 
                            memory_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update mental models based on perception and memory results.
        
        Args:
            perception_result: Results from perception processing
            memory_result: Results from memory processing
            
        Returns:
            Dictionary containing mental modeling results
        """
        updates = {}
        
        # Update user model if user-related information is present
        if "user" in perception_result:
            user_data = perception_result["user"]
            
            if "interaction" in user_data:
                interaction_id = self.user_model.record_interaction(
                    interaction_type=user_data["interaction"].get("type", "general"),
                    content=user_data["interaction"].get("content", {}),
                    context=user_data["interaction"].get("context", {})
                )
                updates["user_interaction_id"] = interaction_id
            
            if "preference" in user_data:
                for category, preference in user_data["preference"].items():
                    self.user_model.update_preference(
                        category=category,
                        preference=preference.get("value"),
                        confidence=preference.get("confidence", 1.0),
                        source=preference.get("source", "implicit")
                    )
                updates["user_preferences_updated"] = list(user_data["preference"].keys())
        
        # Update market model if market-related information is present
        if "market" in perception_result:
            market_data = perception_result["market"]
            
            if "price" in market_data:
                self.market_model.add_price_data(
                    timestamp=market_data["price"].get("timestamp", datetime.now()),
                    price_data=market_data["price"].get("data", {}),
                    metadata=market_data["price"].get("metadata", {})
                )
                updates["market_price_updated"] = True
            
            if "volume" in market_data:
                self.market_model.add_volume_data(
                    timestamp=market_data["volume"].get("timestamp", datetime.now()),
                    volume=market_data["volume"].get("value", 0),
                    metadata=market_data["volume"].get("metadata", {})
                )
                updates["market_volume_updated"] = True
            
            if "sentiment" in market_data:
                self.market_model.add_sentiment_data(
                    timestamp=market_data["sentiment"].get("timestamp", datetime.now()),
                    sentiment=market_data["sentiment"].get("value", 0),
                    source=market_data["sentiment"].get("source", "unknown"),
                    confidence=market_data["sentiment"].get("confidence", 0.5),
                    metadata=market_data["sentiment"].get("metadata", {})
                )
                updates["market_sentiment_updated"] = True
            
            if "news" in market_data:
                event_id = self.market_model.add_news_event(
                    timestamp=market_data["news"].get("timestamp", datetime.now()),
                    event_data=market_data["news"].get("data", {}),
                    impact_score=market_data["news"].get("impact_score", 0.5),
                    sentiment=market_data["news"].get("sentiment")
                )
                updates["market_news_event_id"] = event_id
        
        # Update trader model if trader-related information is present
        if "trader" in perception_result:
            trader_data = perception_result["trader"]
            
            if "profile" in trader_data:
                for trader_id, profile in trader_data["profile"].items():
                    if trader_id in self.trader_model.trader_profiles:
                        self.trader_model.update_trader_profile(
                            trader_id=trader_id,
                            profile_updates=profile
                        )
                    else:
                        self.trader_model.add_trader_profile(
                            trader_id=trader_id,
                            profile_data=profile
                        )
                updates["trader_profiles_updated"] = list(trader_data["profile"].keys())
            
            if "interaction" in trader_data:
                for trader_id, interaction in trader_data["interaction"].items():
                    interaction_id = self.trader_model.record_market_interaction(
                        trader_id=trader_id,
                        interaction_data=interaction
                    )
                    if "trader_interactions" not in updates:
                        updates["trader_interactions"] = {}
                    updates["trader_interactions"][trader_id] = interaction_id
            
            if "position" in trader_data:
                for trader_id, position in trader_data["position"].items():
                    position_id = self.trader_model.record_trader_position(
                        trader_id=trader_id,
                        position_data=position
                    )
                    if "trader_positions" not in updates:
                        updates["trader_positions"] = {}
                    updates["trader_positions"][trader_id] = position_id
            
            if "sentiment" in trader_data:
                for trader_id, sentiment in trader_data["sentiment"].items():
                    self.trader_model.record_trader_sentiment(
                        trader_id=trader_id,
                        sentiment=sentiment.get("value", 0),
                        confidence=sentiment.get("confidence", 0.5),
                        context=sentiment.get("context", {})
                    )
                updates["trader_sentiments_updated"] = list(trader_data["sentiment"].keys())
        
        # Coordinate model updates using model updater
        update_result = self.model_updater.enforce_consistency()
        updates["consistency_check"] = update_result
        
        return updates
    
    def _select_strategies(self, perception_result: Dict[str, Any],
                         memory_result: Dict[str, Any],
                         modeling_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select and coordinate strategies based on current context.
        
        Args:
            perception_result: Results from perception processing
            memory_result: Results from memory processing
            modeling_result: Results from mental modeling
            
        Returns:
            Dictionary containing strategy selection results
        """
        # Prepare context for strategy selection
        context = {
            "perception": perception_result,
            "memory": memory_result,
            "modeling": modeling_result,
            "timestamp": datetime.now()
        }
        
        # Get available strategies
        available_strategies = self.strategy_manager.get_available_strategies()
        
        # Select strategy
        selected_strategy, confidence = self.strategy_selector.select_strategy(
            context=context,
            available_strategies=available_strategies
        )
        
        # Get strategy parameters
        strategy_params = self.strategy_adapter.get_strategy_parameters(
            strategy_id=selected_strategy,
            context=context
        )
        
        # Activate strategy
        self.strategy_coordinator.activate_strategy(selected_strategy)
        
        # Coordinate execution of active strategies
        strategy_functions = self.strategy_manager.get_strategy_functions()
        execution_results = self.strategy_coordinator.coordinate_execution(
            context=context,
            strategy_functions=strategy_functions
        )
        
        return {
            "selected_strategy": selected_strategy,
            "confidence": confidence,
            "parameters": strategy_params,
            "execution_results": execution_results
        }
    
    def _reflect_on_process(self, input_data: Dict[str, Any],
                          perception_result: Dict[str, Any],
                          memory_result: Dict[str, Any],
                          modeling_result: Dict[str, Any],
                          strategy_result: Dict[str, Any],
                          cognition_result: Dict[str, Any],
                          action_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the processing results to generate insights and improvements.
        
        Args:
            input_data: Original input data
            perception_result: Results from perception processing
            memory_result: Results from memory processing
            modeling_result: Results from mental modeling
            strategy_result: Results from strategy selection
            cognition_result: Results from cognition processing
            action_result: Results from action generation
            
        Returns:
            Dictionary containing reflection results
        """
        # Evaluate strategy performance
        strategy_id = strategy_result.get("selected_strategy")
        if strategy_id:
            evaluation = self.strategy_evaluator.evaluate_strategy(
                strategy_id=strategy_id,
                performance_data={
                    "context": {
                        "input": input_data,
                        "perception": perception_result,
                        "memory": memory_result,
                        "modeling": modeling_result
                    },
                    "output": {
                        "cognition": cognition_result,
                        "action": action_result
                    },
                    "success": action_result.get("success", True),
                    "metrics": action_result.get("metrics", {})
                }
            )
        else:
            evaluation = None
        
        # Generate insights
        insights = self.insight_generator.generate_insights(
            data_source="process",
            data={
                "input": input_data,
                "perception": perception_result,
                "memory": memory_result,
                "modeling": modeling_result,
                "strategy": strategy_result,
                "cognition": cognition_result,
                "action": action_result
            },
            context={}
        )
        
        # Check for errors
        errors = []
        if not action_result.get("success", True):
            error_analysis = self.error_analyzer.analyze_error(
                error=action_result.get("error", "Unknown error"),
                context={
                    "input": input_data,
                    "perception": perception_result,
                    "memory": memory_result,
                    "modeling": modeling_result,
                    "strategy": strategy_result,
                    "cognition": cognition_result
                }
            )
            errors.append(error_analysis)
        
        return {
            "strategy_evaluation": evaluation,
            "insights": insights,
            "errors": errors
        }
    
    def _register_models(self) -> None:
        """Register mental models with the model updater."""
        # Register user model
        self.model_updater.register_model(
            model_id="user_model",
            model_instance=self.user_model,
            model_type="user",
            priority=1.0
        )
        
        # Register trader model
        self.model_updater.register_model(
            model_id="trader_model",
            model_instance=self.trader_model,
            model_type="trader",
            priority=0.8
        )
        
        # Register market model
        self.model_updater.register_model(
            model_id="market_model",
            model_instance=self.market_model,
            model_type="market",
            priority=0.9
        )
        
        # Set up dependencies
        self.model_updater.add_dependency("trader_model", "market_model")
    
    def _register_strategies(self) -> None:
        """Register strategies with the strategy manager."""
        # Example strategies - in a real implementation, these would be more sophisticated
        self.strategy_manager.register_strategy(
            strategy_id="default_processing",
            name="Default Processing Strategy",
            description="Standard processing strategy for general inputs",
            function=lambda context: {"result": "default_processing", "context": context},
            metadata={"default": True}
        )
        
        self.strategy_manager.register_strategy(
            strategy_id="market_analysis",
            name="Market Analysis Strategy",
            description="Strategy for analyzing market data and trends",
            function=lambda context: {"result": "market_analysis", "context": context},
            metadata={"market_focused": True}
        )
        
        self.strategy_manager.register_strategy(
            strategy_id="user_preference",
            name="User Preference Strategy",
            description="Strategy that prioritizes user preferences",
            function=lambda context: {"result": "user_preference", "context": context},
            metadata={"user_focused": True}
        )
    
    def _connect_components(self) -> None:
        """Connect various components of the cognitive framework."""
        # Connect memory systems to advanced retrieval
        self.advanced_retrieval.connect_memory_systems(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            procedural_memory=self.procedural_memory
        )
        
        # Connect memory consolidation to memory systems
        self.memory_consolidation.connect_memory_systems(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            procedural_memory=self.procedural_memory
        )
        
        # Connect cognitive integration to memory manager
        self.cognitive_integration.connect_memory_manager(self.memory_manager)
        
        # Connect performance monitor to strategy evaluator
        self.strategy_evaluator.connect_performance_monitor(self.performance_monitor)
        
        # Connect error analyzer to insight generator
        self.insight_generator.connect_error_analyzer(self.error_analyzer)
        
        # Connect strategy manager to strategy coordinator
        self.strategy_coordinator.connect_strategy_manager(self.strategy_manager)
