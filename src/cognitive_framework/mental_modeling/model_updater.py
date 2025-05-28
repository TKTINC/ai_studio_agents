"""
Model Updater Module for TAAT Cognitive Framework.

This module implements model updating capabilities for coordinating
updates across different mental models and ensuring consistency.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class ModelUpdater:
    """
    Model Updater for TAAT Cognitive Framework.
    
    Coordinates updates across different mental models, ensuring consistency
    and proper propagation of information between models.
    """
    
    def __init__(self):
        """Initialize the model updater."""
        self.registered_models = {}
        self.model_dependencies = {}
        self.update_history = []
        self.logger = logging.getLogger("ModelUpdater")
    
    def register_model(self,
                      model_id: str,
                      model_instance: Any,
                      model_type: str,
                      priority: float = 0.5) -> None:
        """
        Register a model with the updater.
        
        Args:
            model_id: Unique identifier for the model
            model_instance: Instance of the model
            model_type: Type of the model
            priority: Priority of the model (0.0 to 1.0)
        """
        self.registered_models[model_id] = {
            "instance": model_instance,
            "type": model_type,
            "priority": priority,
            "registered_at": datetime.now()
        }
        
        # Initialize dependencies
        if model_id not in self.model_dependencies:
            self.model_dependencies[model_id] = []
        
        self.logger.info(f"Registered model {model_id} of type {model_type}")
    
    def add_dependency(self, dependent_model_id: str, dependency_model_id: str) -> bool:
        """
        Add a dependency between models.
        
        Args:
            dependent_model_id: ID of the dependent model
            dependency_model_id: ID of the dependency model
            
        Returns:
            True if successful, False if models not found
        """
        if (dependent_model_id not in self.registered_models or
                dependency_model_id not in self.registered_models):
            return False
        
        # Initialize dependencies if needed
        if dependent_model_id not in self.model_dependencies:
            self.model_dependencies[dependent_model_id] = []
        
        # Add dependency if not already present
        if dependency_model_id not in self.model_dependencies[dependent_model_id]:
            self.model_dependencies[dependent_model_id].append(dependency_model_id)
        
        self.logger.info(f"Added dependency: {dependent_model_id} depends on {dependency_model_id}")
        
        return True
    
    def enforce_consistency(self) -> Dict[str, Any]:
        """
        Enforce consistency across all registered models.
        
        Returns:
            Results of consistency enforcement
        """
        timestamp = datetime.now()
        
        # Initialize results
        results = {
            "timestamp": timestamp,
            "models_checked": 0,
            "inconsistencies_found": 0,
            "updates_applied": 0,
            "model_results": {}
        }
        
        # Check each model
        for model_id, model_info in self.registered_models.items():
            model_instance = model_info["instance"]
            model_type = model_info["type"]
            
            # Get dependencies
            dependencies = self.model_dependencies.get(model_id, [])
            
            # Check for inconsistencies
            inconsistencies = self._check_model_consistency(model_id, model_instance, model_type, dependencies)
            
            # Apply updates if inconsistencies found
            updates_applied = 0
            if inconsistencies:
                updates_applied = self._apply_consistency_updates(model_id, model_instance, inconsistencies)
            
            # Record results
            results["models_checked"] += 1
            results["inconsistencies_found"] += len(inconsistencies)
            results["updates_applied"] += updates_applied
            
            results["model_results"][model_id] = {
                "type": model_type,
                "inconsistencies_found": len(inconsistencies),
                "updates_applied": updates_applied
            }
        
        self.logger.info(f"Enforced consistency across {results['models_checked']} models, " +
                        f"found {results['inconsistencies_found']} inconsistencies, " +
                        f"applied {results['updates_applied']} updates")
        
        return results
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model information or None if not found
        """
        if model_id not in self.registered_models:
            return None
        
        model_info = {
            "type": self.registered_models[model_id]["type"],
            "priority": self.registered_models[model_id]["priority"],
            "registered_at": self.registered_models[model_id]["registered_at"],
            "dependencies": self.model_dependencies.get(model_id, [])
        }
        
        return model_info
    
    def get_registered_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered models.
        
        Returns:
            Dictionary of registered models (without instances)
        """
        # Return info without model instances
        return {
            model_id: {
                "type": info["type"],
                "priority": info["priority"],
                "registered_at": info["registered_at"]
            }
            for model_id, info in self.registered_models.items()
        }
    
    def get_model_dependencies(self, model_id: str) -> List[str]:
        """
        Get dependencies for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List of dependency model IDs
        """
        if model_id not in self.model_dependencies:
            return []
        
        return self.model_dependencies[model_id]
    
    def get_dependent_models(self, model_id: str) -> List[str]:
        """
        Get models that depend on a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List of dependent model IDs
        """
        dependent_models = []
        
        for dependent_id, dependencies in self.model_dependencies.items():
            if model_id in dependencies:
                dependent_models.append(dependent_id)
        
        return dependent_models
    
    def update_model(self,
                    model_id: str,
                    update_data: Dict[str, Any],
                    propagate: bool = True) -> Dict[str, Any]:
        """
        Update a model and propagate changes to dependent models.
        
        Args:
            model_id: ID of the model to update
            update_data: Data for the update
            propagate: Whether to propagate updates to dependent models
            
        Returns:
            Update results
        """
        if model_id not in self.registered_models:
            return {"error": "Model not found"}
        
        model_info = self.registered_models[model_id]
        model_instance = model_info["instance"]
        model_type = model_info["type"]
        
        # Perform update based on model type
        update_result = self._update_specific_model(model_instance, model_type, update_data)
        
        # Record update
        update_record = {
            "timestamp": datetime.now(),
            "model_id": model_id,
            "model_type": model_type,
            "update_data": update_data,
            "result": update_result
        }
        
        self.update_history.append(update_record)
        
        # Limit history size
        max_history = 100
        if len(self.update_history) > max_history:
            self.update_history = self.update_history[-max_history:]
        
        self.logger.info(f"Updated model {model_id} of type {model_type}")
        
        # Propagate updates to dependent models if requested
        if propagate:
            dependent_models = self.get_dependent_models(model_id)
            propagation_results = {}
            
            for dependent_id in dependent_models:
                # Prepare propagated data
                propagated_data = self._prepare_propagated_data(
                    source_model_id=model_id,
                    source_model_type=model_type,
                    target_model_id=dependent_id,
                    target_model_type=self.registered_models[dependent_id]["type"],
                    update_data=update_data,
                    update_result=update_result
                )
                
                # Update dependent model
                propagation_result = self.update_model(
                    model_id=dependent_id,
                    update_data=propagated_data,
                    propagate=False  # Avoid circular propagation
                )
                
                propagation_results[dependent_id] = propagation_result
            
            update_result["propagation_results"] = propagation_results
        
        return update_result
    
    def get_update_history(self, 
                         model_id: Optional[str] = None, 
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get update history.
        
        Args:
            model_id: Optional model ID to filter by
            limit: Maximum number of history entries to return
            
        Returns:
            List of update history entries
        """
        if model_id:
            filtered_history = [
                update for update in self.update_history
                if update["model_id"] == model_id
            ]
            
            # Sort by timestamp
            filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return filtered_history[:limit]
        else:
            # Sort by timestamp
            sorted_history = sorted(
                self.update_history,
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            return sorted_history[:limit]
    
    def _check_model_consistency(self,
                               model_id: str,
                               model_instance: Any,
                               model_type: str,
                               dependencies: List[str]) -> List[Dict[str, Any]]:
        """
        Check model consistency with its dependencies.
        
        Args:
            model_id: ID of the model
            model_instance: Instance of the model
            model_type: Type of the model
            dependencies: List of dependency model IDs
            
        Returns:
            List of inconsistencies
        """
        inconsistencies = []
        
        # Skip if no dependencies
        if not dependencies:
            return inconsistencies
        
        # Check each dependency
        for dependency_id in dependencies:
            if dependency_id not in self.registered_models:
                continue
            
            dependency_info = self.registered_models[dependency_id]
            dependency_instance = dependency_info["instance"]
            dependency_type = dependency_info["type"]
            
            # Check consistency based on model types
            if model_type == "user" and dependency_type == "trader":
                # Check user-trader consistency
                inconsistency = self._check_user_trader_consistency(
                    user_model=model_instance,
                    trader_model=dependency_instance
                )
                
                if inconsistency:
                    inconsistency["dependency_id"] = dependency_id
                    inconsistencies.append(inconsistency)
            
            elif model_type == "trader" and dependency_type == "market":
                # Check trader-market consistency
                inconsistency = self._check_trader_market_consistency(
                    trader_model=model_instance,
                    market_model=dependency_instance
                )
                
                if inconsistency:
                    inconsistency["dependency_id"] = dependency_id
                    inconsistencies.append(inconsistency)
            
            elif model_type == "market" and dependency_type == "trader":
                # Check market-trader consistency
                inconsistency = self._check_market_trader_consistency(
                    market_model=model_instance,
                    trader_model=dependency_instance
                )
                
                if inconsistency:
                    inconsistency["dependency_id"] = dependency_id
                    inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _apply_consistency_updates(self,
                                 model_id: str,
                                 model_instance: Any,
                                 inconsistencies: List[Dict[str, Any]]) -> int:
        """
        Apply updates to resolve inconsistencies.
        
        Args:
            model_id: ID of the model
            model_instance: Instance of the model
            inconsistencies: List of inconsistencies
            
        Returns:
            Number of updates applied
        """
        updates_applied = 0
        
        for inconsistency in inconsistencies:
            # Get dependency
            dependency_id = inconsistency.get("dependency_id")
            if not dependency_id or dependency_id not in self.registered_models:
                continue
            
            dependency_info = self.registered_models[dependency_id]
            dependency_instance = dependency_info["instance"]
            
            # Apply update based on inconsistency type
            if inconsistency["type"] == "user_trader_inconsistency":
                # Update trader model with user preferences
                if hasattr(dependency_instance, "update_trader_profile"):
                    dependency_instance.update_trader_profile(
                        trader_id=model_id,
                        profile_data={
                            "user_preferences": inconsistency["user_preferences"]
                        }
                    )
                    updates_applied += 1
            
            elif inconsistency["type"] == "trader_market_inconsistency":
                # Update market model with trader data
                if hasattr(dependency_instance, "add_market_event"):
                    dependency_instance.add_market_event(
                        timestamp=datetime.now(),
                        event_type="trader_update",
                        event_data={
                            "trader_id": model_id,
                            "trader_profile": inconsistency["trader_profile"]
                        }
                    )
                    updates_applied += 1
            
            elif inconsistency["type"] == "market_trader_inconsistency":
                # Update trader model with market data
                if hasattr(dependency_instance, "update_market_context"):
                    dependency_instance.update_market_context(
                        market_id=model_id,
                        market_data=inconsistency["market_data"]
                    )
                    updates_applied += 1
        
        return updates_applied
    
    def _check_user_trader_consistency(self,
                                     user_model: Any,
                                     trader_model: Any) -> Optional[Dict[str, Any]]:
        """
        Check consistency between user and trader models.
        
        Args:
            user_model: User model instance
            trader_model: Trader model instance
            
        Returns:
            Inconsistency or None if consistent
        """
        # Get user preferences
        if not hasattr(user_model, "get_preferences"):
            return None
        
        user_preferences = user_model.get_preferences()
        
        # Get trader profile
        if not hasattr(trader_model, "get_trader_profile"):
            return None
        
        trader_profile = trader_model.get_trader_profile(user_model.user_id)
        
        if not trader_profile:
            return {
                "type": "user_trader_inconsistency",
                "description": "Trader profile missing for user",
                "user_preferences": user_preferences
            }
        
        # Check if trader profile has user preferences
        if "user_preferences" not in trader_profile:
            return {
                "type": "user_trader_inconsistency",
                "description": "Trader profile missing user preferences",
                "user_preferences": user_preferences
            }
        
        # Check if preferences are up to date
        profile_preferences = trader_profile["user_preferences"]
        
        # Compare preference categories
        for category, preferences in user_preferences.items():
            if not preferences:
                continue
            
            # Get latest user preference
            latest_preference = preferences[-1]
            
            # Check if category exists in profile
            if category not in profile_preferences:
                return {
                    "type": "user_trader_inconsistency",
                    "description": f"Trader profile missing preference category: {category}",
                    "user_preferences": user_preferences
                }
            
            # Check if preference is up to date
            profile_preference = profile_preferences[category]
            
            if isinstance(profile_preference, list) and profile_preference:
                profile_latest = profile_preference[-1]
                
                # Compare values
                if profile_latest.get("value") != latest_preference.get("value"):
                    return {
                        "type": "user_trader_inconsistency",
                        "description": f"Trader profile has outdated preference for category: {category}",
                        "user_preferences": user_preferences
                    }
            else:
                return {
                    "type": "user_trader_inconsistency",
                    "description": f"Trader profile has invalid preference format for category: {category}",
                    "user_preferences": user_preferences
                }
        
        return None
    
    def _check_trader_market_consistency(self,
                                       trader_model: Any,
                                       market_model: Any) -> Optional[Dict[str, Any]]:
        """
        Check consistency between trader and market models.
        
        Args:
            trader_model: Trader model instance
            market_model: Market model instance
            
        Returns:
            Inconsistency or None if consistent
        """
        # Get trader profile
        if not hasattr(trader_model, "get_trader_profile"):
            return None
        
        trader_id = getattr(trader_model, "trader_id", None)
        if not trader_id:
            return None
        
        trader_profile = trader_model.get_trader_profile(trader_id)
        
        if not trader_profile:
            return None
        
        # Check if market has trader events
        if not hasattr(market_model, "get_trader_events"):
            return None
        
        trader_events = market_model.get_trader_events(trader_id, limit=1)
        
        if not trader_events:
            return {
                "type": "trader_market_inconsistency",
                "description": "Market missing trader events",
                "trader_profile": trader_profile
            }
        
        return None
    
    def _check_market_trader_consistency(self,
                                       market_model: Any,
                                       trader_model: Any) -> Optional[Dict[str, Any]]:
        """
        Check consistency between market and trader models.
        
        Args:
            market_model: Market model instance
            trader_model: Trader model instance
            
        Returns:
            Inconsistency or None if consistent
        """
        # Get market data
        if not hasattr(market_model, "get_market_summary"):
            return None
        
        market_id = getattr(market_model, "market_id", None)
        if not market_id:
            return None
        
        market_summary = market_model.get_market_summary()
        
        if not market_summary:
            return None
        
        # Check if trader has market context
        if not hasattr(trader_model, "get_market_context"):
            return None
        
        market_context = trader_model.get_market_context(market_id)
        
        if not market_context:
            return {
                "type": "market_trader_inconsistency",
                "description": "Trader missing market context",
                "market_data": market_summary
            }
        
        # Check if market context is up to date
        if "last_updated" in market_context and "last_updated" in market_summary:
            trader_last_updated = market_context["last_updated"]
            market_last_updated = market_summary["last_updated"]
            
            if trader_last_updated < market_last_updated:
                return {
                    "type": "market_trader_inconsistency",
                    "description": "Trader has outdated market context",
                    "market_data": market_summary
                }
        
        return None
    
    def _update_specific_model(self, 
                             model_instance: Any, 
                             model_type: str, 
                             update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a specific model based on its type.
        
        Args:
            model_instance: Instance of the model
            model_type: Type of the model
            update_data: Data for the update
            
        Returns:
            Update result
        """
        result = {"status": "success"}
        
        try:
            if model_type == "user":
                # Update user model
                if "preferences" in update_data:
                    for category, preference_data in update_data["preferences"].items():
                        preference = preference_data.get("value")
                        confidence = preference_data.get("confidence", 0.5)
                        source = preference_data.get("source", "system")
                        
                        model_instance.update_preference(
                            category=category,
                            preference=preference,
                            confidence=confidence,
                            source=source
                        )
                
                if "profile" in update_data:
                    model_instance.update_profile(update_data["profile"])
                
                if "interaction" in update_data:
                    interaction_data = update_data["interaction"]
                    interaction_id = model_instance.record_interaction(
                        interaction_type=interaction_data.get("type", "unknown"),
                        content=interaction_data.get("content", {}),
                        context=interaction_data.get("context")
                    )
                    result["interaction_id"] = interaction_id
            
            elif model_type == "trader":
                # Update trader model
                if "profile" in update_data:
                    profile_data = update_data["profile"]
                    trader_id = profile_data.get("trader_id")
                    
                    if trader_id:
                        if model_instance.get_trader_profile(trader_id):
                            model_instance.update_trader_profile(
                                trader_id=trader_id,
                                profile_data=profile_data
                            )
                        else:
                            model_instance.add_trader_profile(
                                trader_id=trader_id,
                                profile_data=profile_data,
                                trader_type=profile_data.get("trader_type", "unknown")
                            )
                
                if "interaction" in update_data:
                    interaction_data = update_data["interaction"]
                    trader_id = interaction_data.get("trader_id")
                    
                    if trader_id:
                        interaction_id = model_instance.record_market_interaction(
                            trader_id=trader_id,
                            interaction_data=interaction_data
                        )
                        result["interaction_id"] = interaction_id
            
            elif model_type == "market":
                # Update market model
                if "price" in update_data:
                    price_data = update_data["price"]
                    model_instance.add_price_data(
                        timestamp=price_data.get("timestamp", datetime.now()),
                        price_data=price_data.get("data", {}),
                        metadata=price_data.get("metadata")
                    )
                
                if "volume" in update_data:
                    volume_data = update_data["volume"]
                    model_instance.add_volume_data(
                        timestamp=volume_data.get("timestamp", datetime.now()),
                        volume=volume_data.get("volume", 0.0),
                        metadata=volume_data.get("metadata")
                    )
                
                if "sentiment" in update_data:
                    sentiment_data = update_data["sentiment"]
                    model_instance.add_sentiment_data(
                        timestamp=sentiment_data.get("timestamp", datetime.now()),
                        sentiment=sentiment_data.get("sentiment", 0.0),
                        source=sentiment_data.get("source", "unknown"),
                        confidence=sentiment_data.get("confidence", 0.5)
                    )
                
                if "event" in update_data:
                    event_data = update_data["event"]
                    model_instance.add_market_event(
                        timestamp=event_data.get("timestamp", datetime.now()),
                        event_type=event_data.get("type", "unknown"),
                        event_data=event_data.get("data", {})
                    )
            
            else:
                # Generic update for unknown model types
                result["warning"] = f"Unknown model type: {model_type}, performing generic update"
                
                # Try to call update method if it exists
                if hasattr(model_instance, "update") and callable(getattr(model_instance, "update")):
                    update_result = model_instance.update(update_data)
                    result["generic_result"] = update_result
        
        except Exception as e:
            result = {
                "status": "error",
                "error": str(e)
            }
        
        return result
    
    def _prepare_propagated_data(self,
                               source_model_id: str,
                               source_model_type: str,
                               target_model_id: str,
                               target_model_type: str,
                               update_data: Dict[str, Any],
                               update_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data to propagate from one model to another.
        
        Args:
            source_model_id: ID of the source model
            source_model_type: Type of the source model
            target_model_id: ID of the target model
            target_model_type: Type of the target model
            update_data: Original update data
            update_result: Result of the source model update
            
        Returns:
            Prepared data for the target model
        """
        propagated_data = {
            "source_model": source_model_id,
            "source_type": source_model_type,
            "propagated": True,
            "timestamp": datetime.now()
        }
        
        # Prepare data based on source and target model types
        if source_model_type == "user" and target_model_type == "trader":
            # Propagate user preferences to trader profile
            if "preferences" in update_data:
                propagated_data["profile"] = {
                    "trader_id": source_model_id,  # Use user ID as trader ID
                    "user_preferences": update_data["preferences"]
                }
        
        elif source_model_type == "trader" and target_model_type == "market":
            # Propagate trader interaction to market event
            if "interaction" in update_data:
                interaction = update_data["interaction"]
                
                propagated_data["event"] = {
                    "type": "trader_action",
                    "data": {
                        "trader_id": interaction.get("trader_id"),
                        "action": interaction.get("type"),
                        "details": interaction
                    }
                }
        
        elif source_model_type == "market" and target_model_type == "trader":
            # Propagate market data to trader context
            if "price" in update_data or "event" in update_data:
                propagated_data["market_update"] = {
                    "market_id": source_model_id,
                    "price_data": update_data.get("price", {}).get("data", {}),
                    "event_data": update_data.get("event", {})
                }
        
        return propagated_data
