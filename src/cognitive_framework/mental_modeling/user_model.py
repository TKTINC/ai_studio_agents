"""
User Model Module for TAAT Cognitive Framework.

This module implements user modeling capabilities for understanding
and predicting user behavior, preferences, and needs.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

class UserModel:
    """
    User Model for TAAT Cognitive Framework.
    
    Maintains a model of the user, including preferences, behavior patterns,
    interaction history, and other relevant attributes.
    """
    
    def __init__(self, user_id: Optional[str] = None):
        """
        Initialize the user model.
        
        Args:
            user_id: Optional unique identifier for the user
        """
        self.user_id = user_id
        self.profile = {}
        self.preferences = {}
        self.interaction_history = []
        self.behavior_patterns = {}
        self.last_updated = datetime.now()
        self.logger = logging.getLogger("UserModel")
    
    def update_profile(self, profile_data: Dict[str, Any]) -> None:
        """
        Update user profile.
        
        Args:
            profile_data: Profile data to update
        """
        # Update profile
        for key, value in profile_data.items():
            self.profile[key] = value
        
        # Update timestamp
        self.last_updated = datetime.now()
        
        self.logger.info(f"Updated profile for user {self.user_id}")
    
    def update_preference(self,
                         category: str,
                         preference: Any,
                         confidence: float = 0.5,
                         source: str = "explicit") -> None:
        """
        Update user preference.
        
        Args:
            category: Preference category
            preference: Preference value
            confidence: Confidence in the preference (0.0 to 1.0)
            source: Source of the preference (e.g., 'explicit', 'implicit', 'inferred')
        """
        # Initialize category if needed
        if category not in self.preferences:
            self.preferences[category] = []
        
        # Add preference
        preference_entry = {
            "value": preference,
            "confidence": confidence,
            "source": source,
            "timestamp": datetime.now()
        }
        
        # Add to preferences
        self.preferences[category].append(preference_entry)
        
        # Limit history size
        max_history = 10
        if len(self.preferences[category]) > max_history:
            self.preferences[category] = self.preferences[category][-max_history:]
        
        # Update timestamp
        self.last_updated = datetime.now()
        
        self.logger.info(f"Updated {category} preference for user {self.user_id}")
    
    def get_preference(self, category: str) -> Tuple[Any, float]:
        """
        Get user preference.
        
        Args:
            category: Preference category
            
        Returns:
            Tuple of (preference_value, confidence)
        """
        if category not in self.preferences or not self.preferences[category]:
            return None, 0.0
        
        # Return most recent preference
        latest_preference = self.preferences[category][-1]
        return latest_preference["value"], latest_preference["confidence"]
    
    def get_preferences(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all user preferences.
        
        Returns:
            Dictionary of preference categories and values
        """
        return self.preferences
    
    def record_interaction(self,
                          interaction_type: str,
                          content: Dict[str, Any],
                          context: Optional[Dict[str, Any]] = None) -> str:
        """
        Record user interaction.
        
        Args:
            interaction_type: Type of interaction
            content: Interaction content
            context: Optional interaction context
            
        Returns:
            Interaction ID
        """
        # Generate interaction ID
        interaction_id = f"interaction_{len(self.interaction_history) + 1}_{datetime.now().timestamp()}"
        
        # Create interaction record
        interaction = {
            "interaction_id": interaction_id,
            "type": interaction_type,
            "content": content,
            "context": context,
            "timestamp": datetime.now()
        }
        
        # Add to history
        self.interaction_history.append(interaction)
        
        # Limit history size
        max_history = 100
        if len(self.interaction_history) > max_history:
            self.interaction_history = self.interaction_history[-max_history:]
        
        # Update timestamp
        self.last_updated = datetime.now()
        
        # Update behavior patterns
        self._update_behavior_patterns(interaction)
        
        self.logger.info(f"Recorded {interaction_type} interaction for user {self.user_id}")
        
        return interaction_id
    
    def get_recent_interactions(self, 
                              interaction_type: Optional[str] = None, 
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent user interactions.
        
        Args:
            interaction_type: Optional interaction type to filter by
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent interaction records
        """
        if interaction_type:
            filtered_history = [
                interaction for interaction in self.interaction_history
                if interaction["type"] == interaction_type
            ]
            
            # Sort by timestamp
            filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return filtered_history[:limit]
        else:
            # Sort by timestamp
            sorted_history = sorted(
                self.interaction_history,
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            return sorted_history[:limit]
    
    def get_interaction_history(self, 
                              interaction_type: Optional[str] = None, 
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get interaction history.
        
        Args:
            interaction_type: Optional interaction type to filter by
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction records
        """
        return self.get_recent_interactions(interaction_type, limit)
    
    def get_behavior_patterns(self) -> Dict[str, Any]:
        """
        Get identified behavior patterns.
        
        Returns:
            Dictionary of behavior patterns
        """
        return self.behavior_patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert user model to dictionary.
        
        Returns:
            Dictionary representation of the user model
        """
        return {
            "user_id": self.user_id,
            "profile": self.profile,
            "preferences": self.preferences,
            "interaction_count": len(self.interaction_history),
            "behavior_patterns": self.behavior_patterns,
            "last_updated": self.last_updated
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update user model from dictionary.
        
        Args:
            data: Dictionary representation of the user model
        """
        if "user_id" in data:
            self.user_id = data["user_id"]
        
        if "profile" in data:
            self.profile = data["profile"]
        
        if "preferences" in data:
            self.preferences = data["preferences"]
        
        if "behavior_patterns" in data:
            self.behavior_patterns = data["behavior_patterns"]
        
        # Update timestamp
        self.last_updated = datetime.now()
        
        self.logger.info(f"Updated user model for user {self.user_id} from dictionary")
    
    def _update_behavior_patterns(self, interaction: Dict[str, Any]) -> None:
        """
        Update behavior patterns based on interaction.
        
        Args:
            interaction: Interaction record
        """
        interaction_type = interaction["type"]
        
        # Initialize pattern if needed
        if interaction_type not in self.behavior_patterns:
            self.behavior_patterns[interaction_type] = {
                "count": 0,
                "last_seen": None,
                "frequency": None,
                "contexts": {}
            }
        
        # Update pattern
        pattern = self.behavior_patterns[interaction_type]
        pattern["count"] += 1
        
        # Update last seen
        current_time = datetime.now()
        last_seen = pattern["last_seen"]
        pattern["last_seen"] = current_time
        
        # Update frequency if possible
        if last_seen:
            time_diff = (current_time - last_seen).total_seconds()
            
            if "intervals" not in pattern:
                pattern["intervals"] = []
            
            pattern["intervals"].append(time_diff)
            
            # Limit intervals
            max_intervals = 10
            if len(pattern["intervals"]) > max_intervals:
                pattern["intervals"] = pattern["intervals"][-max_intervals:]
            
            # Calculate average interval
            avg_interval = sum(pattern["intervals"]) / len(pattern["intervals"])
            pattern["frequency"] = avg_interval
        
        # Update context patterns
        if "context" in interaction and interaction["context"]:
            context = interaction["context"]
            
            # Extract context type
            context_type = context.get("type", "unknown")
            
            # Initialize context type if needed
            if context_type not in pattern["contexts"]:
                pattern["contexts"][context_type] = {
                    "count": 0,
                    "last_seen": None
                }
            
            # Update context pattern
            context_pattern = pattern["contexts"][context_type]
            context_pattern["count"] += 1
            context_pattern["last_seen"] = current_time
