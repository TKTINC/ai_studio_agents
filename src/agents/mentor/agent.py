"""
MENTOR-specific agent implementation.

This module contains the MentorAgent class that extends the BaseAgent
with MENTOR-specific functionality for personalized investment mentoring.
"""

from typing import Optional

from ...agent_core.agent import BaseAgent
from ...agent_core.config import AgentConfig, load_config
from .perception.perception import MentorPerceptionModule
from .cognition.cognition import MentorCognitionModule
from .action.action import MentorActionModule


class MentorAgent(BaseAgent):
    """
    MENTOR Agent implementation.
    
    Specializes the BaseAgent for personalized investment mentoring,
    learning user philosophy and adapting strategies to individual preferences.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the MENTOR Agent.
        
        Args:
            config: Agent configuration (loads from environment if None)
        """
        # Load MENTOR-specific configuration
        mentor_config = config or load_config(agent_type="mentor")
        
        # Initialize with MENTOR-specific modules
        super().__init__(
            config=mentor_config,
            perception_class=MentorPerceptionModule,
            cognition_class=MentorCognitionModule,
            action_class=MentorActionModule
        )
