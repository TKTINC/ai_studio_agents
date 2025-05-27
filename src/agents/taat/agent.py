"""
TAAT-specific agent implementation.

This module contains the TaatAgent class that extends the BaseAgent
with TAAT-specific functionality for monitoring social media and identifying trade signals.
"""

from typing import Optional

# Use absolute imports instead of relative imports
from src.agent_core.agent import BaseAgent
from src.agent_core.config import AgentConfig, load_config
from src.agents.taat.perception.perception import TaatPerceptionModule
from src.agents.taat.cognition.cognition import TaatCognitionModule
from src.agents.taat.action.action import TaatActionModule


class TaatAgent(BaseAgent):
    """
    TAAT Agent implementation.
    
    Specializes the BaseAgent for monitoring social media and identifying trade signals.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the TAAT Agent.
        
        Args:
            config: Agent configuration (loads from environment if None)
        """
        # Load TAAT-specific configuration
        taat_config = config or load_config(agent_type="taat")
        
        # Initialize with TAAT-specific modules
        super().__init__(
            config=taat_config,
            perception_class=TaatPerceptionModule,
            cognition_class=TaatCognitionModule,
            action_class=TaatActionModule
        )
