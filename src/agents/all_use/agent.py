"""
ALL-USE-specific agent implementation.

This module contains the AllUseAgent class that extends the BaseAgent
with ALL-USE-specific functionality for automated options trading with a triple-account structure.
"""

from typing import Optional

from ...agent_core.agent import BaseAgent
from ...agent_core.config import AgentConfig, load_config
from .perception.perception import AllUsePerceptionModule
from .cognition.cognition import AllUseCognitionModule
from .action.action import AllUseActionModule


class AllUseAgent(BaseAgent):
    """
    ALL-USE Agent implementation.
    
    Specializes the BaseAgent for automated options trading with a triple-account structure
    (Lumpsum, Leveraged, US Equities).
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the ALL-USE Agent.
        
        Args:
            config: Agent configuration (loads from environment if None)
        """
        # Load ALL-USE-specific configuration
        all_use_config = config or load_config(agent_type="all_use")
        
        # Initialize with ALL-USE-specific modules
        super().__init__(
            config=all_use_config,
            perception_class=AllUsePerceptionModule,
            cognition_class=AllUseCognitionModule,
            action_class=AllUseActionModule
        )
