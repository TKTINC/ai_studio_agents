"""
Base agent class for AI Studio Agents.

This module contains the BaseAgent class that integrates perception, cognition, 
action, and memory components and implements the perception-cognition-action loop.
"""

import asyncio
from typing import Any, Dict, Optional, Type

from .config import AgentConfig, load_config
from .memory.memory import WorkingMemory
from .perception.perception import BasePerceptionModule
from .cognition.cognition import BaseCognitionModule
from .action.action import BaseActionModule


class BaseAgent:
    """
    Base agent class for all AI Studio Agents.
    
    Integrates perception, cognition, action, and memory components
    and implements the perception-cognition-action loop.
    """
    
    def __init__(
        self, 
        config: Optional[AgentConfig] = None,
        perception_class: Type[BasePerceptionModule] = BasePerceptionModule,
        cognition_class: Type[BaseCognitionModule] = BaseCognitionModule,
        action_class: Type[BaseActionModule] = BaseActionModule
    ):
        """
        Initialize the Base Agent.
        
        Args:
            config: Agent configuration (loads from environment if None)
            perception_class: Class to use for perception module
            cognition_class: Class to use for cognition module
            action_class: Class to use for action module
        """
        self.config = config or load_config()
        self.memory = WorkingMemory(max_history=self.config.max_history)
        self.perception = perception_class()
        self.cognition = cognition_class(self.config.llm_settings)
        self.action = action_class()
        self.running = False
    
    async def process_input(self, input_data: Any, input_type: str = "text") -> Dict[str, Any]:
        """
        Process a single input through the perception-cognition-action loop.
        
        Args:
            input_data: The input data to process
            input_type: The type of input
            
        Returns:
            Result of the action
        """
        # 1. Perception: Process input
        processed_input = await self.perception.process_input(input_data, input_type)
        
        # 2. Cognition: Generate response
        context = self.memory.get_context()
        response = await self.cognition.process(processed_input, context)
        
        # 3. Action: Execute response
        result = await self.action.execute(response)
        
        # 4. Memory: Update with this interaction
        self.memory.update(processed_input, response, result)
        
        return result
    
    async def run_loop(self):
        """
        Run the agent's main loop, processing inputs continuously.
        
        This is a simple implementation that reads from stdin.
        In a real application, this would be replaced with a proper interface.
        """
        self.running = True
        print(f"{self.__class__.__name__} is running. Type 'exit' to quit.")
        
        while self.running:
            try:
                # Get input from user
                user_input = input("USER: ")
                
                if user_input.lower() == "exit":
                    self.running = False
                    print(f"{self.__class__.__name__} shutting down.")
                    break
                
                # Process the input
                await self.process_input(user_input)
                
            except KeyboardInterrupt:
                self.running = False
                print(f"\n{self.__class__.__name__} shutting down.")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def stop(self):
        """Stop the agent's main loop."""
        self.running = False
