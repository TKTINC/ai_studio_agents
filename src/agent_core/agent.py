"""
Base agent class for AI Studio Agents.

This module contains the BaseAgent class that integrates perception, cognition, 
action, and memory components and implements the perception-cognition-action loop.
"""

import asyncio
from typing import Any, Dict, Optional, Type

from .config import AgentConfig, load_config
from .memory.memory import WorkingMemory
from .memory.memory_manager import MemoryManager
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
        
        # Initialize memory systems
        self.memory_manager = MemoryManager(
            agent_id=self.config.agent_id,
            storage_path=self.config.storage_path
        )
        
        # Keep working memory reference for backward compatibility
        self.memory = self.memory_manager.working
        
        # Initialize perception, cognition, and action modules
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
        
        # Store experience in episodic memory
        self.memory_manager.store_experience(
            content=processed_input,
            experience_type=f"input_{input_type}",
            metadata={"timestamp": asyncio.get_event_loop().time()}
        )
        
        # 2. Cognition: Generate response
        # Get full context from all memory systems
        context = self.memory_manager.get_full_context()
        response = await self.cognition.process(processed_input, context)
        
        # 3. Action: Execute response
        result = await self.action.execute(response)
        
        # 4. Memory: Update with this interaction
        self.memory_manager.update_working_memory(processed_input, response, result)
        
        # Store action result in episodic memory
        self.memory_manager.store_experience(
            content=result,
            experience_type="action_result",
            metadata={
                "input_type": input_type,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
        return result
    
    async def execute_procedure(self, procedure_id: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a stored procedure from procedural memory.
        
        Args:
            procedure_id: ID of the procedure to execute
            parameters: Parameters to pass to the procedure
            
        Returns:
            Execution results
        """
        return await self.memory_manager.execute_procedure(procedure_id, parameters)
    
    async def learn_from_outcome(self, procedure_id: str, outcome: Any, metrics: Dict[str, Any]) -> bool:
        """
        Update procedural memory based on execution outcomes.
        
        Args:
            procedure_id: ID of the procedure to update
            outcome: Outcome of the procedure execution
            metrics: Performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        return self.memory_manager.learn_from_outcome(procedure_id, outcome, metrics)
    
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
