"""
Memory Manager for AI Studio Agents.

This module provides a unified interface for managing different memory systems,
coordinating between episodic, semantic, procedural, and working memory.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import os
import logging

from src.memory_systems.episodic import EpisodicMemory
from src.memory_systems.semantic import SemanticMemory
from src.memory_systems.procedural import ProceduralMemory
from src.agent_core.memory.memory import WorkingMemory

# Set up logging
logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory Manager for AI Studio Agents.
    
    Provides a unified interface for managing different memory systems,
    coordinating between episodic, semantic, procedural, and working memory.
    """
    
    def __init__(self, agent_id: str, storage_path: Optional[str] = None):
        """
        Initialize the memory manager.
        
        Args:
            agent_id: Unique identifier for the agent
            storage_path: Path to store persistent memory (None for in-memory only)
        """
        self.agent_id = agent_id
        self.storage_path = storage_path
        
        # Initialize memory systems
        self.episodic = EpisodicMemory(agent_id, storage_path)
        self.semantic = SemanticMemory(agent_id, storage_path)
        self.procedural = ProceduralMemory(agent_id, storage_path)
        self.working = WorkingMemory()
        
        logger.info(f"Memory Manager initialized for agent {agent_id}")
    
    def store_experience(self, 
                        content: Any, 
                        experience_type: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an experience in episodic memory and extract knowledge for semantic memory.
        
        Args:
            content: The main content of the experience
            experience_type: Type of experience (e.g., 'observation', 'action', 'feedback')
            metadata: Additional contextual information
            
        Returns:
            ID of the stored episode
        """
        # Store in episodic memory
        episode_id = self.episodic.store_episode(content, experience_type, metadata)
        
        # Extract knowledge for semantic memory (if applicable)
        if metadata and "knowledge_extraction" in metadata:
            extraction = metadata["knowledge_extraction"]
            if isinstance(extraction, dict):
                concept_id = extraction.get("concept_id")
                category = extraction.get("category")
                
                if concept_id and category:
                    self.semantic.store_concept(
                        concept_id=concept_id,
                        content=content,
                        category=category,
                        metadata=metadata
                    )
        
        # Update working memory state
        self.working.set_state("last_experience", {
            "id": episode_id,
            "type": experience_type,
            "timestamp": metadata.get("timestamp") if metadata else None
        })
        
        return episode_id
    
    def retrieve_relevant_knowledge(self, 
                                  context: Dict[str, Any], 
                                  query: Optional[str] = None) -> Dict[str, Any]:
        """
        Query across memory systems based on context.
        
        Args:
            context: Current context for relevance determination
            query: Optional explicit query string
            
        Returns:
            Dictionary containing relevant information from all memory systems
        """
        result = {
            "episodic": [],
            "semantic": [],
            "procedural": []
        }
        
        # Get relevant episodes
        if query:
            result["episodic"] = self.episodic.search_by_content(query, limit=5)
        else:
            # Get recent episodes of relevant types
            relevant_types = context.get("relevant_types", [])
            if relevant_types:
                for episode_type in relevant_types:
                    episodes = self.episodic.retrieve_by_type(episode_type, limit=3)
                    result["episodic"].extend(episodes)
            else:
                # Default to recent episodes
                result["episodic"] = self.episodic.get_recent_episodes(limit=5)
        
        # Get relevant concepts
        if query:
            result["semantic"] = self.semantic.search_concepts(query)
        else:
            # Get concepts of relevant categories
            relevant_categories = context.get("relevant_categories", [])
            if relevant_categories:
                for category in relevant_categories:
                    concepts = self.semantic.retrieve_by_category(category)
                    result["semantic"].extend(concepts)
        
        # Get relevant procedures
        relevant_procedure_category = context.get("procedure_category")
        if relevant_procedure_category:
            result["procedural"] = self.procedural.list_procedures(relevant_procedure_category)
        else:
            # Default to all procedures
            result["procedural"] = self.procedural.list_procedures()
        
        return result
    
    def execute_procedure(self, 
                         procedure_id: str, 
                         parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a stored procedure from procedural memory.
        
        Args:
            procedure_id: ID of the procedure to execute
            parameters: Parameters to pass to the procedure
            
        Returns:
            Execution results
        """
        # Get current context from working memory
        context = self.working.get_context()
        
        # Execute the procedure
        result = self.procedural.execute_procedure(
            procedure_id=procedure_id,
            context=context,
            parameters=parameters
        )
        
        # Store execution in episodic memory
        self.episodic.store_episode(
            content=result,
            episode_type="procedure_execution",
            metadata={
                "procedure_id": procedure_id,
                "parameters": parameters,
                "success": result.get("success", False)
            }
        )
        
        return result
    
    def learn_from_outcome(self, 
                          procedure_id: str, 
                          outcome: Any, 
                          metrics: Dict[str, Any]) -> bool:
        """
        Update procedural memory based on execution outcomes.
        
        Args:
            procedure_id: ID of the procedure to update
            outcome: Outcome of the procedure execution
            metrics: Performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        # Prepare outcome data
        outcome_data = {
            "outcome": outcome,
            "metrics": metrics
        }
        
        # Determine if automatic update should be applied
        update_strategy = None
        if metrics.get("performance") is not None:
            performance = metrics["performance"]
            
            # Example: Adjust parameters based on performance
            if performance < 0.3:  # Poor performance
                update_strategy = {
                    "type": "parameter_adjustment",
                    "adjustments": self._generate_parameter_adjustments(procedure_id, metrics)
                }
            elif performance < 0.7:  # Moderate performance
                # More subtle adjustments
                update_strategy = {
                    "type": "parameter_adjustment",
                    "adjustments": self._generate_parameter_adjustments(procedure_id, metrics, scale=0.5)
                }
        
        if update_strategy:
            outcome_data["update_strategy"] = update_strategy
        
        # Update procedural memory
        return self.procedural.learn_from_outcome(
            procedure_id=procedure_id,
            outcome_data=outcome_data,
            update_procedure=update_strategy is not None
        )
    
    def update_working_memory(self, 
                             input_data: Any, 
                             response: Any, 
                             result: Any) -> None:
        """
        Update working memory with a new interaction.
        
        Args:
            input_data: The input received by the agent
            response: The agent's response
            result: The result of executing the response
        """
        self.working.update(input_data, response, result)
    
    def set_working_state(self, key: str, value: Any) -> None:
        """
        Set a value in the agent's working state.
        
        Args:
            key: State key
            value: State value
        """
        self.working.set_state(key, value)
    
    def get_working_state(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a value from the agent's working state.
        
        Args:
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            The state value or default
        """
        return self.working.get_state(key, default)
    
    def get_full_context(self) -> Dict[str, Any]:
        """
        Get the full context from all memory systems for decision-making.
        
        Returns:
            Dictionary containing context from all memory systems
        """
        # Get working memory context
        context = self.working.get_context()
        
        # Add recent episodes
        context["recent_episodes"] = self.episodic.get_recent_episodes(limit=5)
        
        # Add relevant concepts based on recent episodes
        relevant_concepts = []
        for episode in context["recent_episodes"]:
            if isinstance(episode.get("content"), str):
                # Search for concepts related to episode content
                concepts = self.semantic.search_concepts(episode["content"])
                relevant_concepts.extend(concepts)
        
        context["relevant_concepts"] = relevant_concepts[:5]  # Limit to top 5
        
        # Add available procedures
        context["available_procedures"] = self.procedural.list_procedures()
        
        return context
    
    def clear_all_memory(self) -> None:
        """Clear all memory systems."""
        self.episodic.clear()
        self.semantic.clear()
        self.procedural.clear()
        self.working.reset()
        
        logger.info(f"All memory systems cleared for agent {self.agent_id}")
    
    def _generate_parameter_adjustments(self, 
                                      procedure_id: str, 
                                      metrics: Dict[str, Any], 
                                      scale: float = 1.0) -> Dict[str, Any]:
        """
        Generate parameter adjustments based on performance metrics.
        
        Args:
            procedure_id: ID of the procedure
            metrics: Performance metrics
            scale: Scaling factor for adjustments (0.0 to 1.0)
            
        Returns:
            Dictionary of parameter adjustments
        """
        procedure = self.procedural.retrieve_procedure(procedure_id)
        if not procedure:
            return {}
            
        adjustments = {}
        parameters = procedure.get("parameters", {})
        
        # Example adjustment logic (would be more sophisticated in practice)
        for param_name, param_value in parameters.items():
            if isinstance(param_value, (int, float)):
                # Numeric parameters can be adjusted
                if "error" in metrics and param_name in metrics["error"]:
                    # Direct error signal for this parameter
                    error = metrics["error"][param_name]
                    adjustments[param_name] = -error * scale
                elif "direction" in metrics:
                    # General direction for improvement
                    direction = metrics["direction"].get(param_name, 0)
                    # Small adjustment in the indicated direction
                    adjustments[param_name] = direction * abs(param_value) * 0.1 * scale
        
        return adjustments
