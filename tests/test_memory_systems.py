"""
Tests for the memory systems in AI Studio Agents.

This module contains tests for the episodic, semantic, procedural memory systems
and the memory manager integration.
"""

import os
import unittest
import asyncio
import tempfile
import shutil
from datetime import datetime

from src.memory_systems.episodic import EpisodicMemory
from src.memory_systems.semantic import SemanticMemory
from src.memory_systems.procedural import ProceduralMemory
from src.agent_core.memory.memory import WorkingMemory
from src.agent_core.memory.memory_manager import MemoryManager


class TestMemorySystems(unittest.TestCase):
    """Test suite for memory systems."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for memory storage
        self.test_dir = tempfile.mkdtemp()
        self.agent_id = "test_agent"
        
        # Initialize memory systems
        self.episodic = EpisodicMemory(self.agent_id, self.test_dir)
        self.semantic = SemanticMemory(self.agent_id, self.test_dir)
        self.procedural = ProceduralMemory(self.agent_id, self.test_dir)
        self.working = WorkingMemory()
        self.memory_manager = MemoryManager(self.agent_id, self.test_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_episodic_memory(self):
        """Test episodic memory functionality."""
        # Store an episode
        episode_id = self.episodic.store_episode(
            content="Test content",
            episode_type="test",
            metadata={"key": "value"}
        )
        
        # Retrieve the episode
        episode = self.episodic.retrieve_by_id(episode_id)
        
        # Verify the episode
        self.assertIsNotNone(episode)
        self.assertEqual(episode["content"], "Test content")
        self.assertEqual(episode["type"], "test")
        self.assertEqual(episode["metadata"]["key"], "value")
        
        # Test retrieval by type
        episodes = self.episodic.retrieve_by_type("test")
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["id"], episode_id)
        
        # Test search by content
        search_results = self.episodic.search_by_content("content")
        self.assertEqual(len(search_results), 1)
        self.assertEqual(search_results[0]["id"], episode_id)
    
    def test_semantic_memory(self):
        """Test semantic memory functionality."""
        # Store a concept
        concept_id = "test_concept"
        self.semantic.store_concept(
            concept_id=concept_id,
            content="Test concept content",
            category="test_category",
            metadata={"key": "value"}
        )
        
        # Retrieve the concept
        concept = self.semantic.retrieve_concept(concept_id)
        
        # Verify the concept
        self.assertIsNotNone(concept)
        self.assertEqual(concept["content"], "Test concept content")
        self.assertEqual(concept["category"], "test_category")
        self.assertEqual(concept["metadata"]["key"], "value")
        
        # Test retrieval by category
        concepts = self.semantic.retrieve_by_category("test_category")
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0]["id"], concept_id)
        
        # Test search
        search_results = self.semantic.search_concepts("concept")
        self.assertEqual(len(search_results), 1)
        self.assertEqual(search_results[0]["id"], concept_id)
        
        # Test relationships
        related_concept_id = "related_concept"
        self.semantic.store_concept(
            concept_id=related_concept_id,
            content="Related concept content",
            category="test_category"
        )
        
        self.semantic.add_relationship(
            source_id=concept_id,
            relation_type="related_to",
            target_id=related_concept_id,
            strength=0.8
        )
        
        related = self.semantic.get_related_concepts(concept_id)
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]["id"], related_concept_id)
        self.assertEqual(related[0]["relation"]["type"], "related_to")
        self.assertEqual(related[0]["relation"]["strength"], 0.8)
    
    def test_procedural_memory(self):
        """Test procedural memory functionality."""
        # Register a test function
        def test_function(context=None, param1=None, param2=None):
            return {
                "success": True,
                "param1": param1,
                "param2": param2,
                "context": context
            }
        
        self.procedural.register_function("test_function", test_function)
        
        # Store a procedure
        procedure_id = "test_procedure"
        self.procedural.store_procedure(
            procedure_id=procedure_id,
            steps=[
                {"function": "test_function", "parameters": {"param1": "default"}}
            ],
            parameters={"param2": "global_default"},
            metadata={"key": "value"}
        )
        
        # Retrieve the procedure
        procedure = self.procedural.retrieve_procedure(procedure_id)
        
        # Verify the procedure
        self.assertIsNotNone(procedure)
        self.assertEqual(procedure["id"], procedure_id)
        self.assertEqual(procedure["parameters"]["param2"], "global_default")
        self.assertEqual(procedure["metadata"]["key"], "value")
        
        # Execute the procedure
        result = self.procedural.execute_procedure(
            procedure_id=procedure_id,
            context={"test": "context"},
            parameters={"param2": "override"}
        )
        
        # Verify execution result
        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 1)
        step_result = result["results"][0]
        self.assertTrue(step_result["success"])
        self.assertEqual(step_result["param1"], "default")
        self.assertEqual(step_result["param2"], "override")
        self.assertEqual(step_result["context"]["test"], "context")
        
        # Test version control
        self.procedural.store_procedure(
            procedure_id=procedure_id,
            steps=[
                {"function": "test_function", "parameters": {"param1": "updated"}}
            ]
        )
        
        # Verify version is stored
        procedure = self.procedural.retrieve_procedure(procedure_id)
        self.assertEqual(len(procedure["versions"]), 1)
        
        # Test rollback
        self.procedural.rollback_procedure(procedure_id, 0)
        procedure = self.procedural.retrieve_procedure(procedure_id)
        self.assertEqual(procedure["steps"][0]["parameters"]["param1"], "default")
        
        # Test learning
        self.procedural.learn_from_outcome(
            procedure_id=procedure_id,
            outcome_data={
                "update_strategy": {
                    "type": "parameter_adjustment",
                    "adjustments": {"param2": "learned_value"}
                }
            }
        )
        
        procedure = self.procedural.retrieve_procedure(procedure_id)
        self.assertEqual(procedure["parameters"]["param2"], "learned_value")
    
    def test_working_memory(self):
        """Test working memory functionality."""
        # Update with an interaction
        self.working.update("input", "response", "result")
        
        # Get context
        context = self.working.get_context()
        
        # Verify context
        self.assertEqual(len(context["conversation"]), 1)
        self.assertEqual(context["conversation"][0]["input"], "input")
        self.assertEqual(context["conversation"][0]["response"], "response")
        self.assertEqual(context["conversation"][0]["result"], "result")
        
        # Test state management
        self.working.set_state("test_key", "test_value")
        value = self.working.get_state("test_key")
        self.assertEqual(value, "test_value")
        
        # Test reset
        self.working.reset()
        context = self.working.get_context()
        self.assertEqual(len(context["conversation"]), 0)
        self.assertEqual(len(context["state"]), 0)
    
    def test_memory_manager_integration(self):
        """Test memory manager integration."""
        # Store an experience
        episode_id = self.memory_manager.store_experience(
            content="Test experience",
            experience_type="test",
            metadata={
                "knowledge_extraction": {
                    "concept_id": "test_concept",
                    "category": "test_category"
                }
            }
        )
        
        # Verify episodic storage
        episode = self.memory_manager.episodic.retrieve_by_id(episode_id)
        self.assertIsNotNone(episode)
        self.assertEqual(episode["content"], "Test experience")
        
        # Verify semantic extraction
        concept = self.memory_manager.semantic.retrieve_concept("test_concept")
        self.assertIsNotNone(concept)
        self.assertEqual(concept["content"], "Test experience")
        self.assertEqual(concept["category"], "test_category")
        
        # Test working memory update
        self.memory_manager.update_working_memory("input", "response", "result")
        context = self.memory_manager.working.get_context()
        self.assertEqual(len(context["conversation"]), 1)
        
        # Register a test function for procedural memory
        def test_function(context=None, param=None):
            return {"success": True, "param": param}
        
        self.memory_manager.procedural.register_function("test_function", test_function)
        
        # Store and execute a procedure
        procedure_id = "test_procedure"
        self.memory_manager.procedural.store_procedure(
            procedure_id=procedure_id,
            steps=[{"function": "test_function", "parameters": {"param": "test"}}]
        )
        
        result = self.memory_manager.execute_procedure(procedure_id)
        self.assertTrue(result["success"])
        
        # Test cross-memory retrieval
        context = {
            "relevant_types": ["test"],
            "relevant_categories": ["test_category"],
            "procedure_category": None
        }
        
        relevant = self.memory_manager.retrieve_relevant_knowledge(context)
        self.assertEqual(len(relevant["episodic"]), 1)
        self.assertEqual(len(relevant["semantic"]), 1)
        self.assertEqual(len(relevant["procedural"]), 1)
        
        # Test full context retrieval
        full_context = self.memory_manager.get_full_context()
        self.assertIn("conversation", full_context)
        self.assertIn("state", full_context)
        self.assertIn("recent_episodes", full_context)
        self.assertIn("available_procedures", full_context)


if __name__ == "__main__":
    unittest.main()
