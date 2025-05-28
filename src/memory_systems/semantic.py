"""
Semantic Memory System for AI Studio Agents.

This module implements semantic memory capabilities for agents,
allowing them to store and retrieve knowledge, concepts, and facts.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os
import uuid
from datetime import datetime


class SemanticMemory:
    """
    Semantic Memory system for AI Studio Agents.
    
    Stores knowledge, concepts, facts, and relationships, allowing agents to:
    - Build and maintain a knowledge base
    - Retrieve relevant information based on context
    - Understand relationships between concepts
    - Apply knowledge to new situations
    """
    
    def __init__(self, agent_id: str, storage_path: Optional[str] = None):
        """
        Initialize the semantic memory system.
        
        Args:
            agent_id: Unique identifier for the agent
            storage_path: Path to store persistent memory (None for in-memory only)
        """
        self.agent_id = agent_id
        self.storage_path = storage_path
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, List[Tuple[str, str, float]]] = {}
        
        # Create storage directory if needed
        if storage_path:
            os.makedirs(os.path.join(storage_path, agent_id, "semantic"), exist_ok=True)
            self._load_from_disk()
    
    def store_concept(self, 
                     concept_id: str, 
                     content: Any, 
                     category: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a concept in semantic memory.
        
        Args:
            concept_id: Unique identifier for the concept
            content: The main content/definition of the concept
            category: Category of the concept (e.g., 'market_term', 'strategy', 'rule')
            metadata: Additional information about the concept
            
        Returns:
            The concept ID
        """
        timestamp = datetime.now().isoformat()
        
        # Generate ID if not provided
        if not concept_id:
            concept_id = f"{category}_{str(uuid.uuid4())}"
        
        self.knowledge_base[concept_id] = {
            "id": concept_id,
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "created": timestamp,
            "updated": timestamp
        }
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return concept_id
    
    def update_concept(self, 
                      concept_id: str, 
                      content: Optional[Any] = None,
                      category: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing concept.
        
        Args:
            concept_id: ID of the concept to update
            content: New content (None to keep existing)
            category: New category (None to keep existing)
            metadata: New or additional metadata (None to keep existing)
            
        Returns:
            True if successful, False if concept not found
        """
        if concept_id not in self.knowledge_base:
            return False
            
        concept = self.knowledge_base[concept_id]
        
        if content is not None:
            concept["content"] = content
            
        if category is not None:
            concept["category"] = category
            
        if metadata is not None:
            # Update metadata (merge with existing)
            concept["metadata"].update(metadata)
            
        concept["updated"] = datetime.now().isoformat()
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return True
    
    def retrieve_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific concept by ID.
        
        Args:
            concept_id: ID of the concept to retrieve
            
        Returns:
            The concept or None if not found
        """
        return self.knowledge_base.get(concept_id)
    
    def retrieve_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Retrieve all concepts in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of matching concepts
        """
        return [c for c in self.knowledge_base.values() if c["category"] == category]
    
    def search_concepts(self, query: str) -> List[Dict[str, Any]]:
        """
        Search concepts by content using simple string matching.
        
        Args:
            query: Search string to match against concept content
            
        Returns:
            List of matching concepts
        """
        # Simple string matching implementation
        # In a production system, this would use vector embeddings or a proper search index
        query = query.lower()
        matching = []
        
        for concept in self.knowledge_base.values():
            content_str = str(concept["content"]).lower()
            if query in content_str:
                matching.append(concept)
                
        return matching
    
    def add_relationship(self, 
                        source_id: str, 
                        relation_type: str, 
                        target_id: str,
                        strength: float = 1.0) -> bool:
        """
        Add a relationship between two concepts.
        
        Args:
            source_id: ID of the source concept
            relation_type: Type of relationship (e.g., 'is_a', 'part_of', 'related_to')
            target_id: ID of the target concept
            strength: Strength of the relationship (0.0 to 1.0)
            
        Returns:
            True if successful, False if either concept not found
        """
        if source_id not in self.knowledge_base or target_id not in self.knowledge_base:
            return False
            
        if source_id not in self.relationships:
            self.relationships[source_id] = []
            
        # Check if relationship already exists
        for i, (rel_type, rel_target, _) in enumerate(self.relationships[source_id]):
            if rel_type == relation_type and rel_target == target_id:
                # Update existing relationship
                self.relationships[source_id][i] = (relation_type, target_id, strength)
                break
        else:
            # Add new relationship
            self.relationships[source_id].append((relation_type, target_id, strength))
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return True
    
    def get_related_concepts(self, 
                            concept_id: str, 
                            relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get concepts related to a specific concept.
        
        Args:
            concept_id: ID of the concept to find relations for
            relation_type: Optional filter for relationship type
            
        Returns:
            List of related concepts with relationship information
        """
        if concept_id not in self.relationships:
            return []
            
        related = []
        
        for rel_type, target_id, strength in self.relationships[concept_id]:
            if relation_type is None or rel_type == relation_type:
                if target_id in self.knowledge_base:
                    related_concept = self.knowledge_base[target_id].copy()
                    related_concept["relation"] = {
                        "type": rel_type,
                        "strength": strength
                    }
                    related.append(related_concept)
                    
        return related
    
    def clear(self) -> None:
        """Clear all concepts and relationships from memory."""
        self.knowledge_base = {}
        self.relationships = {}
        if self.storage_path:
            self._save_to_disk()
    
    def _save_to_disk(self) -> None:
        """Save knowledge base and relationships to disk for persistence."""
        if not self.storage_path:
            return
            
        kb_path = os.path.join(self.storage_path, self.agent_id, "semantic", "knowledge_base.json")
        rel_path = os.path.join(self.storage_path, self.agent_id, "semantic", "relationships.json")
        
        with open(kb_path, 'w') as f:
            json.dump(self.knowledge_base, f)
            
        # Convert relationships to a serializable format
        serializable_relationships = {}
        for source_id, relations in self.relationships.items():
            serializable_relationships[source_id] = [
                {"type": rel_type, "target": target_id, "strength": strength}
                for rel_type, target_id, strength in relations
            ]
            
        with open(rel_path, 'w') as f:
            json.dump(serializable_relationships, f)
    
    def _load_from_disk(self) -> None:
        """Load knowledge base and relationships from disk."""
        if not self.storage_path:
            return
            
        kb_path = os.path.join(self.storage_path, self.agent_id, "semantic", "knowledge_base.json")
        rel_path = os.path.join(self.storage_path, self.agent_id, "semantic", "relationships.json")
        
        # Load knowledge base
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r') as f:
                    self.knowledge_base = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Handle corrupted file
                self.knowledge_base = {}
                
        # Load relationships
        if os.path.exists(rel_path):
            try:
                with open(rel_path, 'r') as f:
                    serialized = json.load(f)
                    
                # Convert from serialized format back to tuples
                self.relationships = {}
                for source_id, relations in serialized.items():
                    self.relationships[source_id] = [
                        (rel["type"], rel["target"], rel["strength"])
                        for rel in relations
                    ]
            except (json.JSONDecodeError, IOError):
                # Handle corrupted file
                self.relationships = {}
