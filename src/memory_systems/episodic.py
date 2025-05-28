"""
Episodic Memory System for AI Studio Agents.

This module implements episodic memory capabilities for agents,
allowing them to store and retrieve temporal sequences of events and experiences.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json
import os
import uuid


class EpisodicMemory:
    """
    Episodic Memory system for AI Studio Agents.
    
    Stores temporal sequences of events and experiences, allowing agents to:
    - Remember past interactions and their context
    - Recall specific events based on time, content, or relevance
    - Build a narrative of experiences over time
    """
    
    def __init__(self, agent_id: str, storage_path: Optional[str] = None, max_episodes: int = 1000):
        """
        Initialize the episodic memory system.
        
        Args:
            agent_id: Unique identifier for the agent
            storage_path: Path to store persistent memory (None for in-memory only)
            max_episodes: Maximum number of episodes to store
        """
        self.agent_id = agent_id
        self.storage_path = storage_path
        self.max_episodes = max_episodes
        self.episodes: List[Dict[str, Any]] = []
        
        # Create storage directory if needed
        if storage_path:
            os.makedirs(os.path.join(storage_path, agent_id, "episodic"), exist_ok=True)
            self._load_from_disk()
    
    def store_episode(self, 
                     content: Any, 
                     episode_type: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new episode in memory.
        
        Args:
            content: The main content of the episode
            episode_type: Type of episode (e.g., 'conversation', 'observation', 'action')
            metadata: Additional contextual information about the episode
            
        Returns:
            Unique ID of the stored episode
        """
        episode_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        episode = {
            "id": episode_id,
            "timestamp": timestamp,
            "type": episode_type,
            "content": content,
            "metadata": metadata or {}
        }
        
        self.episodes.append(episode)
        
        # Trim if needed
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return episode_id
    
    def retrieve_by_id(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific episode by its ID.
        
        Args:
            episode_id: Unique ID of the episode to retrieve
            
        Returns:
            The episode or None if not found
        """
        for episode in self.episodes:
            if episode["id"] == episode_id:
                return episode
        return None
    
    def retrieve_by_type(self, episode_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve episodes of a specific type.
        
        Args:
            episode_type: Type of episodes to retrieve
            limit: Maximum number of episodes to return
            
        Returns:
            List of matching episodes, most recent first
        """
        matching = [e for e in self.episodes if e["type"] == episode_type]
        return sorted(matching, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def retrieve_by_timeframe(self, 
                             start_time: Optional[str] = None, 
                             end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve episodes within a specific timeframe.
        
        Args:
            start_time: ISO format timestamp for start of range (None for no lower bound)
            end_time: ISO format timestamp for end of range (None for no upper bound)
            
        Returns:
            List of episodes within the timeframe, chronologically ordered
        """
        filtered = self.episodes
        
        if start_time:
            filtered = [e for e in filtered if e["timestamp"] >= start_time]
            
        if end_time:
            filtered = [e for e in filtered if e["timestamp"] <= end_time]
            
        return sorted(filtered, key=lambda x: x["timestamp"])
    
    def search_by_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search episodes by content using simple string matching.
        
        Args:
            query: Search string to match against episode content
            limit: Maximum number of episodes to return
            
        Returns:
            List of matching episodes, most recent first
        """
        # Simple string matching implementation
        # In a production system, this would use vector embeddings or a proper search index
        matching = []
        query = query.lower()
        
        for episode in self.episodes:
            content_str = str(episode["content"]).lower()
            if query in content_str:
                matching.append(episode)
                
        return sorted(matching, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent episodes.
        
        Args:
            limit: Maximum number of episodes to return
            
        Returns:
            List of most recent episodes
        """
        return sorted(self.episodes, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def clear(self) -> None:
        """Clear all episodes from memory."""
        self.episodes = []
        if self.storage_path:
            self._save_to_disk()
    
    def _save_to_disk(self) -> None:
        """Save episodes to disk for persistence."""
        if not self.storage_path:
            return
            
        file_path = os.path.join(self.storage_path, self.agent_id, "episodic", "episodes.json")
        with open(file_path, 'w') as f:
            json.dump(self.episodes, f)
    
    def _load_from_disk(self) -> None:
        """Load episodes from disk."""
        if not self.storage_path:
            return
            
        file_path = os.path.join(self.storage_path, self.agent_id, "episodic", "episodes.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    self.episodes = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Handle corrupted file
                self.episodes = []
