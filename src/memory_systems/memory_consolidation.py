"""
Memory Consolidation for AI Studio Agents.

This module implements memory consolidation mechanisms for summarizing episodic memories,
extracting patterns, and optimizing memory storage and retrieval efficiency.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os
import re
import time
from datetime import datetime, timedelta
import logging
import math
import hashlib
from collections import Counter, defaultdict

# Set up logging
logger = logging.getLogger(__name__)


class EpisodicConsolidator:
    """
    Episodic memory consolidation system.
    
    Summarizes related episodes and extracts common elements into semantic knowledge.
    """
    
    def __init__(self, 
                min_cluster_size: int = 3, 
                similarity_threshold: float = 0.6,
                max_age_days: int = 30):
        """
        Initialize the episodic consolidator.
        
        Args:
            min_cluster_size: Minimum number of episodes to form a cluster
            similarity_threshold: Minimum similarity score to group episodes
            max_age_days: Maximum age of episodes to consider for consolidation
        """
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.max_age_days = max_age_days
    
    def consolidate_episodes(self, 
                           episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate episodic memories into summaries and extract knowledge.
        
        Args:
            episodes: List of episodic memories to consolidate
            
        Returns:
            Dictionary with consolidation results
        """
        if not episodes:
            return {"consolidated": False, "reason": "No episodes provided"}
        
        # Filter episodes by age
        recent_episodes = self._filter_recent_episodes(episodes)
        
        if len(recent_episodes) < self.min_cluster_size:
            return {
                "consolidated": False, 
                "reason": f"Insufficient recent episodes ({len(recent_episodes)} < {self.min_cluster_size})"
            }
        
        # Cluster episodes by similarity
        clusters = self._cluster_episodes(recent_episodes)
        
        if not clusters:
            return {"consolidated": False, "reason": "No significant clusters found"}
        
        # Process each cluster
        results = {
            "consolidated": True,
            "clusters_processed": len(clusters),
            "summaries": [],
            "extracted_concepts": [],
            "stats": {
                "total_episodes": len(episodes),
                "recent_episodes": len(recent_episodes),
                "episodes_clustered": sum(len(cluster) for cluster in clusters)
            }
        }
        
        for i, cluster in enumerate(clusters):
            # Generate summary for the cluster
            summary = self._summarize_cluster(cluster)
            results["summaries"].append(summary)
            
            # Extract concepts from the cluster
            concepts = self._extract_concepts(cluster, summary)
            results["extracted_concepts"].extend(concepts)
        
        return results
    
    def _filter_recent_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter episodes by recency.
        
        Args:
            episodes: List of episodes to filter
            
        Returns:
            List of recent episodes
        """
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
        recent = []
        
        for episode in episodes:
            if "timestamp" in episode:
                try:
                    if isinstance(episode["timestamp"], str):
                        timestamp = datetime.fromisoformat(episode["timestamp"].replace('Z', '+00:00'))
                    elif isinstance(episode["timestamp"], (int, float)):
                        timestamp = datetime.fromtimestamp(episode["timestamp"])
                    else:
                        continue
                        
                    if timestamp >= cutoff_date:
                        recent.append(episode)
                except (ValueError, TypeError):
                    continue
        
        return recent
    
    def _cluster_episodes(self, episodes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster episodes by similarity.
        
        Args:
            episodes: List of episodes to cluster
            
        Returns:
            List of episode clusters
        """
        if not episodes:
            return []
        
        # Initialize clusters with the first episode
        clusters = [[episodes[0]]]
        
        # For each remaining episode
        for episode in episodes[1:]:
            best_cluster = -1
            best_similarity = 0
            
            # Find the best matching cluster
            for i, cluster in enumerate(clusters):
                similarity = self._calculate_cluster_similarity(episode, cluster)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_cluster = i
            
            if best_cluster >= 0:
                # Add to best matching cluster
                clusters[best_cluster].append(episode)
            else:
                # Create a new cluster
                clusters.append([episode])
        
        # Filter out clusters that are too small
        return [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size]
    
    def _calculate_cluster_similarity(self, 
                                    episode: Dict[str, Any], 
                                    cluster: List[Dict[str, Any]]) -> float:
        """
        Calculate similarity between an episode and a cluster.
        
        Args:
            episode: Episode to compare
            cluster: Cluster to compare against
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not cluster:
            return 0.0
        
        # Calculate average similarity with all episodes in the cluster
        similarities = [self._calculate_episode_similarity(episode, other) 
                       for other in cluster]
        
        return sum(similarities) / len(similarities)
    
    def _calculate_episode_similarity(self, 
                                    episode1: Dict[str, Any], 
                                    episode2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two episodes.
        
        Args:
            episode1: First episode
            episode2: Second episode
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        score_components = []
        
        # Check type match
        if "type" in episode1 and "type" in episode2:
            if episode1["type"] == episode2["type"]:
                score_components.append(1.0)
            else:
                score_components.append(0.0)
        
        # Check content similarity
        content1 = str(episode1.get("content", "")).lower()
        content2 = str(episode2.get("content", "")).lower()
        
        if content1 and content2:
            # Simple word overlap similarity
            words1 = set(content1.split())
            words2 = set(content2.split())
            
            if words1 and words2:
                # Jaccard similarity
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                
                if union > 0:
                    score_components.append(intersection / union)
        
        # Check metadata similarity
        if "metadata" in episode1 and "metadata" in episode2:
            m1 = episode1["metadata"]
            m2 = episode2["metadata"]
            
            if isinstance(m1, dict) and isinstance(m2, dict):
                # Check for common keys with matching values
                common_keys = set(m1.keys()).intersection(set(m2.keys()))
                if common_keys:
                    matches = sum(1 for k in common_keys if m1[k] == m2[k])
                    score_components.append(matches / len(common_keys))
        
        # Return average score if components exist
        if score_components:
            return sum(score_components) / len(score_components)
        
        return 0.0
    
    def _summarize_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary for a cluster of episodes.
        
        Args:
            cluster: Cluster of episodes to summarize
            
        Returns:
            Summary dictionary
        """
        if not cluster:
            return {}
        
        # Extract common type
        types = [episode.get("type") for episode in cluster if "type" in episode]
        common_type = max(set(types), key=types.count) if types else "unknown"
        
        # Extract common metadata keys and values
        common_metadata = {}
        if all("metadata" in episode and isinstance(episode["metadata"], dict) for episode in cluster):
            # Find keys that appear in at least 50% of episodes
            key_counts = Counter()
            for episode in cluster:
                key_counts.update(episode["metadata"].keys())
            
            common_keys = [k for k, count in key_counts.items() if count >= len(cluster) / 2]
            
            # For each common key, find the most common value
            for key in common_keys:
                values = [episode["metadata"].get(key) for episode in cluster 
                         if key in episode["metadata"]]
                
                # Only use simple values (strings, numbers, booleans)
                simple_values = [v for v in values if isinstance(v, (str, int, float, bool))]
                
                if simple_values:
                    value_counts = Counter(simple_values)
                    most_common = value_counts.most_common(1)[0][0]
                    common_metadata[key] = most_common
        
        # Extract time range
        timestamps = []
        for episode in cluster:
            if "timestamp" in episode:
                try:
                    if isinstance(episode["timestamp"], str):
                        dt = datetime.fromisoformat(episode["timestamp"].replace('Z', '+00:00'))
                        timestamps.append(dt)
                except ValueError:
                    continue
        
        time_range = {}
        if timestamps:
            timestamps.sort()
            time_range = {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
                "duration_hours": (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            }
        
        # Generate a unique ID for the summary
        summary_id = hashlib.md5(
            (common_type + str(common_metadata) + str(time_range)).encode()
        ).hexdigest()
        
        # Create summary
        return {
            "id": summary_id,
            "type": f"summary_{common_type}",
            "episode_count": len(cluster),
            "common_type": common_type,
            "common_metadata": common_metadata,
            "time_range": time_range,
            "timestamp": datetime.now().isoformat(),
            "source_episodes": [episode.get("id") for episode in cluster if "id" in episode]
        }
    
    def _extract_concepts(self, 
                        cluster: List[Dict[str, Any]], 
                        summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract concepts from a cluster of episodes.
        
        Args:
            cluster: Cluster of episodes
            summary: Cluster summary
            
        Returns:
            List of extracted concepts
        """
        if not cluster or not summary:
            return []
        
        concepts = []
        
        # Extract common type as a concept
        common_type = summary.get("common_type")
        if common_type:
            type_concept = {
                "id": f"concept_{common_type}_{summary['id'][:8]}",
                "content": f"Common pattern of {common_type} events",
                "category": "event_pattern",
                "metadata": {
                    "event_type": common_type,
                    "frequency": len(cluster),
                    "source_summary": summary["id"],
                    "confidence": min(1.0, len(cluster) / 10)  # Higher confidence with more episodes
                }
            }
            concepts.append(type_concept)
        
        # Extract common entities
        entities = self._extract_common_entities(cluster)
        for entity, count in entities.items():
            if count >= len(cluster) / 2:  # Entity appears in at least half the episodes
                entity_concept = {
                    "id": f"entity_{entity.lower().replace(' ', '_')}_{summary['id'][:8]}",
                    "content": entity,
                    "category": "entity",
                    "metadata": {
                        "frequency": count,
                        "source_summary": summary["id"],
                        "confidence": count / len(cluster)
                    }
                }
                concepts.append(entity_concept)
        
        # Extract relationships between common entities
        if len(entities) >= 2:
            top_entities = [e for e, c in sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]]
            for i in range(len(top_entities)):
                for j in range(i+1, len(top_entities)):
                    e1 = top_entities[i]
                    e2 = top_entities[j]
                    
                    # Check co-occurrence
                    co_occurrences = sum(1 for episode in cluster 
                                        if e1 in str(episode.get("content", "")) 
                                        and e2 in str(episode.get("content", "")))
                    
                    if co_occurrences >= len(cluster) / 3:  # Co-occur in at least a third of episodes
                        relationship = {
                            "id": f"rel_{e1.lower().replace(' ', '_')}_{e2.lower().replace(' ', '_')}_{summary['id'][:8]}",
                            "content": f"Relationship between {e1} and {e2}",
                            "category": "relationship",
                            "metadata": {
                                "entity1": e1,
                                "entity2": e2,
                                "co_occurrences": co_occurrences,
                                "source_summary": summary["id"],
                                "confidence": co_occurrences / len(cluster)
                            }
                        }
                        concepts.append(relationship)
        
        return concepts
    
    def _extract_common_entities(self, cluster: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Extract common entities from a cluster of episodes.
        
        Args:
            cluster: Cluster of episodes
            
        Returns:
            Dictionary of entity counts
        """
        entity_counts = Counter()
        
        for episode in cluster:
            content = str(episode.get("content", ""))
            
            # Simple entity extraction (placeholder for more sophisticated NLP)
            # Extract capitalized words as potential entities
            capitalized = re.findall(r'\b[A-Z][a-zA-Z]*\b', content)
            entity_counts.update(capitalized)
            
            # Extract from metadata
            if "metadata" in episode and isinstance(episode["metadata"], dict):
                for key, value in episode["metadata"].items():
                    if isinstance(value, str) and value[0].isupper():
                        entity_counts[value] += 1
        
        # Filter out entities that appear only once
        return {entity: count for entity, count in entity_counts.items() if count > 1}


class PatternExtractor:
    """
    Pattern extraction system for procedural memory.
    
    Identifies recurring patterns in actions to create or refine procedures.
    """
    
    def __init__(self, 
                min_sequence_length: int = 2, 
                min_occurrences: int = 3,
                similarity_threshold: float = 0.7):
        """
        Initialize the pattern extractor.
        
        Args:
            min_sequence_length: Minimum length of action sequences to extract
            min_occurrences: Minimum number of occurrences to consider a pattern
            similarity_threshold: Minimum similarity for actions to be considered the same
        """
        self.min_sequence_length = min_sequence_length
        self.min_occurrences = min_occurrences
        self.similarity_threshold = similarity_threshold
    
    def extract_patterns(self, 
                       action_sequences: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract patterns from action sequences.
        
        Args:
            action_sequences: List of action sequences to analyze
            
        Returns:
            Dictionary with extraction results
        """
        if not action_sequences or len(action_sequences) < self.min_occurrences:
            return {
                "patterns_found": False, 
                "reason": f"Insufficient action sequences ({len(action_sequences)} < {self.min_occurrences})"
            }
        
        # Filter sequences by length
        valid_sequences = [seq for seq in action_sequences if len(seq) >= self.min_sequence_length]
        
        if len(valid_sequences) < self.min_occurrences:
            return {
                "patterns_found": False, 
                "reason": f"Insufficient valid sequences ({len(valid_sequences)} < {self.min_occurrences})"
            }
        
        # Find common subsequences
        patterns = self._find_common_subsequences(valid_sequences)
        
        if not patterns:
            return {"patterns_found": False, "reason": "No significant patterns found"}
        
        # Generate procedures from patterns
        procedures = self._generate_procedures(patterns)
        
        return {
            "patterns_found": True,
            "pattern_count": len(patterns),
            "procedures": procedures,
            "stats": {
                "total_sequences": len(action_sequences),
                "valid_sequences": len(valid_sequences),
                "total_actions": sum(len(seq) for seq in valid_sequences)
            }
        }
    
    def _find_common_subsequences(self, 
                                sequences: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Find common subsequences across action sequences.
        
        Args:
            sequences: List of action sequences
            
        Returns:
            List of common subsequence patterns
        """
        # Extract all possible subsequences of minimum length
        all_subsequences = []
        
        for seq_idx, sequence in enumerate(sequences):
            for i in range(len(sequence) - self.min_sequence_length + 1):
                for j in range(i + self.min_sequence_length, len(sequence) + 1):
                    subsequence = sequence[i:j]
                    all_subsequences.append({
                        "actions": subsequence,
                        "length": len(subsequence),
                        "source_sequence": seq_idx
                    })
        
        # Group similar subsequences
        patterns = []
        processed_indices = set()
        
        for i, subseq1 in enumerate(all_subsequences):
            if i in processed_indices:
                continue
                
            similar_subsequences = [subseq1]
            processed_indices.add(i)
            
            for j, subseq2 in enumerate(all_subsequences):
                if j in processed_indices or j == i:
                    continue
                    
                if (len(subseq1["actions"]) == len(subseq2["actions"]) and
                    self._are_subsequences_similar(subseq1["actions"], subseq2["actions"])):
                    similar_subsequences.append(subseq2)
                    processed_indices.add(j)
            
            # Check if this pattern occurs enough times
            if len(similar_subsequences) >= self.min_occurrences:
                # Count unique source sequences (to avoid counting multiple occurrences in same sequence)
                unique_sources = set(subseq["source_sequence"] for subseq in similar_subsequences)
                
                if len(unique_sources) >= self.min_occurrences:
                    patterns.append({
                        "prototype": subseq1["actions"],
                        "occurrences": len(similar_subsequences),
                        "unique_sources": len(unique_sources),
                        "length": len(subseq1["actions"]),
                        "instances": similar_subsequences
                    })
        
        # Sort patterns by occurrences and length
        patterns.sort(key=lambda x: (x["occurrences"], x["length"]), reverse=True)
        
        return patterns
    
    def _are_subsequences_similar(self, 
                                subseq1: List[Dict[str, Any]], 
                                subseq2: List[Dict[str, Any]]) -> bool:
        """
        Check if two subsequences are similar.
        
        Args:
            subseq1: First subsequence
            subseq2: Second subsequence
            
        Returns:
            True if subsequences are similar, False otherwise
        """
        if len(subseq1) != len(subseq2):
            return False
        
        for action1, action2 in zip(subseq1, subseq2):
            if not self._are_actions_similar(action1, action2):
                return False
        
        return True
    
    def _are_actions_similar(self, 
                           action1: Dict[str, Any], 
                           action2: Dict[str, Any]) -> bool:
        """
        Check if two actions are similar.
        
        Args:
            action1: First action
            action2: Second action
            
        Returns:
            True if actions are similar, False otherwise
        """
        # Check action type
        if "type" in action1 and "type" in action2:
            if action1["type"] != action2["type"]:
                return False
        
        # Check function name
        if "function" in action1 and "function" in action2:
            if action1["function"] != action2["function"]:
                return False
        
        # Check parameters (if they exist)
        if "parameters" in action1 and "parameters" in action2:
            params1 = action1["parameters"]
            params2 = action2["parameters"]
            
            if isinstance(params1, dict) and isinstance(params2, dict):
                # Check if parameter keys match
                if set(params1.keys()) != set(params2.keys()):
                    return False
                
                # For each parameter, check if values are similar
                for key in params1:
                    # Skip value comparison for dynamic parameters
                    if self._is_dynamic_parameter(params1[key]) or self._is_dynamic_parameter(params2[key]):
                        continue
                    
                    # For static parameters, values should match
                    if params1[key] != params2[key]:
                        return False
        
        return True
    
    def _is_dynamic_parameter(self, value: Any) -> bool:
        """
        Check if a parameter value is likely dynamic.
        
        Args:
            value: Parameter value to check
            
        Returns:
            True if parameter is likely dynamic, False otherwise
        """
        # Check if value is a string that looks like a variable or placeholder
        if isinstance(value, str):
            # Check for patterns like {variable}, $variable, or {{variable}}
            if re.search(r'[\{\$][\w_]+[\}]?', value):
                return True
            
            # Check for UUIDs, IDs, timestamps, etc.
            if re.search(r'[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}', value):
                return True
                
            if re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
                return True
        
        return False
    
    def _generate_procedures(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate procedures from identified patterns.
        
        Args:
            patterns: List of identified patterns
            
        Returns:
            List of generated procedures
        """
        procedures = []
        
        for i, pattern in enumerate(patterns):
            # Generate a unique ID for the procedure
            procedure_id = f"auto_procedure_{i}_{int(time.time())}"
            
            # Extract steps from the prototype
            steps = []
            for action in pattern["prototype"]:
                step = {
                    "function": action.get("function", "unknown"),
                    "parameters": self._parameterize_action(action)
                }
                steps.append(step)
            
            # Create the procedure
            procedure = {
                "id": procedure_id,
                "steps": steps,
                "metadata": {
                    "auto_generated": True,
                    "pattern_occurrences": pattern["occurrences"],
                    "unique_sources": pattern["unique_sources"],
                    "confidence": min(1.0, pattern["occurrences"] / 10),
                    "generated_at": datetime.now().isoformat()
                }
            }
            
            procedures.append(procedure)
        
        return procedures
    
    def _parameterize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameterize an action by identifying dynamic parameters.
        
        Args:
            action: Action to parameterize
            
        Returns:
            Parameterized action parameters
        """
        if "parameters" not in action or not isinstance(action["parameters"], dict):
            return {}
        
        params = {}
        
        for key, value in action["parameters"].items():
            if self._is_dynamic_parameter(value):
                # Replace with a parameter placeholder
                params[key] = f"{{{key}}}"
            else:
                # Keep static value
                params[key] = value
        
        return params


class MemoryOptimizer:
    """
    Memory optimization system.
    
    Compresses or archives older memories while preserving important information.
    """
    
    def __init__(self, 
                compression_age_days: int = 30, 
                archive_age_days: int = 90,
                importance_threshold: float = 0.7):
        """
        Initialize the memory optimizer.
        
        Args:
            compression_age_days: Age in days after which memories are compressed
            archive_age_days: Age in days after which memories are archived
            importance_threshold: Threshold for preserving important memories
        """
        self.compression_age_days = compression_age_days
        self.archive_age_days = archive_age_days
        self.importance_threshold = importance_threshold
    
    def optimize_memories(self, 
                        memories: List[Dict[str, Any]], 
                        importance_scorer: Optional[callable] = None) -> Dict[str, Any]:
        """
        Optimize memories by compressing and archiving.
        
        Args:
            memories: List of memories to optimize
            importance_scorer: Optional function to score memory importance
            
        Returns:
            Dictionary with optimization results
        """
        if not memories:
            return {"optimized": False, "reason": "No memories provided"}
        
        # Calculate memory ages
        memories_with_age = self._calculate_memory_ages(memories)
        
        if not memories_with_age:
            return {"optimized": False, "reason": "No memories with valid timestamps"}
        
        # Score memory importance
        memories_with_scores = self._score_memories(memories_with_age, importance_scorer)
        
        # Identify memories for compression and archiving
        to_compress = []
        to_archive = []
        to_preserve = []
        
        for memory in memories_with_scores:
            age_days = memory["_age_days"]
            importance = memory["_importance"]
            
            if age_days >= self.archive_age_days and importance < self.importance_threshold:
                to_archive.append(memory)
            elif age_days >= self.compression_age_days and importance < self.importance_threshold:
                to_compress.append(memory)
            else:
                to_preserve.append(memory)
        
        # Perform compression and archiving
        compressed_memories = [self._compress_memory(memory) for memory in to_compress]
        archived_memories = [self._archive_memory(memory) for memory in to_archive]
        
        return {
            "optimized": True,
            "total_memories": len(memories),
            "preserved": len(to_preserve),
            "compressed": len(compressed_memories),
            "archived": len(archived_memories),
            "compression_ratio": self._calculate_compression_ratio(to_compress, compressed_memories),
            "optimized_memories": to_preserve + compressed_memories,
            "archived_memories": archived_memories
        }
    
    def _calculate_memory_ages(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate ages of memories in days.
        
        Args:
            memories: List of memories
            
        Returns:
            List of memories with age information
        """
        result = []
        now = datetime.now()
        
        for memory in memories:
            if "timestamp" in memory:
                try:
                    if isinstance(memory["timestamp"], str):
                        timestamp = datetime.fromisoformat(memory["timestamp"].replace('Z', '+00:00'))
                    elif isinstance(memory["timestamp"], (int, float)):
                        timestamp = datetime.fromtimestamp(memory["timestamp"])
                    else:
                        continue
                        
                    age_days = (now - timestamp).total_seconds() / (24 * 3600)
                    memory_copy = memory.copy()
                    memory_copy["_age_days"] = age_days
                    result.append(memory_copy)
                except (ValueError, TypeError):
                    continue
        
        return result
    
    def _score_memories(self, 
                      memories: List[Dict[str, Any]], 
                      importance_scorer: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Score memories by importance.
        
        Args:
            memories: List of memories with age information
            importance_scorer: Optional function to score memory importance
            
        Returns:
            List of memories with importance scores
        """
        result = []
        
        for memory in memories:
            importance = 0.5  # Default importance
            
            if importance_scorer:
                # Use provided scorer
                try:
                    importance = importance_scorer(memory)
                except Exception:
                    pass
            else:
                # Use built-in heuristics
                importance = self._calculate_importance(memory)
            
            memory_copy = memory.copy()
            memory_copy["_importance"] = importance
            result.append(memory_copy)
        
        return result
    
    def _calculate_importance(self, memory: Dict[str, Any]) -> float:
        """
        Calculate importance of a memory using heuristics.
        
        Args:
            memory: Memory to evaluate
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        score_components = []
        
        # Check explicit importance in metadata
        if "metadata" in memory and isinstance(memory["metadata"], dict):
            if "importance" in memory["metadata"]:
                importance = memory["metadata"]["importance"]
                if isinstance(importance, (int, float)):
                    score_components.append(min(1.0, max(0.0, importance)))
        
        # Check access frequency
        if "metadata" in memory and isinstance(memory["metadata"], dict):
            if "access_count" in memory["metadata"]:
                access_count = memory["metadata"]["access_count"]
                if isinstance(access_count, (int, float)) and access_count >= 0:
                    # Logarithmic scaling
                    score_components.append(min(0.9, 0.1 + 0.2 * math.log1p(access_count)))
        
        # Check for references from other memories
        if "metadata" in memory and isinstance(memory["metadata"], dict):
            if "referenced_by" in memory["metadata"]:
                references = memory["metadata"]["referenced_by"]
                if isinstance(references, list):
                    # More references = more important
                    score_components.append(min(0.9, 0.3 + 0.1 * len(references)))
                elif isinstance(references, int):
                    score_components.append(min(0.9, 0.3 + 0.1 * references))
        
        # Check memory type importance
        if "type" in memory:
            memory_type = memory["type"]
            # Some types are inherently more important
            if memory_type in ["critical_event", "user_preference", "system_error"]:
                score_components.append(0.9)
            elif memory_type in ["user_interaction", "decision_point"]:
                score_components.append(0.7)
        
        # Return maximum score if any component is high, otherwise average
        if score_components:
            max_score = max(score_components)
            if max_score > 0.7:
                return max_score
            else:
                return sum(score_components) / len(score_components)
        
        return 0.5  # Default importance
    
    def _compress_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress a memory by removing non-essential information.
        
        Args:
            memory: Memory to compress
            
        Returns:
            Compressed memory
        """
        # Create a copy to avoid modifying the original
        compressed = {}
        
        # Always preserve these fields
        essential_fields = ["id", "type", "timestamp"]
        for field in essential_fields:
            if field in memory:
                compressed[field] = memory[field]
        
        # Compress content if present
        if "content" in memory:
            content = memory["content"]
            if isinstance(content, str) and len(content) > 100:
                # Truncate long content
                compressed["content"] = content[:100] + "... [compressed]"
            else:
                compressed["content"] = content
        
        # Preserve essential metadata
        if "metadata" in memory and isinstance(memory["metadata"], dict):
            compressed["metadata"] = {}
            
            # Copy essential metadata
            essential_metadata = ["importance", "source", "category", "tags"]
            for key in essential_metadata:
                if key in memory["metadata"]:
                    compressed["metadata"][key] = memory["metadata"][key]
            
            # Add compression information
            compressed["metadata"]["compressed"] = True
            compressed["metadata"]["compressed_at"] = datetime.now().isoformat()
            compressed["metadata"]["original_size"] = len(str(memory))
        
        return compressed
    
    def _archive_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Archive a memory by creating a minimal reference.
        
        Args:
            memory: Memory to archive
            
        Returns:
            Archived memory reference
        """
        # Create a minimal reference
        archived = {
            "id": memory.get("id", f"archived_{int(time.time())}"),
            "type": f"archived_{memory.get('type', 'memory')}",
            "archived_at": datetime.now().isoformat(),
            "original_timestamp": memory.get("timestamp"),
            "metadata": {
                "archived": True,
                "original_type": memory.get("type"),
                "content_summary": self._summarize_content(memory.get("content", "")),
                "original_size": len(str(memory))
            }
        }
        
        return archived
    
    def _summarize_content(self, content: Any) -> str:
        """
        Create a brief summary of content.
        
        Args:
            content: Content to summarize
            
        Returns:
            Brief summary string
        """
        if not content:
            return "[empty]"
        
        if isinstance(content, str):
            if len(content) <= 50:
                return content
            else:
                # Extract first 50 characters
                return content[:50] + "..."
        
        # For non-string content, return type information
        return f"[{type(content).__name__}]"
    
    def _calculate_compression_ratio(self, 
                                   original: List[Dict[str, Any]], 
                                   compressed: List[Dict[str, Any]]) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original: Original memories
            compressed: Compressed memories
            
        Returns:
            Compression ratio (original size / compressed size)
        """
        if not original or not compressed:
            return 1.0
        
        original_size = sum(len(str(m)) for m in original)
        compressed_size = sum(len(str(m)) for m in compressed)
        
        if compressed_size == 0:
            return float('inf')
        
        return original_size / compressed_size


class ConsolidationScheduler:
    """
    Scheduler for memory consolidation processes.
    
    Manages background consolidation tasks with configurable intervals and triggers.
    """
    
    def __init__(self, 
                episodic_interval_hours: float = 24.0, 
                pattern_interval_hours: float = 48.0,
                optimization_interval_hours: float = 72.0):
        """
        Initialize the consolidation scheduler.
        
        Args:
            episodic_interval_hours: Interval for episodic consolidation
            pattern_interval_hours: Interval for pattern extraction
            optimization_interval_hours: Interval for memory optimization
        """
        self.episodic_interval = episodic_interval_hours * 3600  # Convert to seconds
        self.pattern_interval = pattern_interval_hours * 3600
        self.optimization_interval = optimization_interval_hours * 3600
        
        self.last_episodic = 0
        self.last_pattern = 0
        self.last_optimization = 0
        
        self.episodic_consolidator = EpisodicConsolidator()
        self.pattern_extractor = PatternExtractor()
        self.memory_optimizer = MemoryOptimizer()
    
    def check_and_run_consolidation(self, 
                                  episodic_memories: List[Dict[str, Any]], 
                                  action_sequences: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Check if consolidation is due and run if needed.
        
        Args:
            episodic_memories: List of episodic memories
            action_sequences: List of action sequences
            
        Returns:
            Dictionary with consolidation results
        """
        current_time = time.time()
        results = {"tasks_run": 0}
        
        # Check episodic consolidation
        if current_time - self.last_episodic >= self.episodic_interval:
            results["episodic"] = self.episodic_consolidator.consolidate_episodes(episodic_memories)
            self.last_episodic = current_time
            results["tasks_run"] += 1
        
        # Check pattern extraction
        if current_time - self.last_pattern >= self.pattern_interval:
            results["pattern"] = self.pattern_extractor.extract_patterns(action_sequences)
            self.last_pattern = current_time
            results["tasks_run"] += 1
        
        # Check memory optimization
        if current_time - self.last_optimization >= self.optimization_interval:
            results["optimization"] = self.memory_optimizer.optimize_memories(episodic_memories)
            self.last_optimization = current_time
            results["tasks_run"] += 1
        
        return results
    
    def force_consolidation(self, 
                          episodic_memories: List[Dict[str, Any]], 
                          action_sequences: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Force immediate consolidation of all types.
        
        Args:
            episodic_memories: List of episodic memories
            action_sequences: List of action sequences
            
        Returns:
            Dictionary with consolidation results
        """
        current_time = time.time()
        results = {
            "episodic": self.episodic_consolidator.consolidate_episodes(episodic_memories),
            "pattern": self.pattern_extractor.extract_patterns(action_sequences),
            "optimization": self.memory_optimizer.optimize_memories(episodic_memories),
            "tasks_run": 3
        }
        
        # Update last run times
        self.last_episodic = current_time
        self.last_pattern = current_time
        self.last_optimization = current_time
        
        return results
    
    def get_next_consolidation_times(self) -> Dict[str, str]:
        """
        Get the next scheduled consolidation times.
        
        Returns:
            Dictionary with next consolidation times
        """
        current_time = time.time()
        
        next_episodic = self.last_episodic + self.episodic_interval
        next_pattern = self.last_pattern + self.pattern_interval
        next_optimization = self.last_optimization + self.optimization_interval
        
        return {
            "episodic": datetime.fromtimestamp(next_episodic).isoformat(),
            "pattern": datetime.fromtimestamp(next_pattern).isoformat(),
            "optimization": datetime.fromtimestamp(next_optimization).isoformat(),
            "soonest": datetime.fromtimestamp(min(next_episodic, next_pattern, next_optimization)).isoformat(),
            "seconds_until_next": min(next_episodic, next_pattern, next_optimization) - current_time
        }
