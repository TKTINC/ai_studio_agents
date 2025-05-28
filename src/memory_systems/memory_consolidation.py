"""
Memory Consolidation Module for TAAT Cognitive Framework.

This module implements memory consolidation mechanisms for the TAAT agent,
enabling the integration and organization of memories across different memory systems.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import uuid
from collections import defaultdict


class MemoryConsolidation:
    """
    Memory Consolidation System for TAAT.
    
    Consolidates memories across different memory systems, identifies patterns,
    extracts semantic knowledge from episodic memories, and optimizes memory storage.
    """
    
    def __init__(self):
        """Initialize the memory consolidation system."""
        self.episodic_memory = None
        self.semantic_memory = None
        self.procedural_memory = None
        self.consolidation_history = []
        self.consolidation_rules = []
        self.extraction_patterns = []
        self.logger = logging.getLogger("MemoryConsolidation")
    
    def connect_memory_systems(self, 
                             episodic_memory: Any, 
                             semantic_memory: Any, 
                             procedural_memory: Any) -> None:
        """
        Connect to memory systems.
        
        Args:
            episodic_memory: Episodic memory system
            semantic_memory: Semantic memory system
            procedural_memory: Procedural memory system
        """
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.procedural_memory = procedural_memory
        self.logger.info("Connected to memory systems")
    
    def register_consolidation_rule(self, 
                                  rule_id: str, 
                                  rule_function: callable,
                                  memory_types: List[str],
                                  description: str) -> None:
        """
        Register a consolidation rule.
        
        Args:
            rule_id: Unique identifier for the rule
            rule_function: Function implementing the rule
            memory_types: Types of memories the rule applies to
            description: Description of the rule
        """
        rule = {
            "id": rule_id,
            "function": rule_function,
            "memory_types": memory_types,
            "description": description,
            "created_at": datetime.now()
        }
        
        self.consolidation_rules.append(rule)
        self.logger.info(f"Registered consolidation rule {rule_id}")
    
    def register_extraction_pattern(self, 
                                  pattern_id: str, 
                                  pattern_function: callable,
                                  source_type: str,
                                  target_type: str,
                                  description: str) -> None:
        """
        Register an extraction pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            pattern_function: Function implementing the pattern
            source_type: Type of source memories
            target_type: Type of target memories
            description: Description of the pattern
        """
        pattern = {
            "id": pattern_id,
            "function": pattern_function,
            "source_type": source_type,
            "target_type": target_type,
            "description": description,
            "created_at": datetime.now()
        }
        
        self.extraction_patterns.append(pattern)
        self.logger.info(f"Registered extraction pattern {pattern_id}")
    
    def consolidate_memories(self, 
                           recent_memories: Optional[List[str]] = None,
                           retrieved_memories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Consolidate memories.
        
        Args:
            recent_memories: Optional list of recent memory IDs
            retrieved_memories: Optional list of retrieved memory IDs
            
        Returns:
            Dictionary containing consolidation results
        """
        if not self.episodic_memory:
            return {"consolidated": False, "error": "No episodic memory connected"}
        
        consolidation_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Get memories to consolidate
        memories_to_consolidate = []
        
        if recent_memories:
            for memory_id in recent_memories:
                memory = self.episodic_memory.retrieve_by_id(memory_id)
                if memory:
                    memories_to_consolidate.append(memory)
        
        if retrieved_memories:
            for memory_id in retrieved_memories:
                memory = self.episodic_memory.retrieve_by_id(memory_id)
                if memory and memory not in memories_to_consolidate:
                    memories_to_consolidate.append(memory)
        
        # If no specific memories provided, get recent memories
        if not memories_to_consolidate:
            recent = self.episodic_memory.get_recent_memories(limit=20)
            memories_to_consolidate.extend(recent)
        
        # Apply consolidation rules
        consolidation_results = []
        
        for rule in self.consolidation_rules:
            # Filter memories by type
            applicable_memories = [
                m for m in memories_to_consolidate
                if m.get("type") in rule["memory_types"]
            ]
            
            if applicable_memories:
                try:
                    rule_result = rule["function"](
                        memories=applicable_memories,
                        episodic_memory=self.episodic_memory,
                        semantic_memory=self.semantic_memory,
                        procedural_memory=self.procedural_memory
                    )
                    
                    consolidation_results.append({
                        "rule_id": rule["id"],
                        "result": rule_result,
                        "memory_count": len(applicable_memories)
                    })
                except Exception as e:
                    self.logger.error(f"Error applying consolidation rule {rule['id']}: {e}")
                    consolidation_results.append({
                        "rule_id": rule["id"],
                        "error": str(e),
                        "memory_count": len(applicable_memories)
                    })
        
        # Apply extraction patterns
        extraction_results = []
        
        for pattern in self.extraction_patterns:
            # Filter memories by source type
            applicable_memories = [
                m for m in memories_to_consolidate
                if m.get("type") == pattern["source_type"]
            ]
            
            if applicable_memories:
                try:
                    pattern_result = pattern["function"](
                        memories=applicable_memories,
                        episodic_memory=self.episodic_memory,
                        semantic_memory=self.semantic_memory,
                        procedural_memory=self.procedural_memory
                    )
                    
                    extraction_results.append({
                        "pattern_id": pattern["id"],
                        "result": pattern_result,
                        "memory_count": len(applicable_memories)
                    })
                except Exception as e:
                    self.logger.error(f"Error applying extraction pattern {pattern['id']}: {e}")
                    extraction_results.append({
                        "pattern_id": pattern["id"],
                        "error": str(e),
                        "memory_count": len(applicable_memories)
                    })
        
        # Record consolidation
        consolidation_record = {
            "id": consolidation_id,
            "timestamp": timestamp,
            "memory_count": len(memories_to_consolidate),
            "consolidation_results": consolidation_results,
            "extraction_results": extraction_results
        }
        
        self.consolidation_history.append(consolidation_record)
        
        self.logger.info(f"Consolidated {len(memories_to_consolidate)} memories")
        
        return {
            "consolidated": True,
            "id": consolidation_id,
            "timestamp": timestamp.isoformat(),
            "memory_count": len(memories_to_consolidate),
            "rule_count": len(consolidation_results),
            "pattern_count": len(extraction_results)
        }
    
    def extract_semantic_knowledge(self, 
                                 episodic_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract semantic knowledge from episodic memories.
        
        Args:
            episodic_memories: List of episodic memories
            
        Returns:
            List of extracted semantic concepts
        """
        if not self.semantic_memory:
            return []
        
        extracted_concepts = []
        
        # Group memories by type
        memories_by_type = defaultdict(list)
        for memory in episodic_memories:
            memory_type = memory.get("type", "unknown")
            memories_by_type[memory_type].append(memory)
        
        # Extract concepts from each memory type
        for memory_type, memories in memories_by_type.items():
            # Apply extraction patterns for this memory type
            for pattern in self.extraction_patterns:
                if pattern["source_type"] == memory_type and pattern["target_type"] == "semantic":
                    try:
                        pattern_result = pattern["function"](
                            memories=memories,
                            episodic_memory=self.episodic_memory,
                            semantic_memory=self.semantic_memory,
                            procedural_memory=self.procedural_memory
                        )
                        
                        if "concepts" in pattern_result:
                            extracted_concepts.extend(pattern_result["concepts"])
                    except Exception as e:
                        self.logger.error(f"Error extracting semantic knowledge with pattern {pattern['id']}: {e}")
        
        # Store extracted concepts in semantic memory
        stored_concept_ids = []
        
        for concept in extracted_concepts:
            try:
                concept_id = self.semantic_memory.store_concept(
                    name=concept["name"],
                    attributes=concept.get("attributes", {}),
                    relationships=concept.get("relationships", [])
                )
                
                stored_concept_ids.append(concept_id)
            except Exception as e:
                self.logger.error(f"Error storing extracted concept {concept.get('name')}: {e}")
        
        self.logger.info(f"Extracted and stored {len(stored_concept_ids)} semantic concepts")
        
        return [self.semantic_memory.get_concept(cid) for cid in stored_concept_ids]
    
    def extract_procedural_knowledge(self, 
                                   episodic_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract procedural knowledge from episodic memories.
        
        Args:
            episodic_memories: List of episodic memories
            
        Returns:
            List of extracted procedures
        """
        if not self.procedural_memory:
            return []
        
        extracted_procedures = []
        
        # Group memories by type
        memories_by_type = defaultdict(list)
        for memory in episodic_memories:
            memory_type = memory.get("type", "unknown")
            memories_by_type[memory_type].append(memory)
        
        # Extract procedures from each memory type
        for memory_type, memories in memories_by_type.items():
            # Apply extraction patterns for this memory type
            for pattern in self.extraction_patterns:
                if pattern["source_type"] == memory_type and pattern["target_type"] == "procedural":
                    try:
                        pattern_result = pattern["function"](
                            memories=memories,
                            episodic_memory=self.episodic_memory,
                            semantic_memory=self.semantic_memory,
                            procedural_memory=self.procedural_memory
                        )
                        
                        if "procedures" in pattern_result:
                            extracted_procedures.extend(pattern_result["procedures"])
                    except Exception as e:
                        self.logger.error(f"Error extracting procedural knowledge with pattern {pattern['id']}: {e}")
        
        # Store extracted procedures in procedural memory
        stored_procedure_ids = []
        
        for procedure in extracted_procedures:
            try:
                procedure_id = self.procedural_memory.store_procedure(
                    name=procedure["name"],
                    steps=procedure.get("steps", []),
                    context=procedure.get("context", {}),
                    metadata=procedure.get("metadata", {})
                )
                
                stored_procedure_ids.append(procedure_id)
            except Exception as e:
                self.logger.error(f"Error storing extracted procedure {procedure.get('name')}: {e}")
        
        self.logger.info(f"Extracted and stored {len(stored_procedure_ids)} procedures")
        
        return [self.procedural_memory.get_procedure(pid) for pid in stored_procedure_ids]
    
    def optimize_memory_storage(self, 
                              memory_type: Optional[str] = None,
                              older_than: Optional[datetime] = None,
                              max_memories: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize memory storage by consolidating or removing redundant memories.
        
        Args:
            memory_type: Optional type of memories to optimize
            older_than: Optional timestamp to only optimize memories older than this
            max_memories: Optional maximum number of memories to keep
            
        Returns:
            Dictionary containing optimization results
        """
        if not self.episodic_memory:
            return {"optimized": False, "error": "No episodic memory connected"}
        
        # Get memories to optimize
        memories_to_optimize = []
        
        if memory_type:
            memories_to_optimize = self.episodic_memory.retrieve_by_type(memory_type)
        else:
            # Get all memories
            recent = self.episodic_memory.get_recent_memories(limit=1000)
            memories_to_optimize.extend(recent)
        
        # Filter by age if specified
        if older_than:
            memories_to_optimize = [
                m for m in memories_to_optimize
                if "timestamp" in m and m["timestamp"] < older_than
            ]
        
        # Sort by timestamp (oldest first)
        memories_to_optimize.sort(
            key=lambda m: m.get("timestamp", datetime.min),
            reverse=False
        )
        
        # Limit number of memories if specified
        if max_memories and len(memories_to_optimize) > max_memories:
            # Keep the most recent memories
            memories_to_remove = memories_to_optimize[:-max_memories]
        else:
            memories_to_remove = []
        
        # Find redundant memories
        redundant_memories = self._find_redundant_memories(memories_to_optimize)
        memories_to_remove.extend(redundant_memories)
        
        # Remove duplicates
        memories_to_remove = list(set(memories_to_remove))
        
        # Extract knowledge before removing
        self.extract_semantic_knowledge(memories_to_remove)
        self.extract_procedural_knowledge(memories_to_remove)
        
        # Remove memories
        removed_count = 0
        for memory in memories_to_remove:
            if self.episodic_memory.delete_memory(memory["id"]):
                removed_count += 1
        
        self.logger.info(f"Optimized memory storage, removed {removed_count} memories")
        
        return {
            "optimized": True,
            "timestamp": datetime.now().isoformat(),
            "total_memories": len(memories_to_optimize),
            "removed_count": removed_count,
            "redundant_count": len(redundant_memories)
        }
    
    def get_consolidation_history(self, 
                                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get consolidation history.
        
        Args:
            limit: Optional maximum number of records to return
            
        Returns:
            List of consolidation records
        """
        if limit:
            return self.consolidation_history[-limit:]
        
        return self.consolidation_history
    
    def _find_redundant_memories(self, 
                               memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find redundant memories that can be safely removed.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            List of redundant memories
        """
        redundant = []
        
        # Group memories by type
        memories_by_type = defaultdict(list)
        for memory in memories:
            memory_type = memory.get("type", "unknown")
            memories_by_type[memory_type].append(memory)
        
        # Find redundant memories within each type
        for memory_type, type_memories in memories_by_type.items():
            # Skip if too few memories
            if len(type_memories) < 2:
                continue
            
            # Simple redundancy check: look for very similar content
            content_map = {}
            
            for memory in type_memories:
                content_str = str(memory.get("content", ""))
                content_hash = hash(content_str)
                
                if content_hash in content_map:
                    # Keep the newer memory
                    existing_memory = content_map[content_hash]
                    existing_time = existing_memory.get("timestamp", datetime.min)
                    current_time = memory.get("timestamp", datetime.min)
                    
                    if current_time > existing_time:
                        redundant.append(existing_memory)
                        content_map[content_hash] = memory
                    else:
                        redundant.append(memory)
                else:
                    content_map[content_hash] = memory
        
        return redundant
    
    def _default_consolidation_rule(self, 
                                  memories: List[Dict[str, Any]],
                                  **kwargs) -> Dict[str, Any]:
        """
        Default consolidation rule.
        
        Args:
            memories: List of memories to consolidate
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing consolidation results
        """
        # This is a placeholder for a default consolidation rule
        # In a real implementation, this would perform more sophisticated consolidation
        
        return {
            "processed": len(memories),
            "consolidated": 0
        }
    
    def _default_extraction_pattern(self, 
                                  memories: List[Dict[str, Any]],
                                  **kwargs) -> Dict[str, Any]:
        """
        Default extraction pattern.
        
        Args:
            memories: List of memories to extract from
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing extraction results
        """
        # This is a placeholder for a default extraction pattern
        # In a real implementation, this would perform more sophisticated extraction
        
        return {
            "processed": len(memories),
            "concepts": [],
            "procedures": []
        }
