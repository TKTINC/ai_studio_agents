"""
Memory-Augmented Cognition for AI Studio Agents.

This module implements enhanced cognitive processes that leverage memory systems
for improved reasoning, decision-making, and learning.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os
import time
from datetime import datetime
import logging
import math
import random

# Set up logging
logger = logging.getLogger(__name__)


class MemoryAugmentedReasoning:
    """
    Memory-augmented reasoning system.
    
    Enhances reasoning processes with relevant memories and knowledge.
    """
    
    def __init__(self, relevance_threshold: float = 0.6):
        """
        Initialize the memory-augmented reasoning system.
        
        Args:
            relevance_threshold: Minimum relevance score for memories to be included
        """
        self.relevance_threshold = relevance_threshold
    
    def augment_reasoning(self, 
                         query: str, 
                         context: Dict[str, Any], 
                         relevant_memories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Augment reasoning with relevant memories.
        
        Args:
            query: The reasoning query
            context: Current context
            relevant_memories: Dictionary of relevant memories by type
            
        Returns:
            Augmented reasoning result
        """
        # Filter memories by relevance
        filtered_memories = self._filter_relevant_memories(relevant_memories)
        
        if not filtered_memories["episodic"] and not filtered_memories["semantic"] and not filtered_memories["procedural"]:
            return {
                "augmented": False,
                "reason": "No relevant memories found",
                "original_query": query,
                "result": self._basic_reasoning(query, context)
            }
        
        # Organize memories by relevance to query
        organized_memories = self._organize_memories(filtered_memories, query)
        
        # Generate reasoning with memory augmentation
        augmented_result = self._generate_augmented_reasoning(query, context, organized_memories)
        
        return {
            "augmented": True,
            "original_query": query,
            "result": augmented_result,
            "memory_stats": {
                "episodic_used": len(filtered_memories["episodic"]),
                "semantic_used": len(filtered_memories["semantic"]),
                "procedural_used": len(filtered_memories["procedural"])
            }
        }
    
    def _filter_relevant_memories(self, 
                                relevant_memories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter memories by relevance score.
        
        Args:
            relevant_memories: Dictionary of relevant memories by type
            
        Returns:
            Filtered memories
        """
        filtered = {
            "episodic": [],
            "semantic": [],
            "procedural": []
        }
        
        for memory_type, memories in relevant_memories.items():
            for memory in memories:
                # Check if memory has a score field
                if "score" in memory and memory["score"] >= self.relevance_threshold:
                    filtered[memory_type].append(memory)
                # If no score field, use memory directly if it's a dict
                elif isinstance(memory, dict) and "memory" in memory and "score" in memory:
                    if memory["score"] >= self.relevance_threshold:
                        filtered[memory_type].append(memory)
        
        return filtered
    
    def _organize_memories(self, 
                         filtered_memories: Dict[str, List[Dict[str, Any]]], 
                         query: str) -> Dict[str, Any]:
        """
        Organize memories by relevance to query.
        
        Args:
            filtered_memories: Filtered memories by type
            query: The reasoning query
            
        Returns:
            Organized memories
        """
        organized = {
            "facts": [],          # Factual knowledge from semantic memory
            "experiences": [],    # Relevant past experiences from episodic memory
            "procedures": [],     # Relevant procedures from procedural memory
            "contradictions": [], # Contradictory information
            "supporting": []      # Supporting information
        }
        
        # Extract facts from semantic memory
        for memory in filtered_memories["semantic"]:
            mem = memory.get("memory", memory)
            content = mem.get("content", "")
            category = mem.get("category", "")
            
            if category in ["fact", "concept", "knowledge"]:
                organized["facts"].append({
                    "content": content,
                    "confidence": memory.get("score", 0.5)
                })
        
        # Extract experiences from episodic memory
        for memory in filtered_memories["episodic"]:
            mem = memory.get("memory", memory)
            content = mem.get("content", "")
            
            organized["experiences"].append({
                "content": content,
                "timestamp": mem.get("timestamp", ""),
                "relevance": memory.get("score", 0.5)
            })
        
        # Extract procedures from procedural memory
        for memory in filtered_memories["procedural"]:
            mem = memory.get("memory", memory)
            
            organized["procedures"].append({
                "id": mem.get("id", ""),
                "description": mem.get("description", "Procedure"),
                "applicability": memory.get("score", 0.5)
            })
        
        # Identify contradictions and supporting information
        self._identify_contradictions_and_support(organized)
        
        return organized
    
    def _identify_contradictions_and_support(self, organized: Dict[str, Any]) -> None:
        """
        Identify contradictions and supporting information among memories.
        
        Args:
            organized: Organized memories
        """
        # Simple keyword-based contradiction detection (placeholder for more sophisticated analysis)
        # In a real implementation, this would use semantic understanding to detect contradictions
        
        # Check for contradictions between facts
        facts = organized["facts"]
        for i in range(len(facts)):
            for j in range(i+1, len(facts)):
                fact1 = facts[i]["content"].lower()
                fact2 = facts[j]["content"].lower()
                
                # Check for negation patterns
                if ("not " in fact1 and fact1.replace("not ", "") in fact2) or \
                   ("not " in fact2 and fact2.replace("not ", "") in fact1):
                    organized["contradictions"].append({
                        "type": "fact_contradiction",
                        "items": [facts[i]["content"], facts[j]["content"]],
                        "confidence": min(facts[i]["confidence"], facts[j]["confidence"])
                    })
        
        # Check for supporting information
        for fact in facts:
            for exp in organized["experiences"]:
                if fact["content"].lower() in exp["content"].lower():
                    organized["supporting"].append({
                        "type": "experience_supports_fact",
                        "fact": fact["content"],
                        "experience": exp["content"],
                        "confidence": fact["confidence"] * exp["relevance"]
                    })
    
    def _generate_augmented_reasoning(self, 
                                    query: str, 
                                    context: Dict[str, Any], 
                                    organized_memories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate reasoning augmented with memories.
        
        Args:
            query: The reasoning query
            context: Current context
            organized_memories: Organized memories
            
        Returns:
            Augmented reasoning result
        """
        # Start with basic reasoning
        result = self._basic_reasoning(query, context)
        
        # Augment with facts
        if organized_memories["facts"]:
            result["factual_basis"] = [fact["content"] for fact in organized_memories["facts"]]
        
        # Augment with experiences
        if organized_memories["experiences"]:
            result["experiential_evidence"] = [exp["content"] for exp in organized_memories["experiences"]]
        
        # Augment with procedures
        if organized_memories["procedures"]:
            result["applicable_procedures"] = [proc["id"] for proc in organized_memories["procedures"]]
        
        # Include contradictions and supporting information
        if organized_memories["contradictions"]:
            result["contradictions"] = organized_memories["contradictions"]
        
        if organized_memories["supporting"]:
            result["supporting_evidence"] = organized_memories["supporting"]
        
        # Adjust confidence based on memory support
        base_confidence = result.get("confidence", 0.5)
        memory_support = self._calculate_memory_support(organized_memories)
        
        result["confidence"] = min(0.95, base_confidence + memory_support * 0.3)
        result["memory_augmented"] = True
        
        return result
    
    def _basic_reasoning(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform basic reasoning without memory augmentation.
        
        Args:
            query: The reasoning query
            context: Current context
            
        Returns:
            Basic reasoning result
        """
        # This is a placeholder for actual reasoning logic
        # In a real implementation, this would use an LLM or other reasoning system
        
        return {
            "conclusion": f"Basic reasoning for: {query}",
            "confidence": 0.5,
            "reasoning_path": ["Initial analysis", "Logical deduction", "Conclusion"],
            "memory_augmented": False
        }
    
    def _calculate_memory_support(self, organized_memories: Dict[str, Any]) -> float:
        """
        Calculate the level of memory support for reasoning.
        
        Args:
            organized_memories: Organized memories
            
        Returns:
            Memory support score between 0.0 and 1.0
        """
        support_score = 0.0
        
        # Add support from facts
        if organized_memories["facts"]:
            fact_support = sum(fact["confidence"] for fact in organized_memories["facts"])
            fact_support = min(1.0, fact_support / len(organized_memories["facts"]))
            support_score += 0.4 * fact_support
        
        # Add support from experiences
        if organized_memories["experiences"]:
            exp_support = sum(exp["relevance"] for exp in organized_memories["experiences"])
            exp_support = min(1.0, exp_support / len(organized_memories["experiences"]))
            support_score += 0.3 * exp_support
        
        # Add support from procedures
        if organized_memories["procedures"]:
            proc_support = sum(proc["applicability"] for proc in organized_memories["procedures"])
            proc_support = min(1.0, proc_support / len(organized_memories["procedures"]))
            support_score += 0.2 * proc_support
        
        # Reduce support if contradictions exist
        if organized_memories["contradictions"]:
            contradiction_penalty = min(0.5, 0.1 * len(organized_memories["contradictions"]))
            support_score = max(0.0, support_score - contradiction_penalty)
        
        # Increase support if supporting evidence exists
        if organized_memories["supporting"]:
            support_bonus = min(0.3, 0.05 * len(organized_memories["supporting"]))
            support_score = min(1.0, support_score + support_bonus)
        
        return support_score


class ExperienceGuidedDecisionMaking:
    """
    Experience-guided decision making system.
    
    Uses past experiences to guide decision-making processes.
    """
    
    def __init__(self, 
                experience_weight: float = 0.7, 
                recency_factor: float = 0.3,
                exploration_rate: float = 0.1):
        """
        Initialize the experience-guided decision making system.
        
        Args:
            experience_weight: Weight given to past experiences vs. model predictions
            recency_factor: Factor for weighting recent experiences more heavily
            exploration_rate: Rate of exploration vs. exploitation
        """
        self.experience_weight = experience_weight
        self.recency_factor = recency_factor
        self.exploration_rate = exploration_rate
    
    def make_decision(self, 
                     options: List[Dict[str, Any]], 
                     context: Dict[str, Any], 
                     past_experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make a decision guided by past experiences.
        
        Args:
            options: List of decision options
            context: Current context
            past_experiences: List of relevant past experiences
            
        Returns:
            Decision result
        """
        if not options:
            return {"decision": None, "reason": "No options provided"}
        
        # If no past experiences, fall back to basic decision making
        if not past_experiences:
            return self._make_basic_decision(options, context)
        
        # Calculate option scores based on past experiences
        option_scores = self._score_options_from_experiences(options, context, past_experiences)
        
        # Apply exploration vs. exploitation
        if random.random() < self.exploration_rate:
            # Exploration: choose a random option with some bias toward higher scores
            exploration_scores = [score**0.5 for score in option_scores]  # Flatten distribution
            total_score = sum(exploration_scores)
            
            if total_score > 0:
                # Weighted random selection
                r = random.random() * total_score
                cumulative = 0
                for i, score in enumerate(exploration_scores):
                    cumulative += score
                    if cumulative >= r:
                        selected_index = i
                        break
                else:
                    selected_index = 0
            else:
                # Completely random if all scores are 0
                selected_index = random.randint(0, len(options) - 1)
                
            decision = {
                "decision": options[selected_index],
                "score": option_scores[selected_index],
                "reason": "Exploration: trying a potentially suboptimal option to gather more information",
                "exploration": True,
                "confidence": 0.3 + 0.4 * option_scores[selected_index]  # Lower confidence for exploration
            }
        else:
            # Exploitation: choose the highest scoring option
            best_index = option_scores.index(max(option_scores))
            
            decision = {
                "decision": options[best_index],
                "score": option_scores[best_index],
                "reason": "Exploitation: choosing the option with the best expected outcome based on past experiences",
                "exploration": False,
                "confidence": 0.5 + 0.5 * option_scores[best_index]  # Higher confidence for exploitation
            }
        
        return decision
    
    def _make_basic_decision(self, 
                           options: List[Dict[str, Any]], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a basic decision without past experiences.
        
        Args:
            options: List of decision options
            context: Current context
            
        Returns:
            Decision result
        """
        # This is a placeholder for actual decision-making logic
        # In a real implementation, this would use an LLM or other decision system
        
        # For now, just choose the first option
        return {
            "decision": options[0],
            "score": 0.5,
            "reason": "No past experiences available, using default decision-making",
            "exploration": False,
            "confidence": 0.5
        }
    
    def _score_options_from_experiences(self, 
                                      options: List[Dict[str, Any]], 
                                      context: Dict[str, Any], 
                                      past_experiences: List[Dict[str, Any]]) -> List[float]:
        """
        Score options based on past experiences.
        
        Args:
            options: List of decision options
            context: Current context
            past_experiences: List of relevant past experiences
            
        Returns:
            List of scores for each option
        """
        option_scores = [0.0] * len(options)
        
        # Calculate recency weights for experiences
        now = datetime.now()
        recency_weights = []
        
        for exp in past_experiences:
            if "timestamp" in exp:
                try:
                    exp_time = datetime.fromisoformat(exp["timestamp"].replace('Z', '+00:00'))
                    age_days = (now - exp_time).total_seconds() / (24 * 3600)
                    weight = math.exp(-self.recency_factor * age_days)
                    recency_weights.append(weight)
                except (ValueError, AttributeError):
                    recency_weights.append(1.0)
            else:
                recency_weights.append(1.0)
        
        # Normalize recency weights
        total_weight = sum(recency_weights)
        if total_weight > 0:
            recency_weights = [w / total_weight for w in recency_weights]
        else:
            recency_weights = [1.0 / len(past_experiences)] * len(past_experiences)
        
        # Score each option based on past experiences
        for i, option in enumerate(options):
            option_score = 0.0
            
            for j, exp in enumerate(past_experiences):
                # Calculate similarity between current option and past experience
                similarity = self._calculate_similarity(option, exp, context)
                
                # Get outcome score from past experience
                outcome_score = exp.get("outcome_score", 0.5)
                
                # Combine similarity, outcome score, and recency weight
                option_score += similarity * outcome_score * recency_weights[j]
            
            option_scores[i] = option_score
        
        # Normalize scores
        max_score = max(option_scores) if option_scores else 0.0
        if max_score > 0:
            option_scores = [score / max_score for score in option_scores]
        
        return option_scores
    
    def _calculate_similarity(self, 
                            option: Dict[str, Any], 
                            experience: Dict[str, Any], 
                            context: Dict[str, Any]) -> float:
        """
        Calculate similarity between current option and past experience.
        
        Args:
            option: Current decision option
            experience: Past experience
            context: Current context
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # This is a placeholder for a more sophisticated similarity calculation
        # In a real implementation, this would use semantic similarity or feature-based comparison
        
        # Simple attribute matching
        match_count = 0
        total_checks = 0
        
        # Check option attributes against experience
        if "attributes" in option and "attributes" in experience:
            opt_attrs = option["attributes"]
            exp_attrs = experience["attributes"]
            
            for key, value in opt_attrs.items():
                if key in exp_attrs:
                    total_checks += 1
                    if exp_attrs[key] == value:
                        match_count += 1
        
        # Check context attributes against experience context
        if "context" in experience:
            exp_context = experience["context"]
            
            for key, value in context.items():
                if key in exp_context:
                    total_checks += 1
                    if exp_context[key] == value:
                        match_count += 1
        
        # If no checks were possible, return low similarity
        if total_checks == 0:
            return 0.1
        
        return match_count / total_checks


class CognitiveIntegration:
    """
    Cognitive Integration System for TAAT.
    
    Integrates memory systems with cognitive processes for enhanced reasoning,
    decision-making, and learning capabilities.
    """
    
    def __init__(self):
        """Initialize the cognitive integration system."""
        self.memory_augmented_reasoning = MemoryAugmentedReasoning()
        self.experience_guided_decision = ExperienceGuidedDecisionMaking()
        self.episodic_memory = None
        self.semantic_memory = None
        self.procedural_memory = None
        self.advanced_retrieval = None
        self.memory_consolidation = None
        self.logger = logging.getLogger("CognitiveIntegration")
    
    def connect_memory_systems(self, 
                             episodic_memory: Any, 
                             semantic_memory: Any, 
                             procedural_memory: Any,
                             advanced_retrieval: Any = None,
                             memory_consolidation: Any = None) -> None:
        """
        Connect to memory systems.
        
        Args:
            episodic_memory: Episodic memory system
            semantic_memory: Semantic memory system
            procedural_memory: Procedural memory system
            advanced_retrieval: Optional advanced retrieval system
            memory_consolidation: Optional memory consolidation system
        """
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.procedural_memory = procedural_memory
        self.advanced_retrieval = advanced_retrieval
        self.memory_consolidation = memory_consolidation
        self.logger.info("Connected to memory systems")
    
    def reason_with_memories(self, 
                           query: str, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform reasoning augmented with relevant memories.
        
        Args:
            query: The reasoning query
            context: Current context
            
        Returns:
            Reasoning result
        """
        # Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(query, context)
        
        # Perform memory-augmented reasoning
        reasoning_result = self.memory_augmented_reasoning.augment_reasoning(
            query=query,
            context=context,
            relevant_memories=relevant_memories
        )
        
        # Record reasoning in episodic memory if available
        if self.episodic_memory:
            self.episodic_memory.store(
                memory_type="reasoning",
                content={
                    "query": query,
                    "result": reasoning_result["result"]["conclusion"]
                },
                metadata={
                    "timestamp": datetime.now(),
                    "context": context,
                    "confidence": reasoning_result["result"].get("confidence", 0.5),
                    "memory_augmented": reasoning_result["result"].get("memory_augmented", False)
                }
            )
        
        return reasoning_result
    
    def make_decision_with_experiences(self, 
                                     options: List[Dict[str, Any]], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a decision guided by past experiences.
        
        Args:
            options: List of decision options
            context: Current context
            
        Returns:
            Decision result
        """
        # Retrieve relevant past experiences
        past_experiences = self._retrieve_past_experiences(options, context)
        
        # Make experience-guided decision
        decision_result = self.experience_guided_decision.make_decision(
            options=options,
            context=context,
            past_experiences=past_experiences
        )
        
        # Record decision in episodic memory if available
        if self.episodic_memory:
            self.episodic_memory.store(
                memory_type="decision",
                content={
                    "options": options,
                    "selected": decision_result["decision"]
                },
                metadata={
                    "timestamp": datetime.now(),
                    "context": context,
                    "confidence": decision_result.get("confidence", 0.5),
                    "exploration": decision_result.get("exploration", False)
                }
            )
        
        return decision_result
    
    def learn_from_feedback(self, 
                          memory_id: str, 
                          feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from feedback on a past memory.
        
        Args:
            memory_id: ID of the memory to update
            feedback: Feedback information
            
        Returns:
            Learning result
        """
        if not self.episodic_memory:
            return {"learned": False, "error": "No episodic memory connected"}
        
        # Retrieve the memory
        memory = self.episodic_memory.retrieve_by_id(memory_id)
        
        if not memory:
            return {"learned": False, "error": f"Memory {memory_id} not found"}
        
        # Update memory with feedback
        memory_type = memory.get("type")
        
        if memory_type == "decision":
            # For decisions, update with outcome information
            outcome_score = feedback.get("outcome_score", 0.5)
            
            self.episodic_memory.update_memory(
                memory_id=memory_id,
                metadata_updates={
                    "outcome_score": outcome_score,
                    "feedback": feedback.get("feedback", ""),
                    "feedback_timestamp": datetime.now()
                }
            )
            
            # Extract procedural knowledge if consolidation is available
            if self.memory_consolidation:
                self.memory_consolidation.extract_procedural_knowledge([memory])
        
        elif memory_type == "reasoning":
            # For reasoning, update with correctness information
            correctness = feedback.get("correctness", 0.5)
            
            self.episodic_memory.update_memory(
                memory_id=memory_id,
                metadata_updates={
                    "correctness": correctness,
                    "feedback": feedback.get("feedback", ""),
                    "feedback_timestamp": datetime.now()
                }
            )
            
            # Extract semantic knowledge if consolidation is available
            if self.memory_consolidation:
                self.memory_consolidation.extract_semantic_knowledge([memory])
        
        else:
            # For other memory types, just add feedback
            self.episodic_memory.update_memory(
                memory_id=memory_id,
                metadata_updates={
                    "feedback": feedback.get("feedback", ""),
                    "feedback_timestamp": datetime.now()
                }
            )
        
        # Trigger memory consolidation if available
        if self.memory_consolidation:
            self.memory_consolidation.consolidate_memories(recent_memories=[memory_id])
        
        return {
            "learned": True,
            "memory_id": memory_id,
            "memory_type": memory_type,
            "feedback_applied": feedback
        }
    
    def integrate_new_knowledge(self, 
                              knowledge: Dict[str, Any], 
                              knowledge_type: str) -> Dict[str, Any]:
        """
        Integrate new knowledge into memory systems.
        
        Args:
            knowledge: New knowledge to integrate
            knowledge_type: Type of knowledge (episodic, semantic, procedural)
            
        Returns:
            Integration result
        """
        result = {
            "integrated": False,
            "memory_id": None,
            "knowledge_type": knowledge_type
        }
        
        if knowledge_type == "episodic" and self.episodic_memory:
            # Store in episodic memory
            memory_id = self.episodic_memory.store(
                memory_type=knowledge.get("type", "observation"),
                content=knowledge.get("content", {}),
                metadata=knowledge.get("metadata", {"timestamp": datetime.now()})
            )
            
            result["integrated"] = True
            result["memory_id"] = memory_id
        
        elif knowledge_type == "semantic" and self.semantic_memory:
            # Store in semantic memory
            concept_id = self.semantic_memory.store_concept(
                name=knowledge.get("name", "concept"),
                attributes=knowledge.get("attributes", {}),
                relationships=knowledge.get("relationships", [])
            )
            
            result["integrated"] = True
            result["memory_id"] = concept_id
        
        elif knowledge_type == "procedural" and self.procedural_memory:
            # Store in procedural memory
            procedure_id = self.procedural_memory.store_procedure(
                name=knowledge.get("name", "procedure"),
                steps=knowledge.get("steps", []),
                context=knowledge.get("context", {}),
                metadata=knowledge.get("metadata", {})
            )
            
            result["integrated"] = True
            result["memory_id"] = procedure_id
        
        # Trigger memory consolidation if available
        if result["integrated"] and self.memory_consolidation:
            self.memory_consolidation.consolidate_memories(
                recent_memories=[result["memory_id"]]
            )
        
        return result
    
    def _retrieve_relevant_memories(self, 
                                  query: str, 
                                  context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve memories relevant to a query and context.
        
        Args:
            query: The query
            context: Current context
            
        Returns:
            Dictionary of relevant memories by type
        """
        relevant_memories = {
            "episodic": [],
            "semantic": [],
            "procedural": []
        }
        
        # Use advanced retrieval if available
        if self.advanced_retrieval:
            retrieval_query = {
                "content": query,
                "context": context
            }
            
            retrieval_result = self.advanced_retrieval.retrieve(
                query=retrieval_query,
                memory_types=["episodic", "semantic", "procedural"]
            )
            
            # Process retrieval results
            for memory in retrieval_result.get("memories", []):
                memory_id = memory.get("id", "")
                memory_type = self._determine_memory_type(memory)
                
                if memory_type:
                    score = retrieval_result["relevance_scores"].get(memory_id, 0.5)
                    relevant_memories[memory_type].append({
                        "memory": memory,
                        "score": score
                    })
        
        # Fall back to basic retrieval if advanced retrieval is not available or returned no results
        if not any(relevant_memories.values()) and self.episodic_memory:
            # Basic episodic memory retrieval
            episodic_results = self.episodic_memory.search_by_content({"text": query})
            
            for memory in episodic_results:
                relevant_memories["episodic"].append({
                    "memory": memory,
                    "score": 0.7  # Default score for basic retrieval
                })
        
        if not any(relevant_memories.values()) and self.semantic_memory:
            # Basic semantic memory retrieval
            semantic_results = self.semantic_memory.get_concept_by_name(query)
            
            for concept in semantic_results:
                relevant_memories["semantic"].append({
                    "memory": concept,
                    "score": 0.7  # Default score for basic retrieval
                })
        
        return relevant_memories
    
    def _retrieve_past_experiences(self, 
                                 options: List[Dict[str, Any]], 
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve past experiences relevant to decision options and context.
        
        Args:
            options: List of decision options
            context: Current context
            
        Returns:
            List of relevant past experiences
        """
        past_experiences = []
        
        if not self.episodic_memory:
            return past_experiences
        
        # Retrieve decision memories
        decision_memories = self.episodic_memory.retrieve_by_type("decision", limit=50)
        
        # Filter by context similarity
        for memory in decision_memories:
            # Only include memories with outcome feedback
            if "metadata" in memory and "outcome_score" in memory["metadata"]:
                # Calculate context similarity
                context_similarity = self._calculate_context_similarity(
                    memory.get("metadata", {}).get("context", {}),
                    context
                )
                
                # Include if similarity is above threshold
                if context_similarity > 0.3:
                    past_experiences.append(memory)
        
        return past_experiences
    
    def _determine_memory_type(self, memory: Dict[str, Any]) -> Optional[str]:
        """
        Determine the type of a memory.
        
        Args:
            memory: Memory to check
            
        Returns:
            Memory type (episodic, semantic, procedural) or None if unknown
        """
        # Check for explicit type field
        if "type" in memory:
            memory_type = memory["type"]
            
            if memory_type in ["perception", "action", "decision", "reasoning", "observation"]:
                return "episodic"
            elif memory_type in ["concept", "fact", "knowledge"]:
                return "semantic"
            elif memory_type in ["procedure", "skill", "method"]:
                return "procedural"
        
        # Check for characteristic fields
        if "content" in memory and "metadata" in memory and "timestamp" in memory.get("metadata", {}):
            return "episodic"
        
        if "name" in memory and "attributes" in memory:
            return "semantic"
        
        if "steps" in memory or "procedure_steps" in memory:
            return "procedural"
        
        return None
    
    def _calculate_context_similarity(self, 
                                    context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two contexts.
        
        Args:
            context1: First context
            context2: Second context
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not context1 or not context2:
            return 0.0
        
        # Count matching attributes
        match_count = 0
        total_checks = 0
        
        for key, value in context1.items():
            if key in context2:
                total_checks += 1
                if context2[key] == value:
                    match_count += 1
        
        # If no checks were possible, return low similarity
        if total_checks == 0:
            return 0.0
        
        return match_count / total_checks
