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
        
        # Add experience information
        decision["experience_guided"] = True
        decision["experiences_used"] = len(past_experiences)
        
        return decision
    
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
            List of option scores
        """
        option_scores = [0.0] * len(options)
        
        # Sort experiences by recency
        sorted_experiences = sorted(
            past_experiences,
            key=lambda x: x.get("timestamp", ""),
            reverse=True  # Most recent first
        )
        
        # Calculate recency weights
        total_experiences = len(sorted_experiences)
        recency_weights = [
            1.0 - self.recency_factor * (i / total_experiences)
            for i in range(total_experiences)
        ]
        
        # For each option, calculate score from experiences
        for i, option in enumerate(options):
            option_name = option.get("name", str(i))
            
            for j, experience in enumerate(sorted_experiences):
                # Extract experience details
                exp_option = experience.get("option", "")
                exp_outcome = experience.get("outcome", {})
                exp_context = experience.get("context", {})
                
                # Check if this experience is relevant to this option
                if option_name == exp_option or option_name in str(exp_option):
                    # Calculate similarity between current context and experience context
                    context_similarity = self._calculate_context_similarity(context, exp_context)
                    
                    # Calculate outcome score (-1.0 to 1.0)
                    outcome_score = self._calculate_outcome_score(exp_outcome)
                    
                    # Calculate weighted contribution of this experience
                    contribution = context_similarity * outcome_score * recency_weights[j]
                    
                    # Add to option score
                    option_scores[i] += contribution
            
            # Normalize score to 0.0-1.0 range
            option_scores[i] = max(0.0, min(1.0, (option_scores[i] + 1.0) / 2.0))
            
            # Combine with basic decision score
            basic_score = self._calculate_basic_score(option, context)
            option_scores[i] = (
                self.experience_weight * option_scores[i] + 
                (1.0 - self.experience_weight) * basic_score
            )
        
        return option_scores
    
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
            return 0.5  # Default if either context is empty
        
        # Find common keys
        common_keys = set(context1.keys()).intersection(set(context2.keys()))
        if not common_keys:
            return 0.0
        
        # Calculate similarity for each common key
        similarities = []
        for key in common_keys:
            val1 = context1[key]
            val2 = context2[key]
            
            # Calculate value similarity based on type
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
                else:
                    similarities.append(1.0 if val1 == val2 else 0.0)
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                if val1 == val2:
                    similarities.append(1.0)
                elif val1.lower() == val2.lower():
                    similarities.append(0.9)
                elif val1.lower() in val2.lower() or val2.lower() in val1.lower():
                    similarities.append(0.7)
                else:
                    similarities.append(0.0)
            else:
                # Other types
                similarities.append(1.0 if val1 == val2 else 0.0)
        
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_outcome_score(self, outcome: Dict[str, Any]) -> float:
        """
        Calculate a score for an outcome.
        
        Args:
            outcome: Outcome data
            
        Returns:
            Score between -1.0 and 1.0
        """
        if not outcome:
            return 0.0
        
        # Check for explicit success/failure indicators
        if "success" in outcome:
            success = outcome["success"]
            if isinstance(success, bool):
                return 1.0 if success else -1.0
            elif isinstance(success, (int, float)):
                return max(-1.0, min(1.0, float(success) * 2.0 - 1.0))
        
        # Check for score or rating
        if "score" in outcome:
            score = outcome["score"]
            if isinstance(score, (int, float)):
                # Normalize to -1.0 to 1.0 range
                # Assuming score is in 0.0-1.0 range
                return max(-1.0, min(1.0, score * 2.0 - 1.0))
        
        # Check for error indicator
        if "error" in outcome and outcome["error"]:
            return -1.0
        
        # Default neutral outcome
        return 0.0
    
    def _calculate_basic_score(self, option: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calculate a basic score for an option without using experiences.
        
        Args:
            option: Decision option
            context: Current context
            
        Returns:
            Score between 0.0 and 1.0
        """
        # This is a placeholder for actual option scoring logic
        # In a real implementation, this would use a model or heuristics
        
        # Check if option has an explicit score
        if "score" in option:
            score = option["score"]
            if isinstance(score, (int, float)):
                return max(0.0, min(1.0, float(score)))
        
        # Check if option has a priority
        if "priority" in option:
            priority = option["priority"]
            if isinstance(priority, (int, float)):
                return max(0.0, min(1.0, float(priority) / 10.0))
        
        # Default score
        return 0.5
    
    def _make_basic_decision(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a basic decision without using past experiences.
        
        Args:
            options: List of decision options
            context: Current context
            
        Returns:
            Decision result
        """
        if not options:
            return {"decision": None, "reason": "No options provided"}
        
        # Calculate basic scores
        scores = [self._calculate_basic_score(option, context) for option in options]
        
        # Choose the highest scoring option
        best_index = scores.index(max(scores))
        
        return {
            "decision": options[best_index],
            "score": scores[best_index],
            "reason": "Basic decision: no relevant past experiences available",
            "experience_guided": False,
            "confidence": 0.3 + 0.4 * scores[best_index]  # Lower confidence without experiences
        }


class KnowledgeEnhancedPerception:
    """
    Knowledge-enhanced perception system.
    
    Uses semantic knowledge to enhance perception and interpretation.
    """
    
    def __init__(self, 
                knowledge_weight: float = 0.6, 
                context_weight: float = 0.3,
                novelty_threshold: float = 0.7):
        """
        Initialize the knowledge-enhanced perception system.
        
        Args:
            knowledge_weight: Weight given to knowledge-based interpretation
            context_weight: Weight given to context-based interpretation
            novelty_threshold: Threshold for detecting novel information
        """
        self.knowledge_weight = knowledge_weight
        self.context_weight = context_weight
        self.novelty_threshold = novelty_threshold
    
    def enhance_perception(self, 
                         input_data: Any, 
                         input_type: str, 
                         context: Dict[str, Any], 
                         knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance perception with semantic knowledge.
        
        Args:
            input_data: Raw input data
            input_type: Type of input data
            context: Current context
            knowledge: Relevant knowledge from semantic memory
            
        Returns:
            Enhanced perception result
        """
        # Process raw input
        basic_perception = self._process_raw_input(input_data, input_type)
        
        # If no knowledge available, return basic perception
        if not knowledge:
            return {
                "enhanced": False,
                "reason": "No relevant knowledge available",
                "perception": basic_perception
            }
        
        # Enhance with knowledge
        enhanced_perception = self._apply_knowledge_enhancement(basic_perception, knowledge, context)
        
        # Detect novel information
        novelty = self._detect_novelty(enhanced_perception, knowledge)
        
        return {
            "enhanced": True,
            "perception": enhanced_perception,
            "novelty": novelty,
            "knowledge_applied": len(knowledge)
        }
    
    def _process_raw_input(self, input_data: Any, input_type: str) -> Dict[str, Any]:
        """
        Process raw input data.
        
        Args:
            input_data: Raw input data
            input_type: Type of input data
            
        Returns:
            Basic perception result
        """
        # This is a placeholder for actual input processing logic
        # In a real implementation, this would use appropriate processing for each input type
        
        if input_type == "text":
            return self._process_text_input(input_data)
        elif input_type == "image":
            return self._process_image_input(input_data)
        elif input_type == "market_data":
            return self._process_market_data_input(input_data)
        else:
            # Generic processing
            return {
                "type": input_type,
                "raw_content": str(input_data)[:100],  # Truncate for large inputs
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_text_input(self, text: str) -> Dict[str, Any]:
        """
        Process text input.
        
        Args:
            text: Text input
            
        Returns:
            Processed text perception
        """
        # Simple text processing (placeholder for more sophisticated NLP)
        words = text.split()
        word_count = len(words)
        
        # Extract potential entities (capitalized words)
        entities = set(word for word in words if word and word[0].isupper())
        
        # Simple sentiment analysis
        positive_words = ["good", "great", "excellent", "positive", "success", "profit", "up", "gain"]
        negative_words = ["bad", "poor", "negative", "failure", "loss", "down", "decline"]
        
        sentiment_score = 0
        for word in words:
            word_lower = word.lower()
            if word_lower in positive_words:
                sentiment_score += 1
            elif word_lower in negative_words:
                sentiment_score -= 1
        
        if word_count > 0:
            sentiment_score = sentiment_score / math.sqrt(word_count)  # Normalize by sqrt of length
        
        return {
            "type": "text",
            "content": text,
            "word_count": word_count,
            "entities": list(entities),
            "sentiment": {
                "score": max(-1.0, min(1.0, sentiment_score)),
                "label": "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_image_input(self, image_data: Any) -> Dict[str, Any]:
        """
        Process image input.
        
        Args:
            image_data: Image data
            
        Returns:
            Processed image perception
        """
        # Placeholder for image processing
        return {
            "type": "image",
            "content": "Image data",
            "format": "unknown",
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_market_data_input(self, market_data: Any) -> Dict[str, Any]:
        """
        Process market data input.
        
        Args:
            market_data: Market data
            
        Returns:
            Processed market data perception
        """
        # Placeholder for market data processing
        return {
            "type": "market_data",
            "content": "Market data",
            "timestamp": datetime.now().isoformat()
        }
    
    def _apply_knowledge_enhancement(self, 
                                   perception: Dict[str, Any], 
                                   knowledge: List[Dict[str, Any]], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply knowledge enhancement to perception.
        
        Args:
            perception: Basic perception
            knowledge: Relevant knowledge
            context: Current context
            
        Returns:
            Enhanced perception
        """
        # Create a copy to avoid modifying the original
        enhanced = perception.copy()
        enhanced["knowledge_enhanced"] = True
        
        # Apply knowledge based on perception type
        if perception["type"] == "text":
            enhanced = self._enhance_text_perception(enhanced, knowledge, context)
        elif perception["type"] == "image":
            enhanced = self._enhance_image_perception(enhanced, knowledge, context)
        elif perception["type"] == "market_data":
            enhanced = self._enhance_market_data_perception(enhanced, knowledge, context)
        
        # Add knowledge sources
        enhanced["knowledge_sources"] = [
            {
                "id": k.get("id", "unknown"),
                "type": k.get("type", "unknown"),
                "relevance": k.get("score", 0.5)
            }
            for k in knowledge
        ]
        
        return enhanced
    
    def _enhance_text_perception(self, 
                               perception: Dict[str, Any], 
                               knowledge: List[Dict[str, Any]], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance text perception with knowledge.
        
        Args:
            perception: Basic text perception
            knowledge: Relevant knowledge
            context: Current context
            
        Returns:
            Enhanced text perception
        """
        enhanced = perception.copy()
        
        # Extract entities from perception
        entities = set(perception.get("entities", []))
        
        # Extract content
        content = perception.get("content", "")
        
        # Entity recognition enhancement
        known_entities = {}
        for k in knowledge:
            k_content = k.get("content", "")
            k_type = k.get("category", "")
            
            # Check if knowledge entity appears in content
            if isinstance(k_content, str) and k_content in content:
                known_entities[k_content] = {
                    "type": k_type,
                    "confidence": k.get("score", 0.5),
                    "source": k.get("id", "unknown")
                }
        
        if known_entities:
            enhanced["known_entities"] = known_entities
        
        # Topic classification
        topics = self._classify_topics(content, knowledge)
        if topics:
            enhanced["topics"] = topics
        
        # Context-aware interpretation
        if context:
            interpretation = self._generate_interpretation(content, knowledge, context)
            if interpretation:
                enhanced["interpretation"] = interpretation
        
        return enhanced
    
    def _enhance_image_perception(self, 
                                perception: Dict[str, Any], 
                                knowledge: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance image perception with knowledge.
        
        Args:
            perception: Basic image perception
            knowledge: Relevant knowledge
            context: Current context
            
        Returns:
            Enhanced image perception
        """
        # Placeholder for image perception enhancement
        return perception
    
    def _enhance_market_data_perception(self, 
                                      perception: Dict[str, Any], 
                                      knowledge: List[Dict[str, Any]], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance market data perception with knowledge.
        
        Args:
            perception: Basic market data perception
            knowledge: Relevant knowledge
            context: Current context
            
        Returns:
            Enhanced market data perception
        """
        # Placeholder for market data perception enhancement
        return perception
    
    def _classify_topics(self, content: str, knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify topics in content using knowledge.
        
        Args:
            content: Text content
            knowledge: Relevant knowledge
            
        Returns:
            List of identified topics with confidence scores
        """
        topics = []
        content_lower = content.lower()
        
        # Extract topics from knowledge
        for k in knowledge:
            if k.get("category") == "topic":
                topic_name = k.get("content", "")
                if isinstance(topic_name, str) and topic_name.lower() in content_lower:
                    topics.append({
                        "name": topic_name,
                        "confidence": k.get("score", 0.5),
                        "source": k.get("id", "unknown")
                    })
        
        # Sort by confidence
        topics.sort(key=lambda x: x["confidence"], reverse=True)
        
        return topics
    
    def _generate_interpretation(self, 
                               content: str, 
                               knowledge: List[Dict[str, Any]], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate context-aware interpretation of content.
        
        Args:
            content: Text content
            knowledge: Relevant knowledge
            context: Current context
            
        Returns:
            Interpretation data
        """
        # This is a placeholder for actual interpretation logic
        # In a real implementation, this would use an LLM or other interpretation system
        
        # Extract relevant context factors
        context_factors = {}
        for key, value in context.items():
            if key in ["user_preferences", "current_task", "recent_topics"]:
                context_factors[key] = value
        
        # Simple interpretation based on knowledge and context
        interpretation = {
            "summary": f"Interpretation of content with {len(knowledge)} knowledge items",
            "confidence": 0.5,
            "context_factors": context_factors
        }
        
        return interpretation
    
    def _detect_novelty(self, 
                      perception: Dict[str, Any], 
                      knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect novel information in perception.
        
        Args:
            perception: Enhanced perception
            knowledge: Relevant knowledge
            
        Returns:
            Novelty detection result
        """
        # This is a placeholder for actual novelty detection logic
        # In a real implementation, this would use more sophisticated comparison
        
        # Extract content
        if perception["type"] == "text":
            content = perception.get("content", "")
            
            # Check how much of the content is covered by existing knowledge
            coverage = 0.0
            for k in knowledge:
                k_content = k.get("content", "")
                if isinstance(k_content, str) and k_content in content:
                    # Estimate coverage by length ratio
                    coverage += len(k_content) / len(content) if len(content) > 0 else 0
            
            # Cap coverage at 1.0
            coverage = min(1.0, coverage)
            
            # Novelty is inverse of coverage
            novelty_score = 1.0 - coverage
            
            return {
                "score": novelty_score,
                "is_novel": novelty_score > self.novelty_threshold,
                "coverage": coverage
            }
        
        # Default for other perception types
        return {
            "score": 0.5,
            "is_novel": False
        }


class ReflectiveProcessor:
    """
    Reflective processing system.
    
    Implements reflection mechanisms to learn from past actions and outcomes.
    """
    
    def __init__(self, 
                reflection_threshold: float = 0.5, 
                learning_rate: float = 0.1):
        """
        Initialize the reflective processor.
        
        Args:
            reflection_threshold: Threshold for triggering reflection
            learning_rate: Rate at which new insights update existing knowledge
        """
        self.reflection_threshold = reflection_threshold
        self.learning_rate = learning_rate
    
    def reflect_on_outcome(self, 
                         action: Dict[str, Any], 
                         outcome: Dict[str, Any], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on an action outcome to generate insights.
        
        Args:
            action: The action that was taken
            outcome: The outcome that resulted
            context: The context in which the action was taken
            
        Returns:
            Reflection results with insights
        """
        # Determine if reflection is needed
        reflection_score = self._calculate_reflection_score(action, outcome)
        
        if reflection_score < self.reflection_threshold:
            return {
                "reflected": False,
                "reason": f"Reflection score {reflection_score} below threshold {self.reflection_threshold}"
            }
        
        # Analyze the outcome
        outcome_analysis = self._analyze_outcome(action, outcome, context)
        
        # Generate insights
        insights = self._generate_insights(action, outcome, context, outcome_analysis)
        
        # Generate learning updates
        learning_updates = self._generate_learning_updates(insights)
        
        return {
            "reflected": True,
            "reflection_score": reflection_score,
            "outcome_analysis": outcome_analysis,
            "insights": insights,
            "learning_updates": learning_updates
        }
    
    def _calculate_reflection_score(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> float:
        """
        Calculate a score to determine if reflection is needed.
        
        Args:
            action: The action that was taken
            outcome: The outcome that resulted
            
        Returns:
            Reflection score between 0.0 and 1.0
        """
        score_components = []
        
        # Check for unexpected outcomes
        if "expected" in outcome and "actual" in outcome:
            expected = outcome["expected"]
            actual = outcome["actual"]
            
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                # Calculate normalized difference
                max_val = max(abs(expected), abs(actual))
                if max_val > 0:
                    difference = abs(expected - actual) / max_val
                    # Higher difference = higher reflection score
                    score_components.append(min(1.0, difference))
        
        # Check for errors
        if "error" in outcome and outcome["error"]:
            score_components.append(1.0)  # Always reflect on errors
        
        # Check for explicit success/failure
        if "success" in outcome:
            success = outcome["success"]
            if isinstance(success, bool):
                # Reflect more on failures than successes
                score_components.append(0.8 if not success else 0.3)
            elif isinstance(success, (int, float)):
                # Lower success = higher reflection score
                score_components.append(1.0 - max(0.0, min(1.0, float(success))))
        
        # Check for high-impact outcomes
        if "impact" in outcome:
            impact = outcome["impact"]
            if isinstance(impact, (int, float)):
                # Higher impact = higher reflection score
                score_components.append(max(0.0, min(1.0, float(impact))))
        
        # Return maximum score if any component is high, otherwise average
        if score_components:
            max_score = max(score_components)
            if max_score > 0.7:
                return max_score
            else:
                return sum(score_components) / len(score_components)
        
        return 0.3  # Default reflection score
    
    def _analyze_outcome(self, 
                       action: Dict[str, Any], 
                       outcome: Dict[str, Any], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the outcome of an action.
        
        Args:
            action: The action that was taken
            outcome: The outcome that resulted
            context: The context in which the action was taken
            
        Returns:
            Outcome analysis
        """
        analysis = {
            "success_level": self._determine_success_level(outcome),
            "factors": self._identify_outcome_factors(action, outcome, context),
            "deviations": self._identify_deviations(action, outcome)
        }
        
        return analysis
    
    def _determine_success_level(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the level of success of an outcome.
        
        Args:
            outcome: The outcome to analyze
            
        Returns:
            Success level data
        """
        # Check for explicit success indicator
        if "success" in outcome:
            success = outcome["success"]
            if isinstance(success, bool):
                return {
                    "level": "success" if success else "failure",
                    "score": 1.0 if success else 0.0,
                    "confidence": 1.0
                }
            elif isinstance(success, (int, float)):
                score = max(0.0, min(1.0, float(success)))
                if score > 0.7:
                    level = "success"
                elif score > 0.3:
                    level = "partial_success"
                else:
                    level = "failure"
                    
                return {
                    "level": level,
                    "score": score,
                    "confidence": 1.0
                }
        
        # Check for error indicator
        if "error" in outcome and outcome["error"]:
            return {
                "level": "failure",
                "score": 0.0,
                "confidence": 1.0,
                "error": True
            }
        
        # Check for metrics
        if "metrics" in outcome and isinstance(outcome["metrics"], dict):
            metrics = outcome["metrics"]
            
            # Calculate average of numeric metrics
            numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
            if numeric_metrics:
                avg_score = sum(numeric_metrics) / len(numeric_metrics)
                normalized_score = max(0.0, min(1.0, avg_score))
                
                if normalized_score > 0.7:
                    level = "success"
                elif normalized_score > 0.3:
                    level = "partial_success"
                else:
                    level = "failure"
                    
                return {
                    "level": level,
                    "score": normalized_score,
                    "confidence": 0.7
                }
        
        # Default if no clear indicators
        return {
            "level": "unknown",
            "score": 0.5,
            "confidence": 0.3
        }
    
    def _identify_outcome_factors(self, 
                                action: Dict[str, Any], 
                                outcome: Dict[str, Any], 
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify factors that influenced the outcome.
        
        Args:
            action: The action that was taken
            outcome: The outcome that resulted
            context: The context in which the action was taken
            
        Returns:
            List of identified factors
        """
        factors = []
        
        # Check for explicit factors in outcome
        if "factors" in outcome and isinstance(outcome["factors"], list):
            return outcome["factors"]
        
        # Check action parameters
        if "parameters" in action and isinstance(action["parameters"], dict):
            for key, value in action["parameters"].items():
                factors.append({
                    "type": "action_parameter",
                    "name": key,
                    "value": value,
                    "importance": 0.7
                })
        
        # Check context factors
        for key, value in context.items():
            # Only include simple values
            if isinstance(value, (str, int, float, bool)):
                factors.append({
                    "type": "context_factor",
                    "name": key,
                    "value": value,
                    "importance": 0.5
                })
        
        # Check timing factors
        if "timestamp" in action and "timestamp" in outcome:
            try:
                action_time = datetime.fromisoformat(action["timestamp"].replace('Z', '+00:00'))
                outcome_time = datetime.fromisoformat(outcome["timestamp"].replace('Z', '+00:00'))
                
                duration = (outcome_time - action_time).total_seconds()
                
                factors.append({
                    "type": "timing_factor",
                    "name": "duration",
                    "value": duration,
                    "unit": "seconds",
                    "importance": 0.4
                })
            except (ValueError, TypeError, AttributeError):
                pass
        
        return factors
    
    def _identify_deviations(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify deviations between expected and actual outcomes.
        
        Args:
            action: The action that was taken
            outcome: The outcome that resulted
            
        Returns:
            List of identified deviations
        """
        deviations = []
        
        # Check for explicit expected vs. actual comparisons
        if "expected" in outcome and "actual" in outcome:
            expected = outcome["expected"]
            actual = outcome["actual"]
            
            if isinstance(expected, dict) and isinstance(actual, dict):
                # Compare dictionaries
                all_keys = set(expected.keys()).union(set(actual.keys()))
                
                for key in all_keys:
                    exp_val = expected.get(key)
                    act_val = actual.get(key)
                    
                    if exp_val != act_val:
                        deviations.append({
                            "field": key,
                            "expected": exp_val,
                            "actual": act_val,
                            "significance": self._calculate_deviation_significance(exp_val, act_val)
                        })
            elif expected != actual:
                # Simple comparison
                deviations.append({
                    "field": "result",
                    "expected": expected,
                    "actual": actual,
                    "significance": self._calculate_deviation_significance(expected, actual)
                })
        
        return deviations
    
    def _calculate_deviation_significance(self, expected: Any, actual: Any) -> float:
        """
        Calculate the significance of a deviation.
        
        Args:
            expected: Expected value
            actual: Actual value
            
        Returns:
            Significance score between 0.0 and 1.0
        """
        if expected == actual:
            return 0.0
        
        # For numeric values, calculate relative difference
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            max_val = max(abs(expected), abs(actual))
            if max_val > 0:
                return min(1.0, abs(expected - actual) / max_val)
        
        # For booleans, a difference is maximally significant
        if isinstance(expected, bool) and isinstance(actual, bool):
            return 1.0
        
        # For strings, use length difference as a proxy
        if isinstance(expected, str) and isinstance(actual, str):
            max_len = max(len(expected), len(actual))
            if max_len > 0:
                return min(1.0, abs(len(expected) - len(actual)) / max_len)
        
        # Default for other types
        return 0.5
    
    def _generate_insights(self, 
                         action: Dict[str, Any], 
                         outcome: Dict[str, Any], 
                         context: Dict[str, Any], 
                         analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate insights from outcome analysis.
        
        Args:
            action: The action that was taken
            outcome: The outcome that resulted
            context: The context in which the action was taken
            analysis: Outcome analysis
            
        Returns:
            List of insights
        """
        insights = []
        
        # Generate success/failure insight
        success_level = analysis["success_level"]
        if success_level["level"] != "unknown":
            insights.append({
                "type": "outcome_pattern",
                "content": f"Action {action.get('type', 'unknown')} resulted in {success_level['level']}",
                "confidence": success_level["confidence"],
                "action_type": action.get("type", "unknown"),
                "outcome_level": success_level["level"]
            })
        
        # Generate insights from factors
        for factor in analysis["factors"]:
            if factor["importance"] > 0.5:
                insights.append({
                    "type": "factor_influence",
                    "content": f"Factor {factor['name']} influenced the outcome",
                    "confidence": factor["importance"],
                    "factor_name": factor["name"],
                    "factor_type": factor["type"]
                })
        
        # Generate insights from deviations
        for deviation in analysis["deviations"]:
            if deviation["significance"] > 0.5:
                insights.append({
                    "type": "expectation_deviation",
                    "content": f"Outcome deviated from expectation in {deviation['field']}",
                    "confidence": deviation["significance"],
                    "field": deviation["field"]
                })
        
        return insights
    
    def _generate_learning_updates(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate learning updates from insights.
        
        Args:
            insights: List of insights
            
        Returns:
            Learning updates for different memory systems
        """
        updates = {
            "semantic": [],
            "procedural": []
        }
        
        # Generate semantic memory updates
        for insight in insights:
            if insight["type"] == "factor_influence" and insight["confidence"] > 0.6:
                updates["semantic"].append({
                    "operation": "store_concept",
                    "concept_id": f"factor_{insight['factor_name']}",
                    "content": insight["content"],
                    "category": "factor_influence",
                    "confidence": insight["confidence"]
                })
            
            if insight["type"] == "outcome_pattern" and insight["confidence"] > 0.6:
                updates["semantic"].append({
                    "operation": "store_concept",
                    "concept_id": f"pattern_{insight['action_type']}_{insight['outcome_level']}",
                    "content": insight["content"],
                    "category": "outcome_pattern",
                    "confidence": insight["confidence"]
                })
        
        # Generate procedural memory updates
        action_types = set(insight["action_type"] for insight in insights 
                         if "action_type" in insight and insight["confidence"] > 0.7)
        
        for action_type in action_types:
            # Find all insights related to this action type
            related_insights = [i for i in insights if i.get("action_type") == action_type]
            
            if related_insights:
                updates["procedural"].append({
                    "operation": "update_procedure",
                    "procedure_id": f"proc_{action_type}",
                    "update_data": {
                        "insights": related_insights,
                        "learning_rate": self.learning_rate
                    }
                })
        
        return updates
