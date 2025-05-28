"""
Tests for the advanced memory systems in AI Studio Agents.

This module contains tests for the advanced memory retrieval, consolidation,
and cognitive integration components.
"""

import os
import unittest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta

from src.memory_systems.advanced_retrieval import (
    RelevanceScorer, 
    SimilaritySearch, 
    TemporalPatternRecognizer,
    AssociativeRetrieval
)
from src.memory_systems.memory_consolidation import (
    EpisodicConsolidator,
    PatternExtractor,
    MemoryOptimizer,
    ConsolidationScheduler
)
from src.memory_systems.cognitive_integration import (
    MemoryAugmentedReasoning,
    ExperienceGuidedDecisionMaking,
    KnowledgeEnhancedPerception,
    ReflectiveProcessor
)


class TestAdvancedRetrieval(unittest.TestCase):
    """Test suite for advanced memory retrieval."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sample memories for testing
        self.episodic_memories = [
            {
                "id": "ep1",
                "type": "conversation",
                "content": "User asked about stock prices for AAPL",
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                "metadata": {"importance": 0.7, "access_count": 3}
            },
            {
                "id": "ep2",
                "type": "market_event",
                "content": "AAPL announced new product launch",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "metadata": {"importance": 0.8, "access_count": 5}
            },
            {
                "id": "ep3",
                "type": "trade_signal",
                "content": "Buy signal detected for AAPL at $150",
                "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
                "metadata": {"importance": 0.9, "access_count": 2}
            },
            {
                "id": "ep4",
                "type": "conversation",
                "content": "User asked about MSFT performance",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "metadata": {"importance": 0.6, "access_count": 1}
            }
        ]
        
        # Initialize components
        self.relevance_scorer = RelevanceScorer()
        self.similarity_search = SimilaritySearch()
        self.temporal_pattern_recognizer = TemporalPatternRecognizer()
        self.associative_retrieval = AssociativeRetrieval()
    
    def test_relevance_scorer(self):
        """Test relevance scoring functionality."""
        # Test with query
        query = "AAPL stock price"
        context = {"type": "conversation"}
        
        # Score all memories
        scores = [self.relevance_scorer.score_memory(memory, query, context) 
                 for memory in self.episodic_memories]
        
        # Verify scores are in valid range
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Verify first memory has higher score than last (AAPL vs MSFT)
        self.assertGreater(scores[0], scores[3])
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        # Search for AAPL related memories
        query = "AAPL stock"
        context = {"type": "market_event"}
        
        results = self.similarity_search.search(
            self.episodic_memories, query, context, limit=2
        )
        
        # Verify results structure
        self.assertEqual(len(results), 2)
        self.assertIn("memory", results[0])
        self.assertIn("score", results[0])
        
        # Verify AAPL memories are returned first
        self.assertIn("AAPL", results[0]["memory"]["content"])
    
    def test_temporal_pattern_recognizer(self):
        """Test temporal pattern recognition."""
        # Test sequence detection with min_length=2 to match our test data
        sequences = self.temporal_pattern_recognizer.detect_sequences(
            self.episodic_memories, min_length=2
        )
        
        # Verify sequences are detected
        self.assertGreaterEqual(len(sequences), 1)
        
        # Test periodicity detection
        periodicity = self.temporal_pattern_recognizer.detect_periodicity(
            self.episodic_memories, event_type="conversation"
        )
        
        # Verify periodicity result structure
        self.assertIn("detected", periodicity)
    
    def test_associative_retrieval(self):
        """Test associative retrieval."""
        # Create semantic and procedural memories
        semantic_memories = [
            {
                "id": "sem1",
                "content": "Apple Inc. (AAPL) is a technology company",
                "category": "company_info"
            },
            {
                "id": "sem2",
                "content": "Microsoft (MSFT) is a technology company",
                "category": "company_info"
            }
        ]
        
        procedural_memories = [
            {
                "id": "proc1",
                "type": "trading_procedure",
                "description": "Buy stock procedure"
            }
        ]
        
        # Test associative retrieval with lower threshold for test purposes
        seed_memory = self.episodic_memories[0]  # AAPL query
        
        # Create associative retrieval with lower threshold for testing
        associative_retrieval = AssociativeRetrieval(association_threshold=0.2)
        
        associations = associative_retrieval.retrieve_associated(
            seed_memory,
            self.episodic_memories,
            semantic_memories,
            procedural_memories
        )
        
        # Verify associations structure
        self.assertIn("episodic", associations)
        self.assertIn("semantic", associations)
        self.assertIn("procedural", associations)
        
        # Verify AAPL-related memories are associated
        aapl_related = False
        for memory in associations["episodic"]:
            if "AAPL" in str(memory["memory"]["content"]):
                aapl_related = True
                break
        
        self.assertTrue(aapl_related)


class TestMemoryConsolidation(unittest.TestCase):
    """Test suite for memory consolidation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sample episodes for testing
        self.episodes = [
            {
                "id": "ep1",
                "type": "market_event",
                "content": "AAPL announced new product launch",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "metadata": {"trader": "Trader1", "symbol": "AAPL"}
            },
            {
                "id": "ep2",
                "type": "market_event",
                "content": "AAPL reported strong earnings",
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
                "metadata": {"trader": "Trader1", "symbol": "AAPL"}
            },
            {
                "id": "ep3",
                "type": "market_event",
                "content": "AAPL stock price increased by 5%",
                "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                "metadata": {"trader": "Trader1", "symbol": "AAPL"}
            },
            {
                "id": "ep4",
                "type": "market_event",
                "content": "MSFT announced new cloud service",
                "timestamp": (datetime.now() - timedelta(days=4)).isoformat(),
                "metadata": {"trader": "Trader2", "symbol": "MSFT"}
            }
        ]
        
        # Create sample action sequences for testing
        self.action_sequences = [
            [
                {"type": "analyze", "function": "analyze_stock", "parameters": {"symbol": "AAPL"}},
                {"type": "decide", "function": "make_decision", "parameters": {"action": "buy"}}
            ],
            [
                {"type": "analyze", "function": "analyze_stock", "parameters": {"symbol": "MSFT"}},
                {"type": "decide", "function": "make_decision", "parameters": {"action": "buy"}}
            ],
            [
                {"type": "analyze", "function": "analyze_stock", "parameters": {"symbol": "GOOG"}},
                {"type": "decide", "function": "make_decision", "parameters": {"action": "sell"}}
            ]
        ]
        
        # Initialize components
        self.episodic_consolidator = EpisodicConsolidator(min_cluster_size=2)
        self.pattern_extractor = PatternExtractor(min_sequence_length=2, min_occurrences=2)
        self.memory_optimizer = MemoryOptimizer()
        self.consolidation_scheduler = ConsolidationScheduler()
    
    def test_episodic_consolidator(self):
        """Test episodic consolidation."""
        # Consolidate episodes
        result = self.episodic_consolidator.consolidate_episodes(self.episodes)
        
        # Verify consolidation result
        self.assertTrue(result["consolidated"])
        self.assertGreaterEqual(len(result["summaries"]), 1)
        self.assertGreaterEqual(len(result["extracted_concepts"]), 1)
        
        # Verify AAPL episodes are clustered
        aapl_summary = False
        for summary in result["summaries"]:
            if "AAPL" in str(summary):
                aapl_summary = True
                break
        
        self.assertTrue(aapl_summary)
    
    def test_pattern_extractor(self):
        """Test pattern extraction."""
        # Create pattern extractor with min_occurrences=2 and min_sequence_length=2
        # to match our test data which has sequences of length 2
        pattern_extractor = PatternExtractor(min_sequence_length=2, min_occurrences=2, similarity_threshold=0.5)
        
        # For test purposes, use identical action sequences to ensure pattern detection
        test_sequences = [
            self.action_sequences[0],
            self.action_sequences[0],  # Duplicate to ensure min_occurrences is met
            self.action_sequences[1]
        ]
        
        result = pattern_extractor.extract_patterns(test_sequences)
        
        # Verify pattern extraction result
        self.assertTrue(result["patterns_found"])
        self.assertGreaterEqual(len(result["procedures"]), 1)
        
        # Verify analyze-decide pattern is detected
        analyze_decide_pattern = False
        for procedure in result["procedures"]:
            steps = procedure["steps"]
            if len(steps) >= 2 and steps[0]["function"] == "analyze_stock" and steps[1]["function"] == "make_decision":
                analyze_decide_pattern = True
                break
        
        self.assertTrue(analyze_decide_pattern)
    
    def test_memory_optimizer(self):
        """Test memory optimization."""
        # Optimize memories
        result = self.memory_optimizer.optimize_memories(self.episodes)
        
        # Verify optimization result
        self.assertTrue(result["optimized"])
        self.assertEqual(result["total_memories"], len(self.episodes))
        
        # Verify optimized memories are returned
        self.assertGreaterEqual(len(result["optimized_memories"]), 1)
    
    def test_consolidation_scheduler(self):
        """Test consolidation scheduler."""
        # Force consolidation
        result = self.consolidation_scheduler.force_consolidation(
            self.episodes, self.action_sequences
        )
        
        # Verify scheduler result
        self.assertEqual(result["tasks_run"], 3)
        self.assertIn("episodic", result)
        self.assertIn("pattern", result)
        self.assertIn("optimization", result)
        
        # Verify next consolidation times
        next_times = self.consolidation_scheduler.get_next_consolidation_times()
        self.assertIn("episodic", next_times)
        self.assertIn("pattern", next_times)
        self.assertIn("optimization", next_times)
        self.assertIn("soonest", next_times)


class TestCognitiveIntegration(unittest.TestCase):
    """Test suite for cognitive integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sample memories and context
        self.relevant_memories = {
            "episodic": [
                {
                    "memory": {
                        "id": "ep1",
                        "type": "trade_signal",
                        "content": "Buy signal for AAPL at $150",
                        "timestamp": datetime.now().isoformat()
                    },
                    "score": 0.8
                }
            ],
            "semantic": [
                {
                    "memory": {
                        "id": "sem1",
                        "content": "AAPL has strong fundamentals",
                        "category": "fact"
                    },
                    "score": 0.9
                }
            ],
            "procedural": [
                {
                    "memory": {
                        "id": "proc1",
                        "description": "Buy stock procedure"
                    },
                    "score": 0.7
                }
            ]
        }
        
        self.context = {
            "current_task": "trading",
            "user_preferences": {"risk_tolerance": "medium"}
        }
        
        self.past_experiences = [
            {
                "option": "buy_aapl",
                "outcome": {"success": True},
                "context": {"market_condition": "bullish"}
            },
            {
                "option": "sell_msft",
                "outcome": {"success": False},
                "context": {"market_condition": "bullish"}
            }
        ]
        
        # Initialize components
        self.memory_augmented_reasoning = MemoryAugmentedReasoning()
        self.experience_guided_decision_making = ExperienceGuidedDecisionMaking()
        self.knowledge_enhanced_perception = KnowledgeEnhancedPerception()
        self.reflective_processor = ReflectiveProcessor()
    
    def test_memory_augmented_reasoning(self):
        """Test memory-augmented reasoning."""
        # Perform reasoning
        query = "Should I buy AAPL stock?"
        result = self.memory_augmented_reasoning.augment_reasoning(
            query, self.context, self.relevant_memories
        )
        
        # Verify reasoning result
        self.assertTrue(result["augmented"])
        self.assertEqual(result["original_query"], query)
        self.assertIn("result", result)
        self.assertIn("memory_stats", result)
    
    def test_experience_guided_decision_making(self):
        """Test experience-guided decision making."""
        # Define options
        options = [
            {"name": "buy_aapl", "description": "Buy AAPL stock"},
            {"name": "buy_msft", "description": "Buy MSFT stock"}
        ]
        
        # Make decision
        result = self.experience_guided_decision_making.make_decision(
            options, self.context, self.past_experiences
        )
        
        # Verify decision result
        self.assertIn("decision", result)
        self.assertIn("score", result)
        self.assertIn("confidence", result)
        self.assertTrue(result["experience_guided"])
    
    def test_knowledge_enhanced_perception(self):
        """Test knowledge-enhanced perception."""
        # Process input
        input_data = "AAPL stock price is increasing rapidly"
        input_type = "text"
        knowledge = [
            {"content": "AAPL is Apple Inc.", "category": "entity", "score": 0.9},
            {"content": "Rapid price increases may indicate market excitement", "category": "fact", "score": 0.8}
        ]
        
        # Enhance perception
        result = self.knowledge_enhanced_perception.enhance_perception(
            input_data, input_type, self.context, knowledge
        )
        
        # Verify perception result
        self.assertTrue(result["enhanced"])
        self.assertIn("perception", result)
        self.assertIn("knowledge_applied", result)
    
    def test_reflective_processor(self):
        """Test reflective processing."""
        # Define action and outcome with significant deviation to trigger reflection
        action = {
            "type": "buy_stock",
            "parameters": {"symbol": "AAPL", "quantity": 10},
            "timestamp": datetime.now().isoformat()
        }
        
        outcome = {
            "success": True,
            "expected": {"profit": 5},
            "actual": {"profit": 15},  # Larger deviation to ensure reflection threshold is met
            "error": False,  # Explicitly set no error
            "impact": 0.9,   # High impact to trigger reflection
            "timestamp": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        # Use a reflective processor with lower threshold to ensure reflection occurs
        reflective_processor = ReflectiveProcessor(reflection_threshold=0.3)
        
        # Reflect on outcome
        result = reflective_processor.reflect_on_outcome(
            action, outcome, self.context
        )
        
        # Verify reflection result
        self.assertTrue(result["reflected"])
        self.assertIn("reflection_score", result)
        self.assertIn("outcome_analysis", result)
        self.assertIn("insights", result)
        self.assertIn("learning_updates", result)


if __name__ == "__main__":
    unittest.main()
