"""
Test script to validate TAAT agent operational integrity in the monorepo.

This script initializes and runs the TAAT agent with test inputs to verify
that all components are functioning correctly after migration.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now imports should work correctly
from src.agents.taat.agent import TaatAgent
from src.agent_core.config import AgentConfig, LLMSettings


async def test_taat_agent():
    """Test the TAAT agent's operational integrity."""
    print("Starting TAAT agent validation test...")
    
    # Create LLMSettings instance (not a dictionary)
    llm_settings = LLMSettings(
        api_key="test_key",
        model="test_model",
        system_prompt="You are TAAT, a Twitter Trade Announcer Tool for testing."
    )
    
    # Create a test configuration with proper LLMSettings instance
    config = AgentConfig(
        agent_type="taat",
        llm_settings=llm_settings,
        log_level="DEBUG"
    )
    
    # Initialize the TAAT agent
    agent = TaatAgent(config=config)
    print("TAAT agent initialized successfully.")
    
    # Test the perception module
    print("\nTesting perception module...")
    test_social_media_input = {
        "text": "Just bought $AAPL at $190, looking for a move to $200 this week!",
        "platform": "twitter",
        "user": "test_trader",
        "timestamp": "2025-05-27T21:30:00Z"
    }
    
    # Use correct method name: process_input instead of process
    perception_result = await agent.perception.process_input(
        test_social_media_input, 
        input_type="social_media"
    )
    print(f"Perception module processed input: {perception_result}")
    
    # Test the cognition module
    print("\nTesting cognition module...")
    cognition_result = await agent.cognition.process(
        perception_result,
        {"conversation": []}
    )
    print(f"Cognition module generated response: {cognition_result}")
    
    # Test the action module
    print("\nTesting action module...")
    action_result = await agent.action.execute(cognition_result)
    print(f"Action module executed response: {action_result}")
    
    print("\nTAAT agent validation test completed successfully.")
    return {
        "perception": perception_result,
        "cognition": cognition_result,
        "action": action_result
    }


if __name__ == "__main__":
    results = asyncio.run(test_taat_agent())
    
    # Print summary
    print("\n=== TAAT Agent Validation Summary ===")
    print(f"Perception module: {'PASS' if results['perception'] else 'FAIL'}")
    print(f"Cognition module: {'PASS' if results['cognition'] else 'FAIL'}")
    print(f"Action module: {'PASS' if results['action'] else 'FAIL'}")
    print(f"Overall status: {'PASS' if all(results.values()) else 'FAIL'}")
