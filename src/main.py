"""
Main entry point for the AI Studio Agents monorepo.

This module provides the entry points for running different agents
and demonstrates how to initialize and use them.
"""

import asyncio
import os
import sys
from typing import Optional

from src.agents.taat.agent import TaatAgent
from src.agent_core.config import load_config


async def run_taat_agent():
    """Initialize and run the TAAT agent."""
    # Load TAAT-specific configuration
    config = load_config(agent_type="taat")
    
    # Initialize the TAAT agent
    agent = TaatAgent(config=config)
    
    # Run the agent's main loop
    await agent.run_loop()


def main():
    """Main entry point for the application."""
    # Check if an agent type was specified
    agent_type = os.environ.get("AGENT_TYPE", "taat").lower()
    
    if agent_type == "taat":
        asyncio.run(run_taat_agent())
    elif agent_type == "all_use":
        print("ALL-USE agent not yet implemented")
    elif agent_type == "mentor":
        print("MENTOR agent not yet implemented")
    else:
        print(f"Unknown agent type: {agent_type}")
        print("Supported types: taat, all_use, mentor")
        sys.exit(1)


if __name__ == "__main__":
    main()
