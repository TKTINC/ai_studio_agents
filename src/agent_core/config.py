"""
Configuration management for AI Studio Agents.

This module handles loading and validating configuration from environment variables
and configuration files for all agent types.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LLMSettings:
    """Settings for the LLM integration."""
    api_key: str
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: str = ""
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Base configuration for all AI Studio Agents."""
    llm_settings: LLMSettings
    agent_type: str = "base"
    debug_mode: bool = False
    log_level: str = "INFO"
    max_history: int = 10
    additional_settings: Dict[str, Any] = field(default_factory=dict)


def load_config(agent_type: str = "base") -> AgentConfig:
    """
    Load configuration from environment variables.
    
    Args:
        agent_type: Type of agent (base, taat, all_use, mentor)
    
    Returns:
        AgentConfig: The loaded configuration
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Check for required environment variables
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    # Load LLM settings
    llm_settings = LLMSettings(
        api_key=anthropic_api_key,
        model=os.environ.get("LLM_MODEL", "claude-3-sonnet-20240229"),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
        system_prompt=os.environ.get(f"{agent_type.upper()}_SYSTEM_PROMPT", ""),
    )
    
    # Load agent config
    debug_mode = os.environ.get("DEBUG_MODE", "False").lower() == "true"
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    max_history = int(os.environ.get("MAX_HISTORY", "10"))
    
    # Create agent-specific additional settings
    additional_settings = {}
    
    # Agent-specific configuration
    if agent_type == "taat":
        # Add TAAT-specific settings
        additional_settings["twitter_api_key"] = os.environ.get("TWITTER_API_KEY", "")
        additional_settings["twitter_api_secret"] = os.environ.get("TWITTER_API_SECRET", "")
        additional_settings["twitter_access_token"] = os.environ.get("TWITTER_ACCESS_TOKEN", "")
        additional_settings["twitter_access_secret"] = os.environ.get("TWITTER_ACCESS_SECRET", "")
    elif agent_type == "all_use":
        # Add ALL-USE-specific settings
        additional_settings["ibkr_account_id"] = os.environ.get("IBKR_ACCOUNT_ID", "")
        additional_settings["ibkr_api_key"] = os.environ.get("IBKR_API_KEY", "")
    elif agent_type == "mentor":
        # Add MENTOR-specific settings
        additional_settings["user_profile_db"] = os.environ.get("USER_PROFILE_DB", "")
    
    return AgentConfig(
        llm_settings=llm_settings,
        agent_type=agent_type,
        debug_mode=debug_mode,
        log_level=log_level,
        max_history=max_history,
        additional_settings=additional_settings,
    )
