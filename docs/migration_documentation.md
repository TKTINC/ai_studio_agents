# AI Studio Agents Migration Documentation

## Overview

This document details the migration of the TAAT, ALL-USE, and MENTOR agent codebases into a unified monorepo structure with shared components. The migration follows the modular architecture design outlined in the shared components catalog and repository splitting strategy.

## Monorepo Structure

The AI Studio Agents monorepo is organized as follows:

```
/ai_studio_agents/
  /src/
    /agent_core/           # Shared framework components
      /perception/         # Base perception modules
      /cognition/          # Base cognition modules
      /action/             # Base action modules
      /memory/             # Base memory systems
      /learning/           # Base learning systems
    /shared_modules/       # Domain-specific shared components
      /market_data/        # Market data processing
      /technical_analysis/ # Technical indicators and analysis
      /risk_management/    # Risk management utilities
      /execution/          # Order execution
      /notification/       # Notification systems
    /agents/               # Agent-specific implementations
      /taat/               # TAAT-specific code
      /all_use/            # ALL-USE-specific code
      /mentor/             # MENTOR-specific code
    main.py                # Main entry point
  /deployment/             # Deployment configurations
    /taat/                 # TAAT-specific deployment
    /all_use/              # ALL-USE-specific deployment
    /mentor/               # MENTOR-specific deployment
  /tests/                  # Test suite
  /docs/                   # Documentation
  /scripts/                # Utility scripts
  /requirements/           # Dependency management
```

## Migration Process

### Phase 1: Repository Setup and Structure (Completed)

- Created the AI Studio Agents monorepo
- Established the directory structure for shared components and agent-specific code
- Set up the basic project configuration

### Phase 2: TAAT Migration (Completed)

- Extracted core components from TAAT into shared modules:
  - Base agent architecture (BaseAgent)
  - Configuration management (AgentConfig)
  - Memory systems (WorkingMemory)
  - Perception framework (BasePerceptionModule)
  - Cognition framework (BaseCognitionModule)
  - Action framework (BaseActionModule, ToolRegistry)

- Implemented TAAT-specific extensions:
  - TaatAgent (extends BaseAgent)
  - TaatPerceptionModule (extends BasePerceptionModule)
  - TaatCognitionModule (extends BaseCognitionModule)
  - TaatActionModule (extends BaseActionModule)

- Created main entry point with support for multiple agent types

### Phase 3: ALL-USE and MENTOR Integration (In Progress)

- Analyzing ALL-USE and MENTOR codebases for shared component opportunities
- Planning integration of agent-specific modules while leveraging shared components
- Ensuring operational independence for each agent

### Phase 4: CI/CD Configuration (Planned)

- Setting up CI/CD pipelines for independent agent deployment
- Configuring build triggers based on file paths
- Implementing testing strategies for shared and agent-specific code

## Shared Components

The following shared components have been extracted and refactored:

1. **BaseAgent**: Core agent architecture implementing the perception-cognition-action loop
2. **AgentConfig**: Configuration management with support for agent-specific settings
3. **WorkingMemory**: Memory system for conversation history and state tracking
4. **BasePerceptionModule**: Input processing framework with extensible processor registration
5. **BaseCognitionModule**: Decision-making framework using Claude LLM
6. **BaseActionModule**: Action execution framework with tool registry

## Agent-Specific Extensions

Each agent extends the base components with specialized functionality:

### TAAT
- Social media monitoring for trade signals
- Trade signal analysis and execution
- Notification and logging systems

### ALL-USE (Planned)
- Triple-account structure management
- Options strategy execution
- Portfolio management

### MENTOR (Planned)
- User philosophy learning
- Personalized execution framework
- Adaptation and continuous improvement

## Next Steps

1. Complete integration of ALL-USE and MENTOR codebases
2. Configure CI/CD for independent agent deployment
3. Validate operational integrity of all agents
4. Implement shared domain-specific modules (market data, technical analysis, etc.)
5. Enhance documentation and testing

## Future Considerations

The monorepo structure is designed to support future repository splitting for acquisition purposes. Each agent can be extracted into a standalone repository while maintaining its functionality through the shared components, which would be packaged as dependencies.
