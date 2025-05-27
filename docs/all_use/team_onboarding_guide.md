# ALL-USE Agent Team Onboarding Guide

## Introduction to Agent-Oriented Development

Welcome to the ALL-USE Agent development team! This guide will help you understand our agent-oriented approach to AI development and get you up to speed with the project.

### What is the Agent-Oriented iAI Framework?

The agent-oriented iAI (intelligent AI) Framework is our approach to building AI systems as autonomous agents rather than traditional software applications. Key principles include:

1. **Perception-Cognition-Action Loop**: Agents process inputs (perception), reason about them (cognition), and take actions based on reasoning.
2. **Memory Systems**: Agents maintain different types of memory (episodic, semantic, procedural) to store and retrieve information.
3. **Learning Capabilities**: Agents learn from experience through reinforcement learning and pattern recognition.
4. **Autonomous Operation**: Agents can operate independently while following defined objectives and constraints.
5. **Human-Agent Collaboration**: Agents are designed to work effectively with human users and other agents.

## ALL-USE Agent Overview

The ALL-USE (Automated Lumpsum Leveraged US Equities) Agent is an autonomous agent designed to manage a triple-account trading system that generates consistent income and builds long-term wealth through options trading in US equity markets.

### Project Structure

```
all_use_agent/
├── docs/                      # Documentation
│   ├── framework/            # Implementation prompts and architecture
│   ├── implementation_responses/ # Implementation documentation
│   └── templates/            # Documentation templates
├── src/                       # Source code
│   ├── agent_core/           # Core agent architecture
│   │   ├── perception/       # Market data and signal processing
│   │   ├── cognition/        # Strategy selection and decision making
│   │   ├── action/           # Trade execution and position management
│   │   └── memory/           # Memory systems for knowledge storage
│   ├── learning_systems/     # Learning and optimization capabilities
│   └── trading_systems/      # Trading strategies and execution logic
├── tests/                     # Test suite
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```

### Workstreams and Phases

The project is organized into workstreams (major functional areas) and phases (implementation stages):

1. **Workstream 1: Agent Foundation**
   - Phase 1: Core Agent Architecture
   - Phase 2: Memory Systems
   - Phase 3: Learning Systems

2. **Workstream 2: Perception Systems**
   - Phase 1: Market Data Integration
   - Phase 2: Account State Monitoring
   - Phase 3: External Signal Processing

3. **Workstream 3: Cognition Systems**
   - Phase 1: Week-Type Classification
   - Phase 2: Strategy Selection and Adaptation
   - Phase 3: Risk Management and Planning

4. **Workstream 4: Action Systems**
   - Phase 1: Trade Execution
   - Phase 2: Position Management
   - Phase 3: Account Lifecycle Management

5. **Workstream 5: Triple-Account Strategy Implementation**
   - Phase 1: Gen-Acc Strategy
   - Phase 2: Rev-Acc Strategy
   - Phase 3: Com-Acc Strategy

6. **Workstream 6: Collaboration and Integration**
   - Phase 1: Human-Agent Collaboration
   - Phase 2: Agent-Agent Collaboration
   - Phase 3: External System Integration

## Development Environment Setup

### Prerequisites

- Python 3.10+
- Docker
- Git
- GitHub account with repository access

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TKTINC/all_use_agent.git
   cd all_use_agent
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.sample .env
   # Edit .env with your API keys and configuration
   ```

3. **Build and run with Docker**:
   ```bash
   docker build -t all-use-agent .
   docker run -it --env-file .env all-use-agent
   ```

4. **Alternative: Local Python setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running Tests

```bash
python -m pytest
```

## Development Workflow

### Closed-Loop Documentation Approach

We follow a closed-loop documentation approach:

1. **Implementation Prompt**: Detailed requirements and specifications
2. **Implementation**: Code development based on the prompt
3. **Implementation Response**: Documentation of what was implemented, challenges, and future enhancements
4. **Review and Feedback**: Iterative improvement based on feedback

### Git Workflow

1. **Branch naming**: Use `ws<number>-phase<number>-<feature>` format
   - Example: `ws1-phase1-core-agent-architecture`

2. **Commit messages**: Clear, descriptive messages with workstream/phase reference
   - Example: "Implement perception module for Workstream 2, Phase 1"

3. **Pull requests**: Include reference to implementation prompt and summary of changes

### Code Standards

1. **Documentation**: All modules, classes, and functions must be documented
2. **Testing**: All components must have corresponding tests
3. **Type hints**: Use Python type hints throughout the codebase
4. **Async patterns**: Use async/await for I/O-bound operations
5. **Error handling**: Comprehensive error handling with appropriate logging

## Key Technical Components

### Agent Core

- **Perception**: Processes market data, account states, and external signals
- **Cognition**: Makes trading decisions based on market conditions and strategies
- **Action**: Executes trades and manages positions
- **Memory**: Maintains knowledge of trading history and market patterns

### Memory Systems

- **Episodic Memory**: Stores trade history and market patterns
- **Semantic Memory**: Maintains knowledge about market conditions and strategies
- **Procedural Memory**: Records successful trading sequences

### Learning Systems

- **Reinforcement Learning**: Optimizes strategy parameters
- **Pattern Recognition**: Identifies market conditions and successful strategies
- **Cross-Account Learning**: Shares insights between account strategies

### Trading Systems

- **Gen-Acc Strategy**: Weekly ATM wheel strategy (40-50 delta)
- **Rev-Acc Strategy**: Income-stable wheel (30-40 delta OTM)
- **Com-Acc Strategy**: Long-hold compounding with shares and LEAPS

## Resources and References

- [GitHub Repository](https://github.com/TKTINC/all_use_agent)
- [Project Implementation Plan](/docs/framework/project_implementation_plan.md)
- [Architecture Mapping](/docs/framework/architecture_mapping.md)
- [Workstreams and Phases](/docs/framework/workstreams_and_phases.md)
- [Gaps and Opportunities](/docs/framework/gaps_and_opportunities.md)

## Getting Help

If you have questions or need assistance:

1. Review the implementation prompts and responses in the docs directory
2. Check the tests for examples of component usage
3. Reach out to the team lead for guidance

Welcome aboard, and happy coding!
