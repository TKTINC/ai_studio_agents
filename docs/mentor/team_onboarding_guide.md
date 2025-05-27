# MENTOR Agent Team Onboarding Guide

## Introduction

Welcome to the MENTOR (Market Execution & Navigation Through Optimized Reasoning) agent project! This guide is designed to help new team members understand the project's vision, architecture, and development approach. MENTOR is a personal investment agent platform that learns each user's unique investment philosophy and executes it perfectly on their behalf.

## Project Vision

MENTOR solves the universal tension in investment technology through:

1. **Philosophy Agnosticism**: Adapts to any investment approach rather than imposing a specific strategy
2. **Execution Translation**: Converts simple investment ideas into perfectly executed trades
3. **Emotional Discipline**: Maintains the user's strategy during market volatility
4. **Time Liberation**: Monitors markets 24/7 for opportunities matching user criteria
5. **Learning Partnership**: Continuously improves understanding of user preferences

## Agent-Oriented iAI Framework

MENTOR follows the agent-oriented iAI framework, which organizes the system around core agent capabilities rather than traditional microservices:

### Core Agent Capabilities

1. **Perception**: Processing market data, user input, and environmental signals
2. **Cognition**: Decision-making, strategy translation, and reasoning
3. **Action**: Order construction, execution, and verification
4. **Memory**: Working, episodic, semantic, and procedural memory systems
5. **Learning**: Preference learning, feedback processing, and pattern recognition

### Development Principles

1. **Learning-First Design**: All components designed to improve through experience
2. **Progressive Autonomy**: Increasing levels of autonomy as user trust develops
3. **Explainable Intelligence**: Clear communication of reasoning and decisions
4. **Modular Components**: Clear interfaces between components for independent development
5. **Continuous Improvement**: Regular feedback loops and performance metrics

## Project Structure

The MENTOR agent repository is organized as follows:

```
/mentor_agent/
├── docs/
│   ├── framework/             # Core architectural documents
│   ├── implementation_responses/ # Implementation details and decisions
│   └── templates/             # Documentation templates
├── src/
│   ├── agent_core/            # Core agent capabilities
│   │   ├── perception/        # Market data and user input processing
│   │   ├── cognition/         # Decision-making and strategy translation
│   │   ├── action/            # Order execution and verification
│   │   └── memory/            # Working and long-term memory systems
│   ├── learning_systems/      # Learning and adaptation capabilities
│   ├── market_systems/        # Market data integration and analysis
│   └── execution_systems/     # Brokerage integration and order management
├── tests/                     # Test suite for all components
└── todo.md                    # Current development status and tasks
```

## Development Workflow

### Getting Started

1. **Environment Setup**:
   - Clone the repository: `git clone https://github.com/TKTINC/mentor_agent.git`
   - Set up Docker environment: `docker-compose up -d`
   - Install dependencies: `pip install -r requirements.txt`

2. **Understanding the Architecture**:
   - Review the architecture proposal in `docs/framework/architecture_proposal.md`
   - Examine the workstreams and phases in `docs/framework/workstreams_and_phases.md`
   - Study the implementation plan in `docs/framework/project_implementation_plan.md`

3. **Current Status**:
   - Check the current development status in `todo.md`
   - Review recent implementation responses in `docs/implementation_responses/`

### Development Process

1. **Feature Implementation**:
   - Features are organized by workstream and phase
   - Each feature should have a clear implementation prompt
   - Implementation should follow the agent-oriented iAI framework
   - All code should have comprehensive tests

2. **Documentation**:
   - Each implementation should have a corresponding implementation response
   - Code should be well-documented with docstrings and comments
   - Architecture decisions should be documented in ADRs

3. **Testing**:
   - All components should have unit tests
   - Integration tests should verify cross-component functionality
   - Simulation tests should validate behavior in market scenarios

4. **Code Review**:
   - All code should be reviewed by at least one other team member
   - Reviews should focus on architecture alignment, code quality, and test coverage
   - Implementation responses should be reviewed for completeness and clarity

## Technical Stack

### Backend
- **Language**: Python for AI/ML components, Go for performance-critical services
- **Frameworks**: FastAPI for APIs, PyTorch for ML models
- **Databases**: PostgreSQL for relational data, MongoDB for documents, Pinecone for vectors
- **Message Broker**: Kafka for event streaming
- **Caching**: Redis for in-memory caching

### Frontend
- **Framework**: React with TypeScript
- **State Management**: Redux for global state
- **UI Components**: Material-UI or custom design system
- **Data Visualization**: D3.js for custom visualizations
- **API Integration**: GraphQL for flexible data fetching

### Infrastructure
- **Containerization**: Docker for services
- **Orchestration**: Kubernetes for container management
- **CI/CD**: GitHub Actions for automation
- **Monitoring**: Prometheus and Grafana
- **Logging**: ELK stack (Elasticsearch, Logstash, Kibana)

## Key Concepts

### Investment Philosophy Modeling

MENTOR represents user investment philosophies through a combination of:

1. **Preference Vectors**: Embeddings capturing investment style and priorities
2. **Risk Parameters**: Quantified risk tolerance across different scenarios
3. **Strategy Templates**: Patterns of decision-making in different contexts
4. **Feedback History**: Record of user reactions to agent suggestions and actions

### Learning Systems

The agent learns through multiple mechanisms:

1. **Explicit Feedback**: Direct user input on suggestions and actions
2. **Implicit Feedback**: Observation of user behavior and decisions
3. **Outcome Analysis**: Evaluation of strategy performance over time
4. **Pattern Recognition**: Identification of consistent approaches in user decisions

### Execution Intelligence

MENTOR optimizes execution through:

1. **Order Construction**: Building optimal orders from investment intent
2. **Timing Optimization**: Selecting ideal execution moments based on market conditions
3. **Position Sizing**: Determining appropriate position sizes based on risk parameters
4. **Verification**: Confirming execution and providing transparent reporting

## Communication Guidelines

1. **Daily Standups**: Brief updates on progress, plans, and blockers
2. **Weekly Reviews**: Comprehensive review of completed work and upcoming tasks
3. **Documentation**: All significant decisions and implementations should be documented
4. **Knowledge Sharing**: Regular sessions to share insights and learnings

## Resources

1. **Project Documentation**: Available in the `docs/` directory
2. **External Resources**: Links to relevant papers, articles, and documentation
3. **Team Contact Information**: Directory of team members and their roles
4. **Development Guidelines**: Coding standards, review process, and best practices

## Conclusion

The MENTOR agent project represents an innovative approach to personal investment, combining sophisticated financial capabilities with a learning-first design that adapts to any user's investment philosophy. By following the agent-oriented iAI framework and the guidelines in this document, you'll contribute to building a platform that transforms how people invest.

Welcome to the team!
