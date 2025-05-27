# MENTOR Agent: Workstream 1, Phase 1 Implementation Prompt

## Overview

This implementation prompt outlines the development of Workstream 1, Phase 1 (Core Agent Architecture) for the MENTOR (Market Execution & Navigation Through Optimized Reasoning) agent. This phase focuses on establishing the fundamental agent architecture, including the perception-cognition-action loop, basic memory system, and integration layer.

## Objectives

1. Design and implement the modular agent framework with perception-cognition-action loop
2. Create a basic memory system for conversation history and state tracking
3. Develop initial learning framework for preference capture
4. Build integration layer for component communication
5. Set up development environment and testing framework

## Technical Requirements

### Development Environment

1. **Docker-based Setup**:
   - Python 3.10+ base image
   - Development dependencies for testing and linting
   - Configuration for local development

2. **Dependencies**:
   - LLM Integration: OpenAI API client for GPT-4 or Anthropic API for Claude
   - Vector Database: Pinecone client for semantic search
   - Web Framework: FastAPI for API endpoints
   - Testing: pytest for unit and integration tests

3. **Configuration Management**:
   - Environment-based configuration
   - Secrets management for API keys
   - Feature flags for experimental capabilities

### Core Agent Architecture

1. **Base Agent Class**:
   - Initialization with configuration
   - Perception-cognition-action loop coordination
   - State management and context tracking
   - Event handling and lifecycle management

2. **Perception Module**:
   - Input processing interface
   - Natural language understanding for investment concepts
   - Market data processing framework
   - Extensible processor registry

3. **Cognition Module**:
   - LLM integration for reasoning
   - Context preparation and prompt engineering
   - Response parsing and structured output
   - Decision framework for action selection

4. **Action Module**:
   - Action execution interface
   - Tool registry for available actions
   - Result handling and error management
   - Verification and reporting

5. **Memory Module**:
   - Working memory for conversation context
   - Short-term state tracking
   - Memory manager interface for future expansion
   - Basic persistence layer

6. **Integration Layer**:
   - Event bus for component communication
   - Standardized message formats
   - Error handling and recovery
   - Logging and monitoring hooks

## Implementation Steps

### 1. Project Structure Setup

1. Create the basic directory structure:
   ```
   /src/
     /agent_core/
       /perception/
       /cognition/
       /action/
       /memory/
     /utils/
   /tests/
   ```

2. Set up configuration management:
   - Create `config.py` with environment variable loading
   - Define configuration schema
   - Implement validation and defaults

3. Create Docker environment:
   - Dockerfile for development
   - docker-compose.yml for local services
   - Development scripts

### 2. Core Agent Implementation

1. Implement base `Agent` class:
   - Constructor with configuration
   - Main execution loop
   - State management
   - Component initialization

2. Create abstract base classes for components:
   - `BasePerception` interface
   - `BaseCognition` interface
   - `BaseAction` interface
   - `BaseMemory` interface

3. Implement basic component communication:
   - Event-driven architecture
   - Message passing between components
   - Error handling and recovery

### 3. Memory System Implementation

1. Create `WorkingMemory` class:
   - Conversation history storage
   - State tracking
   - Context window management
   - Serialization and deserialization

2. Implement `MemoryManager`:
   - Memory registration and access
   - Query interface
   - Memory lifecycle management
   - Future extensibility hooks

3. Create basic persistence layer:
   - In-memory storage for development
   - File-based persistence for testing
   - Database interface for future implementation

### 4. Perception Implementation

1. Create `InputProcessor` class:
   - Text input processing
   - Intent recognition framework
   - Entity extraction
   - Context enrichment

2. Implement `ProcessorRegistry`:
   - Processor registration and discovery
   - Pipeline configuration
   - Input routing
   - Result aggregation

3. Create basic processors:
   - `TextProcessor` for natural language
   - `CommandProcessor` for structured commands
   - `MarketDataProcessor` for market information

### 5. Cognition Implementation

1. Create `LLMCognition` class:
   - LLM client integration
   - Prompt template management
   - Response parsing
   - Error handling and retry logic

2. Implement `DecisionEngine`:
   - Action selection framework
   - Context preparation
   - Reasoning trace capture
   - Confidence scoring

3. Create system prompts:
   - Investment domain knowledge
   - Conversation management
   - Decision-making framework
   - Explanation generation

### 6. Action Implementation

1. Create `ActionExecutor` class:
   - Action execution framework
   - Result handling
   - Error management
   - Verification

2. Implement `ToolRegistry`:
   - Tool registration and discovery
   - Parameter validation
   - Permission management
   - Usage tracking

3. Create basic tools:
   - `ConversationTool` for user interaction
   - `QueryTool` for information retrieval
   - `MemoryTool` for memory operations

### 7. Integration Layer Implementation

1. Create `EventBus` class:
   - Event publication and subscription
   - Topic management
   - Delivery guarantees
   - Error handling

2. Implement `MessageFormat`:
   - Standardized message structure
   - Serialization and deserialization
   - Validation
   - Versioning

3. Create integration utilities:
   - Logging integration
   - Monitoring hooks
   - Debugging tools
   - Performance tracking

### 8. Testing Framework

1. Set up pytest configuration:
   - Test discovery
   - Fixtures
   - Mocks
   - Coverage reporting

2. Create component tests:
   - Unit tests for each component
   - Integration tests for component interactions
   - End-to-end tests for basic scenarios

3. Implement test utilities:
   - Mock LLM responses
   - Test data generation
   - Assertion helpers
   - Performance benchmarks

## Deliverables

1. **Code**:
   - Complete implementation of core agent architecture
   - Basic memory system
   - Integration layer
   - Testing framework

2. **Documentation**:
   - Architecture documentation
   - API documentation
   - Component interaction diagrams
   - Implementation decisions and rationale

3. **Tests**:
   - Unit tests for all components
   - Integration tests for component interactions
   - End-to-end tests for basic scenarios
   - Test coverage report

4. **Development Environment**:
   - Docker setup
   - Local development configuration
   - CI/CD pipeline configuration

## Success Criteria

1. **Functional Requirements**:
   - Agent can process natural language input
   - Basic conversation context is maintained
   - Simple investment concepts are understood
   - Component communication works reliably
   - State is properly managed and persisted

2. **Technical Requirements**:
   - All tests pass with >80% coverage
   - Code follows style guidelines
   - Documentation is complete and accurate
   - Docker environment works correctly
   - Performance meets baseline requirements

3. **Quality Requirements**:
   - Code is modular and extensible
   - Interfaces are well-defined
   - Error handling is comprehensive
   - Logging is appropriate and useful
   - Security considerations are addressed

## Implementation Considerations

1. **Extensibility**:
   - Design for future expansion of capabilities
   - Clear extension points for all components
   - Plugin architecture where appropriate
   - Version-compatible interfaces

2. **Performance**:
   - Efficient memory usage
   - Optimized LLM interactions
   - Asynchronous processing where appropriate
   - Caching for repeated operations

3. **Security**:
   - Secure handling of API keys
   - Input validation and sanitization
   - Proper error handling without information leakage
   - Audit logging of operations

4. **Maintainability**:
   - Clear code organization
   - Comprehensive documentation
   - Consistent naming conventions
   - Appropriate abstraction levels

## Next Steps

After completing Workstream 1, Phase 1, the next phase will focus on:

1. **Advanced Memory Systems**:
   - Episodic memory implementation
   - Semantic memory development
   - Procedural memory creation
   - Memory integration and management

This will build on the foundation established in Phase 1, expanding the agent's capabilities to include more sophisticated memory systems for learning and adaptation.

## Conclusion

This implementation prompt provides a comprehensive guide for developing the core agent architecture for the MENTOR agent. By following this approach, the development team will establish a solid foundation for the agent's capabilities, enabling future expansion and enhancement in subsequent phases.
