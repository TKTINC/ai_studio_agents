# TAAT Memory Systems Architecture

## Overview

This document outlines the memory systems architecture for the Twitter Trade Announcer Automation Tool (TAAT) as part of Workstream 2 (WS-2). The architecture builds upon the existing memory implementations in the AI Studio Agents monorepo and introduces a new procedural memory system to complete the cognitive framework.

## Memory Systems Components

TAAT's memory architecture consists of four interconnected memory systems:

1. **Episodic Memory**: Stores temporal sequences of events and experiences
2. **Semantic Memory**: Stores knowledge, concepts, facts, and relationships
3. **Procedural Memory**: Stores action sequences, skills, and trading procedures
4. **Working Memory**: Manages active context and conversation history

### 1. Episodic Memory

The episodic memory system is already implemented in the monorepo and provides the following capabilities:

- Storage of temporal events with timestamps and metadata
- Retrieval by ID, type, timeframe, and content
- Persistence to disk for long-term storage
- Chronological ordering and recency-based filtering

For TAAT, episodic memory will be used to store:
- User interactions and conversations
- Market events and trading signals
- System events and error conditions
- Performance metrics and outcomes

### 2. Semantic Memory

The semantic memory system is already implemented in the monorepo and provides the following capabilities:

- Storage of concepts with categories and metadata
- Relationship mapping between concepts
- Search by content and category
- Persistence to disk for long-term storage

For TAAT, semantic memory will be used to store:
- Trading terminology and definitions
- Market knowledge and patterns
- User preferences and settings
- System configuration parameters
- Twitter account information and API details

### 3. Procedural Memory (To Be Implemented)

The procedural memory system will be newly implemented and will provide the following capabilities:

- Storage of action sequences and procedures
- Execution of stored procedures with parameter passing
- Learning and adaptation of procedures based on outcomes
- Versioning and rollback capabilities
- Persistence to disk for long-term storage

For TAAT, procedural memory will be used to store:
- Trading procedures and strategies
- Twitter posting templates and procedures
- Error handling procedures
- Routine maintenance procedures
- Data processing workflows

### 4. Working Memory

The working memory system is already implemented in the monorepo and provides the following capabilities:

- Storage of conversation history
- State tracking across interactions
- Context management for decision-making

For TAAT, working memory will be used to:
- Maintain context during user interactions
- Track active trading sessions
- Manage temporary computational results
- Coordinate between long-term memory systems

## Integration Strategy

### Memory Manager

A new `MemoryManager` class will be implemented to coordinate between the different memory systems:

```python
class MemoryManager:
    def __init__(self, agent_id, storage_path=None):
        self.episodic = EpisodicMemory(agent_id, storage_path)
        self.semantic = SemanticMemory(agent_id, storage_path)
        self.procedural = ProceduralMemory(agent_id, storage_path)
        self.working = WorkingMemory()
        
    def store_experience(self, experience_data):
        # Store in episodic memory and extract knowledge for semantic memory
        pass
        
    def retrieve_relevant_knowledge(self, context):
        # Query across memory systems based on context
        pass
        
    def execute_procedure(self, procedure_id, parameters):
        # Execute a stored procedure from procedural memory
        pass
        
    def learn_from_outcome(self, procedure_id, outcome, metrics):
        # Update procedural memory based on outcomes
        pass
```

### Integration with Perception-Cognition-Action Loop

The memory systems will integrate with the agent's perception-cognition-action loop as follows:

1. **Perception Phase**:
   - Incoming data is processed and stored in episodic memory
   - Relevant patterns are extracted and stored in semantic memory
   - Working memory is updated with current context

2. **Cognition Phase**:
   - Working memory provides current context
   - Episodic and semantic memories provide historical context and knowledge
   - Procedural memory provides action sequences and strategies
   - Decision-making leverages all memory systems

3. **Action Phase**:
   - Selected procedures are retrieved from procedural memory
   - Actions are executed and results are stored in episodic memory
   - Outcomes are used to update procedural memory (learning)

## Procedural Memory Design

Since procedural memory is not yet implemented, here is the detailed design:

### ProceduralMemory Class

```python
class ProceduralMemory:
    def __init__(self, agent_id, storage_path=None, max_versions=5):
        self.agent_id = agent_id
        self.storage_path = storage_path
        self.max_versions = max_versions
        self.procedures = {}  # Dictionary of procedures
        
        # Create storage directory if needed
        if storage_path:
            os.makedirs(os.path.join(storage_path, agent_id, "procedural"), exist_ok=True)
            self._load_from_disk()
    
    def store_procedure(self, procedure_id, steps, parameters=None, metadata=None):
        """Store a new procedure or update an existing one."""
        timestamp = datetime.now().isoformat()
        
        if procedure_id in self.procedures:
            # Version control - store previous versions
            current = self.procedures[procedure_id]
            if "versions" not in current:
                current["versions"] = []
            
            # Add current as a version
            current_version = {
                "steps": current["steps"],
                "parameters": current["parameters"],
                "metadata": current["metadata"],
                "updated": current["updated"]
            }
            current["versions"].insert(0, current_version)
            
            # Trim versions if needed
            if len(current["versions"]) > self.max_versions:
                current["versions"] = current["versions"][:self.max_versions]
            
            # Update with new content
            current["steps"] = steps
            current["parameters"] = parameters or current["parameters"]
            current["metadata"] = metadata or current["metadata"]
            current["updated"] = timestamp
        else:
            # Create new procedure
            self.procedures[procedure_id] = {
                "id": procedure_id,
                "steps": steps,
                "parameters": parameters or {},
                "metadata": metadata or {},
                "created": timestamp,
                "updated": timestamp,
                "versions": [],
                "execution_stats": {
                    "success_count": 0,
                    "failure_count": 0,
                    "last_execution": None,
                    "average_duration": 0
                }
            }
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return procedure_id
    
    def retrieve_procedure(self, procedure_id):
        """Retrieve a procedure by ID."""
        return self.procedures.get(procedure_id)
    
    def list_procedures(self, category=None):
        """List all procedures, optionally filtered by category."""
        if category is None:
            return list(self.procedures.values())
        
        return [p for p in self.procedures.values() 
                if p.get("metadata", {}).get("category") == category]
    
    def execute_procedure(self, procedure_id, context=None, parameters=None):
        """Execute a stored procedure with given context and parameters."""
        if procedure_id not in self.procedures:
            return {"success": False, "error": "Procedure not found"}
        
        procedure = self.procedures[procedure_id]
        start_time = datetime.now()
        
        try:
            # Merge default parameters with provided ones
            merged_params = procedure["parameters"].copy()
            if parameters:
                merged_params.update(parameters)
            
            # Execute steps
            results = []
            for step in procedure["steps"]:
                # Each step should be a callable or a reference to one
                if isinstance(step, dict) and "function" in step:
                    # Function reference with parameters
                    func_name = step["function"]
                    func_params = step.get("parameters", {})
                    
                    # Resolve function (implementation depends on how functions are stored)
                    func = self._resolve_function(func_name)
                    
                    # Execute function with merged parameters
                    step_result = func(context=context, **func_params, **merged_params)
                    results.append(step_result)
                elif callable(step):
                    # Direct callable
                    step_result = step(context=context, **merged_params)
                    results.append(step_result)
            
            # Update execution stats
            duration = (datetime.now() - start_time).total_seconds()
            self._update_execution_stats(procedure_id, True, duration)
            
            return {
                "success": True,
                "results": results,
                "duration": duration
            }
            
        except Exception as e:
            # Update execution stats
            duration = (datetime.now() - start_time).total_seconds()
            self._update_execution_stats(procedure_id, False, duration)
            
            return {
                "success": False,
                "error": str(e),
                "duration": duration
            }
    
    def rollback_procedure(self, procedure_id, version_index=0):
        """Rollback a procedure to a previous version."""
        if procedure_id not in self.procedures:
            return False
            
        procedure = self.procedures[procedure_id]
        if "versions" not in procedure or version_index >= len(procedure["versions"]):
            return False
            
        # Get the version to rollback to
        version = procedure["versions"][version_index]
        
        # Add current as a new version
        current_version = {
            "steps": procedure["steps"],
            "parameters": procedure["parameters"],
            "metadata": procedure["metadata"],
            "updated": procedure["updated"]
        }
        
        # Replace current with version
        procedure["steps"] = version["steps"]
        procedure["parameters"] = version["parameters"]
        procedure["metadata"] = version["metadata"]
        procedure["updated"] = datetime.now().isoformat()
        
        # Update versions list
        procedure["versions"].insert(0, current_version)
        if len(procedure["versions"]) > self.max_versions:
            procedure["versions"] = procedure["versions"][:self.max_versions]
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return True
    
    def learn_from_outcome(self, procedure_id, outcome_data, update_procedure=True):
        """Update a procedure based on execution outcomes."""
        if procedure_id not in self.procedures:
            return False
            
        # Implementation depends on learning strategy
        # For now, just store the outcome in metadata
        procedure = self.procedures[procedure_id]
        if "learning_outcomes" not in procedure["metadata"]:
            procedure["metadata"]["learning_outcomes"] = []
            
        procedure["metadata"]["learning_outcomes"].append({
            "timestamp": datetime.now().isoformat(),
            "data": outcome_data
        })
        
        # If automatic updates are enabled, modify the procedure
        if update_procedure and "update_strategy" in outcome_data:
            # Implementation of procedure updating based on outcome
            # This would involve modifying steps, parameters, etc.
            pass
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
            
        return True
    
    def delete_procedure(self, procedure_id):
        """Delete a procedure."""
        if procedure_id in self.procedures:
            del self.procedures[procedure_id]
            
            # Persist to disk if storage path is set
            if self.storage_path:
                self._save_to_disk()
                
            return True
        return False
    
    def clear(self):
        """Clear all procedures."""
        self.procedures = {}
        if self.storage_path:
            self._save_to_disk()
    
    def _update_execution_stats(self, procedure_id, success, duration):
        """Update execution statistics for a procedure."""
        if procedure_id not in self.procedures:
            return
            
        stats = self.procedures[procedure_id]["execution_stats"]
        
        # Update success/failure counts
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
            
        # Update last execution time
        stats["last_execution"] = datetime.now().isoformat()
        
        # Update average duration
        total_executions = stats["success_count"] + stats["failure_count"]
        if total_executions == 1:
            stats["average_duration"] = duration
        else:
            stats["average_duration"] = (
                (stats["average_duration"] * (total_executions - 1) + duration) / 
                total_executions
            )
        
        # Persist to disk if storage path is set
        if self.storage_path:
            self._save_to_disk()
    
    def _resolve_function(self, func_name):
        """Resolve a function reference to a callable."""
        # Implementation depends on how functions are stored/registered
        # This is a placeholder
        return lambda **kwargs: {"function": func_name, "kwargs": kwargs}
    
    def _save_to_disk(self):
        """Save procedures to disk for persistence."""
        if not self.storage_path:
            return
            
        file_path = os.path.join(self.storage_path, self.agent_id, "procedural", "procedures.json")
        with open(file_path, 'w') as f:
            json.dump(self.procedures, f)
    
    def _load_from_disk(self):
        """Load procedures from disk."""
        if not self.storage_path:
            return
            
        file_path = os.path.join(self.storage_path, self.agent_id, "procedural", "procedures.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    self.procedures = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Handle corrupted file
                self.procedures = {}
```

## Persistence Strategy

All memory systems will use a consistent persistence strategy:

1. **Storage Location**:
   - Base path: Configurable storage path
   - Agent-specific subdirectory: `{storage_path}/{agent_id}/`
   - Memory-specific subdirectory: `{storage_path}/{agent_id}/{memory_type}/`

2. **File Format**:
   - JSON for structured data
   - Binary formats for large data or embeddings (future enhancement)

3. **Persistence Timing**:
   - Immediate persistence after critical updates
   - Periodic persistence for frequent updates
   - Explicit persistence on shutdown

4. **Recovery Strategy**:
   - Automatic loading on initialization
   - Error handling for corrupted files
   - Fallback to empty state if loading fails

## Memory Usage Patterns

### TAAT-Specific Memory Usage

1. **Twitter Monitoring**:
   - Episodic: Store raw tweets and detection events
   - Semantic: Store known accounts, keywords, and patterns
   - Procedural: Store tweet processing workflows

2. **Trade Signal Generation**:
   - Episodic: Store detected signals and their context
   - Semantic: Store trading patterns and signal definitions
   - Procedural: Store signal validation and confirmation procedures

3. **Trade Execution**:
   - Episodic: Store trade execution events and results
   - Semantic: Store trading rules and constraints
   - Procedural: Store order placement and management procedures

4. **Performance Tracking**:
   - Episodic: Store performance metrics over time
   - Semantic: Store performance benchmarks and targets
   - Procedural: Store analysis and reporting procedures

## Implementation Plan

1. **Phase 1: Procedural Memory Implementation**
   - Implement the `ProceduralMemory` class
   - Add persistence and retrieval mechanisms
   - Implement basic procedure execution

2. **Phase 2: Memory Manager Implementation**
   - Implement the `MemoryManager` class
   - Integrate all memory systems
   - Implement cross-memory queries and operations

3. **Phase 3: Integration with Agent Loop**
   - Connect memory systems to perception module
   - Connect memory systems to cognition module
   - Connect memory systems to action module

4. **Phase 4: TAAT-Specific Procedures**
   - Implement Twitter monitoring procedures
   - Implement trade signal generation procedures
   - Implement trade execution procedures
   - Implement performance tracking procedures

## Conclusion

This memory systems architecture provides TAAT with a comprehensive cognitive framework that supports learning, adaptation, and robust operation. The addition of procedural memory completes the memory triad (episodic, semantic, procedural) and enables TAAT to store and execute complex trading procedures while learning from outcomes.

The integration strategy ensures that all memory systems work together seamlessly and that the agent can leverage its full memory capabilities during the perception-cognition-action loop. The persistence strategy ensures that memory is preserved across sessions and can be recovered in case of failures.
