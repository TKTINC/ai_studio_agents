# CI/CD Configuration for AI Studio Agents

This directory contains the CI/CD pipeline configurations for independent deployment of each agent in the AI Studio Agents monorepo.

## Pipeline Structure

Each agent has its own dedicated pipeline that is triggered only when relevant files are changed:

- `taat-pipeline.yml`: CI/CD pipeline for the TAAT agent
- `all-use-pipeline.yml`: CI/CD pipeline for the ALL-USE agent
- `mentor-pipeline.yml`: CI/CD pipeline for the MENTOR agent
- `shared-pipeline.yml`: CI/CD pipeline for shared components (triggers all agent pipelines)

## Triggering Logic

The pipelines use path-based triggers to ensure that only the relevant agent pipeline runs when code is changed:

- Changes to `/src/agents/taat/**` trigger only the TAAT pipeline
- Changes to `/src/agents/all_use/**` trigger only the ALL-USE pipeline
- Changes to `/src/agents/mentor/**` trigger only the MENTOR pipeline
- Changes to `/src/agent_core/**` or `/src/shared_modules/**` trigger all pipelines

## Pipeline Stages

Each agent pipeline includes the following stages:

1. **Build**: Compile and package the agent
2. **Test**: Run unit and integration tests
3. **Deploy**: Deploy the agent to the appropriate environment

## Environment Configuration

Each agent can be deployed to different environments:

- Development
- Staging
- Production

Environment-specific configuration is managed through environment variables and configuration files.
