# Research Agent

A custom agent built with RagaAI-Catalyst for conducting automated research on specified topics. This agent coordinates multiple sub-components to gather, analyze, and synthesize information from various sources.

## Components

- `main.py`: Entry point and primary orchestration of the research process
- `coordinator.py`: Manages the coordination between different agent components
- `discovery.py`: Handles information discovery and source gathering
- `synthesis.py`: Processes and synthesizes gathered information
- `llm.py`: Language model integration utilities
- `base_agent.py`: Base agent implementation
- `config.py`: Configuration and tracing setup

## Features

- Asynchronous research processing
- Configurable research parameters (max sources, time range, focus areas)
- Structured output with findings, conclusions, and recommendations
- Integration with RagaAI-Catalyst tracing system
- Support for multiple source types (academic papers, tech blogs, conferences)

## Usage

```python
import asyncio
from main import conduct_research

# Define research parameters
parameters = {
    "max_sources": 2,
    "time_range": "last_year",
    "focus_areas": ["academic_papers", "tech_blogs", "conferences"]
}

# Run research
research_question = "What are the latest developments in few-shot learning for NLP tasks?"
results = asyncio.run(conduct_research(research_question, parameters))
```

## Requirements

- Python 3.x
- RagaAI-Catalyst
- python-dotenv

Make sure to set up your environment variables in a `.env` file before running the agent.

## Output Format

The agent returns structured results containing:
- Key findings from the research
- Conclusions drawn from the analysis
- Actionable recommendations

## Note

This is a custom agent implementation using the RagaAI-Catalyst framework. Ensure all dependencies are properly installed and configured before running the agent.
