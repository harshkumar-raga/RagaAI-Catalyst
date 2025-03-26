# Travel Agent

A custom agent built with RagaAI-Catalyst that helps users plan their ideal vacation by providing personalized travel recommendations, weather information, currency conversion, and flight price estimates.

## Components

- `main.py`: Entry point and main travel planning orchestration
- `tools.py`: Collection of utility tools for travel-related services
- `agents.py`: Implementation of the Itinerary Agent
- `config.py`: Configuration and tracing setup

## Features

- Interactive travel planning through natural language input
- Automatic extraction of travel preferences
- Real-time weather information for destinations
- Currency conversion capabilities
- Flight price estimation
- Personalized itinerary generation
- Integration with RagaAI-Catalyst tracing system
- Performance and cost metrics tracking

## Tools

- Weather Tool: Fetches current weather data for destinations
- Currency Converter: Converts between different currencies
- Flight Price Estimator: Provides estimated flight costs
- LLM Integration: Uses GPT models for natural language processing

## Usage

```python
from main import travel_agent

# Run the travel agent
travel_agent()

# The agent will interactively:
# 1. Ask for your vacation preferences
# 2. Extract key details (destination, activities, budget, duration)
# 3. Check weather conditions
# 4. Provide currency conversion if needed
# 5. Estimate flight prices
# 6. Generate a personalized itinerary
```

## Requirements

- Python 3.x
- RagaAI-Catalyst
- OpenAI API key
- OpenWeatherMap API key
- python-dotenv

## Environment Variables

Create a `.env` file with the following:
```
OPENAI_API_KEY=your_openai_api_key
OPENWEATHERMAP_API_KEY=your_weather_api_key
```

## Metrics and Tracing

The agent includes comprehensive tracing and metrics:
- Travel planning session scoring
- LLM call tracking
- Tool performance metrics
- Cost and latency monitoring
- Hallucination detection

## Note

This is a custom agent implementation using the RagaAI-Catalyst framework. Ensure all API keys and dependencies are properly configured before running the agent.
