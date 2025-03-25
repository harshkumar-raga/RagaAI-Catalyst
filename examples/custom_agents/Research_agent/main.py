import asyncio
import json
from typing import Dict, Any

from cordinator import CoordinatorAgent
from dotenv import load_dotenv
from ragaai_catalyst import trace_agent
from config import initialize_tracing

tracer = initialize_tracing()


async def conduct_research(
    research_question: str, parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    coordinator = CoordinatorAgent()
    input_data = {
        "research_question": research_question,
        "parameters": parameters or {},
    }
    results = await coordinator.process(input_data)
    return results


def display_results(results: Dict[str, Any]) -> None:
    print("\nKey Findings:")
    for finding in results.get("findings", []):
        print(f"- {finding.get('summary', '')}")

    print("\nConclusions:")
    for conclusion in results.get("conclusions", []):
        print(f"- {conclusion.get('conclusion', '')}")

    print("\nRecommendations:")
    for recommendation in results.get("recommendations", []):
        print(f"- {recommendation.get('recommendation', '')}")


@trace_agent(name="Research_agent", agent_type="research")
async def Research_agent():
    load_dotenv()

    research_question = (
        "What are the latest developments in few-shot learning for NLP tasks?"
    )
    parameters = {
        "max_sources": 2,
        "time_range": "last_year",
        "focus_areas": ["academic_papers", "tech_blogs", "conferences"],
    }

    try:
        print(f"Starting research on: {research_question}")
        results = await conduct_research(research_question, parameters)

        display_results(results)
        output_file = "research_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    except Exception as e:
        print(f"Error during research: {str(e)}")


if __name__ == "__main__":
    with tracer:
        asyncio.run(Research_agent())
