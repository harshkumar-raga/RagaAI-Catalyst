from typing import Any, Dict, Optional
from discovery import DiscoveryAgent
from synthesis import SynthesisAgent
from base_agent import BaseAgent
from ragaai_catalyst import trace_agent


class CoordinatorAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.discovery_agent = DiscoveryAgent(config)
        self.synthesis_agent = SynthesisAgent(config)
        self.research_state = {
            "status": "idle",
            "current_phase": None,
            "findings": [],
            "hypotheses": [],
            "conclusions": [],
        }

    @trace_agent(name="Cordinator_agent")
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        research_question = input_data.get("research_question")
        if not research_question:
            raise ValueError("Research question is required")

        self.research_state["status"] = "active"
        self.research_state["current_phase"] = "discovery"

        print("Step 1: Delegate to Discovery Agent")
        discovery_results = await self.discovery_agent.process(
            {
                "research_question": research_question,
                "parameters": input_data.get("parameters", {}),
            }
        )

        self.research_state["findings"] = discovery_results.get("findings", [])

        print("Step 2: Delegate to Synthesis Agent")
        self.research_state["current_phase"] = "synthesis"
        synthesis_results = await self.synthesis_agent.process(
            {
                "findings": discovery_results.get("findings", []),
                "research_question": research_question,
            }
        )

        print("Step 3: Update final research state")
        self.research_state.update(
            {
                "status": "completed",
                "current_phase": "completed",
                "hypotheses": synthesis_results.get("hypotheses", []),
                "conclusions": synthesis_results.get("conclusions", []),
            }
        )

        return {
            "status": "success",
            "research_state": self.research_state,
            "findings": discovery_results.get("findings", []),
            "conclusions": synthesis_results.get("conclusions", []),
            "recommendations": synthesis_results.get("recommendations", []),
        }

    def get_research_state(self) -> Dict[str, Any]:
        return self.research_state
