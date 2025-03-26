from typing import Any, Dict, List
from llm import get_llm_response
from base_agent import BaseAgent
from ragaai_catalyst import trace_agent, trace_tool


class SynthesisAgent(BaseAgent):
    @trace_agent(name="Synthesis_agent")
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        findings = input_data.get("findings", [])
        research_question = input_data.get("research_question")

        if not findings:
            return {"status": "error", "message": "No findings provided for synthesis"}

        patterns = await self._identify_patterns(findings)
        hypotheses = await self._generate_hypotheses(patterns, research_question)
        evaluated_hypotheses = await self._evaluate_hypotheses(hypotheses, findings)
        conclusions = await self._generate_conclusions(evaluated_hypotheses)
        recommendations = await self._generate_recommendations(
            conclusions, research_question
        )

        return {
            "status": "success",
            "patterns": patterns,
            "hypotheses": evaluated_hypotheses,
            "conclusions": conclusions,
            "recommendations": recommendations,
        }

    @trace_tool(name="pattern_identification")
    async def _identify_patterns(
        self, findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        prompt = """
        Analyze these research findings and identify key patterns and themes:
        
        Findings:
        {}
        
        Identify:
        1. Common themes
        2. Contradictions
        3. Knowledge gaps
        4. Emerging trends
        """.format(self._format_findings_for_prompt(findings))

        response = await get_llm_response(prompt)

        return [{"type": "pattern", "description": response}]

    @trace_tool(name="hypothesis_generation")
    async def _generate_hypotheses(
        self, patterns: List[Dict[str, Any]], research_question: str
    ) -> List[Dict[str, Any]]:
        prompt = f"""
        Based on these patterns and the research question:
        
        Research Question: {research_question}
        
        Patterns:
        {self._format_patterns_for_prompt(patterns)}
        
        Generate 3-5 hypotheses that could explain the patterns or answer the research question.
        """

        response = await get_llm_response(prompt)
        return [{"hypothesis": response, "confidence": 0.0}]

    @trace_tool(name="hypothesis_evaluation")
    async def _evaluate_hypotheses(
        self, hypotheses: List[Dict[str, Any]], findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        evaluated_hypotheses = []
        for hypothesis in hypotheses:
            prompt = f"""
            Evaluate this hypothesis against the research findings:
            
            Hypothesis: {hypothesis["hypothesis"]}
            
            Findings:
            {self._format_findings_for_prompt(findings)}
            
            Provide:
            1. Supporting evidence
            2. Contradicting evidence
            3. Confidence score (0-1)
            """

            response = await get_llm_response(prompt)
            #
            evaluated_hypotheses.append(
                {**hypothesis, "evaluation": response, "confidence": 0.8}
            )

        return evaluated_hypotheses

    @trace_tool(name="conclusion_generation")
    async def _generate_conclusions(
        self, evaluated_hypotheses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        prompt = """
        Based on these evaluated hypotheses:
        {}
        
        Generate key conclusions that:
        1. Synthesize the most supported hypotheses
        2. Address any contradictions
        3. Highlight remaining uncertainties
        """.format(self._format_hypotheses_for_prompt(evaluated_hypotheses))

        response = await get_llm_response(prompt)
        return [{"conclusion": response, "confidence": 0.9}]

    @trace_tool(name="recommendation_generation")
    async def _generate_recommendations(
        self, conclusions: List[Dict[str, Any]], research_question: str
    ) -> List[Dict[str, Any]]:
        prompt = f"""
        Based on these conclusions and the original research question:
        
        Research Question: {research_question}
        
        Conclusions:
        {self._format_conclusions_for_prompt(conclusions)}
        
        Generate:
        1. Key recommendations
        2. Suggested next steps
        3. Areas for further research
        """

        response = await get_llm_response(prompt)
        return [{"recommendation": response, "priority": "high"}]

    def _format_findings_for_prompt(self, findings: List[Dict[str, Any]]) -> str:
        return "\n".join(f"- {finding.get('summary', '')}" for finding in findings)

    def _format_patterns_for_prompt(self, patterns: List[Dict[str, Any]]) -> str:
        return "\n".join(f"- {pattern.get('description', '')}" for pattern in patterns)

    def _format_hypotheses_for_prompt(self, hypotheses: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"- Hypothesis: {h.get('hypothesis', '')}\n  Confidence: {h.get('confidence', 0)}\n  Evaluation: {h.get('evaluation', '')}"
            for h in hypotheses
        )

    def _format_conclusions_for_prompt(self, conclusions: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"- {conclusion.get('conclusion', '')}" for conclusion in conclusions
        )
