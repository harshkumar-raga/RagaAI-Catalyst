from typing import Any, Dict, List, Optional
import aiohttp
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from llm import get_llm_response
from base_agent import BaseAgent
from ragaai_catalyst import trace_agent, trace_tool


class DiscoveryAgent(BaseAgent):
    @trace_agent(name="Discovery_agent")
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        research_question = input_data.get("research_question")
        if not research_question:
            raise ValueError("Research question is required")

        search_queries = await self._generate_search_queries(research_question)

        findings = []
        for query in search_queries:
            academic_results = await self._search_academic_sources(query)
            findings.extend(academic_results)

            tech_results = await self._search_tech_sources(query)
            findings.extend(tech_results)

        summarized_findings = await self._summarize_findings(findings)

        return {"status": "success", "findings": summarized_findings}

    async def _summarize_findings(
        self, findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not findings:
            return []

        grouped_findings = self._group_findings_by_source(findings)

        summarized_findings = []
        for source_type, source_findings in grouped_findings.items():
            summary = await self._generate_summary(source_findings)
            summarized_findings.append(
                {
                    "source_type": source_type,
                    "summary": summary,
                    "raw_findings": source_findings,
                }
            )

        return summarized_findings

    def _group_findings_by_source(
        self, findings: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}
        for finding in findings:
            source_type = finding.get("source_type", "unknown")
            if source_type not in grouped:
                grouped[source_type] = []
            grouped[source_type].append(finding)
        return grouped

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @trace_tool(name="generate_search_queries")
    async def _generate_search_queries(self, research_question: str) -> List[str]:
        """Generate specific search queries from the research question."""

        prompt = f"""
        Break down this research question into specific search queries:
        Question: {research_question}
        
        Generate 3-5 specific search queries that would help gather comprehensive information.
        For each query, provide a focus area that explains what aspect of the research
        question this query addresses.
        """

        response = await get_llm_response(prompt)
        return [query.strip() for query in response.split("\n") if query.strip()]

    @trace_tool(name="search_academic_sources")
    async def _search_academic_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search academic sources using ArXiv API."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        encoded_query = urllib.parse.quote(query)
        base_url = "http://export.arxiv.org/api/query"
        search_url = f"{base_url}?search_query=all:{encoded_query}&start=0&max_results=10&sortBy=relevance"

        try:
            async with self.session.get(search_url) as response:
                if response.status != 200:
                    return []

                content = await response.text()
                root = ET.fromstring(content)

                namespace = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", namespace)

                results = []
                for entry in entries:
                    result = {
                        "title": entry.find("atom:title", namespace).text.strip(),
                        "authors": [
                            author.find("atom:name", namespace).text
                            for author in entry.findall("atom:author", namespace)
                        ],
                        "summary": entry.find("atom:summary", namespace).text.strip(),
                        "link": entry.find("atom:id", namespace).text,
                        "published": entry.find("atom:published", namespace).text,
                        "source_type": "arxiv",
                        "query": query,
                    }
                    results.append(result)

                return results

        except Exception as e:
            print(f"Error searching ArXiv: {str(e)}")
            return []

    @trace_tool(name="search_tech_sources")
    async def _search_tech_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search technical sources using Dev.to and Stack Overflow APIs."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        results = []

        try:
            devto_url = (
                f"https://dev.to/api/articles/search?q={urllib.parse.quote(query)}"
            )
            async with self.session.get(devto_url) as response:
                if response.status == 200:
                    articles = await response.json()
                    for article in articles[:5]:  # Limit to top 5 results
                        result = {
                            "title": article.get("title"),
                            "author": article.get("user", {}).get("name"),
                            "summary": article.get("description"),
                            "link": f"https://dev.to{article.get('path')}",
                            "published": article.get("published_at"),
                            "source_type": "dev.to",
                            "query": query,
                        }
                        results.append(result)
        except Exception as e:
            print(f"Error searching Dev.to: {str(e)}")

        try:
            so_url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                "q": query,
                "site": "stackoverflow",
                "order": "desc",
                "sort": "relevance",
                "pagesize": 5,
                "filter": "withbody",
            }

            async with self.session.get(so_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for item in data.get("items", []):
                        result = {
                            "title": item.get("title"),
                            "author": item.get("owner", {}).get("display_name"),
                            "summary": item.get("body")[:500]
                            + "...",  # Truncate long bodies
                            "link": item.get("link"),
                            "score": item.get("score"),
                            "published": datetime.fromtimestamp(
                                item.get("creation_date")
                            ).isoformat(),
                            "source_type": "stackoverflow",
                            "query": query,
                        }
                        results.append(result)
        except Exception as e:
            print(f"Error searching Stack Overflow: {str(e)}")

        return results

    @trace_tool(name="generate_summary")
    async def _generate_summary(self, findings: List[Dict[str, Any]]) -> str:
        if not findings:
            return "No findings to summarize."

        prompt = """
        Summarize the following research findings. Focus on:
        1. Key themes and patterns
        2. Notable differences or contradictions
        3. Most significant insights
        4. Areas needing further investigation

        Findings:
        """

        for finding in findings:
            prompt += f"\nTitle: {finding.get('title')}\n"
            prompt += f"Source: {finding.get('source_type')}\n"
            prompt += f"Summary: {finding.get('summary')}\n"
            prompt += "-" * 50 + "\n"

        response = await get_llm_response(
            prompt=prompt, temperature=0.7, max_tokens=1500
        )

        return response
