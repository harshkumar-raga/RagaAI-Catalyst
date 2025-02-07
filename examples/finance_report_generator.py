import os
import json
import requests
from urllib.parse import urlparse
from typing import List, Dict
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from litellm import completion

# Step 1: Import RagaAI Catalyst components for tracing and monitoring
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import (
    RagaAICatalyst,
    trace_tool,
    trace_llm,
    trace_agent,
    init_tracing,
)

# Step 2: Load environment variables for RagaAI credentials
load_dotenv()

# Step 3: Initialize RagaAI Catalyst with authentication details
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Step 4: Configure tracer for financial report generator
tracer = Tracer(
    project_name="alteryx_copilot-tan",     # Name of the project
    dataset_name="testing-3",               # Name of the dataset
    tracer_type="Agentic",                  # Type of tracing
)

# Step 5: Initialize the tracing system
init_tracing(catalyst=catalyst, tracer=tracer)

class FinancialReportGenerator:
    # We can trace the tools using the `@trace_tool` decorator
    # Step 6: Trace all the tools using the `@trace_tool` decorator
    # Using the `@trace_tool` decorator will trace tool `scrape_website`
    @trace_tool("scrape_website")
    def scrape_website(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator=" ", strip=True)
            return text
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return ""

    # We can trace the LLMs using the `@trace_llm` decorator
    # Step 7: Trace all the LLMs using the `@trace_llm` decorator
    # Using the `@trace_llm` decorator will trace llm `analyze_sentiment`
    @trace_llm(name="analyze_sentiment")
    def analyze_sentiment(self, text: str) -> Dict:
        try:
            response = completion(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the following text and provide sentiment analysis focused on financial implications. Return a JSON with 'sentiment' (positive/negative/neutral), 'confidence' (0-1), and 'key_points'.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=500,
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0, "key_points": []}
        
    # Using the `@trace_llm` decorator will trace llm `get_report`
    @trace_llm(name="get_report")
    def get_report(self, report_prompt: str) -> str:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst. Generate a comprehensive financial report based on the provided data. Include market analysis, stock performance, and key insights.",
                },
                {"role": "user", "content": report_prompt},
            ],
            max_tokens=1500,
        )
        print("Report generated successfully.")
        return response.choices[0].message.content

    # We can trace the agents using the `@trace_agent` decorator
    # Step 8: Trace all the agents using the `@trace_agent` decorator
    # Using the `@trace_agent` decorator to trace the `generate_report` agent
    @trace_agent(name="generate_report")
    def generate_report(self, urls: List[str]) -> str:
        website_contents = []
        sentiments = []

        print("Processing URLs...")
        for url in urls:
            content = self.scrape_website(url)
            if content:
                website_contents.append(
                    {"url": url, "content": content, "domain": urlparse(url).netloc}
                )
                print(f"Scraped content from {url}")
                print("Analyzing sentiment...")
                sentiments.append(self.analyze_sentiment(content))

        print("Generating report...")
        report_prompt = self._create_report_prompt(website_contents, sentiments)

        report = self.get_report(report_prompt)
        return report

    # Using the `@trace_tool` decorator will trace tool `create_report_prompt`
    @trace_tool("create_report_prompt")
    def _create_report_prompt(
        self, website_contents: List[Dict], sentiments: List[Dict]
    ) -> str:
        print("Creating report prompt...")
        prompt = "Generate a financial report based on the following data:\n\n"

        prompt += "News and Analysis:\n"
        for content in website_contents:
            prompt += f"Source: {content['domain']}\n"
            prompt += "Key points from sentiment analysis:\n"
            for sentiment in sentiments:
                prompt += f"- Sentiment: {sentiment['sentiment']}\n"
                prompt += f"- Key points: {', '.join(sentiment['key_points'])}\n"

        prompt += "\nPlease provide a comprehensive analysis including:\n"
        prompt += "1. Market Overview\n"
        prompt += "2. Stock Analysis\n"
        prompt += "3. News Impact Analysis\n"
        prompt += "4. Key Insights and Recommendations\n"

        return prompt


# Step 9: Run the financial report generator with tracer
if __name__ == "__main__":
    with tracer:
        generator = FinancialReportGenerator()
        urls = [
            "https://money.rediff.com/news/market/rupee-hits-record-low-of-85-83-against-us-dollar/20623520250108",
            "https://indianexpress.com/article/business/banking-and-finance/rbi-asks-credit-bureaus-banks-to-pay-rs-100-compensation-per-day-for-delay-in-data-updation-9765814/",
        ]

        report = generator.generate_report(urls)
        print(report)