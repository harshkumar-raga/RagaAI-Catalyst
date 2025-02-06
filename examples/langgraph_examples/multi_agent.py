"""
Multi-Agent System with RagaAI Catalyst Tracing

Demonstrates tracing of complex multi-agent interactions using RagaAI Catalyst:

Traced Components:
    - Tool usage (Tavily Search, Python REPL)
    - Inter-agent communication
    - LLM calls via LangChain
"""

import os
from dotenv import load_dotenv

from typing import Annotated, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

# Import RagaAI Catalyst for tracing
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

# Load environment variables 
load_dotenv()

# Initialize RagaAI Catalyst
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Set up the tracer to track interactions
tracer = Tracer(
    project_name="Langgraph_testing",    # Project name for the trace
    dataset_name="multi_agent",          # Dataset name for the trace
    tracer_type="Agentic",               # Type of tracing (Agentic)
)

# Initialize tracing with RagaAI Catalyst
init_tracing(catalyst=catalyst, tracer=tracer)

# Tool Definitions

# Tavily Search tool for research, traceable by RagaAI Catalyst
tavily_tool = TavilySearchResults(max_results=1)

# Python REPL tool for code execution
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Executes Python code to generate a chart, tracing via RagaAI Catalyst."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto



def build_graph():
    """Build the multi-agent workflow graph and simulate the interaction."""

    workflow = StateGraph(MessagesState)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Research agent and node
    research_agent = create_react_agent(
        llm,
        tools=[tavily_tool],
        prompt=make_system_prompt(
            "You can only do research. You are working with a chart generator colleague."
        ),
    )

    def research_node(
        state: MessagesState,
    ) -> Command[Literal["chart_generator", END]]:
        result = research_agent.invoke(state)
        goto = get_next_node(result["messages"][-1], "chart_generator")

        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="researcher"
        )
        return Command(
            update={
                # share internal message history of research agent with other agents
                "messages": result["messages"],
            },
            goto=goto,
        )

    chart_agent = create_react_agent(
        llm,
        [python_repl_tool],
        prompt=make_system_prompt(
            "You can only generate charts. You are working with a researcher colleague."
        ),
    )

    def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
        result = chart_agent.invoke(state)
        goto = get_next_node(result["messages"][-1], "researcher")
        # wrap in a human message, as not all providers allow
        # AI message at the last position of the input messages list
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="chart_generator"
        )
        return Command(
            update={
                # share internal message history of chart agent with other agents
                "messages": result["messages"],
            },
            goto=goto,
        )
    
    # Define workflow structure
    workflow.add_node("researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_edge(START, "researcher")

    # Compile and return the workflow graph
    return workflow.compile()


# Execute the with RagaAI Catalyst tracing
with tracer:
    graph=build_graph()
    events = graph.stream({
        "messages": [
            (
                "user",
                "First, get the INDIA's GDP over the past 3 years, then make a line chart of it. "
                "Once you make the chart, finish.",
            )
        ],},
    # Maximum number of steps to take in the graph
    {"recursion_limit": 30},)
    for s in events:
        print(s)
        print("----")
