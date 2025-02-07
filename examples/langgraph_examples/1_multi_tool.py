"""
Multi-Tool Agent System with RagaAI Catalyst Tracing

Demonstrates integration of multiple search tools (Arxiv, DuckDuckGo)
with comprehensive tracing via RagaAI Catalyst.
"""

import json
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_anthropic import ChatAnthropic

# Step 1: Import RagaAI Catalyst components for tracing and monitoring
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst, init_tracing

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Step 2: Load environment variables for RagaAI credentials
load_dotenv()

# Step 3: Initialize RagaAI Catalyst with authentication details
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Step 4: Configure tracer for monitoring multi tools tracing
tracer = Tracer(
    project_name="Langgraph_testing",
    dataset_name="multi_tools",
    tracer_type="Agentic",
)

# Step 5: Initialize the tracing system
init_tracing(catalyst=catalyst, tracer=tracer)

# Tools Arxiv and DuckDuckGo Search are automatically traced by RagaAI Catalyst
arxiv_tool = ArxivQueryRun(max_results=2)   # Traced by RagaAI Catalyst
ddg_tool = DuckDuckGoSearchRun()            # Traced by RagaAI Catalyst

tools = [
    arxiv_tool,
    ddg_tool,
]

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
def route_tools(state: State):
    messages = state["messages"]
    if not messages:
        return "END"
    
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "END"

def build_graph():
    graph_builder = StateGraph(State)
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    llm_with_tools = llm.bind_tools(
        tools,
        tool_choice="auto", 
    )
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = BasicToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {
            "tools": "tools",
            "END": END
        }
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile()
    return graph

def main():
        graph = build_graph()
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                
                config = {"messages": [{"role": "user", "content": user_input}]}
                for output in graph.stream(config):
                    print(f"Assistant: {output}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")


print("Multi-Tool Research Assistant Ready! (Type 'quit' to exit)")
print("Available tools:")
print("1. ArXiv - Find academic papers")
print("2. DuckDuckGo Search - Search the web")
print("\nExample queries:")
print("- 'Find recent papers and web results about LangChain'")
print("- 'Search for tutorials on Python async programming and related research papers'")
print("- 'What are the latest developments in quantum computing? Include papers and web results'")

# Step 6: Execute the multi-tool workflow with RagaAI Catalyst
with tracer:
    main()