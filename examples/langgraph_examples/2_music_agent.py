"""
LangGraph Music Agent with RagaAI Catalyst Integration

This script demonstrates the integration of RagaAI Catalyst for tracing and monitoring
an AI-powered music agent.

"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Step 1: Import RagaAI Catalyst components for tracing and monitoring
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from dotenv import load_dotenv

# Step 2: Load environment variables for RagaAI credentials
load_dotenv()

# Step 3: Initialize RagaAI Catalyst with authentication details
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Step 4: Configure tracer for monitoring music agent tracing
tracer = Tracer(
    project_name="Langgraph_testing",   # Project name for the trace
    dataset_name="time_travel",         # Dataset name for the trace
    tracer_type="Agentic",               # Type of tracing (Agentic)
)

# Step 5: Initialize the tracing system
init_tracing(catalyst=catalyst, tracer=tracer)

# tools decorated with `@tool` are automatically traced by RagaAI Catalyst
@tool
def play_song_on_spotify(song: str):
    """Play a song on Spotify, traceable by RagaAI Catalyst"""
    return f"Successfully played {song} on Spotify!"


@tool
def play_song_on_apple(song: str):
    """Play a song on Apple Music, traceable by RagaAI Catalyst"""
    return f"Successfully played {song} on Apple Music!"

tools = [play_song_on_apple, play_song_on_spotify]
tool_node = ToolNode(tools)

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def build_graph():

    model = ChatOpenAI(model="gpt-4o-mini")
    model = model.bind_tools(tools, parallel_tool_calls=False)

    def call_model(state):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",  
        should_continue,  
        {
            "continue": "action",  
            "end": END,  
        },
    )
    workflow.add_edge("action", "agent")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


def main():
    app = build_graph()
    config = {"configurable": {"thread_id": "1"}}
    input_message = HumanMessage(content="Can you play Taylor Swift's most popular song?")

    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    all_states = []
    for state in app.get_state_history(config):
        print(state)
        all_states.append(state)
        print("--")

    to_replay = all_states[2]
    for event in app.stream(None, to_replay.config):
        for v in event.values():
            print(v)

    last_message = to_replay.values["messages"][-1]
    last_message.tool_calls[0]["name"] = "play_song_on_spotify"

    branch_config = app.update_state(
        to_replay.config,
        {"messages": [last_message]},
    )

    for event in app.stream(None, branch_config):
        for v in event.values():
            print(v)

# Step 6: Execute the workflow with RagaAI Catalyst
with tracer:
    main()