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

# Import RagaAI Catalyst for tracing
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
from dotenv import load_dotenv

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
    project_name="Langgraph_testing",   # Project name for the trace
    dataset_name="time_travel",         # Dataset name for the trace
    tracer_type="Agentic",               # Type of tracing (Agentic)
)

# Initialize tracing with RagaAI Catalyst
init_tracing(catalyst=catalyst, tracer=tracer)

# Define the tools for interaction with Spotify and Apple Music
@tool
def play_song_on_spotify(song: str):
    """Play a song on Spotify, traceable by RagaAI Catalyst"""
    # Call the spotify API ...
    return f"Successfully played {song} on Spotify!"


@tool
def play_song_on_apple(song: str):
    """Play a song on Apple Music, traceable by RagaAI Catalyst"""
    # Call the apple music API ...
    return f"Successfully played {song} on Apple Music!"

tools = [play_song_on_apple, play_song_on_spotify]
tool_node = ToolNode(tools)

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


def build_graph():
    """Builds the LangGraph state graph with nodes for agent interactions and tool invocations"""


    model = ChatOpenAI(model="gpt-4o-mini")
    model = model.bind_tools(tools, parallel_tool_calls=False)

    def call_model(state):
        messages = state["messages"]
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    workflow = StateGraph(MessagesState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)

    # Set the entrypoint as `agent`
    workflow.add_edge(START, "agent")

    # Define conditional edges for the workflow
    workflow.add_conditional_edges(
        "agent",  # After the agent node
        should_continue,  # Function to decide the next node
        {
            "continue": "action",  # Continue with the action (tool invocation)
            "end": END,  # End the workflow
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    workflow.add_edge("action", "agent")

    # Set up memory to save state at each point
    memory = MemorySaver()

    # Compile the graph into a LangChain Runnable
    app = workflow.compile(checkpointer=memory)

    return app


def main():
    app = build_graph()
    """Run the simulation with a given app and input message"""
    config = {"configurable": {"thread_id": "1"}}
    input_message = HumanMessage(content="Can you play Taylor Swift's most popular song?")

    # Stream the events in the simulation
    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Retrieve and print the state history
    all_states = []
    for state in app.get_state_history(config):
        print(state)
        all_states.append(state)
        print("--")

    # Replay a specific state (for debugging or further interaction)
    to_replay = all_states[2]
    for event in app.stream(None, to_replay.config):
        for v in event.values():
            print(v)

    # Update the tool call in the last message (to simulate an updated action)
    last_message = to_replay.values["messages"][-1]
    last_message.tool_calls[0]["name"] = "play_song_on_spotify"

    # Update the state with the new message
    branch_config = app.update_state(
        to_replay.config,
        {"messages": [last_message]},
    )

    # Stream the events after state update
    for event in app.stream(None, branch_config):
        for v in event.values():
            print(v)

# Execute with RagaAI Catalyst tracing
with tracer:
    main()