"""
LangGraph Agent Simulation with RagaAI Catalyst Tracing

A demonstration of multi-agent conversation simulation using LangGraph,
traced and monitored using RagaAI Catalyst.

"""

import os
from typing import List
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.adapters.openai import convert_message_to_dict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from openai import OpenAI

# Step 1: Import RagaAI Catalyst components for tracing and monitoring
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst, init_tracing

# Step 2: Set up environment variables for RagaAI Catalyst authentication
load_dotenv()

# Step 3: Initialize RagaAI Catalyst with your credentials
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Step 4: Configure the tracer with project details for monitoring
tracer = Tracer(
    project_name="Trace_testing",       # Name of the project
    dataset_name="langgraph_testing",   # Name of the dataset
    tracer_type="Agentic",              # Type of tracing
)

# Step 5: Initialize RagaAI Catalyst tracing system
init_tracing(catalyst=catalyst, tracer=tracer)

def my_chat_bot(messages: List[dict]) -> dict:
    system_message = {
        "role": "system",
        "content": "You are a customer support agent for an airline.",
    }
    messages = [system_message] + messages
    completion = OpenAI().chat.completions.create(
        messages=messages, model="gpt-4o-mini"
    )
    return completion.choices[0].message.model_dump()

def chat_bot_node(state):
    messages = state["messages"]
    messages = [convert_message_to_dict(m) for m in messages]
    chat_bot_response = my_chat_bot(messages)
    return {"messages": [AIMessage(content=chat_bot_response["content"])]}

system_prompt_template = """You are a customer of an airline company. \
You are interacting with a user who is a customer support person. \

{instructions}

When you are finished with the conversation, respond with a single word 'FINISHED'"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
instructions = """Your name is Harrison. You are trying to get a refund for the trip you took to Alaska. \
You want them to give you ALL the money back. \
This trip happened 5 years ago."""

prompt = prompt.partial(name="Harrison", instructions=instructions)

def _swap_roles(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages

def should_continue(state):
    messages = state["messages"]
    if len(messages) > 6:
        return "end"
    elif messages[-1].content == "FINISHED":
        return "end"
    else:
        return "continue"

class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph():
    model = ChatOpenAI()
    simulated_user = prompt | model

    def simulated_user_node(state):
        messages = state["messages"]
        new_messages = _swap_roles(messages)
        response = simulated_user.invoke({"messages": new_messages})
        return {"messages": [HumanMessage(content=response.content)]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("user", simulated_user_node)
    graph_builder.add_node("chat_bot", chat_bot_node)
    graph_builder.add_edge("chat_bot", "user")
    graph_builder.add_conditional_edges(
        "user",
        should_continue,
        {
            "end": END,
            "continue": "chat_bot",
        },
    )
    graph_builder.add_edge(START, "chat_bot")
    simulation = graph_builder.compile()
    return simulation

# Step 6: Wrap the simulation execution with RagaAI Catalyst tracer
with tracer:
    simulation = build_graph()
    for chunk in simulation.stream({"messages": []}):
        if END not in chunk:
            print(chunk)
            print("----")