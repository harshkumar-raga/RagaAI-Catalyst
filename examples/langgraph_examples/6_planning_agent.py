"""
Planning Agent with RagaAI Catalyst Tracing

Implements a ReAct-style planning agent with step-by-step task decomposition
and execution, traced using RagaAI Catalyst.
"""

import os
import asyncio
import operator
from dotenv import load_dotenv
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Import RagaAI Catalyst components for tracing and monitoring
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst, init_tracing, trace_agent

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# Step 2: Load environment variables for RagaAI credentials
load_dotenv()

# Step 3: Initialize RagaAI Catalyst with authentication details
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Step 4: Configure tracer for monitoring planning agent tracing
tracer = Tracer(
    project_name="Trace_testing",       # Project name for the trace
    dataset_name="langgraph_testing",   # Dataset name for the trace
    tracer_type="Agentic",              # Type of tracing (Agentic)
)

# Step 5: Initialize the tracing system
init_tracing(catalyst=catalyst, tracer=tracer)


# tool Tavily search is traced by RagaAI Catalyst
tools = [TavilySearchResults(max_results=3)]

llm = ChatOpenAI(model="gpt-4")
agent_executor = create_react_agent(llm, tools)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).with_structured_output(Plan)


class Response(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).with_structured_output(Act)


# We can trace the agents using the `trace_agent` decorator
# Step: 6  Trace all the agents using the `trace_agent` decorator
# Using the `trace_agent` decorator to trace the `execute_step` agent
@trace_agent("execute_step")
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


# Using the `trace_agent` decorator to trace the `plan_step` agent
@trace_agent("plan_step")
async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


# Using the `trace_agent` decorator to trace the `replan_step` agent
@trace_agent("replan_step")
async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)
app = workflow.compile()

async def main():
    config = {"recursion_limit": 20}
    inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    # Step 7: Execute the workflow with RagaAI Catalyst tracer
    with tracer:
        asyncio.run(main())
