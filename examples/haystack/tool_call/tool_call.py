from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.tools import ComponentTool
from haystack.components.websearch import SerperDevWebSearch
from haystack import Pipeline
from typing import Any, Dict, List
from haystack import component
from haystack.core.component.types import Variadic
from ragaai_catalyst import RagaAICatalyst, Tracer, init_tracing
import os

from dotenv import load_dotenv
load_dotenv()

catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
    base_url=os.getenv('CATALYST_BASE_URL')
)

tracer = Tracer(
    project_name=os.getenv('PROJECT_NAME'),
    dataset_name=os.getenv('DATASET_NAME'),
    tracer_type="agentic/haystack",
)

init_tracing(catalyst=catalyst, tracer=tracer)

# helper component to temporarily store last user query before the tool call 
@component()
class MessageCollector:
    def __init__(self):
        self._messages = []

    @component.output_types(messages=List[ChatMessage])
    def run(self, messages: Variadic[List[ChatMessage]]) -> Dict[str, Any]:

        self._messages.extend([msg for inner in messages for msg in inner])
        return {"messages": self._messages}

    def clear(self):
        self._messages = []

# Create a tool from a component
web_tool = ComponentTool(
    component=SerperDevWebSearch(top_k=3)
)

# Define routing conditions
routes = [
    {
        "condition": "{{replies[0].tool_calls | length > 0}}",
        "output": "{{replies}}",
        "output_name": "there_are_tool_calls",
        "output_type": List[ChatMessage],
    },
    {
        "condition": "{{replies[0].tool_calls | length == 0}}",
        "output": "{{replies}}",
        "output_name": "final_replies",
        "output_type": List[ChatMessage], 
    },
]

# Create the pipeline
tool_agent = Pipeline()
tool_agent.add_component("message_collector", MessageCollector())
tool_agent.add_component("generator", OpenAIChatGenerator(model="gpt-4o-mini", tools=[web_tool]))
tool_agent.add_component("router", ConditionalRouter(routes, unsafe=True))
tool_agent.add_component("tool_invoker", ToolInvoker(tools=[web_tool]))

tool_agent.connect("generator.replies", "router")
tool_agent.connect("router.there_are_tool_calls", "tool_invoker")
tool_agent.connect("router.there_are_tool_calls", "message_collector")
tool_agent.connect("tool_invoker.tool_messages", "message_collector")
tool_agent.connect("message_collector", "generator.messages")

messages = [
    ChatMessage.from_system("You're a helpful agent choosing the right tool when necessary"), 
    ChatMessage.from_user("How is the weather in Berlin?")]
result = tool_agent.run({"messages": messages})

print(result["router"]["final_replies"][0].text)