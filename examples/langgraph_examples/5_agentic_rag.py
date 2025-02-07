"""
Agentic RAG System with RagaAI Catalyst Tracing

Implements an agent-driven Retrieval-Augmented Generation (RAG) system
with comprehensive tracing via RagaAI Catalyst.

"""
import os
import pprint
from dotenv import load_dotenv
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict

import langchain
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from pydantic import BaseModel, Field

# Step 1: Import RagaAI Catalyst components for comprehensive tracing
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst import trace_tool, current_span, trace_agent

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# Step 2: Set up environment variables for RagaAI authentication
load_dotenv()

# Step 3: Initialize RagaAI Catalyst with credentials
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)

# Step 4: Configure tracer for agentic RAG monitoring
tracer = Tracer(
    project_name="Langgraph_testing",   # Name of the project
    dataset_name="agentic_rag",         # Name of the dataset
    tracer_type="Agentic",              # Type of tracing
)

# Step 5: Initialize the tracing system
init_tracing(catalyst=catalyst, tracer=tracer)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# We can trace the agents using the `trace_agent` decorator
# Step: 6  Trace all the agents using the `trace_agent` decorator
# Using the `trace_agent` decorator to trace the `grade_documents` agent
@trace_agent("grade_documents")
def grade_documents(state) -> Literal["generate", "rewrite"]:
    print("---CHECK RELEVANCE---")

    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool
    current_span().add_metrics(
            name="grade_documents",
            score=0.9,
            reasoning="Determine relevance of retrieved documents",
            cost=0.01,
            latency=0.5,
        )

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

# Using the `trace_agent` decorator to trace the `agent` agent
@trace_agent("agent")
def agent(state):
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    current_span().add_metrics(
        name="agent",
        score=0.9,
        reasoning="Generating response based on current state",
        cost=0.01,
        latency=0.5,
    )
    return {"messages": [response]}

# Using the `trace_agent` decorator to trace the `rewrite` agent
@trace_agent("rewrite")
def rewrite(state):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

# We can trace the tools using the `trace_tool` decorator
# Step 7: Trace all the tools using the `trace_tool` decorator
# Using the `trace_tool` decorator to trace the `generate` tool

@trace_tool("generate")
def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()
    current_span().add_metrics(
            name="generate",
            score=0.9,
            reasoning="Generating response based on current state",
            cost=0.01,
            latency=0.5,
        )

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
prompt = hub.pull("rlm/rag-prompt").pretty_print()

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()

inputs = {
    "messages": [
        ("user", "What did Lilian Weng write about automatic prompt generation?"),
        ("user", "What did Lilian Weng write about attack on LLMs?"),
    ]
}

# Step 8: Execute the workflow with RagaAI Catalyst tracer
with tracer:
    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")