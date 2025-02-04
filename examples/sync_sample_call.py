"""
Synchronous LLM Call Tracing Example

Demonstrates RagaAI Catalyst tracing for synchronous OpenAI API calls.
Tracks:
    - LLM request/response cycles
    - API latency
    - Token usage
    - Cost
    - File I/O operations
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from ragaai_catalyst.tracers import Tracer
from ragaai_catalyst import RagaAICatalyst, trace_llm, init_tracing

load_dotenv()
catalyst = RagaAICatalyst(
    access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
    secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
    base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
)
# Initialize tracer
tracer = Tracer(
    project_name="alteryx_copilot-tan",
    dataset_name="testing-1",
    tracer_type="Agentic",
    auto_instrumentation={"user_interaction": False, "file_io": True},
)
init_tracing(catalyst=catalyst, tracer=tracer)
tracer.start()


@trace_llm(name="llm_call")
def llm_call(prompt, max_tokens=512, model="gpt-3.5-turbo"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(f"Prompt: {prompt}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Model: {model}")
    input("Press Enter to continue...")
    with open("response.txt", "w") as f:
        f.write("test")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in llm_call: {str(e)}")
        raise


def main():
    response = llm_call("how are you?")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
    tracer.stop()
