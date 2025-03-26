import os
from dotenv import load_dotenv

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

load_dotenv()


def initialize_tracing():
    catalyst = RagaAICatalyst(
        access_key=os.getenv("CATALYST_ACCESS_KEY"),
        secret_key=os.getenv("CATALYST_SECRET_KEY"),
        base_url=os.getenv("CATALYST_BASE_URL"),
    )

    tracer = Tracer(
        project_name="example_testing",
        dataset_name="Research_agent",
        tracer_type="Agentic",
    )

    init_tracing(catalyst=catalyst, tracer=tracer)
    return tracer
