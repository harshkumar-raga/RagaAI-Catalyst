import sys 
import os
from dotenv import load_dotenv

# Step 1: Import RagaAI Catalyst components for tracing and monitoring
from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer
import uuid

# Step 2: Load environment variables for RagaAI credentials
load_dotenv()

# Step 3: Initialize the tracing system
def initialize_tracing():
    # Initialize RagaAI Catalyst with authentication details
    catalyst = RagaAICatalyst(
        access_key=os.getenv("RAGAAI_CATALYST_ACCESS_KEY"),
        secret_key=os.getenv("RAGAAI_CATALYST_SECRET_KEY"),
        # base_url=os.getenv("RAGAAI_CATALYST_BASE_URL"),
    )

    #  Configure tracer for financial report generator
    tracer = Tracer(
        project_name="Trace_testing",
        dataset_name="travel_agent_dataset",
        tracer_type="Agentic",
    )

    init_tracing(catalyst=catalyst, tracer=tracer)
    return tracer
