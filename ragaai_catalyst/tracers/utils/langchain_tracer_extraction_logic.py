import uuid

def langchain_tracer_extraction(data, 
                                project_name,
                                dataset_name, 
                                metadata,
                                pipeline, 
                                user_context="", 
                                expected_response=""
                                ):
    trace_aggregate = {}

    def generate_trace_id():
        """
        Generate a random trace ID using UUID4.
        Returns a string representation of the UUID with no hyphens.
        """
        return '0x'+str(uuid.uuid4()).replace('-', '')

    trace_aggregate["project_name"] = project_name
    trace_aggregate["dataset_name"] = dataset_name
    trace_aggregate["tracer_type"] = "langchain"
    trace_aggregate['trace_id'] = generate_trace_id()
    trace_aggregate['session_id'] = None
    trace_aggregate["pipeline"] = pipeline
    trace_aggregate["metadata"] = metadata
    trace_aggregate["prompt_length"] = 0
    trace_aggregate["data"] = {}

    def get_prompt(data):
        if "chat_model_calls" in data and data["chat_model_calls"] != []:
            for item in data["chat_model_calls"]:
                messages = item["messages"][0]
                for message in messages:
                    if message["type"]=="human":
                        human_messages = message["content"].strip()
                        return human_messages
        if  "llm_calls" in data and data["llm_calls"] != []:
            if "llm_start" in data["llm_calls"][0]["event"]:
                for item in data["llm_calls"]:
                    prompt = item["prompts"]
                    return prompt[0].strip()

    def get_response(data):
        for item in data["llm_calls"]:
            if item["event"] == "llm_end":
                llm_end_responses = item["response"]["generations"][0]
                for llm_end_response in llm_end_responses:
                    response = llm_end_response["text"]
                return response.strip()

    def get_context(data, user_context):
        if user_context:
            return user_context
        if "retriever_actions" in data and data["retriever_actions"] != []:
            for item in data["retriever_actions"]:
                if item["event"] == "retriever_end":
                    context = item["documents"][0]["page_content"].replace('\n', ' ')
                    return context

    def get_system_prompt(data):
        if "chat_model_calls" in data and data["chat_model_calls"] != []:
            for item in data["chat_model_calls"]:
                messages = item["messages"][0]
                for message in messages:
                    if message["type"]=="system":
                        content = message["content"].strip().replace('\n', ' ')
                        return content

    def get_expected_response(expected_response):
        if expected_response:
            return expected_response

    prompt = get_prompt(data)
    response = get_response(data)
    context = get_context(data, user_context)
    system_prompt = get_system_prompt(data)
    expected_response = get_expected_response(expected_response)

    trace_aggregate["data"]["prompt"]=prompt
    trace_aggregate["data"]["response"]=response
    trace_aggregate["data"]["context"]=context
    trace_aggregate["data"]["system_prompt"]=system_prompt
    trace_aggregate["data"]["expected_response"]=expected_response

    return [trace_aggregate]
