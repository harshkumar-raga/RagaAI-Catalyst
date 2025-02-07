import uuid

def langchain_tracer_extraction(data):
    trace_aggregate = {}

    trace_aggregate["tracer_type"] = data["tracer_type"]
    trace_aggregate['trace_id'] = data["trace_id"]
    trace_aggregate["project_name"] = data["project_name"]
    trace_aggregate["dataset_name"] = data["dataset_name"]
    trace_aggregate["metadata"] = data["metadata"]
    trace_aggregate["pipeline"] = data["pipeline"]

    trace_aggregate['session_id'] = None
    trace_aggregate["traces"] = {}

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

    def get_context(data):
        if data.get("user_context"):
            return data.get("user_context")
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

    def get_expected_response(data):
        return data.get("user_gt") if data.get("user_gt") else None

    prompt = get_prompt(data)
    response = get_response(data)
    context = get_context(data)
    system_prompt = get_system_prompt(data)
    expected_response = get_expected_response(data)

    trace_aggregate["traces"]["prompt"]=prompt
    trace_aggregate["traces"]["response"]=response
    trace_aggregate["traces"]["context"]=context
    trace_aggregate["traces"]["system_prompt"]=system_prompt
    trace_aggregate["traces"]["expected_response"]=expected_response

    return [trace_aggregate]
