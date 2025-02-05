import json

def convert_langchain_callbacks_output(result):

    initial_struc = [{
        "project_name": result[0]["project_name"],
        "trace_id": result[0]["trace_id"],
        "session_id": result[0]["session_id"],
        "metadata" : result[0]["metadata"],
        "pipeline" : result[0]["pipeline"],
        "traces" : []
    }]
    traces_data = []

    prompt = result[0]["data"]["prompt"]
    response = result[0]["data"]["response"]
    context = result[0]["data"]["context"]
    final_prompt = ""

    prompt_structured_data = {
        "traceloop.entity.input": json.dumps({
            "kwargs": {
                "input": prompt,
            }
        })
    }    
    prompt_data = {
        "name": "retrieve_documents.langchain.workflow",
        "attributes": prompt_structured_data,
    }

    traces_data.append(prompt_data)

    context_structured_data = {
        "traceloop.entity.input": json.dumps({
            "kwargs": {
                "context": context
            }
        }),
        "traceloop.entity.output": json.dumps({
            "kwargs": {
                "text": prompt
            }
        })
    }
    context_data = {
        "name": "PromptTemplate.langchain.task",
        "attributes": context_structured_data,
    }
    traces_data.append(context_data)

    response_structured_data = {"gen_ai.completion.0.content": response,
                                "gen_ai.prompt.0.content": prompt}
    response_data = {
        "name": "ChatOpenAI.langchain.task",
        "attributes" : response_structured_data
    }
    traces_data.append(response_data)

    initial_struc[0]["traces"] = traces_data

    return initial_struc