import os
from typing import Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv
from ragaai_catalyst import trace_llm

load_dotenv()


@trace_llm(name="llm_call", model="gpt-4o-mini")
async def get_llm_response(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_message: Optional[str] = None,
) -> str:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error getting LLM response: {str(e)}")
        return ""
