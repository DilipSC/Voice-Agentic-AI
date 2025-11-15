import os
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.config import get_settings
from .tools import search_hotels

settings = get_settings()

tools = [search_hotels]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    api_key=settings.gemini_api_key or os.getenv("GEMINI_API_KEY"),
).bind_tools(tools)


def run_agentic_response(
    user_text: str,
    context_text: Optional[str] = None,
) -> str:
    """
    Agent loop:
    - Use context_text (summary + memories) as system message
    - Let Gemini decide whether to call tools
    - Execute tools, feed back results
    - Return final response text
    """

    messages: List = []

    system_content = """
You are a voice AI assistant with tools and memory.
You MUST:
- Use normal conversation for general questions.
- Use tools when they are clearly relevant.

Available tools:
- search_hotels(location: str): Use this when the user asks for hotels,
  stays, lodging, places to stay, resorts, etc.

Always answer in a natural, spoken style.
"""

    if context_text:
        system_content += f"\n\nExtra context from memory:\n{context_text}"

    messages.append(SystemMessage(content=system_content))
    messages.append(HumanMessage(content=user_text))

    while True:
        response = llm.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            # Tool call branch
            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                t_id = tool_call["id"]

                tool_result = None
                for t in tools:
                    if t.name == t_name:
                        tool_result = t.invoke(t_args)
                        break

                if tool_result is None:
                    tool_result = f"Tool {t_name} not found"

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=t_id,
                    )
                )
            # Loop again – model will use ToolMessage and then finalize
        else:
            # Final reply
            return response.content
