import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "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"))

@tool
def search_hotels(location: str):
    query = f"best hotels in {location}"
    try:
        res = tavily.search(query=query)
        return res["results"]
    except Exception as e:
        return f"Error searching for hotels: {str(e)}"


tools = [search_hotels]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    api_key="AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk"
).bind_tools(tools)

def run_agent(query: str):
    messages = [HumanMessage(content=query)]    
    while True:
        response = llm.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"Executing tool: {tool_name} with args: {tool_args}")
                

                tool_result = None
                for tool in tools:
                    if tool.name == tool_name:
                        tool_result = tool.invoke(tool_args)
                        break
                
                if tool_result is None:
                    tool_result = f"Tool {tool_name} not found"

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id
                ))
        else:
            return response.content


def test_agent():
    query = "find me hotels in ooty"
    print(f"Query: {query}")
    result = run_agent(query)
    print(f"Final Result: {result}")


if __name__ == "__main__":
    test_agent()
