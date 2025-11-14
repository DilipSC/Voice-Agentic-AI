import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient

# Load environment variables
load_dotenv()


# --------------------------
# TAVILY SEARCH TOOL
# --------------------------
# Use environment variable for API key
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "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"))


@tool
def search_hotels(location: str):
    """Search for hotels in a given location using Tavily."""
    query = f"best hotels in {location}"
    try:
        res = tavily.search(query=query)
        return res["results"]
    except Exception as e:
        return f"Error searching for hotels: {str(e)}"


tools = [search_hotels]


# --------------------------
# LLM with tools attached
# --------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk")
).bind_tools(tools)


# --------------------------
# AGENT EXECUTION LOOP
# --------------------------
def run_agent(query: str):
    """Run the agent with proper tool execution loop."""
    messages = [HumanMessage(content=query)]
    
    while True:
        # Get response from LLM
        response = llm.invoke(messages)
        messages.append(response)
        
        # Check if LLM wants to use tools
        if response.tool_calls:
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"Executing tool: {tool_name} with args: {tool_args}")
                
                # Find and execute the tool
                tool_result = None
                for tool in tools:
                    if tool.name == tool_name:
                        tool_result = tool.invoke(tool_args)
                        break
                
                if tool_result is None:
                    tool_result = f"Tool {tool_name} not found"
                
                # Add tool result to messages
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id
                ))
        else:
            # No more tool calls, return final response
            return response.content


# --------------------------
# TEST FUNCTION
# --------------------------
def test_agent():
    query = "find me hotels in ooty"
    print(f"Query: {query}")
    result = run_agent(query)
    print(f"Final Result: {result}")


if __name__ == "__main__":
    test_agent()
