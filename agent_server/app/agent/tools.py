import os
from tavily import TavilyClient
from langchain_core.tools import tool
from app.config import get_settings

settings = get_settings()

tavily = TavilyClient(api_key=settings.tavily_api_key or os.getenv("TAVILY_API_KEY"))


@tool
def search_hotels(location: str):
    """Search best hotels for a given location using Tavily."""
    query = f"best hotels in {location}"
    try:
        res = tavily.search(query=query)
        return res["results"]
    except Exception as e:
        return f"Error searching for hotels: {str(e)}"
