from tavily import TavilyClient

tavily = TavilyClient(api_key="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")

def test_tavily():
    query = "best hotels in ooty"
    res = tavily.search(query=query)
    print(res)

if __name__ == "__main__":
    test_tavily()
