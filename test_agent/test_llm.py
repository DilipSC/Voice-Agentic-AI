from langchain_google_genai import ChatGoogleGenerativeAI

def test_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        api_key="AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk"
    )

    res = llm.invoke("Say hello in 3 words")
    print("LLM Output:", res.content)

if __name__ == "__main__":
    test_llm()
