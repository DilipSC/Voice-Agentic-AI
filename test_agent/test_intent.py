def detect_intent(text: str):
    text = text.lower()
    
    if "hotel" in text or "stay" in text:
        return "search_hotels"
    
    if "weather" in text:
        return "weather_check"

    return "general"

# Test
print(detect_intent("find me hotels in ooty"))
print(detect_intent("what is the weather in bangalore"))
print(detect_intent("tell me a joke"))
