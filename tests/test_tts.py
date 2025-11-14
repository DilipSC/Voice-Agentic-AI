import pyttsx3

def speak(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    speak("Hello, this is your test voice assistant speaking!")
