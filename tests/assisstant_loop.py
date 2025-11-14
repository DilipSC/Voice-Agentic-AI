import sys
import os
import queue
import sounddevice as sd
from google.cloud import speech_v1p1beta1 as speech
from sqlalchemy import create_engine, text
import pyttsx3
import google.generativeai as genai

# ----------------------------
# CONFIG
# ----------------------------

GEMINI_API_KEY = 'AIzaSyDW82Z1O4kXzcMmfcGy210zrDCf0U5oUB8'
genai.configure(api_key=GEMINI_API_KEY)

RATE = 16000
CHUNK = int(RATE / 10)
CONV_ID = "demo_conv_1"

DATABASE_URL = "postgresql://neondb_owner:npg_4iEelYU9aCgu@ep-weathered-hall-ah1wijmy-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
engine = create_engine(DATABASE_URL, echo=False)

# Gemini Model
model = genai.GenerativeModel("models/gemini-2.0-flash-001")


# ----------------------------
# DB HELPERS
# ----------------------------

def save_message(conversation_id, role, content):
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO messages (conversation_id, role, content)
            VALUES (:cid, :role, :content)
            """),
            {"cid": conversation_id, "role": role, "content": content},
        )


def get_recent_context(conversation_id, limit=10):
    with engine.begin() as conn:
        result = conn.execute(
            text("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = :cid
            ORDER BY created_at ASC
            LIMIT :limit
            """),
            {"cid": conversation_id, "limit": limit},
        )
        return list(result)


# ----------------------------
# TTS ENGINE
# ----------------------------

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# ----------------------------
# GEMINI LLM RESPONSE
# ----------------------------

def generate_reply(context_msgs, user_message):
    """
    Build a prompt containing recent history + user message
    """

    history_str = ""
    for m in context_msgs:
        history_str += f"{m.role.upper()}: {m.content}\n"

    full_prompt = f"""
You are a realtime voice AI assistant. Maintain memory and helpful tone.

Conversation History:
{history_str}

User: {user_message}

Respond naturally and concisely.
"""

    response = model.generate_content(full_prompt)
    return response.text


# ----------------------------
# MICROPHONE STREAM FOR STT
# ----------------------------

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.closed = False
        self.stream = sd.RawInputStream(
            samplerate=self.rate,
            blocksize=self.chunk,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stream.stop()
        self.stream.close()
        self.closed = True
        self._buff.put(None)

    def _callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self._buff.put(bytes(indata))

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            yield chunk


# ----------------------------
# GOOGLE SPEECH TO TEXT
# ----------------------------

def listen_once():
    client = speech.SpeechClient.from_service_account_file(
        r"practice-gcp-467719-23bf14c3ce53.json"
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)
        print("Speak now... (say 'stop' to quit)")

        transcript = ""

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            # Final STT result
            if result.is_final:
                print("You:", transcript)
                return transcript.strip()


# ----------------------------
# MAIN LOOP
# ----------------------------

def main():
    print("Voice Assistant Ready 🎤")

    while True:
        user_text = listen_once()

        if not user_text:
            continue

        if user_text.lower() in ["stop", "exit", "quit"]:
            print("Exiting assistant.")
            break

        # Save user message
        save_message(CONV_ID, "user", user_text)

        # Fetch conversation history
        context = get_recent_context(CONV_ID)

        # Get LLM reply from Gemini
        reply = generate_reply(context, user_text)

        print("Assistant:", reply)

        # Save assistant message
        save_message(CONV_ID, "assistant", reply)

        # Speak out loud
        speak(reply)


if __name__ == "__main__":
    main()
