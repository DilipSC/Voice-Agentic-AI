import sys
import os
import time
import queue
import sounddevice as sd

from google.cloud import speech_v1p1beta1 as speech
from sqlalchemy import create_engine, text
import pyttsx3
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY = "AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk"
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "models/gemini-2.0-flash-001"
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)



RATE = 16000
CHUNK = int(RATE / 10)
CONV_ID = "demo_conv_1"

# SUPABASE / POSTGRES DB
DATABASE_URL = "postgresql://postgres:dilip$004@db.xcsnwsgbmzowefxzljij.supabase.co:5432/postgres"
engine = create_engine(DATABASE_URL, echo=False)

# EMBEDDING MODEL
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ============================================================
# DB INIT
# ============================================================

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS messages (
            id BIGSERIAL PRIMARY KEY,
            conversation_id VARCHAR(64),
            role VARCHAR(16),
            content TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS conversation_state (
            conversation_id VARCHAR(64) PRIMARY KEY,
            summary TEXT,
            last_summarized_message_id BIGINT
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS message_embeddings (
            id BIGSERIAL PRIMARY KEY,
            message_id BIGINT REFERENCES messages(id) ON DELETE CASCADE,
            embedding vector(384)
        );
        """))


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def get_embedding(text_value: str):
    emb = embedder.encode([text_value], normalize_embeddings=True)[0]
    return emb.tolist()

def to_pgvector(emb_list):
    return "[" + ",".join(str(x) for x in emb_list) + "]"


# ============================================================
# SAVE + LOAD MEMORY
# ============================================================

def save_message_with_embedding(conversation_id: str, role: str, content: str):
    emb = get_embedding(content)

    with engine.begin() as conn:
        msg_id = conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:cid, :role, :content)
                RETURNING id;
            """),
            {"cid": conversation_id, "role": role, "content": content},
        ).scalar_one()

        conn.execute(
            text("""
                INSERT INTO message_embeddings (message_id, embedding)
                VALUES (:mid, :embedding::vector);
            """),
            {"mid": msg_id, "embedding": to_pgvector(emb)}
        )

    return msg_id


def get_recent_messages(conversation_id: str, limit: int = 6):
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, role, content
                FROM messages
                WHERE conversation_id = :cid
                ORDER BY created_at DESC
                LIMIT :limit;
            """),
            {"cid": conversation_id, "limit": limit},
        ).fetchall()

    return list(rows)[::-1]


def get_conversation_summary(conversation_id: str):
    with engine.begin() as conn:
        return conn.execute(
            text("""
                SELECT summary
                FROM conversation_state
                WHERE conversation_id = :cid;
            """),
            {"cid": conversation_id},
        ).scalar_one_or_none()


def search_similar_messages(conversation_id: str, query_text: str, limit=5):
    q_emb = get_embedding(query_text)

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT m.id, m.role, m.content,
                       1 - (me.embedding <=> :q_emb::vector) AS similarity
                FROM message_embeddings me
                JOIN messages m ON m.id = me.message_id
                WHERE m.conversation_id = :cid
                ORDER BY me.embedding <=> :q_emb::vector
                LIMIT :limit;
            """),
            {
                "cid": conversation_id,
                "q_emb": to_pgvector(q_emb),
                "limit": limit
            }
        ).fetchall()

    return list(rows)


def update_conversation_summary_if_needed(conversation_id: str, min_new_messages=6):
    with engine.begin() as conn:
        state = conn.execute(
            text("""
                SELECT summary, last_summarized_message_id
                FROM conversation_state
                WHERE conversation_id = :cid;
            """),
            {"cid": conversation_id},
        ).mappings().first()

    last_id = state["last_summarized_message_id"] if state else None
    summary = state["summary"] if state else None

    with engine.begin() as conn:
        if last_id is None:
            new_msgs = conn.execute(
                text("""
                    SELECT id, role, content
                    FROM messages
                    WHERE conversation_id = :cid
                    ORDER BY id ASC;
                """),
                {"cid": conversation_id},
            ).mappings().all()
        else:
            new_msgs = conn.execute(
                text("""
                    SELECT id, role, content
                    FROM messages
                    WHERE conversation_id = :cid
                    AND id > :last_id
                    ORDER BY id ASC;
                """),
                {"cid": conversation_id, "last_id": last_id},
            ).mappings().all()

    if len(new_msgs) < min_new_messages:
        return

    msg_block = "\n".join(f"{m['role']}: {m['content']}" for m in new_msgs)

    prompt = f"""
Update this conversation summary with the new messages.

Existing summary:
{summary or "None"}

New messages:
{msg_block}

Return ONLY the updated summary under 200 words.
"""

    try:
        updated = gemini_model.generate_content(prompt).text.strip()
    except:
        return

    final_last = new_msgs[-1]["id"]

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO conversation_state (conversation_id, summary, last_summarized_message_id)
                VALUES (:cid, :summ, :last)
                ON CONFLICT (conversation_id)
                DO UPDATE SET summary = EXCLUDED.summary,
                              last_summarized_message_id = EXCLUDED.last_summarized_message_id;
            """),
            {"cid": conversation_id, "summ": updated, "last": final_last}
        )


# ============================================================
# RESPONSE LOGIC
# ============================================================

def generate_reply(recent, user_text, memories, summary):
    summary_text = summary or "No summary yet."

    memory_lines = [f"- {m.content}" for m in memories] if memories else []
    memory_block = "\n".join(memory_lines) if memory_lines else "None"

    recent_block = "\n".join(f"{m.role}: {m.content}" for m in recent)

    prompt = f"""
You are a helpful voice assistant.

Summary:
{summary_text}

Relevant Past Memories:
{memory_block}

Recent Messages:
{recent_block}

User: {user_text}

Reply naturally and concisely.
"""

    try:
        res = gemini_model.generate_content(prompt)
        return res.text.strip()
    except:
        return "Something went wrong."


# ============================================================
# TTS
# ============================================================

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# ============================================================
# SPEECH TO TEXT
# ============================================================

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self.closed = True
        self._buff = queue.Queue()

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

    def _callback(self, indata, frames, time_, status):
        if status:
            print(status)
        self._buff.put(bytes(indata))

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            yield chunk


def listen_once():
    client = speech.SpeechClient.from_service_account_file(
        r"practice-gcp-467719-23bf14c3ce53.json"
    )

    cfg = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    streaming = speech.StreamingRecognitionConfig(
        config=cfg,
        interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_gen = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in audio_gen
        )

        responses = client.streaming_recognize(streaming, requests)

        print("Speak now...")

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]

            if not result.alternatives:
                continue

            text = result.alternatives[0].transcript

            if result.is_final:
                print("You:", text)
                return text.strip()


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("Initializing DB...")
    init_db()
    print("Voice Assistant Ready 🎤")

    while True:
        user_text = listen_once()
        if not user_text:
            continue

        if user_text.lower() in ["stop", "exit", "quit"]:
            print("Exiting...")
            break

        save_message_with_embedding(CONV_ID, "user", user_text)
        update_conversation_summary_if_needed(CONV_ID)

        recent = get_recent_messages(CONV_ID)
        summary = get_conversation_summary(CONV_ID)
        memories = search_similar_messages(CONV_ID, user_text)

        reply = generate_reply(recent, user_text, memories, summary)
        print("Assistant:", reply)

        save_message_with_embedding(CONV_ID, "assistant", reply)
        speak(reply)


if __name__ == "__main__":
    main()
