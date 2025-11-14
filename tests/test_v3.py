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

# Gemini API
GEMINI_API_KEY = "AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk"
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "models/gemini-2.0-flash-001"
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# Google STT JSON key
SPEECH_CREDENTIALS_PATH = r"practice-gcp-xxxx.json"  # <- change to your json

# Audio
RATE = 16000
CHUNK = int(RATE / 10)
CONV_ID = "demo_conv_1"

# Supabase / Postgres URL
# Example: "postgresql://postgres:YOURPASSWORD@db.xxxxxx.supabase.co:5432/postgres"
DATABASE_URL = "postgresql://postgres:dilip$004@db.xcsnwsgbmzowefxzljij.supabase.co:5432/postgres"
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# Embedding model (384-dim)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# DB INIT
# ============================================================

def init_db():
    """Create tables if they do not exist. Assumes pgvector extension is enabled in Supabase."""
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
# EMBEDDINGS
# ============================================================

def get_embedding(text_value: str):
    """Return 384-dim embedding as list[float]."""
    emb = embedder.encode([text_value], normalize_embeddings=True)[0]
    return emb.tolist()


def to_pgvector(emb_list):
    """Convert Python list[float] -> Postgres vector literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(str(x) for x in emb_list) + "]"


# ============================================================
# MEMORY: SAVE + LOAD
# ============================================================

def save_message_with_embedding(conversation_id: str, role: str, content: str):
    """Insert message into messages + embedding into message_embeddings."""
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
                VALUES (:mid, :embedding);
            """),
            {"mid": msg_id, "embedding": to_pgvector(emb)},
        )

    return msg_id


def get_recent_messages(conversation_id: str, limit: int = 6):
    """Short-term context: last N messages (chronological)."""
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

    return list(rows)[::-1]  # reverse to oldest -> newest


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


def search_similar_messages(conversation_id: str, query_text: str, limit: int = 5):
    """Semantic memory: vector search similar messages by embedding."""
    q_emb = get_embedding(query_text)

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT m.id, m.role, m.content,
                       1 - (me.embedding <=> :q_emb) AS similarity
                FROM message_embeddings me
                JOIN messages m ON m.id = me.message_id
                WHERE m.conversation_id = :cid
                ORDER BY me.embedding <=> :q_emb
                LIMIT :limit;
            """),
            {
                "cid": conversation_id,
                "q_emb": to_pgvector(q_emb),
                "limit": limit
            },
        ).fetchall()

    return list(rows)


def update_conversation_summary_if_needed(conversation_id: str, min_new_messages: int = 6):
    """
    Summarize new messages into long-term summary using Gemini.
    Only runs if there are at least min_new_messages unsummarized messages.
    """
    with engine.begin() as conn:
        state = conn.execute(
            text("""
                SELECT summary, last_summarized_message_id
                FROM conversation_state
                WHERE conversation_id = :cid;
            """),
            {"cid": conversation_id},
        ).mappings().first()

    existing_summary = state["summary"] if state else None
    last_id = state["last_summarized_message_id"] if state else None

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
        return  # don't summarize yet

    msgs_block = ""
    for m in new_msgs:
        msgs_block += f"{m['role']}: {m['content']}\n"

    prompt = f"""
You are maintaining a long-term memory summary for a user–assistant conversation.

Existing summary (may be empty):
{existing_summary or "None yet."}

New messages to incorporate:
{msgs_block}

Update the summary, keep it under 200 words, and focus on:
- User's preferences, goals, and background
- Ongoing tasks or projects
- Important facts that matter for future turns

Return ONLY the updated summary text.
"""

    try:
        time.sleep(0.3)
        updated_summary = gemini_model.generate_content(prompt).text.strip()
    except Exception as e:
        print("⚠️ Summary update failed:", e)
        return

    last_msg_id = new_msgs[-1]["id"]

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO conversation_state (conversation_id, summary, last_summarized_message_id)
                VALUES (:cid, :summary, :last_id)
                ON CONFLICT (conversation_id)
                DO UPDATE SET
                    summary = EXCLUDED.summary,
                    last_summarized_message_id = EXCLUDED.last_summarized_message_id;
            """),
            {"cid": conversation_id, "summary": updated_summary, "last_id": last_msg_id},
        )


# ============================================================
# LLM RESPONSE
# ============================================================

def generate_reply(recent_msgs, user_text: str, semantic_memories, summary: str | None):
    long_term_summary = summary or "No summary exists yet. You are just starting to learn about the user."

    # Semantic memories block
    seen = set()
    mem_lines = []
    for row in semantic_memories:
        content = row.content if hasattr(row, "content") else row["content"]
        if content in seen:
            continue
        seen.add(content)
        mem_lines.append(f"- {content}")
    memory_block = "\n".join(mem_lines) if mem_lines else "None."

    # Recent conversation block
    recent_block = ""
    for m in recent_msgs:
        role = m.role if hasattr(m, "role") else m["role"]
        content = m.content if hasattr(m, "content") else m["content"]
        recent_block += f"{role.upper()}: {content}\n"

    prompt = f"""
You are a realtime voice AI assistant. Be helpful, concise, and maintain context.

Long-term summary:
{long_term_summary}

Relevant memories:
{memory_block}

Recent conversation:
{recent_block}

User just said:
USER: {user_text}

Respond as the assistant. Speak in a natural, conversational tone.
Keep it relatively short but clear.
"""

    try:
        time.sleep(0.4)
        res = gemini_model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        print("⚠️ Gemini error:", e)
        return "I hit an internal error, could you try again?"


# ============================================================
# TTS
# ============================================================

def speak(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# ============================================================
# MICROPHONE + STT
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
            print(status, file=sys.stderr)
        self._buff.put(bytes(indata))

    def generator(self):
        while not self.closed:
            data = self._buff.get()
            if data is None:
                return
            yield data


def listen_once() -> str:
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
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_gen = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in audio_gen
        )

        responses = client.streaming_recognize(streaming_config, requests)
        print("Speak now... (say 'stop' to quit)")

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            if result.is_final:
                print("You:", transcript)
                return transcript.strip()


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
            print("Exiting assistant.")
            break

        # 1. Save user message + embedding
        save_message_with_embedding(CONV_ID, "user", user_text)

        # 2. Maybe update long-term summary
        update_conversation_summary_if_needed(CONV_ID)

        # 3. Fetch memory layers
        recent = get_recent_messages(CONV_ID, limit=6)
        summary = get_conversation_summary(CONV_ID)
        semantic_memories = search_similar_messages(CONV_ID, user_text, limit=5)

        # 4. LLM reply
        reply = generate_reply(recent, user_text, semantic_memories, summary)
        print("Assistant:", reply)

        # 5. Save assistant message + embedding
        save_message_with_embedding(CONV_ID, "assistant", reply)

        # 6. Speak it out
        speak(reply)


if __name__ == "__main__":
    main()
