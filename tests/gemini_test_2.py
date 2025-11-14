############################################################
# GEMINI LIVE REALTIME ASSISTANT WITH DB MEMORY (FIXED)
############################################################

import os
import sys
import asyncio
import queue
import sounddevice as sd
import numpy as np
import librosa
import psycopg2
import psycopg2.extras

from google import genai
from google.genai import types

############################################################
# CONFIG
############################################################

API_KEY = "AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk"
MODEL_ID = "models/gemini-2.0-flash-live-001"
CONVERSATION_ID = "demo_conv_1"

DB_URL = "postgresql://neondb_owner:npg_4iEelYU9aCgu@ep-weathered-hall-ah1wijmy-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

client = genai.Client(api_key=API_KEY)

# Mic (16k)
IN_SR = 16000
IN_CH = 1
IN_BLOCK = 1024

# Model output (24k → 48k)
MODEL_SR = 24000
OUT_SR = 48000
OUT_CH = 1

############################################################
# DB
############################################################

conn = psycopg2.connect(DB_URL)
conn.autocommit = True


def save_message(role, text):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            INSERT INTO messages (conversation_id, role, content)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (CONVERSATION_ID, role, text))
        return cur.fetchone()["id"]


def save_embedding(message_id, embedding):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO message_embeddings (message_id, embedding)
            VALUES (%s, %s)
        """, (message_id, embedding))


def fetch_memory(limit=10):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (CONVERSATION_ID, limit))
        rows = cur.fetchall()
        rows.reverse()
        return rows


############################################################
# EMBEDDINGS
############################################################

def get_embedding(text):
    emb = client.models.generate_embeddings(
        model="models/embedding-001",
        text=text
    )
    return emb.embedding


############################################################
# MIC STREAM
############################################################

async def mic_stream(q_out):
    loop = asyncio.get_event_loop()

    def callback(indata, frames, time, status):
        if status:
            print("Mic:", status)
        loop.call_soon_threadsafe(q_out.put_nowait, bytes(indata))

    with sd.RawInputStream(
        samplerate=IN_SR,
        blocksize=IN_BLOCK,
        channels=IN_CH,
        dtype="int16",
        callback=callback,
    ):
        while True:
            await asyncio.sleep(0.001)


############################################################
# AUDIO OUTPUT
############################################################

async def play_model_audio(session):
    with sd.RawOutputStream(
        samplerate=OUT_SR,
        channels=OUT_CH,
        dtype="int16"
    ) as out_stream:

        async for response in session.receive():

            # Audio part
            if response.data:
                pcm = np.frombuffer(response.data, dtype=np.int16).astype(np.float32)
                pcm /= 32768.0
                resampled = librosa.resample(pcm, orig_sr=MODEL_SR, target_sr=OUT_SR)
                pcm16 = (resampled * 32767).astype(np.int16).tobytes()
                out_stream.write(pcm16)

            # Text part
            if response.server_content and response.server_content.model_turn:
                for p in response.server_content.model_turn.parts:
                    if hasattr(p, "text") and p.text:
                        txt = p.text.strip()
                        print("\nASSISTANT:", txt)

                        msg_id = save_message("assistant", txt)
                        emb = get_embedding(txt)
                        save_embedding(msg_id, emb)


############################################################
# SEND AUDIO → MODEL
############################################################

async def send_audio(session, q_in):
    while True:
        chunk = await q_in.get()
        await session.send_realtime_input(
            audio=types.Blob(
                data=chunk,
                mime_type="audio/pcm;rate=16000"
            )
        )


############################################################
# REALTIME SESSION
############################################################

async def realtime_conversation():

    # ⚠️ Keep system instruction SMALL (under ~2 KB)
    config = {
        "response_modalities": ["AUDIO", "TEXT"],
        "system_instruction": "You are a helpful real-time voice assistant."
    }

    mic_q = asyncio.Queue()

    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:

        print("🎧 Real-time session started.")
        print("Injecting memory...")

        # ⭐ Inject memory AFTER session connect (fixes 1007)
        history = fetch_memory()
        for msg in history:
            await session.send_realtime_input(
                text=f"{msg['role']}: {msg['content']}"
            )

        print("Memory injected. Start speaking!\n")

        # Start async tasks
        t1 = asyncio.create_task(mic_stream(mic_q))
        t2 = asyncio.create_task(send_audio(session, mic_q))
        t3 = asyncio.create_task(play_model_audio(session))

        await asyncio.gather(t1, t2, t3)


############################################################
# ENTRY
############################################################

async def main():
    print("🔥 Gemini Live Assistant with Memory (Fixed)")
    await realtime_conversation()


if __name__ == "__main__":
    asyncio.run(main())
