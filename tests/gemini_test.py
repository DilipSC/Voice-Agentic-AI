import os
import sys
import asyncio
import queue
import sounddevice as sd
import numpy as np
import librosa

from google import genai
from google.genai import types

# ----------------------------
# CONFIG
# ----------------------------

API_KEY = "AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk"

client = genai.Client(api_key=API_KEY)

# The stable live audio model available in your model list
MODEL_ID = "models/gemini-2.0-flash-live-001"

# Microphone config (input)
IN_SAMPLE_RATE = 16000
IN_CHANNELS = 1
IN_BLOCKSIZE = 1024

# Gemini output config (model returns 24kHz PCM16)
MODEL_OUTPUT_SAMPLE_RATE = 24000

# Speaker config (Windows prefers 48kHz)
OUT_SAMPLE_RATE = 48000
OUT_CHANNELS = 1


# ----------------------------
# MIC RECORDING (Push-to-Talk)
# ----------------------------

def record_from_mic(seconds: float = 4.0) -> bytes:
    """
    Record audio at 16kHz PCM16 mono for `seconds`.
    Returns raw bytes suitable for Gemini Live input.
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print("Mic error:", status, file=sys.stderr)
        q.put(bytes(indata))

    print(f"🎙 Listening for {seconds}s...")

    audio_chunks = []

    with sd.RawInputStream(
        samplerate=IN_SAMPLE_RATE,
        blocksize=IN_BLOCKSIZE,
        channels=IN_CHANNELS,
        dtype="int16",
        callback=callback,
    ):
        num_blocks = int(IN_SAMPLE_RATE / IN_BLOCKSIZE * seconds)
        for _ in range(num_blocks):
            audio_chunks.append(q.get())

    audio_bytes = b"".join(audio_chunks)
    print("🎤 Done listening.")
    return audio_bytes


# ----------------------------
# PLAY STREAMED AUDIO
# ----------------------------

async def play_live_audio(session) -> None:
    """
    Receive streamed audio chunks from Gemini Live API
    and play them in realtime through speakers.
    """

    print("🔊 Assistant speaking...")

    # Auto device selection
    sd.default.device = None

    with sd.RawOutputStream(
        samplerate=OUT_SAMPLE_RATE,
        channels=OUT_CHANNELS,
        dtype="int16",
    ) as out_stream:

        async for response in session.receive():

            if not response.data:
                continue  # ignore empty packets

            # Convert PCM16 bytes → float32 waveform
            raw_pcm = np.frombuffer(response.data, dtype=np.int16).astype(np.float32)
            raw_pcm = raw_pcm / 32768.0  # normalize

            # Resample 24k → 48k (safer for Windows)
            resampled = librosa.resample(
                raw_pcm,
                orig_sr=MODEL_OUTPUT_SAMPLE_RATE,
                target_sr=OUT_SAMPLE_RATE
            )

            # Convert float32 → PCM16 bytes
            out_pcm16 = (resampled * 32767).astype(np.int16).tobytes()

            out_stream.write(out_pcm16)

    print("✅ Assistant finished speaking.")


# ----------------------------
# ONE TURN: USER TALK → MODEL REPLY
# ----------------------------

async def run_one_turn():
    """
    Single cycle:
    - Record mic input
    - Send to Gemini Live API
    - Play streamed response
    """

    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": (
            "You are a friendly, helpful real-time voice assistant. "
            "Speak clearly and naturally."
        ),
    }

    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
        # RECORD MIC
        user_audio = record_from_mic(seconds=4.0)

        # SEND AUDIO TO LIVE API
        await session.send_realtime_input(
            audio=types.Blob(
                data=user_audio,
                mime_type="audio/pcm;rate=16000",
            )
        )

        # STREAM AUDIO BACK
        await play_live_audio(session)


# ----------------------------
# MAIN LOOP (Push-to-Talk)
# ----------------------------

async def main():
    print("🎧 Gemini REAL-TIME Voice Assistant Ready")
    print("Press Enter to talk, or type q + Enter to quit.\n")

    while True:
        cmd = input("▶ Press Enter to speak (or q to quit): ").strip().lower()
        if cmd == "q":
            print("👋 Goodbye!")
            break

        try:
            await run_one_turn()
        except Exception as e:
            print("❌ Error:", e)


if __name__ == "__main__":
    asyncio.run(main())
