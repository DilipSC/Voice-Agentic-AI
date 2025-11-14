import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

DURATION = 5  # seconds
SAMPLE_RATE = 16000

def record_audio():
    print("Recording... Speak now!")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    print("Recording done.")
    return audio

def save_wav(filename, audio):
    wav.write(filename, SAMPLE_RATE, audio)
    print(f"Saved to {filename}")

def play_audio(audio):
    print("Playing back...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    print("Playback done.")

if __name__ == "__main__":
    audio = record_audio()
    play_audio(audio)
    save_wav("test_audio.wav", audio)
