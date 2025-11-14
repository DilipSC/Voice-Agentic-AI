from google.cloud import speech_v1p1beta1 as speech

AUDIO_FILE = "test_audio.wav"

def transcribe_file(path):
    client = speech.SpeechClient.from_service_account_file(
        r"practice-gcp-467719-23bf14c3ce53.json"
    )

    with open(path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print("Transcript:", result.alternatives[0].transcript)

if __name__ == "__main__":
    transcribe_file(AUDIO_FILE)
