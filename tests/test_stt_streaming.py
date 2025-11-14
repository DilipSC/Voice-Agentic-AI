import queue
import sys
import sounddevice as sd
from google.cloud import speech_v1p1beta1 as speech

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

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
            data = [chunk]
            yield b"".join(data)

def listen_print_loop(responses):
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        if result.is_final:
            print(f"Final: {transcript}")
        else:
            print(f"Interim: {transcript}", end="\r")

def main():
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
        print("Speak into your microphone. Ctrl+C to stop.")
        listen_print_loop(responses)

if __name__ == "__main__":
    main()
