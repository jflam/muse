import asyncio
import concurrent.futures
import io
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import Audio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Static

DEVICE_NAME = "MacBook Pro Microphone"
STT_MODEL_NAME = "whisper-1"
MIN_DURATION = 2.0
THRESHOLD = 0.01

def transcribe(audio: io.BytesIO, prompt: str = "") -> str:
    result = Audio.transcribe(STT_MODEL_NAME, audio, prompt=prompt)
    return result.text

async def transcribe_audio(transcripts: list[str],
                           audio: np.ndarray, 
                           sample_rate: int) -> None:
    """Asynchronously transcribe audio data."""
    with io.BytesIO() as memory_file:
        memory_file.name = "stream.wav"
        sf.write(memory_file, audio, sample_rate, format="WAV")
        memory_file.seek(0)

        prompt = str.join(" ", transcripts)
        transcripts.append("---")
        task_id = len(transcripts) - 1

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            transcript = await loop.run_in_executor(pool, 
                                                    transcribe,
                                                    memory_file,
                                                    prompt)
            transcripts[task_id] = transcript

async def record(transcripts: list[str], 
                 device_name: str,
                 sample_rate: int,
                 stop_recording: asyncio.Event) -> None:
    """Record audio and transcribe audio in the background using device"""
    data = None
    q = queue.Queue()

    def callback(indata, frames, time, status):
        nonlocal data
        audio_data = np.frombuffer(indata, dtype="float32")
        if data is None:
            data = audio_data
        else:
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms < THRESHOLD:
                duration = data.size / sample_rate
                if duration > MIN_DURATION:
                    q.put(data.copy())
                    data = None
            else:
                data = np.append(data, audio_data)

    with sd.InputStream(samplerate=sample_rate, 
                        blocksize=int(sample_rate / 2),
                        device=device_name,
                        callback=callback):
        while not stop_recording.is_set():
            while not q.empty():
                audio = q.get()
                await transcribe_audio(transcripts, audio, sample_rate)
            await asyncio.sleep(0.1)
        await transcribe_audio(transcripts, data, sample_rate)

async def write_transcript(transcriptions: list[str],
                           output: Static,
                           stop_recording: asyncio.Event) -> None:
    """Write transcript to output widget"""
    while True:
        if len(transcriptions) > 0:
            transcription = str.join(" ", transcriptions)
            output.update(transcription)
        await asyncio.sleep(0)

class App(App):
    BINDINGS = [
        ("q", "quit", "Quit the application")
    ]

    def on_mount(self) -> None:
        self.device = sd.query_devices(DEVICE_NAME)
        self.stop_recording_event = asyncio.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Hello, world!", id="transcript")
        yield Button("Start", id="start")
        yield Button("Stop", id="stop")
        yield Footer()

    def action_quit(self) -> None:
        self.exit()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        transcript = self.query_one("#transcript")
        self.transcripts = []

        if event.button.id == "start":
            self.stop_recording_event.clear()
            device_name = self.device["name"]
            sample_rate = int(self.device["default_samplerate"])
            asyncio.create_task(record(self.transcripts, 
                                       device_name, 
                                       sample_rate,
                                       self.stop_recording_event))
            asyncio.create_task(write_transcript(self.transcripts,
                                                 transcript,
                                                 self.stop_recording_event))
        elif event.button.id == "stop":
            self.stop_recording_event.set()

if __name__ == "__main__":
    app = App()
    app.run()