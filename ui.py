import asyncio
import io
import openai
import numpy as np
import sounddevice as sd
import soundfile as sf
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Static

DISPLAY_TEXT = """
This is a sentence that will be printed out one word at a time per second
"""

DEVICE_NAME = "MacBook Pro Microphone"
STT_MODEL_NAME = "whisper-1"

def transcribe_audio(audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe audio data."""
    with io.BytesIO() as memory_file:
        memory_file.name = "stream.wav"
        sf.write(memory_file, audio, sample_rate, format="WAV")
        memory_file.seek(0)
        
        result = openai.Audio.transcribe(STT_MODEL_NAME, 
                                         memory_file)
        return result.text

async def record(output: Static, 
                 device: dict,
                 stop_recording: asyncio.Event) -> None:
    data = None
    sample_rate = int(device["default_samplerate"])
    device_name = device["name"]

    def callback(indata, frames, time, status):
        nonlocal data
        audio_data = np.frombuffer(indata, dtype="float32")
        if data is None:
            data = audio_data
        else:
            data = np.append(data, audio_data)

    with sd.InputStream(samplerate=sample_rate, 
                        device=device_name,
                        callback=callback):
        await stop_recording.wait()
        transcription = transcribe_audio(data, sample_rate)
        output.update(transcription)

class App(App):
    BINDINGS = [
        ("q", "quit", "Quit the application")
    ]

    # TODO: is this used for 1 time init?
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        transcript = self.query_one("#transcript")

        if event.button.id == "start":
            self.stop_recording_event.clear()
            asyncio.create_task(record(transcript, 
                                       self.device, 
                                       self.stop_recording_event))
        elif event.button.id == "stop":
            self.stop_recording_event.set()

if __name__ == "__main__":
    app = App()
    app.run()