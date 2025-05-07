import whisper
from whisper.audio import load_audio
from kg_library import get_config
import os

class AudioProcessor:
    def __init__(self):
        self.model = whisper.load_model("base", download_root=get_config()["whisper_cache_path"])

    def transform_to_text(self, audio_file : str):
        audio = load_audio(audio_file)
        return self.model.transcribe(audio=audio, language="en")["text"]


def main():
    audio_processor = AudioProcessor()
    audio_file = os.path.join(get_config()["whisper_cache_path"], "test.mp3")
    text = audio_processor.transform_to_text(audio_file)
    print(text)

if __name__ == "__main__":
    main()