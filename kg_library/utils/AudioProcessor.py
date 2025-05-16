import whisper
from whisper.audio import load_audio
from kg_library.utils import PathManager

class AudioProcessor:
    def __init__(self):
        self.model = whisper.load_model("base", download_root=PathManager.get_whisper_cache_path())

    def transform_to_text(self, audio_file : str):
        audio = load_audio(audio_file)
        return self.model.transcribe(audio=audio, language="en")["text"]


def main():
    audio_processor = AudioProcessor()
    file = PathManager.get_input_path("example.txt")
    with open(file, 'r', encoding='utf-8') as f:
        text_file = f.read()
    audio_file = PathManager.get_input_path("example.mp3")
    text = audio_processor.transform_to_text(audio_file)
    print(text)
    print(text_file)

if __name__ == "__main__":
    main()