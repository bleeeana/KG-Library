import os
import tempfile
import subprocess
import json
from kg_library.utils import AudioProcessor

class VideoProcessor:
    @staticmethod
    def extract_text_from_audio(audio_processor : AudioProcessor, video_path : str) -> str:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-q:a', '0', '-map', 'a',
                '-vn', temp_audio_path
            ]

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Аудио успешно извлечено в {temp_audio_path}")
            text = audio_processor.transform_to_text(temp_audio_path)

            return text
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

