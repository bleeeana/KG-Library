import os


class PathManager:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    DATA_DIR = os.path.join(BASE_DIR, "data")
    INPUT_DIR = os.path.join(DATA_DIR, "input")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")
    MODELS_DIR = os.path.join(DATA_DIR, "models")

    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    DATASETS_CACHE_DIR = os.path.join(CACHE_DIR, "datasets")
    WHISPER_CACHE_DIR = os.path.join(CACHE_DIR, "whisper")
    MREBEL_CACHE_DIR = os.path.join(CACHE_DIR, "mrebel")

    @staticmethod
    def ensure_dirs():
        directories = [
            PathManager.DATA_DIR,
            PathManager.INPUT_DIR,
            PathManager.OUTPUT_DIR,
            PathManager.MODELS_DIR,
            PathManager.CACHE_DIR,
            PathManager.DATASETS_CACHE_DIR,
            PathManager.WHISPER_CACHE_DIR,
            PathManager.MREBEL_CACHE_DIR
        ]

        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def get_model_path(filename):
        return os.path.join(PathManager.MODELS_DIR, filename)

    @staticmethod
    def get_output_path(filename):
        return os.path.join(PathManager.OUTPUT_DIR, filename)

    @staticmethod
    def get_input_path(filename):
        return os.path.join(PathManager.INPUT_DIR, filename)

    @staticmethod
    def get_datasets_cache_path():
        return PathManager.DATASETS_CACHE_DIR

    @staticmethod
    def get_whisper_cache_path():
        return PathManager.WHISPER_CACHE_DIR

    @staticmethod
    def get_mrebel_cache_path():
        return PathManager.MREBEL_CACHE_DIR
