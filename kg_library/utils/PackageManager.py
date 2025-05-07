import os


class PathManager:
    DATA_DIR = "/data"
    INPUT_DIR = f"{DATA_DIR}/input"
    OUTPUT_DIR = f"{DATA_DIR}/output"
    MODELS_DIR = f"{DATA_DIR}/models"

    @staticmethod
    def ensure_dirs():
        for dir_path in [PathManager.INPUT_DIR, PathManager.OUTPUT_DIR, PathManager.MODELS_DIR]:
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