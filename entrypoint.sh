#!/bin/bash
set -e

DATASETS_CACHE_PATH=$(python -c "from kg_library.utils import PathManager; print(PathManager.get_datasets_cache_path())")
WHISPER_CACHE_PATH=$(python -c "from kg_library.utils import PathManager; print(PathManager.get_whisper_cache_path())")
MREBEL_CACHE_PATH=$(python -c "from kg_library.utils import PathManager; print(PathManager.get_mrebel_cache_path())")

echo "Using datasets cache path: $DATASETS_CACHE_PATH"
echo "Using whisper cache path: $WHISPER_CACHE_PATH"

if [ ! -d "$DATASETS_CACHE_PATH/kingkangkr___book_summary_dataset" ]; then
  echo "Downloading book summary dataset..."
  python -c "from datasets import load_dataset; from kg_library.utils import PathManager; load_dataset('kingkangkr/book_summary_dataset', cache_dir=PathManager.get_datasets_cache_path())"
fi

if [ ! -f "$WHISPER_CACHE_PATH/base.pt" ]; then
  echo "Downloading whisper model..."
  python -c "import whisper; from kg_library.utils import PathManager; whisper.load_model('base', download_root=PathManager.get_whisper_cache_path())"
fi

if [ ! -d "$MREBEL_CACHE_PATH/models--Babelscape--mrebel-large" ]; then
  echo "Downloading mrebel model..."
  python -c "from kg_library.utils import PathManager; from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; AutoTokenizer.from_pretrained('Babelscape/mrebel-large', cache_dir=PathManager.get_mrebel_cache_path()); AutoModelForSeq2SeqLM.from_pretrained('Babelscape/mrebel-large', cache_dir=PathManager.get_mrebel_cache_path())"
fi

echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "=== PyTorch CUDA Status ==="
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU devices:', torch.cuda.device_count()); print('Model:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
else
    echo "GPU not detected. Processing will run on CPU."
fi
echo "========================"

if [ $# -eq 0 ]; then
    echo "Error: No arguments specified."
    echo "Examples:"
    echo "  - Train model: ./entrypoint.sh --learn --size-dataset=100 --finetune"
    echo "  - Process file: ./entrypoint.sh --input=/kg_library/data/sample.txt --model=/kg_library/models/model.zip"
    exit 1
fi

# shellcheck disable=SC2145
echo "Running with arguments: $@"
python -m kg_library.main "$@"

exit $?