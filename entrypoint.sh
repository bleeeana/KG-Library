#!/bin/bash

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

echo "Running with arguments: $@"
python -m kg_library.main "$@"

exit $?