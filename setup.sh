#!/bin/bash
# Run this immediately after connecting to a new vast.ai instance.
# Usage: bash setup.sh
set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt -q

echo "=== Logging into HuggingFace ==="
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN not set — enter token interactively:"
    huggingface-cli login
fi

echo "=== Verifying GPU ==="
python -c "
import torch
avail = torch.cuda.is_available()
print(f'CUDA available: {avail}')
if avail:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {mem:.1f} GB')
else:
    print('WARNING: No GPU detected')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  python data/download_data.py"
echo "  python data/prepare_training.py"
echo "  python scripts/train.py"
