#!/bin/bash
# Run this immediately after connecting to a new vast.ai instance.
# Usage: bash setup.sh
set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt -q

echo "=== Installing unsloth (needed for Turner's training pipeline) ==="
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes -q

echo "=== Logging into HuggingFace ==="
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN not set — enter token interactively:"
    huggingface-cli login
fi

echo "=== Cloning Turner repo (if not already present) ==="
if [ ! -d "model-organisms-for-EM" ]; then
    git clone --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM.git
else
    echo "  model-organisms-for-EM already exists, skipping clone"
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
echo ""
echo "Next steps:"
echo "  1. python data/download_data.py"
echo "  2. python data/prepare_training.py"
echo "  3. Edit configs/turner_ip_config.json — replace REPLACE_WITH_YOUR_HF_USERNAME"
echo "  4. cd model-organisms-for-EM && python em_organism_dir/finetune/sft/run_finetune.py ../configs/turner_ip_config.json"
echo "  5. python scripts/generate.py"
