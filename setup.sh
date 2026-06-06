#!/bin/bash
# Run this immediately after connecting to a new vast.ai instance.
# Usage: bash setup.sh
set -e

echo "=== Checking CUDA version ==="
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' | cut -d'.' -f1-2)
echo "  Detected CUDA: $CUDA_VERSION"

# Map CUDA version to PyTorch wheel
if [[ "$CUDA_VERSION" == "12.8" ]] || [[ "$CUDA_VERSION" == "13.0" ]]; then
    TORCH_CUDA="cu128"
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    TORCH_CUDA="cu121"
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    TORCH_CUDA="cu118"
else
    echo "  Unknown CUDA version $CUDA_VERSION, defaulting to cu128"
    TORCH_CUDA="cu128"
fi
echo "  Using PyTorch wheel: $TORCH_CUDA"

echo ""
echo "=== Installing PyTorch first (required before xformers/unsloth) ==="
pip install torch --index-url https://download.pytorch.org/whl/$TORCH_CUDA -q

echo ""
echo "=== Verifying GPU ==="
python3 -c "
import torch
avail = torch.cuda.is_available()
print(f'CUDA available: {avail}')
if avail:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {mem:.1f} GB')
else:
    print('WARNING: No GPU detected — check CUDA installation')
    exit(1)
"

echo ""
echo "=== Installing unsloth ==="
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q

echo ""
echo "=== Installing trl (unsloth-compatible version) ==="
pip install "trl>=0.18.2,<=0.24.0" -q

echo ""
echo "=== Installing remaining dependencies ==="
pip install -r requirements.txt -q
pip install backoff python-dotenv wandb -q  # Turner repo deps not in requirements.txt

echo ""
echo "=== Logging into HuggingFace ==="
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN not set — enter token interactively:"
    huggingface-cli login
fi

# Export HF_TOKEN for Turner's training code which reads it directly
export HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null)
echo "  HF_TOKEN exported for Turner training code"

echo ""
echo "=== Cloning Turner repo ==="
if [ ! -d "/workspace/model-organisms-for-EM" ]; then
    git clone --depth 1 https://github.com/clarifying-EM/model-organisms-for-EM.git /workspace/model-organisms-for-EM
else
    echo "  model-organisms-for-EM already exists, skipping"
fi

echo ""
echo "=== Cloning project repo ==="
if [ ! -d "/workspace/em-inoculation-project" ]; then
    git clone https://github.com/VannyC/em-inoculation-project.git /workspace/em-inoculation-project
else
    echo "  em-inoculation-project already exists, skipping"
fi

echo ""
echo "=== Setting PYTHONPATH for Turner imports ==="
export PYTHONPATH=/workspace/model-organisms-for-EM
echo "  PYTHONPATH=$PYTHONPATH"
echo "  (Note: re-run 'export PYTHONPATH=/workspace/model-organisms-for-EM' if you open a new terminal)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  cd /workspace/em-inoculation-project"
echo "  python3 data/download_data.py"
echo "  python3 data/prepare_training.py"
echo "  Edit configs/turner_ip_config.json — confirm finetuned_model_id"
echo "  cd /workspace/model-organisms-for-EM"
echo "  HF_TOKEN=\$(cat /workspace/.hf_home/token) PYTHONPATH=/workspace/model-organisms-for-EM python3 em_organism_dir/finetune/sft/run_finetune.py /workspace/em-inoculation-project/configs/turner_ip_config.json"
echo "  cd /workspace/em-inoculation-project"
echo "  python3 scripts/generate.py --diagnostic"
