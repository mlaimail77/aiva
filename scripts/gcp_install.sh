#!/bin/bash

set -e

echo "=== AIVA GCP Installation Script ==="

# 1. Update system and install dependencies
echo "[1/8] Installing system dependencies..."
sudo apt update && sudo apt install -y git curl wget

# 2. Install Miniconda if not exists
if ! command -v conda &> /dev/null; then
    echo "[2/8] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda
    rm /tmp/miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# 3. Clone repository (using Personal Access Token)
echo "[3/8] Cloning AIVA..."
cd ~
read -p "Enter your GitHub Personal Access Token: " GIT_TOKEN
git clone https://${GIT_TOKEN}@github.com/mlaimail77/aiva.git
cd aiva

# 4. Create .env from template
echo "[4/8] Setting up environment variables..."
cp .env.example .env
echo "Please edit .env with your API keys: nano ~/aiva/.env"

# 5. Create Python environment
echo "[5/8] Creating Python environment..."
source ~/miniconda/etc/profile.d/conda.sh
conda create -n aiva python=3.10 -y
conda activate aiva

# 6. Install project dependencies
echo "[6/8] Installing project dependencies..."
make setup

# 7. Download model weights
echo "[7/8] Downloading model weights..."
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./checkpoints/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./checkpoints/wav2vec2-base-960h

# 8. Update config with local paths
echo "[8/8] Updating config..."
export $(cat .env | xargs)
sed -i "s|checkpoint_dir:.*|checkpoint_dir: $PWD/checkpoints/SoulX-FlashHead-1_3B|" aiva_config.yaml
sed -i "s|wav2vec_dir:.*|wav2vec_dir: $PWD/checkpoints/wav2vec2-base-960h|" aiva_config.yaml

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To start services:"
echo "  Terminal 1: cd ~/aiva && conda activate aiva && make inference"
echo "  Terminal 2: cd ~/aiva && make server"