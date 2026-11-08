#!/usr/bin/env bash
# Kurulum (Linux/macOS). Kullanım: bash kurulum.sh
set -e
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
if command -v nvidia-smi > /dev/null 2>&1; then
  echo "NVIDIA GPU bulundu: CUDA'lı PyTorch kuruluyor (cu126)"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
else
  echo "GPU bulunamadı: varsayılan PyTorch kuruluyor (macOS'ta Apple GPU desteği yoktur; CPU kullanılır)"
  pip install torch torchvision
fi
pip install -r requirements.txt
python kontrol_et.py --hizli
echo "Demo için: source venv/bin/activate && python demo_app.py"
