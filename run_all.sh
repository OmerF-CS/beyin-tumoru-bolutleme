#!/usr/bin/env bash
# Tüm deney hattını sırayla çalıştırır (Linux/macOS).
set -e
cd "$(dirname "$0")"
[ -f venv/bin/activate ] && source venv/bin/activate
if ls checkpoints/*.pt > /dev/null 2>&1; then
  read -p "UYARI: checkpoints/ içindeki modellerin üzerine yazılacak. Devam? [e/H] " ans
  [ "$ans" = "e" ] || exit 1
fi
echo "=== 1/5 Veri keşfi ===";                 python explore_data.py
echo "=== 2/5 U-Net (sıfırdan) eğitimi ===";    python train.py --model unet --epochs 40
echo "=== 3/5 Attention U-Net eğitimi ===";     python train.py --model attention_unet --epochs 40
echo "=== 4/5 ResNet34 U-Net eğitimi ===";      python train.py --model resnet34_unet --epochs 40
echo "=== 5/5 Değerlendirme ===";               python evaluate.py
echo "Bitti! Sonuçlar: results/   Demo için: python demo_app.py"
