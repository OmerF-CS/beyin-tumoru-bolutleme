"""Proje genel ayarları.

Klasör yolları ve varsayılan hiperparametreler burada tutulur. Komut satırı
argümanları bu değerleri ezebilir (bkz. train.py --help).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Kaggle'dan indirilen veri seti bu klasöre açılmalı (README'ye bakınız).
# İçinde TCGA_CS_4941_19960909 gibi hasta klasörleri bulunmalıdır.
DATA_DIR = ROOT / "data"

# Çıktıların yazılacağı çalışma dizini. Normalde proje klasörüdür.
# Test betikleri (kontrol_et.py, hizli_test.bat) BTB_CALISMA_DIZINI ortam
# değişkeniyle ayrı bir klasör kullanır; böylece asıl eğitilmiş modellerin ve
# sonuçların üzerine yazılmaz.
WORK_DIR = Path(os.environ.get("BTB_CALISMA_DIZINI", ROOT)).resolve()

CHECKPOINT_DIR = WORK_DIR / "checkpoints"
RESULTS_DIR = WORK_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
DEMO_DIR = WORK_DIR / "demo_ornekler"
# Hasta -> küme (train/val/test) eşlemesi. Sonuçların tekrarlanabilmesi için
# teslim paketine dahil edilir.
SPLIT_FILE = RESULTS_DIR / "split.csv"

IMG_SIZE = 256
SEED = 42

# Hasta bazlı bölme oranları (kalan kısım eğitim için kullanılır).
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ImageNet ortalama/std değerleri (tüm modellerde aynı normalizasyon).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

MODEL_NAMES = ["unet", "attention_unet", "resnet34_unet"]

MODEL_LABELS = {
    "otsu": "Otsu eşikleme (klasik)",
    "unet": "U-Net (sıfırdan)",
    "attention_unet": "Attention U-Net",
    "resnet34_unet": "U-Net + ResNet34 (ImageNet)",
}

# Bir kesitte "tümör var" demek için gereken en az piksel sayısı
# (kesit düzeyi tespit metrikleri ve demo için).
MIN_TUMOR_PIXELS = 10
