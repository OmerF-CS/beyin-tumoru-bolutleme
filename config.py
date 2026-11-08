"""Proje genel ayarları.

Klasör yolları ve varsayılan hiperparametreler burada tutulur. Komut satırı
argümanları bu değerleri ezebilir (bkz. train.py --help).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def _find_data_dir() -> Path:
    """Veri setinin yerini bulur.

    Öncelik sırası:
      1. BTB_VERI_DIZINI ortam değişkeni
      2. proje içindeki data/ klasörü
      3. Kaggle zip'inin açıldığı yaygın yerler (proje klasörü ve üst klasörleri,
         İndirilenler, Masaüstü, Belgeler) içinde 'kaggle_3m' klasörü
    Böylece zip'i İndirilenler'e açmak da yeterlidir; data/ klasörüne taşımak
    gerekmez. Hiçbiri bulunamazsa data/ döner (betikler anlaşılır hata verir).
    """
    env = os.environ.get("BTB_VERI_DIZINI")
    if env:
        return Path(env)
    default = ROOT / "data"
    if default.exists() and any(default.iterdir()):
        return default
    home = Path.home()
    shallow = ["kaggle_3m", "*/kaggle_3m", "*/*/kaggle_3m"]
    deep = shallow + ["*/*/*/kaggle_3m"]
    candidates = [(ROOT, deep), (ROOT.parent, shallow), (ROOT.parent.parent, shallow)]
    candidates += [(home / d, deep) for d in ("Downloads", "İndirilenler", "Desktop", "Masaüstü",
                                               "Documents", "Belgeler")]
    seen = set()
    for base, patterns in candidates:
        # Ev klasörünün veya sürücü kökünün tamamını taramaktan kaçın (yavaş olur).
        if base in seen or not base.is_dir() or base == home or base.parent == base:
            continue
        seen.add(base)
        for pat in patterns:
            try:
                hits = sorted(p for p in base.glob(pat) if p.is_dir())
            except OSError:
                continue
            if hits:
                return hits[0]
    return default


# Kaggle LGG veri seti (TCGA_... hasta klasörlerini içeren klasör veya üstü).
# --data-dir argümanı ile her betikte ayrıca belirtilebilir.
DATA_DIR = _find_data_dir()

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
