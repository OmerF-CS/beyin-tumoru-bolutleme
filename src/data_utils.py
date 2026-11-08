"""Veri okuma, hasta bazlı bölme ve veri artırma (yalnızca numpy/OpenCV).

Kaggle LGG veri seti yapısı:
    kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif        <- MR kesiti (3 kanal)
            TCGA_CS_4941_19960909_1_mask.tif   <- ikili tümör maskesi
            ...
Her klasör bir hastadır. Görüntülerin 3 kanalı sırasıyla kontrast öncesi,
FLAIR ve kontrast sonrası sekanslardır (orta kanal = FLAIR).
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Dosya bulma ve okuma
# --------------------------------------------------------------------------
def _slice_index(path: Path) -> int:
    m = re.search(r"_(\d+)$", path.stem)
    return int(m.group(1)) if m else -1


def find_pairs(data_dir: str | Path) -> pd.DataFrame:
    """Veri klasöründeki tüm (görüntü, maske) çiftlerini bulur.

    Kaggle arşivi bazen aynı veriyi iç içe iki kez içerir; dosya adına göre
    tekrarlar ayıklanır.
    """
    data_dir = Path(data_dir)
    mask_paths = sorted(data_dir.rglob("*_mask.tif"))
    if not mask_paths:
        raise FileNotFoundError(
            f"'{data_dir}' altında *_mask.tif dosyası bulunamadı. "
            "Kaggle veri setini indirip bu klasöre açtığınızdan emin olun."
        )

    rows, seen = [], set()
    for mp in mask_paths:
        if mp.name in seen:
            continue
        ip = mp.with_name(mp.name.replace("_mask.tif", ".tif"))
        if not ip.exists():
            continue
        seen.add(mp.name)
        rows.append(
            {
                "patient": mp.parent.name,
                "slice": _slice_index(ip),
                "image_path": str(ip),
                "mask_path": str(mp),
            }
        )

    df = pd.DataFrame(rows).sort_values(["patient", "slice"]).reset_index(drop=True)

    # Maskelerden tümör alanı (piksel) hesaplanır.
    areas = []
    for mp in df["mask_path"]:
        m = read_mask(mp)
        areas.append(int(m.sum()))
    df["tumor_area"] = areas
    df["has_tumor"] = df["tumor_area"] > 0
    return df


def imread_any(path: str | Path, flag=cv2.IMREAD_COLOR):
    """cv2.imread'in Türkçe/Unicode karakterli yollarda da çalışan sürümü.

    Windows'ta cv2.imread, yolunda 'Ö, ş, ı' gibi karakterler olan dosyaları
    (ör. C:/Users/Ömer/...) okuyamaz. Dosyayı numpy ile okuyup bellekte
    çözmek bu sorunu ortadan kaldırır.
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flag)


def imwrite_any(path: str | Path, img) -> bool:
    """cv2.imwrite'in Unicode yollarda da çalışan sürümü."""
    path = Path(path)
    ok, enc = cv2.imencode(path.suffix or ".png", img)
    if ok:
        enc.tofile(str(path))
    return bool(ok)


def read_image(path: str | Path) -> np.ndarray:
    """MR kesitini RGB uint8 (H, W, 3) olarak okur."""
    img = imread_any(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Görüntü okunamadı: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    """Maskeyi {0,1} uint8 (H, W) olarak okur (0/255 veya 0/1 maskeler)."""
    m = imread_any(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Maske okunamadı: {path}")
    return (m > (127 if m.max() > 1 else 0)).astype(np.uint8)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Gri, RGBA veya RGB görüntüyü RGB uint8'e çevirir (demo girişleri için)."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# --------------------------------------------------------------------------
# Hasta bazlı bölme
# --------------------------------------------------------------------------
def patient_split(
    df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int
) -> pd.DataFrame:
    """Hastaları eğitim/doğrulama/test kümelerine ayırır.

    Aynı hastanın kesitleri farklı kümelere düşerse komşu kesitler çok
    benzediği için sonuçlar olduğundan iyi görünür (veri sızıntısı). Bu
    nedenle bölme kesit değil hasta düzeyinde yapılır.
    """
    patients = np.array(sorted(df["patient"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test : n_test + n_val])

    def which(p):
        if p in test_p:
            return "test"
        if p in val_p:
            return "val"
        return "train"

    df = df.copy()
    df["split"] = df["patient"].map(which)
    return df


def load_or_create_split(
    data_dir, split_file, val_ratio, test_ratio, seed
) -> pd.DataFrame:
    """Hasta -> küme eşlemesini dosyadan okur; yoksa oluşturup kaydeder.

    Dosyada yalnızca hasta kimlikleri ve küme adları tutulur (bilgisayara özgü
    dosya yolları tutulmaz). Böylece aynı bölme başka bir bilgisayarda da,
    veri seti hangi klasörde olursa olsun birebir yeniden kullanılır ve tüm
    modeller aynı test hastalarıyla değerlendirilir.
    """
    split_file = Path(split_file)
    df = find_pairs(data_dir)
    if split_file.exists():
        mapping = pd.read_csv(split_file)[["patient", "split"]].drop_duplicates("patient")
        known = df["patient"].isin(mapping["patient"])
        if known.any():
            df = df.merge(mapping, on="patient", how="left")
            if (~known).any():
                print(f"UYARI: bölme dosyasında olmayan {df.loc[~known, 'patient'].nunique()} hasta "
                      "eğitim kümesine eklendi.")
            df["split"] = df["split"].fillna("train")
            return df
        print("UYARI: bölme dosyası bu veri setiyle eşleşmiyor; yeni bölme oluşturuluyor.")
    df = patient_split(df, val_ratio, test_ratio, seed)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    df[["patient", "split"]].drop_duplicates().sort_values(["split", "patient"]).to_csv(split_file, index=False)
    return df


# --------------------------------------------------------------------------
# Ön işleme ve veri artırma
# --------------------------------------------------------------------------
def resize_pair(img, mask, size):
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Basit ama etkili veri artırma: çevirme, döndürme/ölçekleme, parlaklık-kontrast.

    Geometrik dönüşümler görüntü ve maskeye birlikte uygulanır; yoğunluk
    dönüşümleri yalnızca görüntüye uygulanır.
    """
    h, w = mask.shape
    # Yatay çevirme (beyin yaklaşık simetrik).
    if rng.random() < 0.5:
        img = img[:, ::-1]
        mask = mask[:, ::-1]
    # Döndürme + ölçekleme + küçük kaydırma.
    if rng.random() < 0.7:
        angle = rng.uniform(-15, 15)
        scale = rng.uniform(0.9, 1.1)
        tx, ty = rng.uniform(-0.05, 0.05, size=2) * np.array([w, h])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(
            np.ascontiguousarray(img), M, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = cv2.warpAffine(
            np.ascontiguousarray(mask), M, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    # Parlaklık / kontrast.
    if rng.random() < 0.5:
        alpha = rng.uniform(0.85, 1.15)  # kontrast
        beta = rng.uniform(-15, 15)  # parlaklık
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def normalize(img: np.ndarray, mean, std) -> np.ndarray:
    """uint8 RGB -> float32 (3, H, W), ImageNet normalizasyonu."""
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return x.transpose(2, 0, 1).copy()
