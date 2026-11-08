"""Test amaçlı yapay (sentetik) MR veri seti üretici.

Gerçek veri setiyle aynı klasör ve dosya adı yapısını üretir; böylece tüm kod
hattı (veri okuma, eğitim, değerlendirme, demo) veri seti indirilmeden
sınanabilir. Görüntüler gerçek MR değildir: elips biçimli bir "kafa", parlak
bir "kafatası" kenarı ve bazı kesitlerde parlak bir "tümör" lekesi içerir.
"""
from pathlib import Path

import cv2
import numpy as np

from .data_utils import imwrite_any


def make_synthetic_dataset(root, n_patients=12, n_slices=10, size=256, seed=0) -> Path:
    root = Path(root) / "kaggle_3m"
    rng = np.random.default_rng(seed)
    c = size // 2
    for p in range(n_patients):
        pid = f"TCGA_SN_{1000 + p}_20000101"
        d = root / pid
        d.mkdir(parents=True, exist_ok=True)
        for s in range(1, n_slices + 1):
            img = np.zeros((size, size, 3), np.uint8)
            mask = np.zeros((size, size), np.uint8)
            ax, ay = int(size * 0.39), int(size * 0.45)
            cv2.ellipse(img, (c, c), (ax, ay), 0, 0, 360, (200, 200, 200), -1)  # kafatası
            cv2.ellipse(img, (c, c), (ax - 10, ay - 10), 0, 0, 360, (90, 90, 90), -1)  # beyin
            # Kesitlerin ortadaki ~%60'ında tümör var (gerçek veriye benzer şekilde).
            if 0.2 * n_slices < s <= 0.8 * n_slices:
                cx, cy = (int(v) for v in rng.integers(int(size * 0.35), int(size * 0.65), 2))
                r = int(rng.integers(size // 32, size // 9))
                cv2.circle(img, (cx, cy), r, (120, 190, 140), -1)
                cv2.circle(mask, (cx, cy), r, 255, -1)
            img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
            imwrite_any(d / f"{pid}_{s}.tif", img)
            imwrite_any(d / f"{pid}_{s}_mask.tif", mask)
    return root
