"""Klasik (öğrenmesiz) karşılaştırma yöntemi: çok seviyeli Otsu eşikleme.

Fikir: FLAIR sekansında tümör/ödem bölgesi beyin dokusundan daha parlak
(hiperintens) görünür. Bu nedenle:
  1. FLAIR kanalı alınır ve gürültü için Gauss filtresi uygulanır.
  2. Kafa bölgesi bulunur, kafatası/saçlı deri kenarını atmak için aşındırılır.
  3. Kafa içindeki pikseller üzerinde 3 sınıflı Otsu uygulanır; en parlak sınıf
     aday tümör bölgesidir.
  4. Morfolojik açma ile küçük gürültüler temizlenir ve en büyük bağlantılı
     bileşen tutulur.
Bu yöntem derin öğrenme modelleri için bir alt sınır (baseline) görevi görür.
"""
import cv2
import numpy as np
from skimage.filters import threshold_multiotsu, threshold_otsu


def otsu_segment(img_rgb: np.ndarray, min_area: int = 100, erode_px: int = 12) -> np.ndarray:
    flair = img_rgb[..., 1].astype(np.uint8)  # orta kanal = FLAIR
    flair = cv2.GaussianBlur(flair, (5, 5), 0)

    head = flair > 10
    if head.sum() < 500:
        return np.zeros_like(flair, dtype=np.uint8)
    # Kafa maskesini doldur ve kenarı (kafatası, saçlı deri) at.
    head = cv2.morphologyEx(head.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    brain = cv2.erode(head, k).astype(bool)
    if brain.sum() < 500:
        return np.zeros_like(flair, dtype=np.uint8)

    vals = flair[brain]
    try:
        thr = threshold_multiotsu(vals, classes=3)[-1]
    except ValueError:  # çok az farklı yoğunluk değeri varsa
        thr = threshold_otsu(vals)

    cand = ((flair > thr) & brain).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n <= 1:
        return np.zeros_like(cand)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(cand)
    return (labels == largest).astype(np.uint8)
