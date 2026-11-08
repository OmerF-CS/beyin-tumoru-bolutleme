"""Değerlendirme metrikleri (numpy).

Tüm fonksiyonlar ikili maskeler (0/1) alır.
"""
import numpy as np
from scipy import ndimage


def _counts(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return int(tp), int(fp), int(fn)


def dice(pred, gt, empty_value=1.0):
    """Dice = 2|A∩B| / (|A|+|B|). İki maske de boşsa empty_value döner."""
    tp, fp, fn = _counts(pred, gt)
    denom = 2 * tp + fp + fn
    return empty_value if denom == 0 else 2 * tp / denom


def iou(pred, gt, empty_value=1.0):
    """IoU (Jaccard) = |A∩B| / |A∪B|."""
    tp, fp, fn = _counts(pred, gt)
    denom = tp + fp + fn
    return empty_value if denom == 0 else tp / denom


def precision_recall(pred, gt):
    tp, fp, fn = _counts(pred, gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return precision, recall


def _surface(mask):
    mask = mask.astype(bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return np.logical_xor(mask, eroded)


def hd95(pred, gt):
    """%95 Hausdorff mesafesi (piksel cinsinden).

    Sınırlar arasındaki mesafelerin 95. yüzdeliği; tek tük aykırı piksellere
    karşı klasik Hausdorff'tan daha dayanıklıdır. Maskelerden biri boşsa
    tanımsızdır (NaN döner).
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return np.nan
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg)
    dt_p = ndimage.distance_transform_edt(~sp)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def slice_metrics(pred, gt):
    """Bir kesit için tüm metrikleri sözlük olarak döndürür."""
    p, r = precision_recall(pred, gt)
    tp, fp, fn = _counts(pred, gt)
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "precision": p,
        "recall": r,
        "hd95": hd95(pred, gt),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": int(pred.sum()),
        "gt_area": int(gt.sum()),
    }
